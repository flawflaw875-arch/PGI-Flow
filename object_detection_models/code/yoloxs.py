import json

# ---------------------------------------------------------
# 1. Common operation definitions and graph initialization
# ---------------------------------------------------------
OPS = {
    "Conv_1x1": 0, "Conv_3x3": 1, "Conv_3x3_DS": 2, "Conv_7x7_DS": 3,
    "Conv_3x3_Dil": 4, "DWConv_3x3": 5, "DWConv_3x3_DS": 6, "DWConv_5x5": 7,
    "MaxPool_2x2": 8, "MaxPool_SPP": 9, "AvgPool_Global": 10, "Upsample_2x": 11,
    "RoIAlign_7x7": 12, "BatchNorm": 13, "LayerNorm": 14, "Linear": 15,
    "ReLU": 16, "SiLU": 17, "Sigmoid": 18, "Concat": 19, "Add": 20,
    "Split_Half": 21, "Flatten": 22, "Reshape_Heads": 23, "Transpose": 24,
    "Gather_TopK": 25, "MatMul": 26, "SoftMax": 27, "Mul": 28, "Div": 29,
    "Sub": 30, "Exp": 31
}

nodes = []
edges = []
node_counter = 0

def add_node(op_name, input_ids=[], annotation=""):
    """Helper function that adds nodes to the graph and connects edges"""
    global node_counter
    current_id = node_counter
    op_idx = OPS[op_name]
    
    nodes.append({
        "id": current_id,
        "op_idx": op_idx,
        "op_name": op_name,
        "annotation": annotation
    })
    
    for inp_id in input_ids:
        if inp_id >= 0:
            edges.append([inp_id, current_id])
        
    node_counter += 1
    return current_id

# ---------------------------------------------------------
# 2. YOLOX Component Builder (Atomic Op unit decomposition)
# ---------------------------------------------------------

def make_conv_module(input_id, k=1, s=1, act=True, name_prefix=""):
    """Conv -> BN -> SiLU"""
    # 1. Convolution
    if k == 1: op = "Conv_1x1"
    elif k == 3 and s == 1: op = "Conv_3x3"
    elif k == 3 and s == 2: op = "Conv_3x3_DS"
    else: op = "Conv_3x3" 
    
    c_id = add_node(op, [input_id], f"{name_prefix}_Conv")
    
    # 2. BatchNorm
    bn_id = add_node("BatchNorm", [c_id], f"{name_prefix}_BN")
    
    # 3. Activation
    # YOLOX uses SiLU (Swish) as its standard activation
    if act:
        out_id = add_node("SiLU", [bn_id], f"{name_prefix}_SiLU")
    else:
        out_id = bn_id
    return out_id

def make_bottleneck(input_id, shortcut=True, name_prefix=""):
    """Bottleneck: Conv_1x1 -> Conv_3x3 -> Add"""
    # First Conv 1x1
    c1 = make_conv_module(input_id, k=1, s=1, name_prefix=f"{name_prefix}_cv1")
    # Second Conv 3x3
    c2 = make_conv_module(c1, k=3, s=1, name_prefix=f"{name_prefix}_cv2")
    
    if shortcut:
        add_id = add_node("Add", [input_id, c2], f"{name_prefix}_Add")
        return add_id
    else:
        return c2

def make_csp_block(input_id, n=1, name_prefix=""):
    """CSPDarknet/C3-like Module: Conv1x1 || Bottlenecks(n) -> Concat -> Conv1x1"""
    
    # 1. cv1 (1x1 Conv) - Route for residual path
    route = make_conv_module(input_id, k=1, s=1, name_prefix=f"{name_prefix}_route")
    
    # 2. cv2 (1x1 Conv) - Input for Bottlenecks
    x = make_conv_module(input_id, k=1, s=1, name_prefix=f"{name_prefix}_main")
    
    # 3. Bottlenecks (n times)
    last_tensor = x
    for i in range(n):
        # YOLOX uses a shortcut in its Bottlenecks
        last_tensor = make_bottleneck(last_tensor, shortcut=True, name_prefix=f"{name_prefix}_bottle_{i}")
        
    # 4. Concat (route + bottlenecks output)
    concat = add_node("Concat", [route, last_tensor], f"{name_prefix}_Concat")
    
    # 5. Final cv3 (1x1 Conv)
    out = make_conv_module(concat, k=1, s=1, name_prefix=f"{name_prefix}_cv3")
    return out

def make_sppf(input_id, name_prefix=""):
    """SPPF: Conv -> MaxPools -> Concat -> Conv"""
    # 1. Conv 1x1
    cv1 = make_conv_module(input_id, k=1, s=1, name_prefix=f"{name_prefix}_cv1")
    
    # 2. Sequential MaxPools (k=5)
    p1 = add_node("MaxPool_SPP", [cv1], f"{name_prefix}_pool1") 
    p2 = add_node("MaxPool_SPP", [p1], f"{name_prefix}_pool2")
    p3 = add_node("MaxPool_SPP", [p2], f"{name_prefix}_pool3")
    
    # 3. Concat (original input + 3 pooled outputs)
    concat = add_node("Concat", [cv1, p1, p2, p3], f"{name_prefix}_Concat")
    
    # 4. Final Conv 1x1
    cv2 = make_conv_module(concat, k=1, s=1, name_prefix=f"{name_prefix}_cv2")
    return cv2

# ---------------------------------------------------------
# 3. Compose Architecture (YOLOX Scaling)
# ---------------------------------------------------------

depths = [1, 3, 3, 1] # YOLOX-s

# === Backbone ===
# Stem (using Conv 3x3 s=2)
stem = make_conv_module(input_id=-1, k=3, s=2, name_prefix="Stem") 

# P2: Conv(s=2) -> CSP(n=depths[0])
p2_conv = make_conv_module(stem, k=3, s=2, name_prefix="B_P2_Conv")
p2_out = make_csp_block(p2_conv, n=depths[0], name_prefix="B_P2_CSP") # Feature P2

# P3: Conv(s=2) -> CSP(n=depths[1])
p3_conv = make_conv_module(p2_out, k=3, s=2, name_prefix="B_P3_Conv")
p3_out = make_csp_block(p3_conv, n=depths[1], name_prefix="B_P3_CSP") # Feature P3

# P4: Conv(s=2) -> CSP(n=depths[2])
p4_conv = make_conv_module(p3_out, k=3, s=2, name_prefix="B_P4_Conv")
p4_out = make_csp_block(p4_conv, n=depths[2], name_prefix="B_P4_CSP") # Feature P4

# P5: Conv(s=2) -> CSP(n=depths[0]) -> SPPF (using YOLOX's SPP structure)
p5_conv = make_conv_module(p4_out, k=3, s=2, name_prefix="B_P5_Conv")
p5_csp = make_csp_block(p5_conv, n=depths[0], name_prefix="B_P5_CSP")
# Note: YOLOX uses a simpler SPP structure, but for structural consistency 
# and given the atomic ops, we model it similar to SPPF.
p5_out = make_sppf(p5_csp, name_prefix="B_P5_SPP") 

# === Neck (PANet/FPN) ===
# 1. Upsample P5 -> Concat P4 -> CSP(n=depths[0])
up_p5 = add_node("Upsample_2x", [p5_out], "N_Up_P5")
concat_p4 = add_node("Concat", [up_p5, p4_out], "N_Concat_P4")
csp_p4_neck = make_csp_block(concat_p4, n=depths[0], name_prefix="N_P4_CSP") 

# 2. Upsample P4_neck -> Concat P3 -> CSP(n=depths[0])
up_p4 = add_node("Upsample_2x", [csp_p4_neck], "N_Up_P4")
concat_p3 = add_node("Concat", [up_p4, p3_out], "N_Concat_P3")
csp_p3_out = make_csp_block(concat_p3, n=depths[0], name_prefix="N_P3_CSP_Head") # Head Input 0 (Small)

# 3. Downsample P3_neck -> Concat P4_neck -> CSP(n=depths[0])
down_p3 = make_conv_module(csp_p3_out, k=3, s=2, name_prefix="N_Down_P3")
concat_p4_2 = add_node("Concat", [down_p3, csp_p4_neck], "N_Concat_P4_2")
csp_p4_out = make_csp_block(concat_p4_2, n=depths[0], name_prefix="N_P4_CSP_Head") # Head Input 1 (Medium)

# 4. Downsample P4_neck_out -> Concat P5 -> CSP(n=depths[0])
down_p4 = make_conv_module(csp_p4_out, k=3, s=2, name_prefix="N_Down_P4")
concat_p5_2 = add_node("Concat", [down_p4, p5_out], "N_Concat_P5_2")
csp_p5_out = make_csp_block(concat_p5_2, n=depths[0], name_prefix="N_P5_CSP_Head") # Head Input 2 (Large)

# === Head (Decoupled Detect) ===
# YOLOX uses a Decoupled Head structure (like YOLOv8) with separate Conv stacks.
head_inputs = [(csp_p3_out, "P3"), (csp_p4_out, "P4"), (csp_p5_out, "P5")]

for h_in, name in head_inputs:
    # Processing Convs (shared across branches)
    shared_cv1 = make_conv_module(h_in, k=3, s=1, name_prefix=f"H_{name}_Shared_1")
    shared_cv2 = make_conv_module(shared_cv1, k=3, s=1, name_prefix=f"H_{name}_Shared_2")

    # Box/Reg Branch (uses shared convs)
    box_out = add_node("Conv_1x1", [shared_cv2], f"H_{name}_Box_Out") # Raw Box Output
    
    # Class/Cls Branch (uses shared convs)
    cls_final = add_node("Conv_1x1", [shared_cv2], f"H_{name}_Cls_Out")
    cls_out = add_node("Sigmoid", [cls_final], f"H_{name}_Cls_Sigmoid") # Sigmoid for final classification output

# ---------------------------------------------------------
# 4. create JSON file
# ---------------------------------------------------------
graph_data = {
    "architecture": "YOLOX-s",
    "total_nodes": len(nodes),
    "total_edges": len(edges),
    "nodes": nodes,
    "edges": edges
}

file_path = "../Graph/yoloxs_graph.json"
with open(file_path, "w", encoding="utf-8") as f:
    json.dump(graph_data, f, indent=2)

print(f"File saved: {file_path}")
print(f"Graph Stat: Nodes={len(nodes)}, Edges={len(edges)}")