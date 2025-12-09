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
# 2. YOLOv8 Component Builder (Atomic Op unit decomposition)
# ---------------------------------------------------------

def make_conv_module(input_id, k=1, s=1, act=True, name_prefix=""):
    """Conv -> BN -> SiLU"""
    # 1. Convolution
    if k == 1: op = "Conv_1x1"
    elif k == 3 and s == 1: op = "Conv_3x3"
    elif k == 3 and s == 2: op = "Conv_3x3_DS"
    else: op = "Conv_3x3" # Fallback
    
    c_id = add_node(op, [input_id], f"{name_prefix}_Conv")
    
    # 2. BatchNorm
    bn_id = add_node("BatchNorm", [c_id], f"{name_prefix}_BN")
    
    # 3. Activation
    if act:
        out_id = add_node("SiLU", [bn_id], f"{name_prefix}_SiLU")
    else:
        out_id = bn_id
    return out_id

def make_bottleneck(input_id, shortcut=True, name_prefix=""):
    """Standard Bottleneck: Conv3x3 -> Conv3x3 -> Add"""
    # YOLOv8 Bottleneck: 3x3 (k=3) -> 3x3 (k=3)
    # Note: YOLOv8 uses two 3x3 convs in bottleneck usually
    
    # First Conv 3x3
    c1 = make_conv_module(input_id, k=3, s=1, name_prefix=f"{name_prefix}_cv1")
    # Second Conv 3x3
    c2 = make_conv_module(c1, k=3, s=1, name_prefix=f"{name_prefix}_cv2")
    
    if shortcut:
        add_id = add_node("Add", [input_id, c2], f"{name_prefix}_Add")
        return add_id
    else:
        return c2

def make_c2f(input_id, n=1, shortcut=True, name_prefix=""):
    """C2f Module: Split -> Bottlenecks -> Concat -> Conv"""
    # 1. cv1 (1x1 Conv)
    cv1 = make_conv_module(input_id, k=1, s=1, name_prefix=f"{name_prefix}_cv1")
    
    # 2. Split (Using Split_Half logic, assuming split into 2 * (0.5c + 0.5n))
    # In graph, we represent split as a node that produces multiple outputs implicitly
    # or just separate flows. Let's use Split_Half node.
    split_node = add_node("Split_Half", [cv1], f"{name_prefix}_Split")
    
    # YOLOv8 C2f Logic:
    # y = list(cv1(x).chunk(2, 1))
    # y.extend(m(y[-1]) for m in m)
    # z = torch.cat(y, 1)
    # return cv2(z)
    
    # Branch 0 (x0 from split)
    x0 = split_node # Conceptually first output
    # Branch 1 (x1 from split) -> goes into bottlenecks
    x1 = split_node # Conceptually second output
    
    branches = [x0, x1]
    
    # Bottlenecks
    last_tensor = x1
    for i in range(n):
        out_b = make_bottleneck(last_tensor, shortcut=shortcut, name_prefix=f"{name_prefix}_bottle_{i}")
        branches.append(out_b)
        last_tensor = out_b
        
    # Concat all branches
    concat = add_node("Concat", branches, f"{name_prefix}_Concat")
    
    # Final cv2 (1x1 Conv)
    out = make_conv_module(concat, k=1, s=1, name_prefix=f"{name_prefix}_cv2")
    return out

def make_sppf(input_id, name_prefix=""):
    """SPPF: Conv -> MaxPools -> Concat -> Conv"""
    cv1 = make_conv_module(input_id, k=1, s=1, name_prefix=f"{name_prefix}_cv1")
    
    p1 = add_node("MaxPool_SPP", [cv1], f"{name_prefix}_pool1") # k=5, s=1, p=2
    p2 = add_node("MaxPool_SPP", [p1], f"{name_prefix}_pool2")
    p3 = add_node("MaxPool_SPP", [p2], f"{name_prefix}_pool3")
    
    concat = add_node("Concat", [cv1, p1, p2, p3], f"{name_prefix}_Concat")
    cv2 = make_conv_module(concat, k=1, s=1, name_prefix=f"{name_prefix}_cv2")
    return cv2

# ---------------------------------------------------------
# 3. Compose Architecture (Graph Construction)
# ---------------------------------------------------------

# Depth config
depths = [1, 2, 2, 1]  # YOLOv8-n

# Input Placeholder (Optional, usually implied, but let's assume specific input node is not in 32 ops)
# We start with the first operation.

# === Backbone ===
# P1: Stem
# Conv 3x3, s=2
stem = make_conv_module(input_id=-1 if len(nodes)==0 else nodes[-1]['id'], k=3, s=2, name_prefix="Stem") 
# Note: For the first node, input_id is dummy or external. 
# Here we treat the first added node as receiving external input.

# P2: Conv(s=2) -> C2f(n=depths[0])
p2_conv = make_conv_module(stem, k=3, s=2, name_prefix="B_P2_Conv")
p2_out = make_c2f(p2_conv, n=depths[0], shortcut=True, name_prefix="B_P2_C2f")

# P3: Conv(s=2) -> C2f(n=depths[1])
p3_conv = make_conv_module(p2_out, k=3, s=2, name_prefix="B_P3_Conv")
p3_out = make_c2f(p3_conv, n=depths[1], shortcut=True, name_prefix="B_P3_C2f") # Feature P3

# P4: Conv(s=2) -> C2f(n=depths[2])
p4_conv = make_conv_module(p3_out, k=3, s=2, name_prefix="B_P4_Conv")
p4_out = make_c2f(p4_conv, n=depths[2], shortcut=True, name_prefix="B_P4_C2f") # Feature P4

# P5: Conv(s=2) -> C2f(n=depths[3]) -> SPPF
p5_conv = make_conv_module(p4_out, k=3, s=2, name_prefix="B_P5_Conv")
p5_c2f = make_c2f(p5_conv, n=depths[3], shortcut=True, name_prefix="B_P5_C2f")
p5_out = make_sppf(p5_c2f, name_prefix="B_P5_SPPF") # Feature P5

# === Neck (YOLOv8 PA-FPN) ===
# 1. Upsample P5 -> Concat P4
up_p5 = add_node("Upsample_2x", [p5_out], "N_Up_P5")
concat_p4 = add_node("Concat", [up_p5, p4_out], "N_Concat_P4")
c2f_p4 = make_c2f(concat_p4, n=depths[0], shortcut=False, name_prefix="N_P4_C2f") # Head Input 1 (Medium)

# 2. Upsample P4_neck -> Concat P3
up_p4 = add_node("Upsample_2x", [c2f_p4], "N_Up_P4")
concat_p3 = add_node("Concat", [up_p4, p3_out], "N_Concat_P3")
c2f_p3 = make_c2f(concat_p3, n=depths[0], shortcut=False, name_prefix="N_P3_C2f_Head") # Head Input 0 (Small) -> P3 Output

# 3. Downsample P3_neck -> Concat P4_neck
down_p3 = make_conv_module(c2f_p3, k=3, s=2, name_prefix="N_Down_P3")
concat_p4_2 = add_node("Concat", [down_p3, c2f_p4], "N_Concat_P4_2")
c2f_p4_out = make_c2f(concat_p4_2, n=depths[0], shortcut=False, name_prefix="N_P4_C2f_Head") # Head Input 1 (Medium) -> P4 Output

# 4. Downsample P4_neck_out -> Concat P5
down_p4 = make_conv_module(c2f_p4_out, k=3, s=2, name_prefix="N_Down_P4")
concat_p5_2 = add_node("Concat", [down_p4, p5_out], "N_Concat_P5_2")
c2f_p5_out = make_c2f(concat_p5_2, n=depths[0], shortcut=False, name_prefix="N_P5_C2f_Head") # Head Input 2 (Large) -> P5 Output

# === Head (Decoupled Detect) ===
# 3 Outputs: c2f_p3 (Small), c2f_p4_out (Medium), c2f_p5_out (Large)
head_inputs = [(c2f_p3, "P3"), (c2f_p4_out, "P4"), (c2f_p5_out, "P5")]
detect_outputs = []

for h_in, name in head_inputs:
    # Regression Branch (Box)
    # Conv3x3 -> Conv3x3 -> Conv1x1
    box_cv1 = make_conv_module(h_in, k=3, s=1, name_prefix=f"H_{name}_Box_cv1")
    box_cv2 = make_conv_module(box_cv1, k=3, s=1, name_prefix=f"H_{name}_Box_cv2")
    box_out = add_node("Conv_1x1", [box_cv2], f"H_{name}_Box_Out") # Raw Box Output (dist)
    
    # Class Branch (Cls)
    # Conv3x3 -> Conv3x3 -> Conv1x1 -> Sigmoid (Implicit in Loss, but graph ends at logits or sigmoid)
    cls_cv1 = make_conv_module(h_in, k=3, s=1, name_prefix=f"H_{name}_Cls_cv1")
    cls_cv2 = make_conv_module(cls_cv1, k=3, s=1, name_prefix=f"H_{name}_Cls_cv2")
    cls_final = add_node("Conv_1x1", [cls_cv2], f"H_{name}_Cls_Logits")
    cls_out = add_node("Sigmoid", [cls_final], f"H_{name}_Cls_Out") # Optional, depending on where we define 'architecture' ends
    
    detect_outputs.append({"box": box_out, "cls": cls_out})

# ---------------------------------------------------------
# 4. create JSON file
# ---------------------------------------------------------
graph_data = {
    "architecture": "YOLOv8-n",
    "total_nodes": len(nodes),
    "total_edges": len(edges),
    "nodes": nodes,
    "edges": edges
}

file_path = "../Graph/yolov8n_graph.json"
with open(file_path, "w", encoding="utf-8") as f:
    json.dump(graph_data, f, indent=2)

print(f"File saved: {file_path}")
print(f"Graph Stat: Nodes={len(nodes)}, Edges={len(edges)}")