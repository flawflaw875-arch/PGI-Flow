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
# 2. EfficientNet/BiFPN Component Builder
# ---------------------------------------------------------

def make_conv_bn_act(input_id, k=3, s=1, act=True, op_type_annotation=""):
    """Conv -> BN -> SiLU (Standard Module)"""
    if k == 1 and s == 1: op = "Conv_1x1"
    elif k == 3 and s == 1: op = "Conv_3x3"
    elif k == 3 and s == 2: op = "Conv_3x3_DS"
    elif k == 7 and s == 2: op = "Conv_7x7_DS" 
    else: op = "Conv_3x3" 
    
    c_id = add_node(op, [input_id], op_type_annotation + "_Conv")
    bn_id = add_node("BatchNorm", [c_id], op_type_annotation + "_BN")
    
    if act:
        out_id = add_node("SiLU", [bn_id], op_type_annotation + "_SiLU")
    else:
        out_id = bn_id
    return out_id

def make_se_module(input_id, name_prefix):
    """Squeeze-and-Excitation Module"""
    avg_pool = add_node("AvgPool_Global", [input_id], f"{name_prefix}_SE_AvgPool")
    c1_reduce = add_node("Conv_1x1", [avg_pool], f"{name_prefix}_SE_Conv_Reduce")
    si = add_node("SiLU", [c1_reduce], f"{name_prefix}_SE_SiLU")
    c2_expand = add_node("Conv_1x1", [si], f"{name_prefix}_SE_Conv_Expand")
    sig = add_node("Sigmoid", [c2_expand], f"{name_prefix}_SE_Sigmoid")
    output = add_node("Mul", [input_id, sig], f"{name_prefix}_SE_Mul") 
    return output

def make_mbconv_block(input_id, name_prefix, stride=1, use_residual=True):
    """MBConv Block"""
    expand_conv = make_conv_bn_act(input_id, k=1, s=1, op_type_annotation=f"{name_prefix}_Expand_1x1")
    
    dw_conv = add_node("DWConv_3x3", [expand_conv], f"{name_prefix}_DWConv3x3")
    dw_bn = add_node("BatchNorm", [dw_conv], f"{name_prefix}_DWBN")
    dw_act = add_node("SiLU", [dw_bn], f"{name_prefix}_DWAct")
    
    se_out = make_se_module(dw_act, f"{name_prefix}_SE")
    
    proj_conv = add_node("Conv_1x1", [se_out], f"{name_prefix}_Proj_1x1")
    proj_bn = add_node("BatchNorm", [proj_conv], f"{name_prefix}_Proj_BN")
    
    if use_residual:
        add_id = add_node("Add", [input_id, proj_bn], f"{name_prefix}_Add")
        return add_id
    else:
        return proj_bn

def make_bifpn_fusion_node(inputs, name_prefix):
    """
    BiFPN Fusion Node: Weighted Sum (Mul/Add/Div) -> ReLU -> DWConv3x3 -> BN -> SiLU
    """
    
    # 1. Weighted Sum (Captures the flow of all inputs)
    current_sum = inputs[0]
    # Mul for the first input
    current_sum = add_node("Mul", [current_sum], f"{name_prefix}_Mul_0")
    
    # Process remaining inputs
    for i, inp_id in enumerate(inputs[1:]):
        mul_id = add_node("Mul", [inp_id], f"{name_prefix}_Mul_{i+1}")
        current_sum = add_node("Add", [current_sum, mul_id], f"{name_prefix}_Sum_{i+1}")
    
    # Normalization (Div)
    norm_div = add_node("Div", [current_sum], f"{name_prefix}_Normalize")
    
    # 2. Refinement
    relu_id = add_node("ReLU", [norm_div], f"{name_prefix}_ReLU")
    
    dw_conv = add_node("DWConv_3x3", [relu_id], f"{name_prefix}_DWConv3x3")
    dw_bn = add_node("BatchNorm", [dw_conv], f"{name_prefix}_DWBN")
    
    final_out = add_node("SiLU", [dw_bn], f"{name_prefix}_SiLU")
    return final_out

# ---------------------------------------------------------
# 3. Compose Architecture (EfficientDet-D3)
# ---------------------------------------------------------

# D3 Scaling (Depth ~1.6):
# Backbone MBConv blocks: [3, 3, 5, 5] (Stages C3 to C6, roughly)
# BiFPN Repetitions: 4
BIFPN_REPETITIONS = 4
NUM_HEAD_CONVS = 4 # Head layers

# === Backbone: EfficientNet B3 ===
current_id = -1 
feature_outputs = {}

# 1. Stem (Conv 3x3 s=2)
current_id = make_conv_bn_act(current_id, k=3, s=2, op_type_annotation="Stem_3x3") 
current_id = add_node("MaxPool_2x2", [current_id], "Stem_Pool") 

# Stage C2 (3 blocks)
for i in range(3): 
    current_id = make_mbconv_block(current_id, name_prefix=f"C2_MB_{i+1}", stride=1, use_residual=True)

# Stage C3 (3 blocks, P3 feature)
c3_out = make_mbconv_block(current_id, name_prefix="C3_MB_1", stride=2, use_residual=False) # Downsample
current_id = c3_out
for i in range(2): 
    current_id = make_mbconv_block(current_id, name_prefix=f"C3_MB_{i+2}", stride=1)
feature_outputs['C3'] = current_id

# Stage C4 (5 blocks, P4 feature)
c4_out = make_mbconv_block(current_id, name_prefix="C4_MB_1", stride=2, use_residual=False) # Downsample
current_id = c4_out
for i in range(4): 
    current_id = make_mbconv_block(current_id, name_prefix=f"C4_MB_{i+2}", stride=1)
feature_outputs['C4'] = current_id

# Stage C5 (5 blocks, P5 feature)
c5_out = make_mbconv_block(current_id, name_prefix="C5_MB_1", stride=2, use_residual=False) # Downsample
current_id = c5_out
for i in range(4): 
    current_id = make_mbconv_block(current_id, name_prefix=f"C5_MB_{i+2}", stride=1)
feature_outputs['C5'] = current_id

# P6 and P7 Extensions
p6_out = make_conv_bn_act(feature_outputs['C5'], k=3, s=2, op_type_annotation="P6_Downsample")
p7_out = make_conv_bn_act(p6_out, k=3, s=2, op_type_annotation="P7_Downsample")

# === Neck: BiFPN (4 Iterations) ===
current_features = {
    'P3': feature_outputs['C3'], 
    'P4': feature_outputs['C4'], 
    'P5': feature_outputs['C5'], 
    'P6': p6_out, 
    'P7': p7_out
}

# The BiFPN block is complex, we model the sequence of nodes for 4 iterations
final_features = {}

for iteration in range(BIFPN_REPETITIONS):
    iter_prefix = f"BiFPN_Iter{iteration+1}"
    
    # Store inputs for the next pass
    p3_in, p4_in, p5_in, p6_in, p7_in = current_features['P3'], current_features['P4'], current_features['P5'], current_features['P6'], current_features['P7']
    
    # 1. Top-Down Path
    # P7_td (only 2 inputs: P7_in, Down(P8_in)) - simplified to P7_in only for graph flow start
    p6_up = add_node("Upsample_2x", [p7_in], f"{iter_prefix}_P6_Upsample_TD")
    p6_td = make_bifpn_fusion_node([p6_in, p7_in, p6_up], name_prefix=f"{iter_prefix}_P6_TD") 

    p5_up = add_node("Upsample_2x", [p6_td], f"{iter_prefix}_P5_Upsample_TD")
    p5_td = make_bifpn_fusion_node([p5_in, p6_td, p5_up], name_prefix=f"{iter_prefix}_P5_TD")

    p4_up = add_node("Upsample_2x", [p5_td], f"{iter_prefix}_P4_Upsample_TD")
    p4_td = make_bifpn_fusion_node([p4_in, p5_td, p4_up], name_prefix=f"{iter_prefix}_P4_TD")

    p3_up = add_node("Upsample_2x", [p4_td], f"{iter_prefix}_P3_Upsample_TD")
    p3_out = make_bifpn_fusion_node([p3_in, p4_td, p3_up], name_prefix=f"{iter_prefix}_P3_Final")

    # 2. Bottom-Up Path (Uses P3_out and intermediate P4_td, P5_td, P6_td)
    
    # P4_bu
    p4_down_bu = make_conv_bn_act(p3_out, k=3, s=2, op_type_annotation=f"{iter_prefix}_P4_Downsample_BU")
    p4_out = make_bifpn_fusion_node([p4_in, p4_td, p4_down_bu], name_prefix=f"{iter_prefix}_P4_Final")

    # P5_bu
    p5_down_bu = make_conv_bn_act(p4_out, k=3, s=2, op_type_annotation=f"{iter_prefix}_P5_Downsample_BU")
    p5_out = make_bifpn_fusion_node([p5_in, p5_td, p5_down_bu], name_prefix=f"{iter_prefix}_P5_Final")

    # P6_bu
    p6_down_bu = make_conv_bn_act(p5_out, k=3, s=2, op_type_annotation=f"{iter_prefix}_P6_Downsample_BU")
    p6_out = make_bifpn_fusion_node([p6_in, p6_td, p6_down_bu], name_prefix=f"{iter_prefix}_P6_Final")

    # P7_bu
    p7_down_bu = make_conv_bn_act(p6_out, k=3, s=2, op_type_annotation=f"{iter_prefix}_P7_Downsample_BU")
    p7_out = make_bifpn_fusion_node([p7_in, p7_down_bu], name_prefix=f"{iter_prefix}_P7_Final") # Only 2 inputs

    # Update features for the next iteration (connects the cascade)
    current_features = {'P3': p3_out, 'P4': p4_out, 'P5': p5_out, 'P6': p6_out, 'P7': p7_out}

# === Head (Shared for Cls/Box, 4 Conv Layers) ===
final_fpn_features = [current_features['P3'], current_features['P4'], current_features['P5'], current_features['P6'], current_features['P7']]
efficientdet_outputs = []
NUM_HEAD_CONVS = 4 

for i, fm_id in enumerate(final_fpn_features):
    scale_name = f"P{i+3}"
    
    # 1. Classification Head (4 Conv stack)
    cls_current = fm_id
    for j in range(NUM_HEAD_CONVS):
        cls_current = make_conv_bn_act(cls_current, k=3, s=1, op_type_annotation=f"{scale_name}_Cls_Subnet_{j+1}")
    cls_final_conv = add_node("Conv_3x3", [cls_current], f"{scale_name}_Cls_Final_Conv")
    cls_output = add_node("Sigmoid", [cls_final_conv], f"{scale_name}_Cls_Sigmoid") 
    
    # 2. Box Regression Head (4 Conv stack)
    reg_current = fm_id
    for j in range(NUM_HEAD_CONVS):
        reg_current = make_conv_bn_act(reg_current, k=3, s=1, op_type_annotation=f"{scale_name}_Reg_Subnet_{j+1}")
    reg_output = add_node("Conv_3x3", [reg_current], f"{scale_name}_Reg_Final_Conv")
    
    efficientdet_outputs.extend([cls_output, reg_output])

# ---------------------------------------------------------
# 4. create JSON file
# ---------------------------------------------------------
graph_data = {
    "architecture": "EfficientDet-D3",
    "total_nodes": len(nodes),
    "total_edges": len(edges),
    "nodes": nodes,
    "edges": edges,
    "final_output_ids": efficientdet_outputs
}

file_path = "../Graph/efficientdet_d3_graph.json"
with open(file_path, "w", encoding="utf-8") as f:
    json.dump(graph_data, f, indent=2)

print(f"File saved: {file_path}")
print(f"Graph Stat: Nodes={len(nodes)}, Edges={len(edges)}")