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
    """EfficientNet Style: Conv -> BN -> SiLU"""
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
    """Squeeze-and-Excitation Module: AvgPool -> Conv1x1 -> SiLU -> Conv1x1 -> Sigmoid -> Mul"""
    
    avg_pool = add_node("AvgPool_Global", [input_id], f"{name_prefix}_SE_AvgPool")
    
    # Reduction 1x1 Conv (linear)
    c1_reduce = add_node("Conv_1x1", [avg_pool], f"{name_prefix}_SE_Conv_Reduce")
    si = add_node("SiLU", [c1_reduce], f"{name_prefix}_SE_SiLU")
    
    # Expansion 1x1 Conv (linear)
    c2_expand = add_node("Conv_1x1", [si], f"{name_prefix}_SE_Conv_Expand")
    sig = add_node("Sigmoid", [c2_expand], f"{name_prefix}_SE_Sigmoid")
    
    # Scale: Element-wise Multiplication with input feature map
    output = add_node("Mul", [input_id, sig], f"{name_prefix}_SE_Mul") 
    return output

def make_mbconv_block(input_id, name_prefix, stride=1, use_residual=True):
    """Mobile Inverted Bottleneck Conv (MBConv) Block"""
    
    # 1. Expansion (Inverted bottleneck: 1x1 Conv -> BN -> SiLU)
    expand_conv = make_conv_bn_act(input_id, k=1, s=1, op_type_annotation=f"{name_prefix}_Expand_1x1")
    
    # 2. Depthwise Conv (DWConv_3x3 -> BN -> SiLU). Stride applies here.
    dw_conv = add_node("DWConv_3x3", [expand_conv], f"{name_prefix}_DWConv3x3")
    dw_bn = add_node("BatchNorm", [dw_conv], f"{name_prefix}_DWBN")
    
    dw_act = add_node("SiLU", [dw_bn], f"{name_prefix}_DWAct")
    
    # 3. SE Module
    se_out = make_se_module(dw_act, f"{name_prefix}_SE")
    
    # 4. Projection (Conv1x1 -> BN, NO Activation)
    proj_conv = add_node("Conv_1x1", [se_out], f"{name_prefix}_Proj_1x1")
    proj_bn = add_node("BatchNorm", [proj_conv], f"{name_prefix}_Proj_BN")
    
    # 5. Residual Connection
    if use_residual:
        add_id = add_node("Add", [input_id, proj_bn], f"{name_prefix}_Add")
        return add_id
    else:
        return proj_bn

def make_bifpn_fusion_node(inputs, name_prefix):
    """
    BiFPN Fusion Node (Top-down or Bottom-up): Weighted Sum + ReLU + DWConv3x3
    Fusion: (w1*P1 + w2*P2) / (w1 + w2 + epsilon) -> ReLU -> DWConv
    We simplify the fusion to (P1 + P2) / 2 for structural flow, but use Mul/Add/Div ops.
    We assume the two main inputs are provided in the list (in1, in2, and potentially in3 for intermediate nodes)
    """
    
    # 1. Weighted Sum (Simplified to capture Add/Mul/Div flow)
    # We use Mul for weighted inputs, Add for summation, and Div for normalization.
    
    # Assuming two main inputs for simplicity in modeling the BiFPN *node* flow
    in1, in2 = inputs[0], inputs[1]
    
    # w1 * in1 (Mul - represents weighted input)
    mul1 = add_node("Mul", [in1], f"{name_prefix}_Mul1")
    
    # w2 * in2 (Mul - represents weighted input)
    mul2 = add_node("Mul", [in2], f"{name_prefix}_Mul2")
    
    # Summation (Add)
    sum_add = add_node("Add", [mul1, mul2], f"{name_prefix}_Sum")
    
    # Normalization (Div - represents dividing by w1 + w2 + epsilon)
    norm_div = add_node("Div", [sum_add], f"{name_prefix}_Normalize")
    
    # 2. Refinement: ReLU -> DWConv3x3 -> BN -> SiLU (using MBConv projection steps)
    relu_id = add_node("ReLU", [norm_div], f"{name_prefix}_ReLU")
    
    dw_conv = add_node("DWConv_3x3", [relu_id], f"{name_prefix}_DWConv3x3")
    dw_bn = add_node("BatchNorm", [dw_conv], f"{name_prefix}_DWBN")
    
    final_out = add_node("SiLU", [dw_bn], f"{name_prefix}_SiLU")
    return final_out

# ---------------------------------------------------------
# 3. Compose Architecture (EfficientDet-D0)
# ---------------------------------------------------------

# EfficientNet B0 Stage Configuration (D0 uses 1.0 depth factor)
MB_BLOCKS = [
    (1, 1), # Stage 2 (P2)
    (2, 2), # Stage 3 (P3)
    (2, 3), # Stage 4 (P4)
    (3, 3), # Stage 5 (P5)
    (3, 4), # Stage 6 (P6)
    (4, 1), # Stage 7 (P7)
]
# P3, P4, P5 features from the backbone are used for BiFPN

# === Backbone: EfficientNet B0 ===
current_id = -1 
feature_outputs = {} # C3, C4, C5 output feature IDs

# 1. Stem (Conv 3x3 s=2)
current_id = make_conv_bn_act(current_id, k=3, s=2, op_type_annotation="Stem_3x3") 
current_id = add_node("MaxPool_2x2", [current_id], "Stem_Pool") # Initial Pool

# 2. Stage C2 (P2 feature is rarely used in EfficientDet D0 implementation, start P3)
current_id = make_mbconv_block(current_id, name_prefix="C2_MB", stride=1, use_residual=False)
current_id = make_mbconv_block(current_id, name_prefix="C2_MB2", stride=1, use_residual=False) # 2 blocks

# 3. Stage C3 (P3 feature)
c3_out = make_mbconv_block(current_id, name_prefix="C3_MB", stride=2, use_residual=False) # Downsample
current_id = c3_out
for i in range(1): # 2 blocks total
    current_id = make_mbconv_block(current_id, name_prefix=f"C3_MB_{i+2}", stride=1)
feature_outputs['C3'] = current_id

# 4. Stage C4 (P4 feature)
c4_out = make_mbconv_block(current_id, name_prefix="C4_MB", stride=2, use_residual=False) # Downsample
current_id = c4_out
for i in range(1): # 2 blocks total
    current_id = make_mbconv_block(current_id, name_prefix=f"C4_MB_{i+2}", stride=1)
feature_outputs['C4'] = current_id

# 5. Stage C5 (P5 feature)
c5_out = make_mbconv_block(current_id, name_prefix="C5_MB", stride=2, use_residual=False) # Downsample
current_id = c5_out
for i in range(2): # 3 blocks total
    current_id = make_mbconv_block(current_id, name_prefix=f"C5_MB_{i+2}", stride=1)
feature_outputs['C5'] = current_id

# P6 and P7 Extensions
# P6: Conv 3x3 s=2 on P5 output
p6_out = make_conv_bn_act(feature_outputs['C5'], k=3, s=2, op_type_annotation="P6_Downsample")
# P7: Conv 3x3 s=2 on P6 output
p7_out = make_conv_bn_act(p6_out, k=3, s=2, op_type_annotation="P7_Downsample")

# === Neck: BiFPN (Single Iteration) ===
p3_in, p4_in, p5_in, p6_in, p7_in = feature_outputs['C3'], feature_outputs['C4'], feature_outputs['C5'], p6_out, p7_out
bi_fpn_outputs = {}

# 1. Top-Down Path
# P7_td: Fusion(P7_in + Up(P8_td -> P7_in)) -> P7_out (Simplified to use only P7_in, no P8)
p7_up = add_node("Upsample_2x", [p7_in], "P7_Upsample_TD")
p6_in_td_inputs = [p6_in, p7_up] 
p6_td = make_bifpn_fusion_node(p6_in_td_inputs, name_prefix="P6_TD")

p6_up = add_node("Upsample_2x", [p6_td], "P6_Upsample_TD")
p5_in_td_inputs = [p5_in, p6_up] 
p5_td = make_bifpn_fusion_node(p5_in_td_inputs, name_prefix="P5_TD")

p5_up = add_node("Upsample_2x", [p5_td], "P5_Upsample_TD")
p4_in_td_inputs = [p4_in, p5_up] 
p4_td = make_bifpn_fusion_node(p4_in_td_inputs, name_prefix="P4_TD")

p4_up = add_node("Upsample_2x", [p4_td], "P4_Upsample_TD")
p3_in_td_inputs = [p3_in, p4_up] 
p3_out = make_bifpn_fusion_node(p3_in_td_inputs, name_prefix="P3_Final")
bi_fpn_outputs['P3'] = p3_out

# 2. Bottom-Up Path
# P4_bu: Fusion(P4_in + P4_td + Down(P3_out))
p4_down_bu = make_conv_bn_act(p3_out, k=3, s=2, op_type_annotation="P4_Downsample_BU")
p4_in_bu_inputs = [p4_in, p4_td, p4_down_bu] # Assuming 3 inputs for intermediate nodes, simplest model uses 2. Using 3 inputs to the Fusion node (Mul/Add/Div will be repeated 3 times)
p4_out = make_bifpn_fusion_node(p4_in_bu_inputs[:2], name_prefix="P4_BU") # Simplification: Use 2 dominant paths for modeling

# P5_bu
p5_down_bu = make_conv_bn_act(p4_out, k=3, s=2, op_type_annotation="P5_Downsample_BU")
p5_in_bu_inputs = [p5_in, p5_td, p5_down_bu]
p5_out = make_bifpn_fusion_node(p5_in_bu_inputs[:2], name_prefix="P5_BU")

# P6_bu
p6_down_bu = make_conv_bn_act(p5_out, k=3, s=2, op_type_annotation="P6_Downsample_BU")
p6_in_bu_inputs = [p6_in, p6_td, p6_down_bu]
p6_out = make_bifpn_fusion_node(p6_in_bu_inputs[:2], name_prefix="P6_BU")

# P7_bu
p7_down_bu = make_conv_bn_act(p6_out, k=3, s=2, op_type_annotation="P7_Downsample_BU")
p7_in_bu_inputs = [p7_in, p7_down_bu]
p7_out = make_bifpn_fusion_node(p7_in_bu_inputs, name_prefix="P7_Final")
bi_fpn_outputs['P4'] = p4_out
bi_fpn_outputs['P5'] = p5_out
bi_fpn_outputs['P6'] = p6_out
bi_fpn_outputs['P7'] = p7_out

final_fpn_features = [bi_fpn_outputs['P3'], bi_fpn_outputs['P4'], bi_fpn_outputs['P5'], bi_fpn_outputs['P6'], bi_fpn_outputs['P7']]

# === Head (Shared for Cls/Box) ===
efficientdet_outputs = []
NUM_HEAD_CONVS = 3 # D0 uses 3 conv layers

for i, fm_id in enumerate(final_fpn_features):
    scale_name = f"P{i+3}"
    
    # 1. Classification Head (Shared Conv stack)
    cls_current = fm_id
    for j in range(NUM_HEAD_CONVS):
        cls_current = make_conv_bn_act(cls_current, k=3, s=1, op_type_annotation=f"{scale_name}_Cls_Subnet_{j+1}")
    cls_final_conv = add_node("Conv_3x3", [cls_current], f"{scale_name}_Cls_Final_Conv")
    cls_output = add_node("Sigmoid", [cls_final_conv], f"{scale_name}_Cls_Sigmoid") 
    
    # 2. Box Regression Head (Shared Conv stack)
    reg_current = fm_id
    for j in range(NUM_HEAD_CONVS):
        reg_current = make_conv_bn_act(reg_current, k=3, s=1, op_type_annotation=f"{scale_name}_Reg_Subnet_{j+1}")
    reg_output = add_node("Conv_3x3", [reg_current], f"{scale_name}_Reg_Final_Conv")
    
    efficientdet_outputs.extend([cls_output, reg_output])

# ---------------------------------------------------------
# 4. create JSON file
# ---------------------------------------------------------
graph_data = {
    "architecture": "EfficientDet-D0",
    "total_nodes": len(nodes),
    "total_edges": len(edges),
    "nodes": nodes,
    "edges": edges,
    "final_output_ids": efficientdet_outputs
}

file_path = "../Graph/efficientdet_d0_graph.json"
with open(file_path, "w", encoding="utf-8") as f:
    json.dump(graph_data, f, indent=2)

print(f"File saved: {file_path}")
print(f"Graph Stat: Nodes={len(nodes)}, Edges={len(edges)}")