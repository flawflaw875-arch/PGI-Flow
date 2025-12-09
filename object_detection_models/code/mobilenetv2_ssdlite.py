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
# 2. MobileNetV2 / SSDLite Component Builder
# ---------------------------------------------------------

def make_conv_bn_relu6(input_id, k=3, s=1, dw=False, op_type_annotation=""):
    """
    Standard Conv or DWConv block with BN and ReLU6 (mapped to ReLU)
    """
    if dw:
        op = "DWConv_3x3_DS" if s == 2 else "DWConv_3x3"
    else:
        if k == 1: op = "Conv_1x1"
        elif k == 3: op = "Conv_3x3_DS" if s == 2 else "Conv_3x3"
        else: op = "Conv_3x3"

    c_id = add_node(op, [input_id], op_type_annotation + "_Conv")
    bn_id = add_node("BatchNorm", [c_id], op_type_annotation + "_BN")
    out_id = add_node("ReLU", [bn_id], op_type_annotation + "_ReLU6") # ReLU6 -> ReLU mapping
    return out_id

def make_inverted_residual(input_id, expansion_factor, stride, use_residual, name_prefix):
    """
    MobileNetV2 Inverted Residual Block:
    1x1 Expand -> 3x3 DWConv -> 1x1 Project (Linear)
    """
    current_id = input_id
    
    # 1. Expansion (1x1 Conv -> BN -> ReLU6)
    if expansion_factor > 1:
        current_id = make_conv_bn_relu6(current_id, k=1, s=1, dw=False, op_type_annotation=f"{name_prefix}_Expand")
    
    # 2. Depthwise (3x3 DW -> BN -> ReLU6)
    current_id = make_conv_bn_relu6(current_id, k=3, s=stride, dw=True, op_type_annotation=f"{name_prefix}_DW")
    
    # 3. Projection (1x1 Conv -> BN) - No Non-linearity
    proj_c = add_node("Conv_1x1", [current_id], f"{name_prefix}_Proj_Conv")
    proj_bn = add_node("BatchNorm", [proj_c], f"{name_prefix}_Proj_BN")
    
    # 4. Residual Connection
    if use_residual:
        out_id = add_node("Add", [input_id, proj_bn], f"{name_prefix}_Add")
        return out_id
    else:
        return proj_bn

def make_ssdlite_extra_layer(input_id, name_prefix):
    """
    SSDLite Extra Layer:
    Inverted Bottleneck style downsampling or simplified DW+PW block
    Typically: 1x1 Conv (reduce) -> ReLU -> 3x3 DWConv (s=2) -> ReLU -> 1x1 Conv (expand) -> ReLU
    Simplified SSDLite block often used: Conv1x1 -> BN -> ReLU -> DWConv3x3(s=2) -> BN -> ReLU -> Conv1x1 -> BN -> ReLU
    """
    # 1x1 Reduce
    l1 = make_conv_bn_relu6(input_id, k=1, s=1, dw=False, op_type_annotation=f"{name_prefix}_1x1_Reduce")
    # 3x3 DW Downsample
    l2 = make_conv_bn_relu6(l1, k=3, s=2, dw=True, op_type_annotation=f"{name_prefix}_DW_Down")
    # 1x1 Expand
    l3 = make_conv_bn_relu6(l2, k=1, s=1, dw=False, op_type_annotation=f"{name_prefix}_1x1_Expand")
    return l3

def make_separable_head(input_id, name_prefix):
    """
    SSDLite Prediction Head (Separable Conv):
    3x3 DWConv -> BN -> ReLU6 -> 1x1 Conv -> (Output)
    """
    # 1. Depthwise 3x3
    dw = add_node("DWConv_3x3", [input_id], f"{name_prefix}_Head_DW")
    dw_bn = add_node("BatchNorm", [dw], f"{name_prefix}_Head_DW_BN")
    dw_relu = add_node("ReLU", [dw_bn], f"{name_prefix}_Head_DW_ReLU")
    
    # 2. Pointwise 1x1 (Prediction)
    pw = add_node("Conv_1x1", [dw_relu], f"{name_prefix}_Head_PW_Pred")
    return pw

# ---------------------------------------------------------
# 3. Compose Architecture (MobileNetV2-SSDLite)
# ---------------------------------------------------------

# === Backbone: MobileNetV2 ===
# Config: [t, c, n, s]
# t: expansion, c: output channels, n: repeats, s: stride (first block)
mb_config = [
    [1, 16, 1, 1],
    [6, 24, 2, 2],
    [6, 32, 3, 2],
    [6, 64, 4, 2],
    [6, 96, 3, 1],
    [6, 160, 3, 2], # Feature extraction point 1 (Layer 19)
    [6, 320, 1, 1]
]

current_id = -1
feature_maps = {}

# 1. Stem (Conv 3x3 s=2)
current_id = make_conv_bn_relu6(current_id, k=3, s=2, dw=False, op_type_annotation="Stem")

# 2. Inverted Residual Blocks
block_idx = 0
for t, c, n, s in mb_config:
    for i in range(n):
        stride = s if i == 0 else 1
        # Residual connection is used if stride == 1 (channels matching is assumed implicit or handled by projection)
        use_res = (stride == 1) 
        
        current_id = make_inverted_residual(current_id, expansion_factor=t, stride=stride, use_residual=use_res, name_prefix=f"MBBlock_{block_idx}")
        block_idx += 1
        
    # Capture feature maps for SSD heads
    # Standard SSD-MobileNetV2 uses output of layer with stride 16 (c=96? No, usually expansion of it) and stride 32.
    # Typically: Expansion of block 13 (stride 16) and Block 17 (stride 32, final)
    if c == 96: # ~19x19 map
        feature_maps['L1_19x19'] = current_id # Actually often taken from expansion of next block, simplifying to this output
    if c == 320: # ~10x10 map
        # MobileNetV2 often adds a 1x1 Conv 1280 layer at the end
        conv_1280 = make_conv_bn_relu6(current_id, k=1, s=1, dw=False, op_type_annotation="Conv_1x1_1280")
        feature_maps['L2_10x10'] = conv_1280
        current_id = conv_1280

# === Extra Layers (SSDLite) ===
# 4 more layers to get 5x5, 3x3, 2x2, 1x1 maps
l3 = make_ssdlite_extra_layer(current_id, "Extra_L3")
feature_maps['L3_5x5'] = l3

l4 = make_ssdlite_extra_layer(l3, "Extra_L4")
feature_maps['L4_3x3'] = l4

l5 = make_ssdlite_extra_layer(l4, "Extra_L5")
feature_maps['L5_2x2'] = l5

l6 = make_ssdlite_extra_layer(l5, "Extra_L6")
feature_maps['L6_1x1'] = l6

# === Prediction Heads (Separable Convs) ===
outputs = []
for name, fm_id in feature_maps.items():
    # Box Head
    box_out = make_separable_head(fm_id, f"{name}_Box")
    # Class Head
    cls_conv = make_separable_head(fm_id, f"{name}_Cls")
    # Optional Sigmoid for class probability
    cls_out = add_node("Sigmoid", [cls_conv], f"{name}_Cls_Sigmoid")
    
    outputs.extend([box_out, cls_out])

# ---------------------------------------------------------
# 4. create JSON file
# ---------------------------------------------------------
graph_data = {
    "architecture": "MobileNetV2_SSDLite",
    "total_nodes": len(nodes),
    "total_edges": len(edges),
    "nodes": nodes,
    "edges": edges,
    "final_output_ids": outputs
}

file_path = "../Graph/mobilenetv2_ssdlite_graph.json"
with open(file_path, "w", encoding="utf-8") as f:
    json.dump(graph_data, f, indent=2)

print(f"File saved: {file_path}")
print(f"Graph Stat: Nodes={len(nodes)}, Edges={len(edges)}")