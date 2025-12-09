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
# 2. SSD300 Component Builder (VGG16, No BatchNorm)
# ---------------------------------------------------------

def make_conv_relu(input_id, k=3, s=1, is_dilated=False, act=True, op_type_annotation=""):
    """VGG Style Conv -> ReLU (No BN)"""
    # 1. Convolution
    if is_dilated:
        op = "Conv_3x3_Dil"
    elif k == 1:
        op = "Conv_1x1"
    elif s == 2:
        op = "Conv_3x3_DS"
    else:
        op = "Conv_3x3"
    
    c_id = add_node(op, [input_id], op_type_annotation)
    
    # 2. Activation (VGG uses ReLU)
    if act:
        out_id = add_node("ReLU", [c_id], op_type_annotation)
    else:
        out_id = c_id
    return out_id, out_id # Return (last_conv_id, last_output_id)

def make_vgg_block(input_id, num_convs, name_prefix, is_dilated=False):
    """VGG Stage: Conv_ReLU * N -> MaxPool"""
    current_id = input_id
    for i in range(num_convs):
        current_id, _ = make_conv_relu(current_id, k=3, s=1, is_dilated=is_dilated, op_type_annotation=f"{name_prefix}_{i+1}")
    
    # Max pooling for VGG stages (except modified Pool5)
    if name_prefix != "Conv5":
        pool_id = add_node("MaxPool_2x2", [current_id], f"{name_prefix}_Pool")
    else:
        # VGG Pool5 is modified: MaxPool_2x2 -> MaxPool_2x2 with k=3, s=1 (SSD paper's convention)
        # We will model it as MaxPool_SPP (Idx 9) which is 5x5, s=1, p=2 (closest structural match in the provided ops)
        pool_id = add_node("MaxPool_SPP", [current_id], f"{name_prefix}_Pool_Modified") 
        
    return pool_id

# ---------------------------------------------------------
# 3. Compose Architecture (SSD300 VGG16)
# ---------------------------------------------------------

# === Backbone: VGG16 (Modified) ===
current_id = -1
feature_maps = {} # Stores IDs for Multi-Box Heads

# Stage 1: Conv1_1, Conv1_2, Pool1
current_id = make_vgg_block(current_id, num_convs=2, name_prefix="Conv1") 

# Stage 2: Conv2_1, Conv2_2, Pool2
current_id = make_vgg_block(current_id, num_convs=2, name_prefix="Conv2")

# Stage 3: Conv3_1, Conv3_2, Conv3_3, Pool3
current_id = make_vgg_block(current_id, num_convs=3, name_prefix="Conv3")

# Stage 4: Conv4_1, Conv4_2, Conv4_3, Pool4 (Feature extraction point 1)
conv4_3_out, _ = make_conv_relu(current_id, k=3, s=1, op_type_annotation="Conv4_3_Feature") 
current_id = conv4_3_out
current_id = make_conv_relu(current_id, k=3, s=1, op_type_annotation="Conv4_4_Hidden")[1]
current_id = make_vgg_block(current_id, num_convs=0, name_prefix="Conv4") # Only pooling

# Note: VGG4_3 feature map (before L2-Norm/ReLU) is used as the first prediction layer.
# We take the output ID of the last Conv in this stage.
feature_maps['L1_Conv4_3'] = conv4_3_out
current_id = add_node("MaxPool_2x2", [current_id], "Conv4_Pool")

# Stage 5: Conv5_1, Conv5_2, Conv5_3 (Dilated Convs)
# We use Conv_3x3_Dil (Idx 4) for these
current_id, _ = make_conv_relu(current_id, k=3, s=1, is_dilated=True, op_type_annotation="Conv5_1_Dilated")
current_id, _ = make_conv_relu(current_id, k=3, s=1, is_dilated=True, op_type_annotation="Conv5_2_Dilated")
conv5_3_out, _ = make_conv_relu(current_id, k=3, s=1, is_dilated=True, op_type_annotation="Conv5_3_Dilated")
current_id = conv5_3_out
pool5_out = add_node("MaxPool_SPP", [current_id], "Conv5_Pool_Modified")
current_id = pool5_out 

# Stage 6 (FC6 equivalent, converted to Conv 3x3 Dilated)
current_id, _ = make_conv_relu(current_id, k=3, s=1, is_dilated=True, op_type_annotation="Conv6_Dilated") 

# Stage 7 (FC7 equivalent, converted to Conv 1x1) (Feature extraction point 2)
conv7_out, _ = make_conv_relu(current_id, k=1, s=1, op_type_annotation="Conv7_Feature")
feature_maps['L2_Conv7'] = conv7_out
current_id = conv7_out

# === Extra Layers (Feature Extraction Points 3, 4, 5, 6) ===

# Layer 8 (Feature extraction point 3: Conv8_2)
# Conv 1x1 s=1 -> Conv 3x3 s=2 (Downsampling)
conv8_1, _ = make_conv_relu(current_id, k=1, s=1, op_type_annotation="Conv8_1")
conv8_2, _ = make_conv_relu(conv8_1, k=3, s=2, op_type_annotation="Conv8_2_Feature")
feature_maps['L3_Conv8_2'] = conv8_2
current_id = conv8_2

# Layer 9 (Feature extraction point 4: Conv9_2)
# Conv 1x1 s=1 -> Conv 3x3 s=2 (Downsampling)
conv9_1, _ = make_conv_relu(current_id, k=1, s=1, op_type_annotation="Conv9_1")
conv9_2, _ = make_conv_relu(conv9_1, k=3, s=2, op_type_annotation="Conv9_2_Feature")
feature_maps['L4_Conv9_2'] = conv9_2
current_id = conv9_2

# Layer 10 (Feature extraction point 5: Conv10_2)
# Conv 1x1 s=1 -> Conv 3x3 s=1 (Padding 0, but s=1)
conv10_1, _ = make_conv_relu(current_id, k=1, s=1, op_type_annotation="Conv10_1")
conv10_2, _ = make_conv_relu(conv10_1, k=3, s=1, op_type_annotation="Conv10_2_Feature")
feature_maps['L5_Conv10_2'] = conv10_2
current_id = conv10_2

# Layer 11 (Feature extraction point 6: Conv11_2)
# Conv 1x1 s=1 -> Conv 3x3 s=1 (Padding 0, but s=1)
conv11_1, _ = make_conv_relu(current_id, k=1, s=1, op_type_annotation="Conv11_1")
conv11_2, _ = make_conv_relu(conv11_1, k=3, s=1, op_type_annotation="Conv11_2_Feature")
feature_maps['L6_Conv11_2'] = conv11_2
current_id = conv11_2


# === Multi-Box Head (6 parallel prediction layers) ===
# For each feature map, apply two parallel Conv 3x3 operations: one for Location, one for Classification.
ssd_outputs = []
for name, fm_id in feature_maps.items():
    # 1. Location (Box) Prediction Head
    loc_conv = make_conv_relu(fm_id, k=3, s=1, act=False, op_type_annotation=f"{name}_Loc_Conv")[0] # Final projection is a Conv without ReLU
    
    # 2. Classification Prediction Head
    cls_conv = make_conv_relu(fm_id, k=3, s=1, act=False, op_type_annotation=f"{name}_Cls_Conv")[0] # Final projection is a Conv without ReLU
    
    ssd_outputs.extend([loc_conv, cls_conv]) # Store the final output node IDs

# ---------------------------------------------------------
# 4. create JSON file
# ---------------------------------------------------------
graph_data = {
    "architecture": "SSD300_VGG16",
    "total_nodes": len(nodes),
    "total_edges": len(edges),
    "nodes": nodes,
    "edges": edges,
    "feature_map_ids": feature_maps # For verification
}

file_path = "../Graph/ssd300_vgg16_graph.json"
with open(file_path, "w", encoding="utf-8") as f:
    json.dump(graph_data, f, indent=2)

print(f"File saved: {file_path}")
print(f"Graph Stat: Nodes={len(nodes)}, Edges={len(edges)}")