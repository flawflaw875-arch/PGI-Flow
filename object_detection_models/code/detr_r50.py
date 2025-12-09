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
# 2. ResNet Component Builder (R50)
# ---------------------------------------------------------

def make_conv_bn_relu(input_id, k=3, s=1, act=True, op_type_annotation=""):
    """ResNet Style: Conv -> BN -> ReLU"""
    if k == 1 and s == 1: op = "Conv_1x1"
    elif k == 3 and s == 1: op = "Conv_3x3"
    elif k == 3 and s == 2: op = "Conv_3x3_DS"
    elif k == 7 and s == 2: op = "Conv_7x7_DS" 
    else: op = "Conv_3x3" 
    
    c_id = add_node(op, [input_id], op_type_annotation + "_Conv")
    bn_id = add_node("BatchNorm", [c_id], op_type_annotation + "_BN")
    
    if act:
        out_id = add_node("ReLU", [bn_id], op_type_annotation + "_ReLU")
    else:
        out_id = bn_id
    return out_id

def make_bottleneck_block(input_id, name_prefix, downsample=False, projection_shortcut=False):
    """ResNet Bottleneck Block"""
    s_conv = 2 if downsample else 1
    
    # Main Path
    b1_c1 = make_conv_bn_relu(input_id, k=1, s=1, op_type_annotation=f"{name_prefix}_b1_1x1_a")
    b1_c2 = make_conv_bn_relu(b1_c1, k=3, s=s_conv, op_type_annotation=f"{name_prefix}_b1_3x3")
    b1_c3 = make_conv_bn_relu(b1_c2, k=1, s=1, act=False, op_type_annotation=f"{name_prefix}_b1_1x1_b")

    # Shortcut Path
    shortcut_id = input_id
    if downsample or projection_shortcut:
        shortcut_id = make_conv_bn_relu(input_id, k=1, s=s_conv, act=False, op_type_annotation=f"{name_prefix}_shortcut_proj")

    # Merge
    add_id = add_node("Add", [shortcut_id, b1_c3], f"{name_prefix}_Add")
    out_id = add_node("ReLU", [add_id], f"{name_prefix}_ReLU")
    return out_id

def make_resnet_stage(input_id, name_prefix, num_blocks):
    current_id = input_id
    for i in range(num_blocks):
        downsample = (i == 0 and name_prefix != "C2") 
        projection_shortcut = (i == 0)
        current_id = make_bottleneck_block(current_id, f"{name_prefix}_b{i+1}", downsample, projection_shortcut)
    return current_id

# ---------------------------------------------------------
# 3. Transformer Component Builder (LayerNorm, Attention, FFN)
# ---------------------------------------------------------
# Note: In Transformer, BN/ReLU is replaced by LayerNorm/Add

def make_layer_norm_add(input_id, residual_id, name_prefix):
    """Residual Connection + LayerNorm"""
    add_id = add_node("Add", [input_id, residual_id], f"{name_prefix}_Add")
    ln_id = add_node("LayerNorm", [add_id], f"{name_prefix}_LayerNorm")
    return ln_id

def make_multi_head_attention(input_id, name_prefix):
    """Simplified MHA flow: Projections -> Reshape -> MatMul -> SoftMax -> MatMul -> Final Linear"""
    
    # 1. Linear Projections (Q, K, V) - For self or cross attention, input can be same or different.
    q = add_node("Linear", [input_id], f"{name_prefix}_Linear_Q")
    k = add_node("Linear", [input_id], f"{name_prefix}_Linear_K")
    v = add_node("Linear", [input_id], f"{name_prefix}_Linear_V")
    
    # 2. Reshape/Transpose (Prepare for MatMul)
    q_r = add_node("Reshape_Heads", [q], f"{name_prefix}_Reshape_Q")
    k_r = add_node("Reshape_Heads", [k], f"{name_prefix}_Reshape_K")
    v_r = add_node("Reshape_Heads", [v], f"{name_prefix}_Reshape_V")
    k_t = add_node("Transpose", [k_r], f"{name_prefix}_Transpose_K")
    
    # 3. MatMul (QK^T)
    qk_t = add_node("MatMul", [q_r, k_t], f"{name_prefix}_MatMul_QK")
    
    # 4. SoftMax (Attention Weights)
    attn_w = add_node("SoftMax", [qk_t], f"{name_prefix}_SoftMax")
    
    # 5. MatMul (Attention Weights * V)
    output_a = add_node("MatMul", [attn_w, v_r], f"{name_prefix}_MatMul_AV")
    
    # 6. Reshape/Transpose back (Combine heads)
    output_c = add_node("Reshape_Heads", [output_a], f"{name_prefix}_Reshape_Combined")
    
    # 7. Final Linear Projection
    output_f = add_node("Linear", [output_c], f"{name_prefix}_Linear_Final")
    return output_f

def make_ffn(input_id, name_prefix):
    """Feed-Forward Network: Linear -> ReLU -> Linear"""
    l1 = add_node("Linear", [input_id], f"{name_prefix}_FFN_Linear1")
    r = add_node("ReLU", [l1], f"{name_prefix}_FFN_ReLU")
    l2 = add_node("Linear", [r], f"{name_prefix}_FFN_Linear2")
    return l2

def make_encoder_layer(input_id, name_prefix):
    """Transformer Encoder Layer (Self-Attention + FFN)"""
    residual_in = input_id
    
    # 1. MHA Block (Self-Attention)
    attn_out = make_multi_head_attention(input_id, f"{name_prefix}_MHA")
    
    # 2. Residual + LayerNorm (Add + LayerNorm)
    ln1_out = make_layer_norm_add(attn_out, residual_in, f"{name_prefix}_LN1")
    
    # 3. FFN Block
    ffn_out = make_ffn(ln1_out, f"{name_prefix}_FFN")
    
    # 4. Residual + LayerNorm (Add + LayerNorm)
    ln2_out = make_layer_norm_add(ffn_out, ln1_out, f"{name_prefix}_LN2")
    
    return ln2_out

def make_decoder_layer(input_id, enc_output_id, name_prefix):
    """Transformer Decoder Layer (Self-Attention + Cross-Attention + FFN)"""
    residual_in_sa = input_id
    
    # 1. Masked Self-Attention Block (Attending to Queries)
    sa_out = make_multi_head_attention(input_id, f"{name_prefix}_SA") # SA for Queries
    
    # 2. Residual + LayerNorm
    ln1_out = make_layer_norm_add(sa_out, residual_in_sa, f"{name_prefix}_LN1")
    
    residual_in_ca = ln1_out
    
    # 3. Cross-Attention Block (Attending to Encoder Output)
    # Cross-Attention Q=input, K/V=Encoder output
    ca_out = make_multi_head_attention(ln1_out, f"{name_prefix}_CA") # CA Q=Input, K/V=Enc
    
    # 4. Residual + LayerNorm
    ln2_out = make_layer_norm_add(ca_out, residual_in_ca, f"{name_prefix}_LN2")
    
    # 5. FFN Block
    ffn_out = make_ffn(ln2_out, f"{name_prefix}_FFN")
    
    # 6. Residual + LayerNorm
    ln3_out = make_layer_norm_add(ffn_out, ln2_out, f"{name_prefix}_LN3")
    
    return ln3_out

# ---------------------------------------------------------
# 4. Compose Architecture (DETR R50)
# ---------------------------------------------------------

# === Backbone: ResNet-50 (Only C5 output used) ===
current_id = -1 

# Stem and C1
stem_conv = make_conv_bn_relu(current_id, k=7, s=2, op_type_annotation="Stem_7x7") 
c1_out = add_node("MaxPool_2x2", [stem_conv], "C1_Pool") 

# Stage C2, C3, C4, C5 (only final C5 output is passed to Transformer)
c2_out = make_resnet_stage(c1_out, "C2", num_blocks=3) 
c3_out = make_resnet_stage(c2_out, "C3", num_blocks=4)
c4_out = make_resnet_stage(c3_out, "C4", num_blocks=6)
c5_out = make_resnet_stage(c4_out, "C5", num_blocks=3)

# === Neck: Projection & Flatten ===
# 1x1 Conv Projection
proj_conv = add_node("Conv_1x1", [c5_out], "Backbone_Proj_1x1")
proj_bn = add_node("BatchNorm", [proj_conv], "Backbone_Proj_BN")
proj_relu = add_node("ReLU", [proj_bn], "Backbone_Proj_ReLU")

# Flatten (for input to Transformer, positional embedding addition is implicit)
enc_input_id = add_node("Flatten", [proj_relu], "Encoder_Input_Flatten") 

# Positional Encoding (Implicit input, added via Add node)
enc_input_with_pe = add_node("Add", [enc_input_id], "Encoder_Input_Add_PE")

# === Transformer Encoder (6 Layers) ===
encoder_output = enc_input_with_pe
NUM_ENCODER_LAYERS = 6
for i in range(NUM_ENCODER_LAYERS):
    encoder_output = make_encoder_layer(encoder_output, f"Encoder_L{i+1}")

# === Transformer Decoder (6 Layers) ===
# Decoder Input: Object Queries (Implicit input, modeled by Linear projection)
# The Object Queries are Learnable Parameters, often initialized via a Linear Layer.
query_embed_id = add_node("Linear", [], "Decoder_Input_ObjectQueries") # Input is empty list as it's a learned embedding

decoder_output = query_embed_id
NUM_DECODER_LAYERS = 6
enc_output_id = encoder_output # Output of the Encoder is the K/V source

for i in range(NUM_DECODER_LAYERS):
    # Cross-attention K/V source (enc_output_id) is used implicitly in the helper
    decoder_output = make_decoder_layer(decoder_output, enc_output_id, f"Decoder_L{i+1}")

# === Head: Prediction Feed-Forward Network (PFFN) ===
# Applied to the final output of the decoder (for each object query)
# PFFN: Linear -> ReLU -> Linear -> ReLU -> Final Linear Projection
pffn_inputs = [decoder_output]
detr_outputs = []

# For simplicity, we model the PFFN once, assuming the graph captures the flow applied N times (N=Queries)
# The output is parallel Classification and Regression

# Shared PFFN Stack
pffn_l1 = add_node("Linear", pffn_inputs, "PFFN_Linear1")
pffn_r1 = add_node("ReLU", [pffn_l1], "PFFN_ReLU1")
pffn_l2 = add_node("Linear", [pffn_r1], "PFFN_Linear2")
pffn_r2 = add_node("ReLU", [pffn_l2], "PFFN_ReLU2")
pffn_shared = pffn_r2 

# 1. Classification Output
cls_output = add_node("Linear", [pffn_shared], "Head_Final_Cls")
detr_outputs.append(cls_output)

# 2. Box Regression Output
reg_output = add_node("Linear", [pffn_shared], "Head_Final_Reg")
detr_outputs.append(reg_output)

# ---------------------------------------------------------
# 5. create JSON file
# ---------------------------------------------------------
graph_data = {
    "architecture": "DETR_R50",
    "total_nodes": len(nodes),
    "total_edges": len(edges),
    "nodes": nodes,
    "edges": edges,
    "final_output_ids": detr_outputs
}

file_path = "../Graph/detr_r50_graph.json"
with open(file_path, "w", encoding="utf-8") as f:
    json.dump(graph_data, f, indent=2)

print(f"File saved: {file_path}")
print(f"Graph Stat: Nodes={len(nodes)}, Edges={len(edges)}")