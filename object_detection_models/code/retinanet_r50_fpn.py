import json

# ---------------------------------------------------------
# 1. 공통 연산 정의 및 그래프 초기화
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
    """그래프에 노드를 추가하고 엣지를 연결하는 헬퍼 함수"""
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
# 2. ResNet/FPN 컴포넌트 빌더 
# ---------------------------------------------------------

def make_conv_bn_relu(input_id, k=3, s=1, act=True, op_type_annotation=""):
    """ResNet Style: Conv -> BN -> ReLU (or just Conv -> BN)"""
    # 1. Convolution
    if k == 1 and s == 1: op = "Conv_1x1"
    elif k == 3 and s == 1: op = "Conv_3x3"
    elif k == 3 and s == 2: op = "Conv_3x3_DS" # For downsampling in ResNet
    elif k == 7 and s == 2: op = "Conv_7x7_DS" # For the initial stem
    else: op = "Conv_3x3" 
    
    c_id = add_node(op, [input_id], op_type_annotation + "_Conv")
    
    # 2. BatchNorm
    bn_id = add_node("BatchNorm", [c_id], op_type_annotation + "_BN")
    
    # 3. Activation
    if act:
        out_id = add_node("ReLU", [bn_id], op_type_annotation + "_ReLU")
    else:
        out_id = bn_id
    return out_id

def make_bottleneck_block(input_id, name_prefix, downsample=False, projection_shortcut=False, initial_stage=False):
    """
    ResNet Bottleneck Block: 
    1x1 -> 3x3 -> 1x1 (main path) + (identity or projection) (shortcut) -> Add -> ReLU
    """
    s_conv = 2 if downsample else 1 # Stride for the first 1x1 in the shortcut/3x3 in main
    
    # === 1. Main Path: 1x1 -> 3x3 -> 1x1 ===
    
    # 1. First 1x1 Conv (Stride is 1)
    b1_c1 = make_conv_bn_relu(input_id, k=1, s=1, op_type_annotation=f"{name_prefix}_b1_1x1_a")
    
    # 2. 3x3 Conv (Stride applies here if downsample is True)
    b1_c2 = make_conv_bn_relu(b1_c1, k=3, s=s_conv, op_type_annotation=f"{name_prefix}_b1_3x3")
    
    # 3. Second 1x1 Conv (Expansion, NO ReLU yet)
    b1_c3 = make_conv_bn_relu(b1_c2, k=1, s=1, act=False, op_type_annotation=f"{name_prefix}_b1_1x1_b")

    # === 2. Shortcut Path ===
    shortcut_id = input_id
    if downsample or projection_shortcut:
        # Projection shortcut: 1x1 Conv, s=2 if downsample is needed
        shortcut_id = make_conv_bn_relu(input_id, k=1, s=s_conv, act=False, op_type_annotation=f"{name_prefix}_shortcut_proj")

    # === 3. Merge and Final Activation ===
    
    # Add (Residual Connection)
    add_id = add_node("Add", [shortcut_id, b1_c3], f"{name_prefix}_Add")
    
    # Final ReLU
    out_id = add_node("ReLU", [add_id], f"{name_prefix}_ReLU")
    return out_id

def make_resnet_stage(input_id, name_prefix, num_blocks):
    """Builds a ResNet stage with multiple bottleneck blocks."""
    current_id = input_id
    output_ids = []
    
    for i in range(num_blocks):
        downsample = (i == 0 and name_prefix != "C2") # Downsampling happens at the first block of C3, C4, C5
        projection_shortcut = (i == 0) # Projection shortcut used at first block to match channels/dim
        
        current_id = make_bottleneck_block(
            current_id, 
            name_prefix=f"{name_prefix}_b{i+1}", 
            downsample=downsample,
            projection_shortcut=projection_shortcut
        )
        output_ids.append(current_id)
        
    return current_id, output_ids # returns final output ID and list of all block outputs

# ---------------------------------------------------------
# 3. 아키텍처 구성 (RetinaNet R50-FPN)
# ---------------------------------------------------------

# === Backbone: ResNet-50 ===
current_id = -1 

# 1. Stem (Conv 7x7 s=2 -> BN -> ReLU -> MaxPool 2x2)
# Conv 7x7 s=2
stem_conv = make_conv_bn_relu(current_id, k=7, s=2, op_type_annotation="Stem_7x7") 
# MaxPool 2x2 (MaxPool_2x2, Idx 8)
current_id = add_node("MaxPool_2x2", [stem_conv], "Stem_Pool")

# 2. Stage C2 (3 blocks)
c2_out, _ = make_resnet_stage(current_id, "C2", num_blocks=3) 
current_id = c2_out

# 3. Stage C3 (4 blocks)
c3_out, _ = make_resnet_stage(current_id, "C3", num_blocks=4)
current_id = c3_out

# 4. Stage C4 (6 blocks)
c4_out, _ = make_resnet_stage(current_id, "C4", num_blocks=6)
current_id = c4_out

# 5. Stage C5 (3 blocks)
c5_out, _ = make_resnet_stage(current_id, "C5", num_blocks=3)
c5_final = c5_out

# === Neck: Feature Pyramid Network (FPN) ===

# L1: C5 -> P5 (Lateral 1x1 Conv)
p5_lateral = make_conv_bn_relu(c5_final, k=1, s=1, act=False, op_type_annotation="P5_Lateral")

# L2: C4 -> P4 (Lateral 1x1 Conv + Top-down merge)
c4_lateral = make_conv_bn_relu(c4_out, k=1, s=1, act=False, op_type_annotation="P4_Lateral")
p5_up = add_node("Upsample_2x", [p5_lateral], "P5_Upsample")
p4_merged = add_node("Add", [c4_lateral, p5_up], "P4_Merge")
p4_final = make_conv_bn_relu(p4_merged, k=3, s=1, op_type_annotation="P4_Final") # Final P4 feature

# L3: C3 -> P3 (Lateral 1x1 Conv + Top-down merge)
c3_lateral = make_conv_bn_relu(c3_out, k=1, s=1, act=False, op_type_annotation="P3_Lateral")
p4_up = add_node("Upsample_2x", [p4_final], "P4_Upsample")
p3_merged = add_node("Add", [c3_lateral, p4_up], "P3_Merge")
p3_final = make_conv_bn_relu(p3_merged, k=3, s=1, op_type_annotation="P3_Final") # Final P3 feature

# L4: P6 (Extension from C5)
# P6 is typically P5 output using Conv 3x3 s=2 (Downsample)
p6_down = make_conv_bn_relu(c5_final, k=3, s=2, op_type_annotation="P6_Downsample")

# L5: P7 (Extension from P6)
# P7 is typically P6 output using ReLU + Conv 3x3 s=2 (Downsample)
p7_down = make_conv_bn_relu(p6_down, k=3, s=2, op_type_annotation="P7_Downsample")

fpn_outputs = [p3_final, p4_final, p5_lateral, p6_down, p7_down] # P5 uses the lateral conv output before final 3x3

# === Head: Classification and Box Regression Subnets ===
retinanet_outputs = []
NUM_HEAD_CONVS = 4 

for i, fm_id in enumerate(fpn_outputs):
    scale_name = f"P{i+3}"
    
    # 1. Classification Subnet (4 Conv3x3 -> ReLU/BN stack + final Conv3x3)
    cls_current = fm_id
    for j in range(NUM_HEAD_CONVS):
        # Conv 3x3 -> BN -> ReLU (using ResNet style module)
        cls_current = make_conv_bn_relu(cls_current, k=3, s=1, op_type_annotation=f"{scale_name}_Cls_Subnet_{j+1}")
        
    # Final Classification Projection (Conv 3x3, NO BN/ReLU)
    cls_final_conv = add_node("Conv_3x3", [cls_current], f"{scale_name}_Cls_Final_Conv")
    cls_output = add_node("Sigmoid", [cls_final_conv], f"{scale_name}_Cls_Sigmoid") # Final classification scores
    
    # 2. Box Regression Subnet (4 Conv3x3 -> ReLU/BN stack + final Conv3x3)
    reg_current = fm_id
    for j in range(NUM_HEAD_CONVS):
        # Conv 3x3 -> BN -> ReLU
        reg_current = make_conv_bn_relu(reg_current, k=3, s=1, op_type_annotation=f"{scale_name}_Reg_Subnet_{j+1}")
        
    # Final Regression Projection (Conv 3x3, NO BN/ReLU/Sigmoid)
    reg_output = add_node("Conv_3x3", [reg_current], f"{scale_name}_Reg_Final_Conv")
    
    retinanet_outputs.extend([cls_output, reg_output]) # Store 10 total final output node IDs

# ---------------------------------------------------------
# 4. JSON 파일 생성
# ---------------------------------------------------------
graph_data = {
    "architecture": "RetinaNet_R50_FPN",
    "total_nodes": len(nodes),
    "total_edges": len(edges),
    "nodes": nodes,
    "edges": edges,
    "fpn_output_ids": [id for id in fpn_outputs],
    "final_output_ids": [id for id in retinanet_outputs]
}

file_path = "retinanet_r50_fpn_graph.json"
with open(file_path, "w", encoding="utf-8") as f:
    json.dump(graph_data, f, indent=2)

print(f"File saved: {file_path}")
print(f"Graph Stat: Nodes={len(nodes)}, Edges={len(edges)}")