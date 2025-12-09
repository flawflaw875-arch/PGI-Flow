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
# 2. ResNet/FPN/Head 컴포넌트 빌더 
# ---------------------------------------------------------

def make_conv_bn_relu(input_id, k=3, s=1, act=True, op_type_annotation=""):
    """ResNet Style: Conv -> BN -> ReLU (or just Conv -> BN)"""
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
    """ResNet Bottleneck Block: 1x1 -> 3x3 -> 1x1 + (shortcut) -> Add -> ReLU"""
    s_conv = 2 if downsample else 1
    
    # 1. Main Path
    b1_c1 = make_conv_bn_relu(input_id, k=1, s=1, op_type_annotation=f"{name_prefix}_b1_1x1_a")
    b1_c2 = make_conv_bn_relu(b1_c1, k=3, s=s_conv, op_type_annotation=f"{name_prefix}_b1_3x3")
    b1_c3 = make_conv_bn_relu(b1_c2, k=1, s=1, act=False, op_type_annotation=f"{name_prefix}_b1_1x1_b")

    # 2. Shortcut Path
    shortcut_id = input_id
    if downsample or projection_shortcut:
        shortcut_id = make_conv_bn_relu(input_id, k=1, s=s_conv, act=False, op_type_annotation=f"{name_prefix}_shortcut_proj")

    # 3. Merge and Final Activation
    add_id = add_node("Add", [shortcut_id, b1_c3], f"{name_prefix}_Add")
    out_id = add_node("ReLU", [add_id], f"{name_prefix}_ReLU")
    return out_id

def make_resnet_stage(input_id, name_prefix, num_blocks):
    """Builds a ResNet stage with multiple bottleneck blocks."""
    current_id = input_id
    
    for i in range(num_blocks):
        # Stage C2 is special, having no downsampling in the first block
        downsample = (i == 0 and name_prefix != "C2") 
        projection_shortcut = (i == 0)
        
        current_id = make_bottleneck_block(
            current_id, 
            name_prefix=f"{name_prefix}_b{i+1}", 
            downsample=downsample,
            projection_shortcut=projection_shortcut
        )
        
    return current_id

def make_rpn_head(fm_id, name_prefix):
    """
    RPN Head: 3x3 Conv Shared -> 1x1 Conv (Cls) || 1x1 Conv (Reg)
    RPN output is implicitly passed to the RoIAlign operation
    """
    # 1. Shared 3x3 Conv (uses BN/ReLU since it follows FPN)
    shared_conv = make_conv_bn_relu(fm_id, k=3, s=1, op_type_annotation=f"{name_prefix}_RPN_Shared")
    
    # 2. RPN Classification (Objectness)
    cls_conv = add_node("Conv_1x1", [shared_conv], f"{name_prefix}_RPN_Cls")
    cls_output = add_node("Sigmoid", [cls_conv], f"{name_prefix}_RPN_Cls_Sigmoid") # Objectness scores
    
    # 3. RPN Regression (Box Delta)
    reg_output = add_node("Conv_1x1", [shared_conv], f"{name_prefix}_RPN_Reg")
    
    return cls_output, reg_output # Return last nodes of both branches

# ---------------------------------------------------------
# 3. 아키텍처 구성 (Faster R-CNN R50-FPN)
# ---------------------------------------------------------

# === Backbone: ResNet-50 ===
current_id = -1 

# 1. Stem
stem_conv = make_conv_bn_relu(current_id, k=7, s=2, op_type_annotation="Stem_7x7") 
c1_out = add_node("MaxPool_2x2", [stem_conv], "C1_Pool") # C1 output
current_id = c1_out

# 2. Stage C2 (3 blocks) - P2 feature is extracted here
c2_out = make_resnet_stage(current_id, "C2", num_blocks=3) 
current_id = c2_out

# 3. Stage C3 (4 blocks)
c3_out = make_resnet_stage(current_id, "C3", num_blocks=4)
current_id = c3_out

# 4. Stage C4 (6 blocks)
c4_out = make_resnet_stage(current_id, "C4", num_blocks=6)
current_id = c4_out

# 5. Stage C5 (3 blocks)
c5_out = make_resnet_stage(current_id, "C5", num_blocks=3)
c5_final = c5_out

# === Neck: Feature Pyramid Network (FPN) ===
# Note: FPN combines features from C2, C3, C4, C5 (unlike RetinaNet that uses C3-C5 for FPN part)
fpn_inputs = {"C2": c2_out, "C3": c3_out, "C4": c4_out, "C5": c5_final}
fpn_outputs = {}

# L1: C5 -> P5 (Lateral 1x1 Conv)
p5_lateral = make_conv_bn_relu(c5_final, k=1, s=1, act=False, op_type_annotation="P5_Lateral")
fpn_outputs['P5'] = p5_lateral

# L2-L4: Top-down path
prev_p_id = p5_lateral
for stage_name, c_out_id in [("C4", c4_out), ("C3", c3_out), ("C2", c2_out)]:
    p_name = "P" + stage_name[1]
    
    # Lateral 1x1 Conv
    c_lateral = make_conv_bn_relu(c_out_id, k=1, s=1, act=False, op_type_annotation=f"{p_name}_Lateral")
    
    # Top-down merge: Upsample -> Add
    p_up = add_node("Upsample_2x", [prev_p_id], f"{p_name}_Upsample")
    p_merged = add_node("Add", [c_lateral, p_up], f"{p_name}_Merge")
    
    # Final Px feature (3x3 Conv)
    p_final = make_conv_bn_relu(p_merged, k=3, s=1, op_type_annotation=f"{p_name}_Final") 
    fpn_outputs[p_name] = p_final
    prev_p_id = p_final

# L5: P6 (Extension from C5/P5)
# P6 is C5 output using Conv 3x3 s=2 (Downsample)
p6_down = make_conv_bn_relu(c5_final, k=3, s=2, op_type_annotation="P6_Downsample")
fpn_outputs['P6'] = p6_down

# FPN outputs for RPN/RoI Head (P2, P3, P4, P5, P6)
fpn_rpn_inputs = [fpn_outputs['P2'], fpn_outputs['P3'], fpn_outputs['P4'], fpn_outputs['P5'], fpn_outputs['P6']]

# === Stage 1: Region Proposal Network (RPN) ===
rpn_outputs = [] # RPN outputs (Cls/Reg) for each scale

for i, fm_id in enumerate(fpn_rpn_inputs):
    p_name = f"P{i+2}"
    cls_out, reg_out = make_rpn_head(fm_id, name_prefix=p_name)
    rpn_outputs.extend([cls_out, reg_out]) 

# === Stage 2: Detection Head (RoIAlign -> FC Layers -> Final Projection) ===

# 1. RoIAlign (Takes proposals implicitly, and features from P2-P6)
# RoIAlign receives inputs from ALL FPN levels used by RPN
roi_align_id = add_node("RoIAlign_7x7", fpn_rpn_inputs, "RoIAlign_Head")

# 2. Flatten
flatten_id = add_node("Flatten", [roi_align_id], "Head_Flatten")

# 3. Fully Connected Layers (Two FC layers, common for R50-FPN)
# FC1
fc1_id = add_node("Linear", [flatten_id], "Head_FC1")
fc1_relu = add_node("ReLU", [fc1_id], "Head_FC1_ReLU")
# FC2
fc2_id = add_node("Linear", [fc1_relu], "Head_FC2")
fc2_relu = add_node("ReLU", [fc2_id], "Head_FC2_ReLU")
current_head = fc2_relu

# 4. Final Classification and Regression Projections
# Final Classification (SoftMax is implicit in loss, Linear is the projection)
final_cls_id = add_node("Linear", [current_head], "Head_Final_Cls")

# Final Regression (Linear is the projection)
final_reg_id = add_node("Linear", [current_head], "Head_Final_Reg")

# The final outputs of the whole graph are the RPN projections + Head projections
faster_rcnn_outputs = rpn_outputs + [final_cls_id, final_reg_id]

# ---------------------------------------------------------
# 4. JSON 파일 생성
# ---------------------------------------------------------
graph_data = {
    "architecture": "Faster_R-CNN_R50_FPN",
    "total_nodes": len(nodes),
    "total_edges": len(edges),
    "nodes": nodes,
    "edges": edges,
    "final_output_ids": faster_rcnn_outputs
}

file_path = "faster_rcnn_r50_fpn_graph.json"
with open(file_path, "w", encoding="utf-8") as f:
    json.dump(graph_data, f, indent=2)

print(f"File saved: {file_path}")
print(f"Graph Stat: Nodes={len(nodes)}, Edges={len(edges)}")