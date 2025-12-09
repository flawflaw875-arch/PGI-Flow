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
# 2. Component Builder (ResNet, RPN, RoIHead)
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

def make_rpn_head(fm_id, name_prefix):
    """RPN Head"""
    shared_conv = make_conv_bn_relu(fm_id, k=3, s=1, op_type_annotation=f"{name_prefix}_RPN_Shared")
    cls_output = add_node("Sigmoid", [add_node("Conv_1x1", [shared_conv], f"{name_prefix}_RPN_Cls")], f"{name_prefix}_RPN_Cls_Sigmoid")
    reg_output = add_node("Conv_1x1", [shared_conv], f"{name_prefix}_RPN_Reg")
    return cls_output, reg_output 

def make_roi_head_stage(fpn_features, proposal_inputs, stage_idx):
    """
    Builds one stage of the Cascade RoI Head.
    Inputs: FPN features (list), Proposal inputs (list of IDs acting as boxes)
    """
    prefix = f"Cascade_Stage{stage_idx}"
    
    # 1. RoIAlign: Takes Features + Proposals
    # In graph, we represent this dependency by connecting proposal source nodes to RoIAlign
    roi_inputs = fpn_features + proposal_inputs
    roi_out = add_node("RoIAlign_7x7", roi_inputs, f"{prefix}_RoIAlign")
    
    # 2. FC Head (Standard 2FC)
    flat = add_node("Flatten", [roi_out], f"{prefix}_Flatten")
    fc1 = add_node("ReLU", [add_node("Linear", [flat], f"{prefix}_FC1")], f"{prefix}_FC1_ReLU")
    fc2 = add_node("ReLU", [add_node("Linear", [fc1], f"{prefix}_FC2")], f"{prefix}_FC2_ReLU")
    
    # 3. Output Branches
    # Classification
    cls_score = add_node("Linear", [fc2], f"{prefix}_Cls_Score")
    
    # Regression (Box Refinement)
    bbox_pred = add_node("Linear", [fc2], f"{prefix}_BBox_Pred")
    
    return cls_score, bbox_pred

# ---------------------------------------------------------
# 3. Compose Architecture (Cascade R-CNN R50-FPN)
# ---------------------------------------------------------

# === Backbone: ResNet-50 ===
current_id = -1 
stem_conv = make_conv_bn_relu(current_id, k=7, s=2, op_type_annotation="Stem_7x7") 
c1_out = add_node("MaxPool_2x2", [stem_conv], "C1_Pool") 

c2_out = make_resnet_stage(c1_out, "C2", num_blocks=3) 
c3_out = make_resnet_stage(c2_out, "C3", num_blocks=4)
c4_out = make_resnet_stage(c3_out, "C4", num_blocks=6)
c5_out = make_resnet_stage(c4_out, "C5", num_blocks=3)

# === Neck: FPN ===
fpn_outputs = {}
p5_lateral = make_conv_bn_relu(c5_out, k=1, s=1, act=False, op_type_annotation="P5_Lateral")
fpn_outputs['P5'] = p5_lateral

prev_p_id = p5_lateral
for stage_name, c_out_id in [("C4", c4_out), ("C3", c3_out), ("C2", c2_out)]:
    p_name = "P" + stage_name[1]
    c_lateral = make_conv_bn_relu(c_out_id, k=1, s=1, act=False, op_type_annotation=f"{p_name}_Lateral")
    p_up = add_node("Upsample_2x", [prev_p_id], f"{p_name}_Upsample")
    p_merged = add_node("Add", [c_lateral, p_up], f"{p_name}_Merge")
    fpn_outputs[p_name] = make_conv_bn_relu(p_merged, k=3, s=1, op_type_annotation=f"{p_name}_Final")
    prev_p_id = fpn_outputs[p_name]

fpn_outputs['P6'] = make_conv_bn_relu(c5_out, k=3, s=2, op_type_annotation="P6_Downsample")

fpn_features_list = [fpn_outputs[k] for k in ['P2', 'P3', 'P4', 'P5']] # RoIAlign typically uses P2-P5

# === RPN ===
rpn_reg_outputs = [] # To be used as initial proposals
rpn_outputs_all = []
for i, key in enumerate(['P2', 'P3', 'P4', 'P5', 'P6']):
    cls_out, reg_out = make_rpn_head(fpn_outputs[key], f"P{i+2}")
    rpn_reg_outputs.append(reg_out)
    rpn_outputs_all.extend([cls_out, reg_out])

# === Cascade RoI Heads (3 Stages) ===
cascade_final_outputs = []

# Initial Proposals: From RPN Regression outputs
current_proposals = rpn_reg_outputs 

# Stage 1
s1_cls, s1_reg = make_roi_head_stage(fpn_features_list, current_proposals, stage_idx=1)
cascade_final_outputs.extend([s1_cls, s1_reg])

# Stage 2 (Input proposals come from Stage 1 Regression)
s2_cls, s2_reg = make_roi_head_stage(fpn_features_list, [s1_reg], stage_idx=2)
cascade_final_outputs.extend([s2_cls, s2_reg])

# Stage 3 (Input proposals come from Stage 2 Regression)
s3_cls, s3_reg = make_roi_head_stage(fpn_features_list, [s2_reg], stage_idx=3)
cascade_final_outputs.extend([s3_cls, s3_reg])


# Total outputs: RPN outputs + Cascade Stage 1, 2, 3 outputs
final_output_ids = rpn_outputs_all + cascade_final_outputs

# ---------------------------------------------------------
# 4. create JSON file
# ---------------------------------------------------------
graph_data = {
    "architecture": "Cascade_R-CNN_R50_FPN",
    "total_nodes": len(nodes),
    "total_edges": len(edges),
    "nodes": nodes,
    "edges": edges,
    "final_output_ids": final_output_ids
}

file_path = "../Graph/cascade_rcnn_r50_fpn_graph.json"
with open(file_path, "w", encoding="utf-8") as f:
    json.dump(graph_data, f, indent=2)

print(f"File saved: {file_path}")
print(f"Graph Stat: Nodes={len(nodes)}, Edges={len(edges)}")