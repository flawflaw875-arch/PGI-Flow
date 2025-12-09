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

def add_node(op_name, input_ids=[]):
    """그래프에 노드를 추가하고 엣지를 연결하는 헬퍼 함수"""
    global node_counter
    current_id = node_counter
    op_idx = OPS[op_name]
    
    nodes.append({
        "id": current_id,
        "op_idx": op_idx,
        "op_name": op_name
    })
    
    for inp_id in input_ids:
        edges.append([inp_id, current_id])
        
    node_counter += 1
    return current_id

# ---------------------------------------------------------
# 2. YOLOv5-s 컴포넌트 빌더 (Atomic Op 단위 분해)
# ---------------------------------------------------------

def make_conv_module(input_id, k=1, s=1, act=True):
    """Conv -> BN -> SiLU (CBL module)"""
    # 1. Convolution
    if k == 1: op = "Conv_1x1"
    elif k == 3 and s == 1: op = "Conv_3x3"
    elif k == 3 and s == 2: op = "Conv_3x3_DS"
    else: op = "Conv_3x3" 
    
    c_id = add_node(op, [input_id])
    
    # 2. BatchNorm
    bn_id = add_node("BatchNorm", [c_id])
    
    # 3. Activation
    if act:
        out_id = add_node("SiLU", [bn_id])
    else:
        out_id = bn_id
    return out_id

def make_bottleneck(input_id, shortcut=True):
    """Bottleneck: Conv_1x1 -> Conv_3x3 -> Add"""
    # First Conv 1x1
    c1 = make_conv_module(input_id, k=1, s=1)
    # Second Conv 3x3
    c2 = make_conv_module(c1, k=3, s=1)
    
    if shortcut:
        add_id = add_node("Add", [input_id, c2])
        return add_id
    else:
        return c2

def make_c3(input_id, n=1, shortcut=True):
    """C3 Module: Conv1x1 || Bottlenecks(n) -> Concat -> Conv1x1"""
    
    # 1. cv1 (1x1 Conv) - Route for residual path
    route = make_conv_module(input_id, k=1, s=1)
    
    # 2. cv2 (1x1 Conv) - Input for Bottlenecks
    x = make_conv_module(input_id, k=1, s=1)
    
    # 3. Bottlenecks (n times)
    last_tensor = x
    for _ in range(n):
        # C3 uses the bottleneck with shortcut internally
        last_tensor = make_bottleneck(last_tensor, shortcut=True) 
        
    # 4. Concat (route + bottlenecks output)
    concat = add_node("Concat", [route, last_tensor])
    
    # 5. Final cv3 (1x1 Conv)
    out = make_conv_module(concat, k=1, s=1)
    return out

def make_sppf(input_id):
    """SPPF: Conv -> MaxPools -> Concat -> Conv"""
    cv1 = make_conv_module(input_id, k=1, s=1)
    
    p1 = add_node("MaxPool_SPP", [cv1]) # k=5, s=1, p=2
    p2 = add_node("MaxPool_SPP", [p1])
    p3 = add_node("MaxPool_SPP", [p2])
    
    concat = add_node("Concat", [cv1, p1, p2, p3])
    cv2 = make_conv_module(concat, k=1, s=1)
    return cv2

# ---------------------------------------------------------
# 3. 아키텍처 구성 (YOLOv5-s Scaling 적용)
# ---------------------------------------------------------
# YOLOv5-s Depth Factors (n):
# Backbone C3 blocks: [1, 3, 3, 1]
# Neck C3 blocks: [1, 1, 1, 1]

# === Backbone ===
# Focus Module Equivalent (Stem):
# Using Conv_3x3_DS as the initial downsampling layer
stem = make_conv_module(input_id=-1, k=3, s=2) 

# P2: Conv(s=2) -> C3(n=1)
p2_conv = make_conv_module(stem, k=3, s=2)
p2_out = make_c3(p2_conv, n=1, shortcut=True) # Feature P2

# P3: Conv(s=2) -> C3(n=3)
p3_conv = make_conv_module(p2_out, k=3, s=2)
p3_out = make_c3(p3_conv, n=3, shortcut=True) # Feature P3

# P4: Conv(s=2) -> C3(n=3)
p4_conv = make_conv_module(p3_out, k=3, s=2)
p4_out = make_c3(p4_conv, n=3, shortcut=True) # Feature P4

# P5: Conv(s=2) -> C3(n=1) -> SPPF
p5_conv = make_conv_module(p4_out, k=3, s=2)
p5_c3 = make_c3(p5_conv, n=1, shortcut=True)
p5_out = make_sppf(p5_c3) # Feature P5

# === Neck (YOLOv5 PANet/FPN) ===
# 1. Upsample P5 -> Concat P4 -> C3(n=1)
up_p5 = add_node("Upsample_2x", [p5_out])
concat_p4 = add_node("Concat", [up_p5, p4_out])
c3_p4_neck = make_c3(concat_p4, n=1, shortcut=False) 

# 2. Upsample P4_neck -> Concat P3 -> C3(n=1)
up_p4 = add_node("Upsample_2x", [c3_p4_neck])
concat_p3 = add_node("Concat", [up_p4, p3_out])
c3_p3_out = make_c3(concat_p3, n=1, shortcut=False) # Head Input 0 (Small)

# 3. Downsample P3_neck -> Concat P4_neck -> C3(n=1)
# Note: Downsampling is typically Conv 3x3 s=2
down_p3 = make_conv_module(c3_p3_out, k=3, s=2)
concat_p4_2 = add_node("Concat", [down_p3, c3_p4_neck])
c3_p4_out = make_c3(concat_p4_2, n=1, shortcut=False) # Head Input 1 (Medium)

# 4. Downsample P4_neck_out -> Concat P5 -> C3(n=1)
down_p4 = make_conv_module(c3_p4_out, k=3, s=2)
concat_p5_2 = add_node("Concat", [down_p4, p5_out])
c3_p5_out = make_c3(concat_p5_2, n=1, shortcut=False) # Head Input 2 (Large)

# === Head (Coupled Detect) ===
# The head consists of a single 1x1 Conv for objectness, class, and box prediction at each scale.
head_inputs = [c3_p3_out, c3_p4_out, c3_p5_out]

for h_in in head_inputs:
    # Single Conv_1x1 layer projects features to the final prediction vector
    # This vector contains [Box, Objectness, Classes]
    final_output = add_node("Conv_1x1", [h_in])
    # For FGP, we stop at the raw output layer before non-linear ops specific to the loss function.
    # However, for consistency with YOLOv8 (which used Sigmoid), we include a placeholder Sigmoid on the final output.
    _ = add_node("Sigmoid", [final_output]) 

# ---------------------------------------------------------
# 4. JSON 파일 생성
# ---------------------------------------------------------
graph_data = {
    "architecture": "YOLOv5-s",
    "total_nodes": len(nodes),
    "total_edges": len(edges),
    "nodes": nodes,
    "edges": edges
}

file_path = "yolov5s_graph.json"
with open(file_path, "w", encoding="utf-8") as f:
    json.dump(graph_data, f, indent=2)

print(f"File saved: {file_path}")
print(f"Graph Stat: Nodes={len(nodes)}, Edges={len(edges)}")