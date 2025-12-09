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

def make_conv_module(input_id, k=1, s=1, act=True):
    """Conv -> BN -> SiLU"""
    # 1. Convolution
    if k == 1: op = "Conv_1x1"
    elif k == 3 and s == 1: op = "Conv_3x3"
    elif k == 3 and s == 2: op = "Conv_3x3_DS"
    else: op = "Conv_3x3" 
    
    c_id = add_node(op, [input_id])
    
    # 2. BatchNorm
    bn_id = add_node("BatchNorm", [c_id])
    
    # 3. Activation (SiLU)
    if act:
        out_id = add_node("SiLU", [bn_id])
    else:
        out_id = bn_id
    return out_id

def make_bottleneck(input_id, shortcut=True):
    """Bottleneck: Conv_1x1 -> Conv_3x3 -> Add"""
    c1 = make_conv_module(input_id, k=1, s=1)
    c2 = make_conv_module(c1, k=3, s=1)
    
    if shortcut:
        add_id = add_node("Add", [input_id, c2])
        return add_id
    else:
        return c2

def make_csp_block(input_id, n=1):
    """CSPDarknet/C3-like Module"""
    
    # 1. cv1 (1x1 Conv) - Route for residual path
    route = make_conv_module(input_id, k=1, s=1)
    
    # 2. cv2 (1x1 Conv) - Input for Bottlenecks
    x = make_conv_module(input_id, k=1, s=1)
    
    # 3. Bottlenecks (n times)
    last_tensor = x
    for _ in range(n):
        last_tensor = make_bottleneck(last_tensor, shortcut=True) 
        
    # 4. Concat 
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
# 3. 아키텍처 구성 (YOLOX-l Scaling 적용)
# ---------------------------------------------------------
# YOLOX-l Depth Factors (n):
# Backbone CSP blocks: [3, 9, 9, 3] (Depth = 1.0)
# Neck CSP blocks: [3, 3, 3, 3]

# === Backbone ===
stem = make_conv_module(input_id=-1, k=3, s=2) 

# P2: Conv(s=2) -> CSP(n=3)  <-- Increased depth
p2_conv = make_conv_module(stem, k=3, s=2)
p2_out = make_csp_block(p2_conv, n=3) # Feature P2

# P3: Conv(s=2) -> CSP(n=9)
p3_conv = make_conv_module(p2_out, k=3, s=2)
p3_out = make_csp_block(p3_conv, n=9) # Feature P3

# P4: Conv(s=2) -> CSP(n=9)
p4_conv = make_conv_module(p3_out, k=3, s=2)
p4_out = make_csp_block(p4_conv, n=9) # Feature P4

# P5: Conv(s=2) -> CSP(n=3) -> SPPF
p5_conv = make_conv_module(p4_out, k=3, s=2)
p5_csp = make_csp_block(p5_conv, n=3)
p5_out = make_sppf(p5_csp) # Feature P5

# === Neck (PANet/FPN) ===
# 1. Upsample P5 -> Concat P4 -> CSP(n=3)  <-- Increased depth
up_p5 = add_node("Upsample_2x", [p5_out])
concat_p4 = add_node("Concat", [up_p5, p4_out])
csp_p4_neck = make_csp_block(concat_p4, n=3) 

# 2. Upsample P4_neck -> Concat P3 -> CSP(n=3)
up_p4 = add_node("Upsample_2x", [csp_p4_neck])
concat_p3 = add_node("Concat", [up_p4, p3_out])
csp_p3_out = make_csp_block(concat_p3, n=3) # Head Input 0 (Small)

# 3. Downsample P3_neck -> Concat P4_neck -> CSP(n=3)
down_p3 = make_conv_module(csp_p3_out, k=3, s=2)
concat_p4_2 = add_node("Concat", [down_p3, csp_p4_neck])
csp_p4_out = make_csp_block(concat_p4_2, n=3) # Head Input 1 (Medium)

# 4. Downsample P4_neck_out -> Concat P5 -> CSP(n=3)
down_p4 = make_conv_module(csp_p4_out, k=3, s=2)
concat_p5_2 = add_node("Concat", [down_p4, p5_out])
csp_p5_out = make_csp_block(concat_p5_2, n=3) # Head Input 2 (Large)

# === Head (Decoupled Detect) ===
head_inputs = [csp_p3_out, csp_p4_out, csp_p5_out]

for h_in in head_inputs:
    # Processing Convs (shared across branches)
    shared_cv1 = make_conv_module(h_in, k=3, s=1)
    shared_cv2 = make_conv_module(shared_cv1, k=3, s=1)

    # Box/Reg Branch
    box_out = add_node("Conv_1x1", [shared_cv2])
    
    # Class/Cls Branch
    cls_final = add_node("Conv_1x1", [shared_cv2])
    cls_out = add_node("Sigmoid", [cls_final])

# ---------------------------------------------------------
# 4. JSON 파일 생성
# ---------------------------------------------------------
graph_data = {
    "architecture": "YOLOX-l",
    "total_nodes": len(nodes),
    "total_edges": len(edges),
    "nodes": nodes,
    "edges": edges
}

file_path = "yoloxl_graph.json"
with open(file_path, "w", encoding="utf-8") as f:
    json.dump(graph_data, f, indent=2)

print(f"File saved: {file_path}")
print(f"Graph Stat: Nodes={len(nodes)}, Edges={len(edges)}")