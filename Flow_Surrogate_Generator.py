import torch
import torch.nn.functional as F
import numpy as np
import networkx as nx
import json

class PGFlowEngine:
    """
    [Step 2] PG-Flow Surrogate Generator
    UWEF 포맷의 웨이트와 아키텍처 그래프를 받아,
    설정된 Gating 방식에 따라 Flow Surrogate (s)를 생성하는 엔진.
    """
    def __init__(self, dim_k=16, beta=0.2, lam=1.0, epsilon=1e-6):
        self.dim_k = dim_k
        self.beta = beta      # 게이트 변동 폭 (1.0 +/- beta)
        self.lam = lam        # Z-score 민감도
        self.epsilon = epsilon
        
        # FGP Random Variables (Lazy Init)
        self.P = None 
        self.W = None
        self.r = None

    def _init_random_projections(self, num_ops=32, seed=None):
        """FGP 논문의 Random Projection 초기화"""
        if seed is not None:
            torch.manual_seed(seed)
            
        if self.P is None:
            # 논문에 따라 작은 분산(sigma) 사용
            sigma = 0.05
            self.P = torch.normal(0, sigma, size=(num_ops, self.dim_k))
            self.W = torch.normal(0, sigma, size=(2 * self.dim_k, self.dim_k))
            self.r = torch.rand(self.dim_k)

    def _calculate_stats_from_uwef(self, tensor_obj):
        """
        UWEF 포맷의 텐서 객체에서 통계량(Norm, Fan-in 등)을 즉석 계산
        """
        if not tensor_obj or 'data' not in tensor_obj:
            return None
            
        # 1. Reconstruct Tensor (Flattened List -> Tensor)
        data = tensor_obj['data']
        shape = tensor_obj.get('shape', [len(data)])
        
        # 계산 효율을 위해 Tensor로 변환하지 않고 Numpy/List 상태에서 계산 가능하지만,
        # 정확한 L2 Norm 등을 위해 PyTorch Tensor로 변환
        tensor = torch.tensor(data, dtype=torch.float32)
        
        # 2. Statistics Calculation
        l2_norm = torch.norm(tensor).item()
        num_params = tensor.numel()
        
        # Sparsity (G3용, Threshold 1e-3)
        zero_count = torch.sum(torch.abs(tensor) < 1e-3).item()
        sparsity = zero_count / num_params if num_params > 0 else 0.0
        
        # Fan-in (G2용)
        # shape 정보: [Out, In, K, K] or [Out, In]
        fan_in = 0
        if len(shape) == 4:   # Conv2d
            fan_in = shape[1] * shape[2] * shape[3]
        elif len(shape) == 2: # Linear
            fan_in = shape[1]
        elif len(shape) == 1: # BN/LN
            fan_in = 1
            
        return {
            "l2_norm": l2_norm,
            "num_params": num_params,
            "fan_in": fan_in,
            "sparsity": sparsity
        }

    def _compute_gate_scalar(self, stats, gate_type, gamma):
        """
        통계량 -> 대표값(Representative Value, e_i) 변환
        """
        if stats is None: return None

        norm_f = stats['l2_norm']
        num_params = stats['num_params']
        
        # G1: Relative Norm Gate
        if gate_type == "rel_norm": 
            val = norm_f / (np.sqrt(num_params) + self.epsilon)
            e_i = np.log(val + self.epsilon)
            
        # G2: Scale-Invariant Norm Gate
        elif gate_type == "scale_norm": 
            fan_in = stats['fan_in'] if stats['fan_in'] > 0 else num_params
            val = norm_f / (np.sqrt(fan_in) + self.epsilon)
            e_i = np.log(val + self.epsilon)
            
        # G3: Norm + Sparsity Gate
        elif gate_type == "norm_sparsity": 
            val = norm_f / (np.sqrt(num_params) + self.epsilon)
            e_i_norm = np.log(val + self.epsilon)
            e_i = e_i_norm - gamma * stats['sparsity']
        
        else:
            raise ValueError(f"Unknown gate type: {gate_type}")
            
        return e_i

    def run(self, graph_json, weights_json, gate_type="rel_norm", gating_pattern="outgoing", gamma=0.5, seed=42):
        """
        [메인 실행 함수]
        :param graph_json: 아키텍처 그래프 (Nodes, Edges)
        :param weights_json: UWEF 포맷의 웨이트 데이터
        :param gate_type: 'rel_norm' | 'scale_norm' | 'norm_sparsity'
        :param gating_pattern: 'outgoing' | 'incoming'
        :return: Surrogate Vector s (Tensor)
        """
        self._init_random_projections(seed=seed)
        
        # ---------------------------------------------------------
        # 1. Gate Calculation Phase
        # ---------------------------------------------------------
        e_values = {}
        valid_nodes = []
        node_weights = weights_json.get("node_weights", {})
        
        # 모든 노드를 순회하며 e_i(대표값) 계산
        for node_id, w_data in node_weights.items():
            if not w_data['has_weight'] or 'weight' not in w_data['tensors']:
                continue
            
            # (1) Raw Data -> Stats
            stats = self._calculate_stats_from_uwef(w_data['tensors']['weight'])
            
            # (2) Stats -> e_i
            e_i = self._compute_gate_scalar(stats, gate_type, gamma)
            
            if e_i is not None:
                e_values[str(node_id)] = e_i
                valid_nodes.append(str(node_id))
        
        # Z-score Normalization & Tanh Activation
        gates = {}
        if valid_nodes:
            vals = np.array([e_values[nid] for nid in valid_nodes])
            mu = np.mean(vals)
            sigma = np.std(vals) + self.epsilon
            
            # 그래프의 모든 노드에 대해 게이트 할당
            for node in graph_json['nodes']:
                nid = str(node['id'])
                if nid in e_values:
                    # 학습된 웨이트가 있는 노드: 통계 기반 게이팅
                    z = (e_values[nid] - mu) / sigma
                    g = 1.0 + self.beta * np.tanh(self.lam * z)
                else:
                    # 웨이트 없는 노드(ReLU, Pooling): 중립 게이트
                    g = 1.0
                gates[nid] = float(g)
        else:
            # 웨이트 정보가 아예 없는 경우 모두 1.0
            gates = {str(n['id']): 1.0 for n in graph_json['nodes']}

        # ---------------------------------------------------------
        # 2. Flow Simulation Phase
        # ---------------------------------------------------------
        G = nx.DiGraph()
        for node in graph_json['nodes']:
            G.add_node(node['id'], op_idx=node['op_idx'])
        for src, dst in graph_json['edges']:
            G.add_edge(src, dst)
            
        try:
            topo_order = list(nx.topological_sort(G))
        except:
            print("[Error] Graph has cycles. Returning zero vector.")
            return torch.zeros(self.dim_k)

        # Forward Pass Memory
        f_msgs = {}
        
        for node_id in topo_order:
            op_idx = G.nodes[node_id]['op_idx']
            h_i = self.P[op_idx]
            g_i = gates.get(str(node_id), 1.0)
            
            preds = list(G.predecessors(node_id))
            
            if not preds:
                # Input Nodes
                f_msgs[node_id] = self.r
            else:
                m_i = torch.zeros(self.dim_k)
                
                # --- Gating Pattern Application ---
                if gating_pattern == "outgoing":
                    # P1: Source(pred)의 힘(gate)에 비례해서 정보 수신
                    for pred in preds:
                        m_i += gates.get(str(pred), 1.0) * f_msgs[pred]
                        
                elif gating_pattern == "incoming":
                    # P2: 일단 다 받고, 나의 수용력(gate)만큼 반영
                    for pred in preds:
                        m_i += f_msgs[pred]
                    m_i = m_i * g_i
                
                # FGP Transform (Gate와 무관한 고정 변환)
                concat = torch.cat([h_i, m_i], dim=0)
                transformed = F.relu(torch.matmul(concat, self.W))
                
                # Update State
                f_msgs[node_id] = 0.5 * m_i + 0.5 * transformed
                
                # Safety Clamp (수치 안정성)
                f_msgs[node_id] = torch.clamp(f_msgs[node_id], -10, 10)

        # ---------------------------------------------------------
        # 3. Final Aggregation
        # ---------------------------------------------------------
        # Sink Nodes (출력 노드)들의 합을 Surrogate s로 사용
        sink_nodes = [n for n in G.nodes if G.out_degree(n) == 0]
        s = torch.zeros(self.dim_k)
        
        for sink in sink_nodes:
            s += f_msgs[sink]
            
        # Normalize Result
        if torch.norm(s) > 1e-6:
            s = s / torch.norm(s)
            
        return s

# --- 사용 예시 ---
if __name__ == "__main__":
    # 1. 파일 로드 (예시)
    with open("yolov5s_graph.json", "r") as f:
        graph_data = json.load(f)
        
    # [가정] Step 1에서 생성된 웨이트 파일이 있다고 가정
    # with open("yolov5s_weights_uwef.json", "r") as f:
    #     weights_data = json.load(f)
    
    # 2. 엔진 초기화
    engine = PGFlowEngine(dim_k=16)
    
    # 3. 실행 (옵션 선택 가능)
    # Case A: G1 게이트 + Outgoing 패턴
    # s_vector = engine.run(graph_data, weights_data, 
    #                       gate_type="rel_norm", 
    #                       gating_pattern="outgoing")
                          
    # Case B: G2 게이트 + Incoming 패턴 (Conv 모델에 추천)
    # s_vector = engine.run(graph_data, weights_data, 
    #                       gate_type="scale_norm", 
    #                       gating_pattern="incoming")
    
    print("PG-Flow Engine code generated successfully.")