# PG-Flow-N Implementation Plan
_Norm-based Parameter-Gated Flow Surrogate_

이 문서는 **PG-Flow-N** 방법론(데이터 비의존, weight-only gate)을 실제 코드로 구현하기 위한  
**실질적인 액션 플랜**을 정리한 것이다.

목표:

> 훈련된 모델 \(M\) (예: YOLO/DETR/Faster R-CNN 백본, 또는 NAS cell)에 대해  
> **아키텍처 DAG + weight 기반 gate**를 사용하는  
> **weight-aware flow surrogate** \(s(M)\) 계산 모듈을 구현한다.

PG-Flow-N의 전제:

- **데이터(activation, gradient)를 쓰지 않는다.**
- 오직 **weight 분포만으로 gate \(g_i\)**를 구성한다.
- gate는 **G¹/G²/G³ 타입 중 하나**, **gating 패턴은 outgoing / incoming 중 하나**를 선택한다.

---

## 0. 전체 아키텍처 개요

구현하고자 하는 구성 요소는 다음 5개다.

1. **그래프 빌더** (`graph_builder.py`)  
   → PyTorch 모델 → 모듈 단위 DAG(`nodes`, `edges`, `topo_order`)

2. **웨이트 추출기** (`weight_extractor.py`)  
   → 각 노드(Node)에 대해 effective weight \(W_i\) 추출

3. **Gate 함수 모듈** (`gate_functions.py`)  
   → \(W_i\)들로부터 대표값 \(e_i\) 계산  
   → z-score → gate \(g_i\)  
   → **Gate 타입 G¹/G²/G³ 중 선택 가능**

4. **Flow 시뮬레이터** (`flow_simulator.py`)  
   → DAG와 gate를 이용해 FGP-style random flow를 흘림  
   → **gating 패턴(Outgoing / Incoming)을 선택해서 적용**

5. **Surrogate 계산 래퍼** (`surrogate_computer.py`)  
   → `model` 하나를 받아 위 1–4단계를 호출해 \(s(M)\) 반환

+ 실험 스크립트:

- `experiments/compute_surrogates.py`  
  → 여러 모델에 대해 s′ 일괄 계산

---

## 1. 파일/디렉토리 구조

```text
project_root/
├── pgflow/
│   ├── __init__.py
│   ├── graph_builder.py        # 모델 → DAG
│   ├── weight_extractor.py     # Node → W_i
│   ├── gate_functions.py       # Gate(W) 정의 (G¹~G³)
│   ├── flow_simulator.py       # gate가 적용된 flow 시뮬레이션 (P¹/P² 선택)
│   └── surrogate_computer.py   # model → s(M)
└── experiments/
    └── compute_surrogates.py   # 여러 모델에 대해 s(M) 계산
```

---

## 2. 그래프 빌더 설계 (`graph_builder.py`)

### 2.1 Node 자료구조 정의

논리적 모듈 → Node:

- `id: int` – 그래프 내 유니크 ID  
- `name: str` – 모듈 경로 / 이름  
- `module: nn.Module` – 실제 PyTorch 모듈  
- `op_type: str` – `'conv3x3'`, `'block_C2f'`, `'encoder_layer'` 등  
- `in_channels, out_channels, kernel_size, stride`  
- `extra: Dict` – 추가 메타데이터

### 2.2 TODO: build_graph_from_model(model)

입력:

- 학습된 PyTorch `model`

출력:

- `nodes: List[Node]`  
- `edges: List[(src_id, dst_id)]`  
- `topo_order: List[node_id]`

**액션 아이템**

- 우선 한 종류의 모델부터 지원  
  - 예: YOLO 백본 또는 NAS cell  
  - 각 연산/블록을 Node로 만들고, feature 흐름을 edge로 구성  
  - 입력/출력 노드 명시적 추가  
  - topological sort 구현 (Kahn 알고리즘)
- 이후 다른 detector(backbone/neck/head)도 확장

---

## 3. Weight 추출기 설계 (`weight_extractor.py`)

### 3.1 get_effective_weight(node: Node)

역할:

- Node가 가리키는 모듈에서 **effective weight** \(W_i\)를 가져온다.

1차 버전:

- `hasattr(module, "weight")` && `weight is not None`  
  → `module.weight` 그대로 사용
- Conv+BN, composite block은 나중에 확장

### 3.2 정책 포인트

- Skip / Identity:
  - `W_i = None`로 두고 gate 계산 시 `g_i = 1.0`으로 강제  
- Zero op / Dropout:
  - gate를 0 근처로 두는 정책 고려

**액션 아이템**

- Conv/Linear 기본 처리  
- Skip/Zero 정책 정의  
- 필요 시 Conv+BN folding

---

## 4. Gate 값 설계 (Layer 1) – `gate_functions.py`

### 4.0 Gate 설계 원칙

- gate는 **오직 weight \(W\)** 만 사용
- Gate(W)는 다음을 만족:
  - (1) **상대적 크기**만 사용 → z-score  
  - (2) 값 범위 **[1−β, 1+β]** → flow가 터지거나 0으로 죽지 않게  
  - (3) weight 크기가 커질수록 gate 증가 (단조 증가)

이를 위해 세 가지 Gate 타입을 정의:

```python
s_prime = compute_pgflow_surrogate(
    model,
    gate_type="rel_norm",      # "rel_norm" | "scale_norm" | "norm_sparsity"
    beta=0.2,
    lam=1.0,
    gating_pattern="outgoing", # "outgoing" | "incoming"
)
```

- **G¹: `rel_norm`** – 상대적인 L2 norm 기반 (기본형)  
- **G²: `scale_norm`** – fan-in을 고려한 scale-invariant norm  
- **G³: `norm_sparsity`** – norm + sparsity로 dead-layer 억제

---

### 4.1 G¹: Relative Norm Gate (`gate_type="rel_norm"`)

**한 줄 요약:**  
> “평균보다 weight 에너지가 큰 모듈은 gate ↑, 작은 모듈은 gate ↓”

1. 대표값 \(e_i\)  
   \[
   e_i = \log\left( rac{\|W_i\|_F}{\sqrt{|	heta_i|}} + arepsilon 
ight)
   \]

2. z-score  
   \[
   \hat e_i = rac{e_i - \mu_e}{\sigma_e}
   \]

3. gate  
   \[
   g_i = 1 + eta 	anh(\lambda \hat e_i)
   \]  
   - 예: \(eta = 0.2,\ \lambda = 1.0\)

**구현 함수**

- `compute_node_stats_rel_norm(nodes)` → {node_id: e_i}  
- `normalize_stats(stats)` → {node_id: e_hat_i}  
- `gate_tanh(normed_stats, beta, lam)` → {node_id: g_i}

---

### 4.2 G²: Scale-Invariant Norm Gate (`gate_type="scale_norm"`)

**한 줄 요약:**  
> “입력 채널/커널 크기까지 고려해서, **입력 대비 얼마나 크게 학습됐는지** 본다”

1. Conv weight 형상:  
   \(W_i \in \mathbb{R}^{C_{out}	imes C_{in}	imes k 	imes k}\)

2. fan-in:  
   \(	ext{fan\_in}_i = C_{in} \cdot k^2\)

3. 대표값:  
   \[
   e_i = \log\left( rac{\|W_i\|_F}{\sqrt{	ext{fan\_in}_i}} + arepsilon 
ight)
   \]

4. 이후 z-score, gate는 G¹과 동일

**구현 함수**

- `compute_node_stats_scale_norm(nodes)` → {node_id: e_i}  
  - Conv가 아니면 G¹ 방식으로 fallback  
- `normalize_stats`, `gate_tanh` 재사용

---

### 4.3 G³: Norm + Sparsity Gate (`gate_type="norm_sparsity"`)

**한 줄 요약:**  
> “norm이 커도 대부분 0이면 낮게 보고, dense하게 살아 있는 레이어를 강조”

1. norm term (G¹):  
   \[
   e_i^{(N)} = \log\left( rac{\|W_i\|_F}{\sqrt{|	heta_i|}} + arepsilon 
ight)
   \]

2. sparsity term:  
   \[
   s_i = rac{1}{|	heta_i|} \sum_{	heta \in i} \mathbf{1}(|	heta| < 	au)
   \]  
   - \(	au pprox 10^{-3}\)  
   - \(s_i \in [0,1]\)

3. 합성 대표값:  
   \[
   e_i = e_i^{(N)} - \gamma s_i
   \]  
   - \(\gamma > 0\) (예: 0.5)

4. 이후 z-score, gate는 동일

**구현 함수**

- `compute_node_stats_norm_sparsity(nodes, tau, gamma)` → {node_id: e_i}  
- `normalize_stats`, `gate_tanh` 재사용

---

## 5. Gate 적용 패턴 (Layer 2) – `flow_simulator.py`

Gate 값 \(g_i\) (G¹~G³ 중 하나로 계산)가 준비되면,  
이제 **flow 안에서 어디에 곱할지**를 결정한다.  
이 선택은 `gating_pattern`으로 컨트롤한다.

```python
s_prime = simulate_pgflow(
    nodes,
    edges,
    gates,
    d_hidden=64,
    gating_pattern="outgoing",  # 또는 "incoming"
)
```

### 5.1 공통 입력

- `nodes: List[Node]`  
- `edges: List[(src_id, dst_id)]` (Node.id 기준)  
- `gates: Dict[node_id -> g_i]`  
- `d_hidden: int`  
- `gating_pattern: "outgoing" | "incoming"`

### 5.2 P¹: Outgoing Gating (`gating_pattern="outgoing"`)

메시지 합산:

\[
m_i = \sum_{j \in \mathcal{N}_	ext{in}(i)} g_j f_j
\]

- 해석: **보내는 쪽(j)이 얼마나 크게 말하는지**를 gate로 반영

### 5.3 P²: Incoming Gating (`gating_pattern="incoming"`)

메시지 합산:

\[
m_i = g_i \cdot \sum_{j} f_j
\]

- 해석: **받는 쪽(i)이 upstream 정보를 얼마나 받아들일지**를 gate로 반영

### 5.4 Flow 구현 절차

1. Node.id → 0..N-1 index 매핑  
2. edges를 index space로 변환 → `edges_idx`  
3. indegree 계산 → input 노드 인덱스 찾기  
4. input 노드에 `torch.randn`로 초기 메시지 할당  
5. `topological_sort(num_nodes, edges_idx)`로 순서 구하기  
6. topo order대로:
   - incoming 인덱스 목록 찾기
   - `gating_pattern`에 따라  
     - `"outgoing"`: `msgs = f_j * g_j`, sum  
     - `"incoming"`: `sum(f_j)` 후 `* g_i`
7. input 노드들의 메시지 sum → `s_prime`  
8. \(s(M) = s_	ext{prime} / (\|s_	ext{prime}\| + arepsilon)\)

**액션 아이템**

- `topological_sort(num_nodes, edges_idx)` 구현  
- `simulate_pgflow(..., gating_pattern)` 분기 구현  

---

## 6. Surrogate 래퍼 – `surrogate_computer.py`

### 6.1 compute_pgflow_surrogate(model, ...)

```python
def compute_pgflow_surrogate(
    model,
    gate_type="rel_norm",      # G¹/G²/G³ 선택
    beta=0.2,
    lam=1.0,
    d_hidden=64,
    gating_pattern="outgoing", # P¹ / P² 선택
):
    nodes, edges, topo = build_graph_from_model(model)

    # 1) Gate 타입 선택
    if gate_type == "rel_norm":
        stats = compute_node_stats_rel_norm(nodes)
    elif gate_type == "scale_norm":
        stats = compute_node_stats_scale_norm(nodes)
    elif gate_type == "norm_sparsity":
        stats = compute_node_stats_norm_sparsity(nodes, tau=1e-3, gamma=0.5)
    else:
        raise ValueError(...)

    # 2) z-score 정규화
    normed = normalize_stats(stats)

    # 3) gate 계산
    gates = gate_tanh(normed, beta=beta, lam=lam)

    # 4) flow 시뮬레이션
    s_prime = simulate_pgflow(
        nodes=nodes,
        edges=edges,
        gates=gates,
        d_hidden=d_hidden,
        gating_pattern=gating_pattern,
    )
    return s_prime
```

**사용 예시**

```python
# 기본: G¹ + Outgoing
s1 = compute_pgflow_surrogate(
    model,
    gate_type="rel_norm",
    gating_pattern="outgoing",
)

# ablation: G² + Incoming
s2 = compute_pgflow_surrogate(
    model,
    gate_type="scale_norm",
    gating_pattern="incoming",
)
```

---

## 7. 실험 스크립트 – `experiments/compute_surrogates.py`

### 역할

- 여러 모델에 대해:
  - 모델 로딩
  - `compute_pgflow_surrogate` 호출
  - 결과 `s(M)`을 npz/pickle로 저장

**액션 아이템**

- `load_model_list()` 구현 (예: `(model_id, model_instance)` 리스트)  
- 루프에서 s′ 계산 후 dict에 저장  
- `np.savez("pgflow_surrogates_*.npz", **results)` 저장  

---

## 8. 구현 체크리스트 (Gate 타입 & Gating 패턴 동일 우선순위)

아래 항목들은 **동일 우선순위**로, PG-Flow-N 기능을 완성하기 위해 모두 구현하는 것을 목표로 한다.

- [ ] `Node` 정의 + `build_graph_from_model` (한 종류 모델에 대해 동작)  
- [ ] `get_effective_weight` (Conv/Linear 기본 지원 + Skip/Zero 정책)  
- [ ] G¹: Relative Norm Gate (`compute_node_stats_rel_norm`, `normalize_stats`, `gate_tanh`)  
- [ ] G²: Scale-Invariant Norm Gate (`compute_node_stats_scale_norm`)  
- [ ] G³: Norm + Sparsity Gate (`compute_node_stats_norm_sparsity`)  
- [ ] `topological_sort(num_nodes, edges_idx)` 구현  
- [ ] `simulate_pgflow`에서 `gating_pattern="outgoing"` / `"incoming"` 두 패턴 모두 지원  
- [ ] `compute_pgflow_surrogate`에서 `gate_type`과 `gating_pattern` 둘 다 인자로 받아 분기 처리  
- [ ] `experiments/compute_surrogates.py`에서 다양한 `(gate_type, gating_pattern)` 조합으로 s′ 계산 가능하게 구성  

---

이 계획에 따라 구현하면,  
- **Gate 타입(G¹/G²/G³)**와  
- **Gating 패턴(Outgoing / Incoming)**을  
동일한 우선순위에서 자유롭게 선택할 수 있는 PG-Flow-N 파이프라인을 갖추게 된다.
