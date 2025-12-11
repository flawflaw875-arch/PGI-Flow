# PG-Flow Implementation Plan
_Norm-based Parameter-Gated Flow Surrogate_

이 문서는 **PG-Flow** 방법론을 실제 코드로 구현하기 위한  
**실질적인 액션 플랜**을 정리한 것이다.

목표:

> 훈련된 모델 $(M)$ (예: YOLO/DETR/Faster R-CNN 등)에 대해  
> **아키텍처 DAG + weight 기반 gate**를 사용하는  
> **parameter-gated flow surrogate** $(s(M))$ 계산 모듈을 구현한다.

PG-Flow의 전제:

- weight를 이용하여 gate $(g_i)$를 구성한다.
- gate는 **G¹/G²/G³ 타입 중 하나**, **gating 패턴은 outgoing / incoming 중 하나**를 선택한다.

---

## 0. 전체 아키텍처 개요

구현하고자 하는 구성 요소는 다음 4개다.

1. **그래프 빌더** (`./object_detection/models/code`)  
   → object detection 모델 아키텍처 → Neural architecture graph (`nodes feature(X)`, `edges(E)`, `Operator(O)`)

2. **웨이트 추출기** (`weight_extractor.py`)  
   → 각 노드(Node)에 대해 effective weight $(W_i)$ 추출

3. **Flow 시뮬레이터** (`Flow_Surrogate_Generator.py`)  
   * Type of Parameter-Gate  
   → $(W_i)$들로부터 대표값 $(e_i)$ 계산  
   → z-score → gate $(g_i)$  
   → **Gate 타입 G¹/G²/G³ 중 선택 가능**
   * Gating pattern  
   → DAG와 gate를 이용해 information flow를 흘림  
   → **gating 패턴(Outgoing / Incoming)을 선택해서 적용** 

4. **Surrogate 계산 래퍼**  
   → `model`을 받아 위 1–3단계를 호출해 $(s(M))$ 반환

---

## 1. 파일/디렉토리 구조

```text
PG-Flow/
├── object_detection_models/
│   ├── code/                     # 모델 → DAG 코드
│   ├── Graph/                    # 생성된 아키텍처별 DAG
│   └── Ops.json                  # Op 리스트
├── parameter_regenerator/        # 아키텍처별 웨이트에 대해 DAG에 적합한 형식으로 변환
└── compute_surrogates.py         # 여러 모델 아키텍처와 웨이트에 대해 s(M) 계산
```

---

## 2. 아키텍처별 그래프 설계 (`./object_detection_models/code/`)

### 2.1 Operator 리스트 정의

- `0: Conv_1x1`
- `1: Conv_3x3`
- `2: Conv_3x3_DS`
- `3: Conv_7x7_DS`
- `4: Conv_3x3_Dil`
- `5: DWConv_3x3`
- `6: DWConv_3x3_DS`
- `7: DWConv_5x5`
- `8: MaxPool_2x2`
- `9: MaxPool_SPP`
- `10: AvgPool_Global`
- `11: Upsample_2x`
- `12: RoIAlign_7x7`
- `13: BatchNorm`
- `14: LayerNorm`
- `15: Linear`
- `16: ReLU`
- `17: SiLU`
- `18: Sigmoid`
- `19: Concat`
- `20: Add`
- `21: Split_Half`
- `22: Flatten`
- `23: Reshape_Heads`
- `24: Transpose`
- `25: Gather_TopK`
- `26: MatMul`
- `27: SoftMax`
- `28: Mul`
- `29: Div`
- `30: Sub`
- `31: Exp`

### 2.2 Node 및 Edge 자료구조 정의

**Node 구조:**
- `id: int` – 그래프 내 node ID  
- `op_idx: int` – node별 Operator ID  
- `op_name: str` – Operator 이름  
- `annotation: str` – 기타 설명  

**Edge 구조:**
- $[id_i, id_j]$ – $node_i$(정보출력 노드) → $node_j$(정보입력 노드)

### 2.3 Architecture 종류

**YOLO Family (Real-time One-stage)**
- YOLOv8 (Anchor-free, SOTA): YOLOv8-n, YOLOv8-m
- YOLOv5 (Anchor-based, Industry Standard): YOLOv5-s, YOLOv5-x
- YOLOX (Decoupled Head, Anchor-free): YOLOX-s, YOLOX-l

**R-CNN Family (Two-stage)**
- Faster R-CNN (Standard Baseline): R50-FPN, R101-FPN
- Cascade R-CNN (High Quality): R50-FPN

**Legacy One-stage (Baselines)**
- SSD: SSD300-VGG16, MobileNetV2-SSDLite
- RetinaNet: R50-FPN (Focal Loss Base)

**EfficientDet (Scalable)**
- EfficientDet: D0, D3 (BiFPN + Compound Scaling)

**Transformer (End-to-End)**
- DETR: R50 (Encoder-Decoder Attention)

---

## 3. 공통 웨이트 교환 포맷(UWEF) 정의 (`./parameter_regenerator/`)

### 3.1 파일 구조 (Root Schema)

파일의 최상위 루트는 메타데이터(meta)와 가중치 데이터(node_weights) 두 가지 키로 구성됩니다.

```json
{
  "meta": {
    "architecture": "string",
    "format_version": "string",
    "source_framework": "string",
    "created_at": "string"
  },
  "node_weights": {
    "NODE_ID_1": { },
    "NODE_ID_2": { },
    "...": "..."
  }
}
```

### 3.2 노드 가중치 객체 (node_weights)

그래프 정의 파일(*_graph.json)에 명시된 Node ID를 Key로 사용합니다. 각 노드는 해당 연산에 필요한 텐서들을 포함합니다.

#### 3.2.1 개별 노드 구조

```json
{
  "0": {
    "op_type": "Conv_3x3",
    "has_weight": true,
    "tensors": {
      "weight": { },
      "bias": { },
      "running_mean": { },
      "running_var": { }
    }
  }
}
```

### 3.3 텐서 객체 (tensors)

실제 가중치 값을 담는 객체입니다. 텐서의 **차원 정보(Shape)**와 **데이터(Data)**를 포함합니다.

```json
{
  "weight": {
    "dtype": "float32",
    "shape": [64, 3, 3, 3],
    "data": [0.12, -0.05, 0.0, "..."]
  }
}
```

data는 다차원 텐서를 view(-1) 또는 flatten()하여 1차원 리스트로 저장합니다. 로드 시 shape를 이용해 복원합니다.

### 3.4 연산 타입별 표준 스키마

변환 코드 작성 시, 연산 타입에 따라 다음 키(Key) 이름을 준수해야 합니다.

**A. 컨볼루션 / 선형 레이어 (Conv2d, Linear)**
- 필수 텐서: weight
- 선택 텐서: bias
- Shape 규칙:
  - Conv: [Out_Channels, In_Channels, Kernel_H, Kernel_W]
  - Linear: [Out_Features, In_Features]

**B. 정규화 레이어 (BatchNorm, LayerNorm)**
- 필수 텐서:
  - weight: Scale 파라미터 ($\gamma$)
  - bias: Shift 파라미터 ($\beta$)
- 선택 텐서 (BN):
  - running_mean: 이동 평균
  - running_var: 이동 분산

**C. 가중치가 없는 연산 (ReLU, Pooling, Add)**
- has_weight: false
- tensors: {} (빈 객체)

### 3.5 특수 케이스 처리 규칙 (Implementation Rules)

변환기(Exporter) 구현 시 반드시 지켜야 할 규칙입니다.

**1. Transformer Q/K/V 분할 저장 원칙:**

프레임워크 내부에서 하나의 큰 텐서(예: in_proj_weight)로 합쳐져 있더라도, 그래프 노드 정의에 맞춰 잘라서(Slicing) 저장해야 합니다.

예: Q_Node에는 텐서의 앞부분(:embed_dim), K_Node에는 중간 부분(embed_dim:2*embed_dim)을 저장합니다.

**2. 공유 파라미터 중복 저장 원칙 (Deep Copy):**

EfficientDet의 Head처럼 여러 노드가 동일한 파라미터를 공유하더라도, 각 노드 ID 항목에 데이터를 중복하여 기록합니다.

이는 엔진(Step 2)이 복잡한 참조 로직 없이 노드 ID만으로 데이터를 로드할 수 있게 하기 위함입니다.

---

## 4. Gate 값 설계 (Layer 1) – `gate_functions.py`

### 4.0 Gate 설계 원칙

- gate는 **오직 weight $(W)$** 만 사용
- Gate(W)는 다음을 만족:
  - (1) **상대적 크기**만 사용 → z-score  
  - (2) 값 범위 **[1−β, 1+β]** → flow가 터지거나 0으로 죽지 않게  
  - (3) weight 크기가 커질수록 gate 증가 (단조 증가)

이를 위해 세 가지 Gate 타입을 정의:

```python
s_prime = compute_pgflow_surrogate(
    model,
    gate_type="rel_norm",
    beta=0.2,
    lam=1.0,
    gating_pattern="outgoing",
)
```

- **G¹: `rel_norm`** – 상대적인 L2 norm 기반 (기본형)  
- **G²: `scale_norm`** – fan-in을 고려한 scale-invariant norm  
- **G³: `norm_sparsity`** – norm + sparsity로 dead-layer 억제

---

## 4.1 G¹: Relative Norm Gate (`rel_norm`)

> **💡 핵심 컨셉**
> "평균보다 Weight **에너지**가 큰 모듈은 Gate를 열고($\uparrow$), 작은 모듈은 닫는다($\downarrow$)."

| 단계 | 수식 (Formula) | 설명 |
| :--- | :--- | :--- |
| **1. 대표값 ($e_i$)** | $e_i = \log\left( \frac{\lVert W_i \rVert_F}{\sqrt{\|\theta_i\|}} + \epsilon \right)$ | 전체 파라미터 수($\|\theta_i\|$)로 정규화된 Frobenius Norm |
| **2. Z-Score ($\hat{e}_i$)** | $\hat{e}_i = \frac{e_i - \mu_e}{\sigma_e}$ | 전체 레이어 분포 내에서의 상대적 위치 산출 |
| **3. Gate ($g_i$)** | $g_i = 1 + \beta \tanh(\lambda \hat{e}_i)$ | $\beta=0.2, \lambda=1.0$ (예시) |

**🛠 구현 함수 매핑**
* `compute_node_stats_rel_norm(nodes)` → $\{node_{id}: e_i\}$
* `normalize_stats(stats)` → $\{node_{id}: \hat{e}_i\}$
* `gate_tanh(normed_stats, beta, lam)` → $\{node_{id}: g_i\}$

---

## 4.2 G²: Scale-Invariant Norm Gate (`scale_norm`)

> **💡 핵심 컨셉**
> "단순 크기가 아니라, **입력(Fan-in) 대비** 얼마나 크게 학습되었는지를 본다."

| 단계 | 수식 (Formula) | 설명 |
| :--- | :--- | :--- |
| **1. Fan-in** | $fan\_in_i = C_{in} \cdot k^2$ | $W_i \in \mathbb{R}^{C_{out} \times C_{in} \times k \times k}$ 일 때 입력 수용 영역 크기 |
| **2. 대표값** | $e_i = \log\left( \frac{\lVert W_i \rVert_F}{\sqrt{fan\_in_i}} + \epsilon \right)$ | $\|\theta_i\|$ 대신 **Fan-in**으로 나누어 Scale-Invariant 특성 확보 |
| **3. 이후** | $G^1$과 동일 (Z-score → Gate) | |

**🛠 구현 함수 매핑**
* `compute_node_stats_scale_norm(nodes)` → $\{node_{id}: e_i\}$
* *Note:* Conv 레이어가 아닌 경우 $G^1$ 방식으로 Fallback 처리.

---

## 4.3 G³: Norm + Sparsity Gate (`norm_sparsity`)

> **💡 핵심 컨셉**
> "Norm이 커도 대부분이 0(Dead)이라면 낮게 평가하고, **Dense하게 살아있는** 레이어를 강조한다."

| 항목 | 수식 (Formula) | 설명 |
| :--- | :--- | :--- |
| **Norm Term** | $e_i^{(N)} = \log\left( \frac{\lVert W_i \rVert_F}{\sqrt{\|\theta_i\|}} + \epsilon \right)$ | $G^1$의 대표값과 동일 |
| **Sparsity Term** | $s_i = \frac{1}{\|\theta_i\|} \sum_{\theta \in i} \mathbf{1}(\|\theta\| < \tau)$ | $\tau \approx 10^{-3}$, Dead Parameter 비율 |
| **최종 대표값** | $e_i = e_i^{(N)} - \gamma s_i$ | $\gamma > 0$ (예: 0.5), Sparsity가 높을수록 대표값 차감 |

**🛠 구현 함수 매핑**
* `compute_node_stats_norm_sparsity(nodes, tau, gamma)` → $\{node_{id}: e_i\}$

---

## 📊 요약: Gate 타입 비교

| 타입 | 코드명 (`gate_type`) | 고려 요소 | 주요 특징 |
| :--- | :--- | :--- | :--- |
| **$G^1$** | `rel_norm` | **에너지 (Norm)** | 가장 기본적이며, 파라미터 전체의 평균적인 크기를 반영 |
| **$G^2$** | `scale_norm` | **입력 스케일 (Fan-in)** | Conv 필터 구조를 고려함. 입력 대비 증폭률을 중시 |
| **$G^3$** | `norm_sparsity` | **밀도 (Density)** | Dead neuron이 많은 레이어의 중요도를 낮춤 |

---

## 5. Gate 적용 패턴 (Layer 2) – `flow_simulator.py`

Gate 값 $(g_i)$ (G¹~G³ 중 하나로 계산)가 준비되면, **Flow 안에서 어디에 곱할지**를 결정합니다. 이 선택은 `gating_pattern`으로 컨트롤합니다.

```python
s_prime = simulate_pgflow(
    nodes,
    edges,
    gates,
    d_hidden=64,
    gating_pattern="outgoing",
)
```

### 5.1 공통 입력

| 변수명 | 타입 | 설명 |
| :--- | :--- | :--- |
| `nodes` | `List[Node]` | 그래프 노드 리스트 |
| `edges` | `List[(src_id, dst_id)]` | Node ID 기준의 엣지 리스트 |
| `gates` | `Dict[node_id -> g_i]` | 계산된 Gate 값 딕셔너리 |
| `d_hidden` | `int` | 히든 벡터 차원 |
| `gating_pattern` | `str` | `"outgoing"` 또는 `"incoming"` |

### 5.2 Gating 패턴 비교 (P¹ vs P²)

| 패턴 | 코드명 | 수식 | 해석 |
| :--- | :--- | :--- | :--- |
| **P¹: Outgoing** | `"outgoing"` | $m_i = \sum_{j \in \mathcal{N}_{\text{in}}(i)} g_j f_j$ | **보내는 쪽**($j$)이 얼마나 크게 말하는지를 Gate로 반영 |
| **P²: Incoming** | `"incoming"` | $m_i = g_i \cdot \sum_{j} f_j$ | **받는 쪽**($i$)이 정보를 얼마나 받아들일지를 Gate로 반영 |

### 5.3 Flow 구현 절차

1. **인덱스 매핑:** Node.id → 0..N-1 인덱스 변환
2. **엣지 변환:** edges → edges_idx
3. **초기화:** Input 노드에 `torch.randn` 초기 메시지 할당
4. **위상 정렬:** topological_sort(num_nodes, edges_idx) 수행
5. **순회 및 계산:**
   - **Outgoing:** $msgs = f_j \cdot g_j$ 후 Sum
   - **Incoming:** $\sum(f_j)$ 후 $\cdot g_i$
6. **최종 산출:** Input 노드 메시지 합산 및 정규화

$$s(M) = \frac{s_{\text{prime}}}{\|s_{\text{prime}}\| + \epsilon}$$




# PG-Flow Implementation Plan
_Norm-based Parameter-Gated Flow Surrogate_

This document organizes a **practical action plan** for implementing the **PG-Flow** methodology in actual code.

Objective:

> For a trained model $(M)$ (e.g., YOLO/DETR/Faster R-CNN, etc.)  
> implement a **parameter-gated flow surrogate** $(s(M))$ calculation module  
> using **architecture DAG + weight-based gate**.

PG-Flow Premises:

- Construct a gate $(g_i)$ using weights.
- Gate is **one of G¹/G²/G³ types**, and **gating pattern is either outgoing or incoming**.

---

## 0. Overall Architecture Overview

There are 4 components to implement:

1. **Graph Builder** (`./object_detection/models/code`)  
   → Object detection model architecture → Neural architecture graph (`nodes feature(X)`, `edges(E)`, `Operator(O)`)

2. **Weight Extractor** (`weight_extractor.py`)  
   → Extract effective weight $(W_i)$ for each node

3. **Flow Simulator** (`Flow_Surrogate_Generator.py`)  
   * Type of Parameter-Gate  
   → Compute representative value $(e_i)$ from $(W_i)$  
   → z-score → gate $(g_i)$  
   → **Selectable among G¹/G²/G³ gate types**
   * Gating pattern  
   → Flow information through DAG and gate  
   → **Apply gating pattern (Outgoing / Incoming)** 

4. **Surrogate Computation Wrapper**  
   → Receive `model` and call steps 1–3 to return $(s(M))$

---

## 1. File/Directory Structure

```text
PG-Flow/
├── object_detection_models/
│   ├── code/                     # Model → DAG code
│   ├── Graph/                    # Generated architecture-specific DAGs
│   └── Ops.json                  # Operator list
├── parameter_regenerator/        # Convert architecture-specific weights to DAG-compatible format
└── compute_surrogates.py         # Compute s(M) for multiple model architectures and weights
```

---

## 2. Architecture-Specific Graph Design (`./object_detection_models/code/`)

### 2.1 Operator List Definition

- `0: Conv_1x1`
- `1: Conv_3x3`
- `2: Conv_3x3_DS`
- `3: Conv_7x7_DS`
- `4: Conv_3x3_Dil`
- `5: DWConv_3x3`
- `6: DWConv_3x3_DS`
- `7: DWConv_5x5`
- `8: MaxPool_2x2`
- `9: MaxPool_SPP`
- `10: AvgPool_Global`
- `11: Upsample_2x`
- `12: RoIAlign_7x7`
- `13: BatchNorm`
- `14: LayerNorm`
- `15: Linear`
- `16: ReLU`
- `17: SiLU`
- `18: Sigmoid`
- `19: Concat`
- `20: Add`
- `21: Split_Half`
- `22: Flatten`
- `23: Reshape_Heads`
- `24: Transpose`
- `25: Gather_TopK`
- `26: MatMul`
- `27: SoftMax`
- `28: Mul`
- `29: Div`
- `30: Sub`
- `31: Exp`

### 2.2 Node and Edge Data Structure Definition

**Node Structure:**
- `id: int` – Node ID within the graph  
- `op_idx: int` – Operator ID per node  
- `op_name: str` – Operator name  
- `annotation: str` – Additional description  

**Edge Structure:**
- $[id_i, id_j]$ – $node_i$ (information output node) → $node_j$ (information input node)

### 2.3 Architecture Types

**YOLO Family (Real-time One-stage)**
- YOLOv8 (Anchor-free, SOTA): YOLOv8-n, YOLOv8-m
- YOLOv5 (Anchor-based, Industry Standard): YOLOv5-s, YOLOv5-x
- YOLOX (Decoupled Head, Anchor-free): YOLOX-s, YOLOX-l

**R-CNN Family (Two-stage)**
- Faster R-CNN (Standard Baseline): R50-FPN, R101-FPN
- Cascade R-CNN (High Quality): R50-FPN

**Legacy One-stage (Baselines)**
- SSD: SSD300-VGG16, MobileNetV2-SSDLite
- RetinaNet: R50-FPN (Focal Loss Base)

**EfficientDet (Scalable)**
- EfficientDet: D0, D3 (BiFPN + Compound Scaling)

**Transformer (End-to-End)**
- DETR: R50 (Encoder-Decoder Attention)

---

## 3. Universal Weight Exchange Format (UWEF) Definition (`./parameter_regenerator/`)

### 3.1 File Structure (Root Schema)

The top-level root of the file consists of two keys: metadata (meta) and weight data (node_weights).

```json
{
  "meta": {
    "architecture": "string",
    "format_version": "string",
    "source_framework": "string",
    "created_at": "string"
  },
  "node_weights": {
    "NODE_ID_1": { },
    "NODE_ID_2": { },
    "...": "..."
  }
}
```

### 3.2 Node Weight Object (node_weights)

Use Node IDs specified in the graph definition file (*_graph.json) as keys. Each node contains tensors required for the corresponding operation.

#### 3.2.1 Individual Node Structure

```json
{
  "0": {
    "op_type": "Conv_3x3",
    "has_weight": true,
    "tensors": {
      "weight": { },
      "bias": { },
      "running_mean": { },
      "running_var": { }
    }
  }
}
```

### 3.3 Tensor Object (tensors)

An object containing actual weight values. Includes **dimension information (Shape)** and **data**.

```json
{
  "weight": {
    "dtype": "float32",
    "shape": [64, 3, 3, 3],
    "data": [0.12, -0.05, 0.0, "..."]
  }
}
```

Data stores multidimensional tensors as 1D lists using view(-1) or flatten(). Shape is used to restore the tensor when loaded.

### 3.4 Standard Schema by Operation Type

When writing conversion code, respect the following key names according to operation type.

**A. Convolution / Linear Layers (Conv2d, Linear)**
- Required tensors: weight
- Optional tensors: bias
- Shape rules:
  - Conv: [Out_Channels, In_Channels, Kernel_H, Kernel_W]
  - Linear: [Out_Features, In_Features]

**B. Normalization Layers (BatchNorm, LayerNorm)**
- Required tensors:
  - weight: Scale parameter ($\gamma$)
  - bias: Shift parameter ($\beta$)
- Optional tensors (BN):
  - running_mean: Running mean
  - running_var: Running variance

**C. Operations Without Weights (ReLU, Pooling, Add)**
- has_weight: false
- tensors: {} (empty object)

### 3.5 Special Case Handling Rules (Implementation Rules)

Rules that must be observed when implementing the converter (Exporter).

**1. Transformer Q/K/V Split Storage Principle:**

Even if internally merged into a single large tensor (e.g., in_proj_weight), slice and store according to the graph node definition.

Example: Store the front part (:embed_dim) of the tensor in Q_Node, and the middle part (embed_dim:2*embed_dim) in K_Node.

**2. Shared Parameter Duplicate Storage Principle (Deep Copy):**

Even if multiple nodes share the same parameter (e.g., in EfficientDet Head), record the data redundantly for each node ID entry.

This allows the engine (Step 2) to load data using only the node ID without complex reference logic.

---

## 4. Gate Value Design (Layer 1) – `gate_functions.py`

### 4.0 Gate Design Principles

- Gate uses **only weight $(W)$**
- Gate(W) satisfies:
  - (1) **Use only relative magnitude** → z-score  
  - (2) Value range **[1−β, 1+β]** → Flow does not explode or die to 0  
  - (3) Gate increases monotonically as weight magnitude increases

To achieve this, three gate types are defined:

```python
s_prime = compute_pgflow_surrogate(
    model,
    gate_type="rel_norm",
    beta=0.2,
    lam=1.0,
    gating_pattern="outgoing",
)
```

- **G¹: `rel_norm`** – Relative L2 norm based (basic form)  
- **G²: `scale_norm`** – Scale-invariant norm considering fan-in  
- **G³: `norm_sparsity`** – Norm + sparsity to suppress dead-layer

---

## 4.1 G¹: Relative Norm Gate (`rel_norm`)

> **💡 Core Concept**
> "Modules with weight **energy** larger than average open the gate ($\uparrow$), modules with smaller energy close it ($\downarrow$)."

| Step | Formula | Description |
| :--- | :--- | :--- |
| **1. Representative Value ($e_i$)** | $e_i = \log\left( \frac{\lVert W_i \rVert_F}{\sqrt{\|\theta_i\|}} + \epsilon \right)$ | Frobenius Norm normalized by total parameter count ($\|\theta_i\|$) |
| **2. Z-Score ($\hat{e}_i$)** | $\hat{e}_i = \frac{e_i - \mu_e}{\sigma_e}$ | Relative position within the entire layer distribution |
| **3. Gate ($g_i$)** | $g_i = 1 + \beta \tanh(\lambda \hat{e}_i)$ | $\beta=0.2, \lambda=1.0$ (example) |

**🛠 Implementation Function Mapping**
* `compute_node_stats_rel_norm(nodes)` → $\{node_{id}: e_i\}$
* `normalize_stats(stats)` → $\{node_{id}: \hat{e}_i\}$
* `gate_tanh(normed_stats, beta, lam)` → $\{node_{id}: g_i\}$

---

## 4.2 G²: Scale-Invariant Norm Gate (`scale_norm`)

> **💡 Core Concept**
> "Not just absolute magnitude, but **relative to input (Fan-in)** how much was learned."

| Step | Formula | Description |
| :--- | :--- | :--- |
| **1. Fan-in** | $fan\_in_i = C_{in} \cdot k^2$ | Input receptive field size when $W_i \in \mathbb{R}^{C_{out} \times C_{in} \times k \times k}$ |
| **2. Representative Value** | $e_i = \log\left( \frac{\lVert W_i \rVert_F}{\sqrt{fan\_in_i}} + \epsilon \right)$ | Divide by **Fan-in** instead of $\|\theta_i\|$ to ensure scale-invariance |
| **3. Afterwards** | Same as $G^1$ (Z-score → Gate) | |

**🛠 Implementation Function Mapping**
* `compute_node_stats_scale_norm(nodes)` → $\{node_{id}: e_i\}$
* *Note:* Fallback to $G^1$ approach for non-Conv layers.

---

## 4.3 G³: Norm + Sparsity Gate (`norm_sparsity`)

> **💡 Core Concept**
> "Even if Norm is large, if most values are 0 (Dead), evaluate low; emphasize **densely alive** layers."

| Item | Formula | Description |
| :--- | :--- | :--- |
| **Norm Term** | $e_i^{(N)} = \log\left( \frac{\lVert W_i \rVert_F}{\sqrt{\|\theta_i\|}} + \epsilon \right)$ | Same as $G^1$ representative value |
| **Sparsity Term** | $s_i = \frac{1}{\|\theta_i\|} \sum_{\theta \in i} \mathbf{1}(\|\theta\| < \tau)$ | $\tau \approx 10^{-3}$, ratio of dead parameters |
| **Final Representative Value** | $e_i = e_i^{(N)} - \gamma s_i$ | $\gamma > 0$ (e.g., 0.5), deduct representative value when sparsity is high |

**🛠 Implementation Function Mapping**
* `compute_node_stats_norm_sparsity(nodes, tau, gamma)` → $\{node_{id}: e_i\}$

---

## 📊 Summary: Gate Type Comparison

| Type | Code Name (`gate_type`) | Consideration Factors | Key Characteristics |
| :--- | :--- | :--- | :--- |
| **$G^1$** | `rel_norm` | **Energy (Norm)** | Most basic, reflects average magnitude of parameters |
| **$G^2$** | `scale_norm` | **Input Scale (Fan-in)** | Accounts for Conv filter structure; emphasizes amplification relative to input |
| **$G^3$** | `norm_sparsity` | **Density** | Lowers importance of layers with many dead neurons |

---

## 5. Gate Application Pattern (Layer 2) – `flow_simulator.py`

Once gate values $(g_i)$ (computed from one of G¹~G³) are ready, decide **where in the Flow to multiply them**. This choice is controlled by `gating_pattern`.

```python
s_prime = simulate_pgflow(
    nodes,
    edges,
    gates,
    d_hidden=64,
    gating_pattern="outgoing",
)
```

### 5.1 Common Inputs

| Variable Name | Type | Description |
| :--- | :--- | :--- |
| `nodes` | `List[Node]` | List of graph nodes |
| `edges` | `List[(src_id, dst_id)]` | Edge list based on node IDs |
| `gates` | `Dict[node_id -> g_i]` | Dictionary of computed gate values |
| `d_hidden` | `int` | Hidden vector dimension |
| `gating_pattern` | `str` | `"outgoing"` or `"incoming"` |

### 5.2 Gating Pattern Comparison (P¹ vs P²)

| Pattern | Code Name | Formula | Interpretation |
| :--- | :--- | :--- | :--- |
| **P¹: Outgoing** | `"outgoing"` | $m_i = \sum_{j \in \mathcal{N}_{\text{in}}(i)} g_j f_j$ | Gate reflects **how loudly the sender** ($j$) speaks |
| **P²: Incoming** | `"incoming"` | $m_i = g_i \cdot \sum_{j} f_j$ | Gate reflects **how much the receiver** ($i$) accepts |

### 5.3 Flow Implementation Procedure

1. **Index Mapping:** Convert node.id → 0..N-1 indices
2. **Edge Conversion:** edges → edges_idx
3. **Initialization:** Assign initial message `torch.randn` to input nodes
4. **Topological Sort:** Perform topological_sort(num_nodes, edges_idx)
5. **Iteration and Computation:**
   - **Outgoing:** $msgs = f_j \cdot g_j$ then Sum
   - **Incoming:** Sum$(f_j)$ then $\cdot g_i$
6. **Final Output:** Aggregate and normalize input node messages

$$s(M) = \frac{s_{\text{prime}}}{\|s_{\text{prime}}\| + \epsilon}$$