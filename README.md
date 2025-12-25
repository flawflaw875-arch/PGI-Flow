# PGI-Flow Surrogates
Predicting Object Detection Transferability via Parameter-Gated Information Flow Surrogates

## 1. Research Background and Problem Definition

Object detection models in construction sites experience significant performance degradation when deployed in new sites (Target Sites) after being trained on a specific site (Source Site) due to differences in lighting, camera angles, equipment types, and worker attire. Re-training the model for each new site or testing all models individually incurs substantial time and cost.

This project aims to address this issue by developing a system that predicts the transfer performance of a trained model using only the parameter information of the model and the visual features of the target site, without actual deployment or re-training.

## 2. Key Ideas and Methodology

This research proposes the following key ideas:

1.  **'Parameter-Gated Information Flow Surrogate' (`s`)**:
    *   **Overcoming Limitations of Existing Research:** Previous studies primarily analyzed the architecture's structure or used only the final output/single-layer features to predict transfer performance. This often failed to reflect how the model was actually trained (parameter information), leading to the underestimation of the potential of 'shallow expert' models.
    *   **Differentiation of This Study:** We integrate the statistical characteristics of the trained model's parameters (e.g., weight norms, variance) into the information flow simulation. This generates a unique **'fingerprint' (`s`)** formed by the results of training on a specific dataset, representing not only the model's structure but also the intrinsic characteristics and learning experiences of the model.

2.  **Target Dataset Features (`f`)**:
    *   Visual features are extracted from unlabeled images of the new target site to create a **feature vector (`f`)** that represents the dataset of that site. This process identifies the characteristics of the site without labeling costs.

3.  **Learning the 'Discriminator'(`P`)**:
    *   `P` is a regression model that takes the fingerprint of the source model `s` and the features of the target dataset `f` as input to predict the expected transfer performance of the source model in the target site.
    *   **Supporting Strategic Decision-Making:** `P` goes beyond merely predicting performance; it provides strategic insights into which type, single-site expert model or multi-site generalist model, is more suitable for the new site. of model

## 3. Definition of Parameter-Gated Information Flow Surrogate

Neural networks can be represented as Directed Acyclic Graphs (DAGs). Hwang et al. proposed a method to define the node features matrix $\mathbf{X} \in \\{0, 1\\}^{|\mathcal{V}| \times |\mathcal{O}|}$ and edges matrix $\mathcal{E} \in \\{0, 1\\}^{|\mathcal{V}| \times |\mathcal{V}|}$ for neural network architectures [1]. In this matrix, each column corresponds to a specific operation, and each row is a one-hot vector indicating the operation type associated with the corresponding node.

![The way of Graph representation](./images/GR.png)

Extending the work of Hwang et al., Kim et al. proposed a method to estimate the potential performance of a neural architecture on a specific dataset by generating a 'Flow Surrogate' through information flow simulation [2]. 

![Topological Order Assignment](./images/TOA.png)

First, a topological order is assigned to the nodes of the architecture. Specifically, if the maximum order among the incoming nodes connected to a target node is $N$, the order of that target node is defined as $N+1$. 

To mimic the information flow, node embeddings and input messages are randomly initialized, and a random matrix $\mathbf{P} \in \\mathbb{R}^{|\mathcal{O}| \times k}$ is generated. 

Here, $P = \\{P_1, \dots, P_{|\mathcal{O}|}\\}$ denotes a set of vectors representing distinct operation primitives. Each vector $P_i$ is randomly sampled from a normal distribution:

$$
P_i \sim \mathcal{N}(0, \sigma^2)
$$

Subsequently, the node embedding matrix $H$ is computed as:

$$
H = \mathbf{P}\mathbf{X}
$$

Finally, let $f_i \in \mathbb{R}^k$ denote the info-message of each node. We initialize the info-messages of order-1 nodes with a randomly sampled vector $r \in \mathbb{R}^k$:

$$
f_j = r, \quad \forall v_j \in \mathcal{V}^{(1)}
$$

Subsequently, by iterating through the topological orders, we compute the arriving info-message $m_i$ for a node $v_i \in \mathcal{V}^{(T)}$ as follows:

$$
m_i = \sum_{v_j \in \mathcal{N}^{(i)}} f_j, \quad \text{where } \mathcal{N}^{(i)} = \{v_j : E_{ji} = 1\}
$$

Subsequently, we convert the arriving info-message $m_i$ based on the operation of the node $v_i$:

$$
f_i = \alpha m_i + (1 - \alpha)\text{ReLU}([h_i \| m_i]W)
$$

Here, $[a \| b]$ represents the concatenation of vectors $a$ and $b$, $W \in \mathbb{R}^{2k \times k}$ is a fixed projection matrix, and $\alpha \in [0, 1]$ is a weighting hyperparameter.

Finally, the info-message appearing at the output constitutes the **Information Flow Surrogate**.

Extending the framework of Kim et al., we propose the **Parameter-Gated Information Flow Surrogate ($s$)** to evaluate the transferability of trained models. Our approach introduces two key modifications:

1.  **Gating Mechanism:** We utilize pre-trained weights as gates during the propagation of info-messages, allowing the system to amplify or modulate existing signals.
2.  **Target-Aware Initialization:** Instead of using random initialization, we employ embeddings derived from the target dataset as the input values.

## 4. Implementation Plan

**Action Plan**

> About trained mode $(M)$ (ex. YOLO/DETR/Faster R-CNN etc)  
> Using **Architecture DAG + weight based gate** 
> Build calculator module **parameter-gated information flow surrogate** $(s(M))$

### 4.0. Overall Architecture Overview

There are 4 components to implement:

1. **Graph Builder** (`./object_detection/models/code`)  
   → Object detection model architecture → Neural architecture graph: nodes features $(\mathbf{X}$), edges $(\mathcal{E}$), Operator $(\mathcal{O}$)

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

### 4.1. File/Directory Structure

```text
PGI-Flow/
├── object_detection_models/
│   ├── code/                     # Model → DAG code
│   ├── Graph/                    # Generated architecture-specific DAGs
│   └── Ops.json                  # Operator list
├── parameter_regenerator/        # Convert architecture-specific weights to DAG-compatible format
└── compute_surrogates.py         # Compute s(M) for multiple model architectures and weights
```

---

### 4.2. Architecture-Specific Graph Design (`./object_detection_models/code/`)

#### 4.2.1 Operator List Definition

| ID | Operator       | 수식(대표 정의)                                                                    | Trainable?                         | 사용 아키텍처(구체)                                                                                                                                                                                                |
| -: | -------------- | ---------------------------------------------------------------------------- | ---------------------------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
|  0 | Conv_1x1       | (y_{o,h,w}=\sum_i W_{o,i}x_{i,h,w}+b_o)                                      | Yes (W,b*)                         | **전부**                                                                                                                                                                                                     |
|  1 | Conv_3x3       | (y_{o,h,w}=\sum_{i,u,v} W_{o,i,u,v}x_{i,h+u,w+v}+b_o)                        | Yes                                | **전부**                                                                                                                                                                                                     |
|  2 | Conv_3x3_DS    | stride=2: (y_{o,h,w}=\sum_{i,u,v} W_{o,i,u,v}x_{i,2h+u,2w+v}+b_o)            | Yes                                | **전부** *(SSD extra layer에도 stride2 conv 존재)* ([mmdetection.readthedocs.io][1])                                                                                                                             |
|  3 | Conv_7x7_DS    | stride=2 7×7 conv                                                            | Yes                                | **FR50/FR101/CR50/CR101/Retina**                                                                                                                                                                           |
|  4 | Conv_3x3_Dil   | dilation d: (x_{i,h+d u,w+d v}) 참조                                           | Yes                                | **SSD** (MMDet SSDVGG에 dilation=6 conv 존재) ([mmdetection.readthedocs.io][1])                                                                                                                               |
|  5 | DWConv_3x3     | depthwise: (y_{c,h,w}=\sum_{u,v}K_{c,u,v}x_{c,h+u,w+v}+b_c)                  | Yes                                | **SSDLite, ED0, ED3**                                                                                                                                                                                      |
|  6 | DWConv_3x3_DS  | stride=2 depthwise                                                           | Yes                                | **SSDLite, ED0, ED3**                                                                                                                                                                                      |
|  7 | DWConv_5x5     | depthwise 5×5                                                                | Yes                                | **ED0, ED3** *(EfficientNet backbone에 k=5 MBConv stage 존재 — 일반적 구성)*                                                                                                                                       |
|  8 | DWConv_5x5_DS  | stride=2 depthwise 5×5                                                       | Yes                                | **ED0, ED3**                                                                                                                                                                                               |
|  9 | MaxPool_2x2_DS | (y_{c,h,w}=\max_{(u,v)\in2\times2}x_{c,2h+u,2w+v})                           | No                                 | **SSD** (VGG pool1~ 등)                                                                                                                                                                                     |
| 10 | MaxPool_3x3    | (보통 s=1) (y=\max_{(u,v)\in3\times3}x)                                        | No                                 | **SSD** (MMDet SSDVGG에 k=3,s=1,p=1 maxpool 존재) ([mmdetection.readthedocs.io][1])                                                                                                                           |
| 11 | MaxPool_3x3_DS | stride=2 (y=\max x)                                                          | No                                 | **FR50/FR101/CR50/CR101/Retina** (ResNet stem maxpool k=3,s=2,p=1) ([GitHub][2]) / **ED0/ED3**(BiFPN downsample에서 흔함)                                                                                      |
| 12 | MaxPool_5x5    | stride=1,pad=2 (y=\max_{5\times5}x)                                          | No                                 | **Y5s/Y5x** (SPPF: MaxPool2d k=5,s=1,p=2) ([Hugging Face][3]) / **Y8n/Y8m** (Ultralytics SPPF 동일 구현) ([docs.ultralytics.com][4]) / **X-s/X-l** (SPP의 kernel_sizes에 5 포함) ([mmdetection.readthedocs.io][5]) |
| 13 | MaxPool_9x9    | stride=1,pad=4                                                               | No                                 | **X-s/X-l** (MMDet CSPDarknet SPP default kernel_sizes=(5,9,13)) ([mmdetection.readthedocs.io][5])                                                                                                         |
| 14 | MaxPool_13x13  | stride=1,pad=6                                                               | No                                 | **X-s/X-l** (동일) ([mmdetection.readthedocs.io][5])                                                                                                                                                         |
| 15 | AvgPool_Global | (y_c=\frac{1}{HW}\sum_{h,w}x_{c,h,w})                                        | No                                 | **ED0/ED3** (SE 등)                                                                                                                                                                                         |
| 16 | Upsample_2x    | nearest: (y_{c,2h+i,2w+j}=x_{c,h,w})                                         | No                                 | **Y8n/Y8m/Y5s/Y5x/X-s/X-l**, **FR/CR/Retina**, **ED0/ED3**                                                                                                                                                 |
| 17 | RoIAlign_7x7   | bin 샘플링 + bilinear: (y=\sum_p w_p x(p))                                      | No                                 | **FR50/FR101/CR50/CR101**                                                                                                                                                                                  |
| 18 | BatchNorm      | (y=\gamma\frac{x-\mu}{\sqrt{\sigma^2+\epsilon}}+\beta)                       | Yes (γ,β)                          | **Y8/Y5/X**, **FR/CR**, **SSDLite**, **Retina(구현/설정에 따라 head BN 포함 가능)**, **ED0/ED3**                                                                                                                      |
| 19 | ReLU           | (y=\max(0,x))                                                                | No                                 | **FR/CR/Retina**, **SSD**, (ED0/ED3에서 fusion-weight 비음수화에 ReLU를 쓰는 fast fusion이면 포함) ([CVF Open Access][6])                                                                                                |
| 20 | ReLU6          | (y=\min(\max(0,x),6))                                                        | No                                 | **SSDLite**                                                                                                                                                                                                |
| 21 | SiLU           | (y=x\cdot\sigma(x))                                                          | No                                 | **Y8/Y5/X**, **ED0/ED3**                                                                                                                                                                                   |
| 22 | Sigmoid        | (\sigma(x)=\frac{1}{1+e^{-x}})                                               | No                                 | **ED0/ED3** (SE gating 등 “네트워크 내부”에서 확실)                                                                                                                                                                   |
| 23 | Add            | (y=x_1+x_2)                                                                  | No                                 | **Y8/Y5/X**, **FR/CR/Retina**, **SSDLite**, **ED0/ED3**                                                                                                                                                    |
| 24 | Concat         | (y=\text{concat}(x_1,x_2,\dots)) (채널)                                        | No                                 | **Y8/Y5/X** *(CSP, PAN, SPP/SPPF에서 필수)*                                                                                                                                                                    |
| 25 | Split_Channel  | (x\rightarrow (x^{(1)},x^{(2)})) (채널 분기)                                     | No                                 | **Y8/Y5/X** *(CSP류 구조에서 필수)*                                                                                                                                                                               |
| 26 | Mul            | (y=x_1\odot x_2)                                                             | No(연산) / (단, 곱해지는 **w**가 학습일 수 있음) | **ED0/ED3** (SE, BiFPN weighted fusion)                                                                                                                                                                    |
| 27 | Div            | (y=\frac{x_1}{x_2+\epsilon})                                                 | No                                 | **ED0/ED3** (BiFPN fast normalized fusion: (w_i/(\epsilon+\sum w_j))) ([CVF Open Access][6])                                                                                                               |
| 28 | Flatten        | (x\in\mathbb{R}^{C\times7\times7}\rightarrow \mathbb{R}^{C\cdot49})          | No                                 | **FR/CR**                                                                                                                                                                                                  |
| 29 | Reshape        | 텐서 shape 변경(값 불변)                                                            | No                                 | **Y5**                                                                                                                                                                                    |
| 30 | Linear         | (y=Wx+b)                                                                     | Yes                                | **FR/CR** (RoI FC head)                                                                                                                                                                                    |
| 31 | L2Norm         | (위치별) (y_{c,h,w}=s_c\frac{x_{c,h,w}}{\sqrt{\sum_{c'}x_{c',h,w}^2+\epsilon}}) | **Yes** (채널별 (s_c))                | **SSD** (MMDet SSDVGG에 L2Norm 모듈 + trainable weight 존재) ([mmdetection.readthedocs.io][1])                                                                                                                  |
| 32 | out            | 최종 결과 선정(Decode/NMS/TopK/Threshold/score 정규화 등) 전체                           | (정의상) No                           | **전부(공통)**                                                                                                                                                                                                 |

#### 4.2.2 Node and Edge Data Structure Definition

**Node Structure:**
- `id: int` – Node ID within the graph  
- `op_idx: int` – Operator ID per node  
- `op_name: str` – Operator name  
- `annotation: str` – Additional description  

**Edge Structure:**
- $[id_i, id_j]$ – $node_i$ (information output node) → $node_j$ (information input node)

#### 4.2.3 Architecture Types

**YOLO Family (Real-time One-stage)**
- YOLOv8 (Anchor-free, SOTA): YOLOv8-n, YOLOv8-m
- YOLOv5 (Anchor-based, Industry Standard): YOLOv5-s, YOLOv5-x
- YOLOX (Decoupled Head, Anchor-free): YOLOX-s, YOLOX-l

**R-CNN Family (Two-stage)**
- Faster R-CNN (Standard Baseline): R50-FPN, R101-FPN
- Cascade R-CNN (High Quality): R50-FPN, R101-FPN

**Legacy One-stage (Baselines)**
- SSD: SSD300-VGG16, MobileNetV2-SSDLite
- RetinaNet: R50-FPN (Focal Loss Base)

**EfficientDet (Scalable)**
- EfficientDet: D0, D3 (BiFPN + Compound Scaling)

---

### 4.3. Universal Weight Exchange Format (UWEF) Definition (`./parameter_regenerator/`)

#### 4.3.1 File Structure (Root Schema)

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

#### 4.3.2 Node Weight Object (node_weights)

Use Node IDs specified in the graph definition file (*_graph.json) as keys. Each node contains tensors required for the corresponding operation.

* Individual Node Structure

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

#### 4.3.3 Tensor Object (tensors)

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

#### 4.3.4 Standard Schema by Operation Type

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

#### 4.3.5 Special Case Handling Rules (Implementation Rules)

Rules that must be observed when implementing the converter (Exporter).

**1. Transformer Q/K/V Split Storage Principle:**

Even if internally merged into a single large tensor (e.g., in_proj_weight), slice and store according to the graph node definition.

Example: Store the front part (:embed_dim) of the tensor in Q_Node, and the middle part (embed_dim:2*embed_dim) in K_Node.

**2. Shared Parameter Duplicate Storage Principle (Deep Copy):**

Even if multiple nodes share the same parameter (e.g., in EfficientDet Head), record the data redundantly for each node ID entry.

This allows the engine (Step 2) to load data using only the node ID without complex reference logic.

---

### 4.4. Gate Value Design (Layer 1) – `gate_functions.py`

#### 4.4.0 Gate Design Principles

- Gate uses **only weight $(W)$**
- Gate(W) satisfies:
  - (1) **Use only relative magnitude** → z-score  
  - (2) Value range **[1−β, 1+β]** → Flow does not explode or die to 0  
  - (3) Gate increases monotonically as weight magnitude increases

To achieve this, three gate types are defined:

```python
s_prime = compute_pgiflow_surrogate(
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

#### 4.4.1 G¹: Relative Norm Gate (`rel_norm`)

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

#### 4.4.2 G²: Scale-Invariant Norm Gate (`scale_norm`)

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

#### 4.4.3 G³: Norm + Sparsity Gate (`norm_sparsity`)

> **💡 Core Concept**
> "Even if Norm is large, if most values are 0 (Dead), evaluate low; emphasize **densely alive** layers."

| Item | Formula | Description |
| :--- | :--- | :--- |
| **1. Norm Term** | $e_i^{(N)} = \log\left( \frac{\lVert W_i \rVert_F}{\sqrt{\|\theta_i\|}} + \epsilon \right)$ | Same as $G^1$ representative value |
| **2. Sparsity Term** | $s_i = \frac{1}{\|\theta_i\|} \sum_{\theta \in i} \mathbf{1}(\|\theta\| < \tau)$ | $\tau \approx 10^{-3}$, ratio of dead parameters |
| **3. Final Representative Value** | $e_i = e_i^{(N)} - \gamma s_i$ | $\gamma > 0$ (e.g., 0.5), deduct representative value when sparsity is high |
| **4. Afterwards** | Same as $G^1$ (Z-score → Gate) | |

**🛠 Implementation Function Mapping**
* `compute_node_stats_norm_sparsity(nodes, tau, gamma)` → $\{node_{id}: e_i\}$

---

#### 📊 Summary: Gate Type Comparison

| Type | Code Name (`gate_type`) | Consideration Factors | Key Characteristics |
| :--- | :--- | :--- | :--- |
| **$G^1$** | `rel_norm` | **Energy (Norm)** | Most basic, reflects average magnitude of parameters |
| **$G^2$** | `scale_norm` | **Input Scale (Fan-in)** | Accounts for Conv filter structure; emphasizes amplification relative to input |
| **$G^3$** | `norm_sparsity` | **Density** | Lowers importance of layers with many dead neurons |

---

### 4.5. Gate Application Pattern (Layer 2) – `flow_simulator.py`

Once gate values $(g_i)$ (computed from one of G¹~G³) are ready, decide **where in the Flow to multiply them**. This choice is controlled by `gating_pattern`.

```python
s_prime = simulate_pgiflow(
    nodes,
    edges,
    gates,
    d_hidden=64,
    gating_pattern="outgoing",
)
```

#### 4.5.1 Common Inputs

| Variable Name | Type | Description |
| :--- | :--- | :--- |
| `nodes` | `np.array` | N x D |
| `edges` | `np.array` | N x N |
| `gates` | `Dict[node_id -> g_i]` | Dictionary of computed gate values |
| `d_hidden` | `int` | Hidden vector dimension |
| `gating_pattern` | `str` | `"outgoing"` or `"incoming"` |

#### 4.5.2 Gating Pattern Comparison (P¹ vs P²)

| Pattern | Code Name | Formula | Interpretation |
| :--- | :--- | :--- | :--- |
| **P¹: Outgoing** | `"outgoing"` | $m_i = \sum_{j \in \mathcal{N}_{\text{in}}(i)} g_j f_j$ | Gate reflects **how loudly the sender** ($j$) speaks |
| **P²: Incoming** | `"incoming"` | $m_i = g_i \cdot \sum_{j} f_j$ | Gate reflects **how much the receiver** ($i$) accepts |

#### 4.5.3 Flow Implementation Procedure

1. **Index Mapping:** Convert node.id → 0..N-1 indices
2. **Edge Conversion:** edges → edges_idx
3. **Initialization:** Assign initial message `torch.randn` to input nodes
4. **Topological Sort:** Perform topological_sort(num_nodes, edges_idx)
5. **Iteration and Computation:**
   - **Outgoing:** $m = f_j \cdot g_j$ then Sum
   - **Incoming:** $m = sum f_j$ then $\cdot g_i$
6. **Final Output:** Aggregate and normalize input node messages

## Reference

[1] Hwang, Dongyeong, et al. "Flowerformer: Empowering neural architecture encoding using a flow-aware graph transformer." Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition. 2024.
[2] Kim, Sunwoo, Hyunjin Hwang, and Kijung Shin. "Learning to Flow from Generative Pretext Tasks for Neural Architecture Encoding." arXiv preprint arXiv:2510.18360 (2025).
