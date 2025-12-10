📄 공통 웨이트 교환 포맷 명세서 (UWEF) v1.0
이 문서는 다양한 아키텍처(YOLO, DETR, EfficientDet 등)의 가중치 정보를 PG-Flow 엔진이 읽을 수 있도록 표준화한 JSON 파일 형식을 정의합니다.

1. 파일 구조 (Root Schema)
파일의 최상위 루트는 **메타데이터(meta)**와 가중치 데이터(node_weights) 두 가지 키로 구성됩니다.

{
  "meta": {
    "architecture": "string",       // 예: "detr_r50"
    "format_version": "string",     // 예: "1.0"
    "source_framework": "string",   // 예: "pytorch"
    "created_at": "string"          // 예: "2025-10-21T14:30:00"
  },
  "node_weights": {
    "NODE_ID_1": { ... },           // 그래프의 Node ID (문자열)
    "NODE_ID_2": { ... },
    ...
  }
}

2. 노드 가중치 객체 (node_weights)
그래프 정의 파일(*_graph.json)에 명시된 Node ID를 Key로 사용합니다. 각 노드는 해당 연산에 필요한 텐서들을 포함합니다.

2.1 개별 노드 구조

"0": {
  "op_type": "Conv_3x3",   // 연산 종류 (필수)
  "has_weight": true,      // 가중치 존재 여부 (필수)
  "tensors": {             // 텐서 데이터 맵 (필수)
    "weight": { ... },     // 핵심 가중치 (Key 이름 고정)
    "bias": { ... },       // (선택) 바이어스
    "running_mean": { ... }, // (선택) BN 통계
    "running_var": { ... }   // (선택) BN 통계
  }
}

3. 텐서 객체 (tensors)
실제 가중치 값을 담는 객체입니다. 텐서의 **차원 정보(Shape)**와 **데이터(Data)**를 포함합니다.

"weight": {
  "dtype": "float32",            // 데이터 타입 ("float32", "float16")
  "shape": [64, 3, 3, 3],        // 텐서 형상 [Out, In, K, K] 등
  "data": [0.12, -0.05, 0.0, ...] // Flattened (1차원) 리스트
}

* data: 다차원 텐서를 view(-1) 또는 flatten()하여 1차원 리스트로 저장합니다. 로드 시 shape를 이용해 복원합니다.

4. 연산 타입별 표준 스키마
변환 코드 작성 시, 연산 타입에 따라 다음 키(Key) 이름을 준수해야 합니다.

A. 컨볼루션 / 선형 레이어 (Conv2d, Linear)
* 필수 텐서: weight

* 선택 텐서: bias

* Shape 규칙:
    * Conv: [Out_Channels, In_Channels, Kernel_H, Kernel_W]
    * Linear: [Out_Features, In_Features]

B. 정규화 레이어 (BatchNorm, LayerNorm)
* 필수 텐서:
    * weight: Scale 파라미터 ($\gamma$)
    * bias: Shift 파라미터 ($\beta$)

* 선택 텐서 (BN):
    * running_mean: 이동 평균
    * running_var: 이동 분산

C. 가중치가 없는 연산 (ReLU, Pooling, Add)
* has_weight: false
* tensors: {} (빈 객체)

5. 특수 케이스 처리 규칙 (Implementation Rules)
변환기(Exporter) 구현 시 반드시 지켜야 할 규칙입니다.

 1. Transformer Q/K/V 분할 저장 원칙:

    * 프레임워크 내부에서 하나의 큰 텐서(예: in_proj_weight)로 합쳐져 있더라도, 그래프 노드 정의에 맞춰 잘라서(Slicing) 저장해야 합니다.

    * 예: Q_Node에는 텐서의 앞부분(:embed_dim), K_Node에는 중간 부분(embed_dim:2*embed_dim)을 저장합니다.

 2. 공유 파라미터 중복 저장 원칙 (Deep Copy):

    * EfficientDet의 Head처럼 여러 노드가 동일한 파라미터를 공유하더라도, 각 노드 ID 항목에 데이터를 중복하여 기록합니다.

    * 이는 엔진(Step 2)이 복잡한 참조 로직 없이 노드 ID만으로 데이터를 로드할 수 있게 하기 위함입니다.