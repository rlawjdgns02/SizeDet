# SizeDet: 크기 기반 정보를 통한 회전 객체 탐지 개선 연구

> 본 저장소는 회전 객체 탐지(Oriented Object Detection)에서 크기에 기반한 수정 및 적용을 통해 정확도(localization accuracy)를 개선하는 연구를 담고 있습니다.

## 개요

본 프로젝트는 회전 객체 탐지에서 **크기 예측 오차를 줄이는 것을 타겟**으로 진행되는 방법론을 검증하는 연구입니다.
**DOTA v2.0** 데이터셋(위성이미지) 을 사용합니다.

## 연구 배경

### 초기 동기
- **목표**: DOTA 데이터셋에서 tiny object detection 성능 개선
- **가설**: 객체의 통상적인 크기 정보를 활용할 수 있다면 이게 기반하여 더욱 빠르고 정교한 bbox 예측이 가능할 것

### 현재 연구 초점
Tiny object detection을 직접 다루기보다는, **localization 정확도와 detection 성능의 근본적인 관계**를 검증하는 데 집중하고 있습니다.

## 방법론

### 주요 수정 사항

1. **Area+Ratio Loss 항 추가** ([gsd_loss.py](mmdet/models/losses/gsd_loss.py))
   - 예측값을 통하여 ground truth의 면적값과의 오차인 Area Loss 활용
   - 같은 면적값을 만드는 정답이 아닌 조합들에 대비하여 width/height의 비율 활용을 위한 Ratio Loss 활용

2-1. **FC Layer 분리** ([obb_decoup_convfc_bbox_head.py](mmdet/models/roi_heads/bbox_heads/obb/obb_decoup_convfc_bbox_head.py))
   - 중심점(cx, cy, θ)과 크기(w, h) 예측 브랜치를 분리
   - Oriented R-CNN의 2nd stage에서 중심점과 크기 예측 간 간섭(interference) 감소

2-2. **Layer Normalization 적용**
  - 분리된 브랜치에 LayerNorm 적용
  - 각 예측 태스크의 feature distribution 안정화

## 실험 결과

### 시도한 방법들
1. **Area + Ratio Loss 추가**: width와 height에 대한 명시적 loss term 추가 - ❌ 효과 없음
2-1. **FC Layer 분리**: 중심점 vs 크기 예측을 위한 shared FC layer 분리 - ❌ 효과 없음
2-2. **분리 + LayerNorm**: 분리된 브랜치에 LayerNorm 적용 - ⚠️ 역설적 결과

### 역설적인 발견 (**방법 2.2 케이스**)
- ✅ **평균 크기 오차 (픽셀)**: 감소
- ✅ **평균 중심점 오차 (픽셀)**: 감소
- ❌ **mAP**: 개선되지 않음 (때로는 하락)

**핵심 질문**: 왜 더 나은 localization이 더 나은 detection 성능으로 이어지지 않는가? 추가 분석이 필요합니다.

## 설치 방법

### 사전 요구사항
- Ubuntu 20.04+ (Ubuntu 24.04에서 테스트됨)
- NVIDIA GPU with CUDA 지원
- Python 3.7

### 방법 1: Conda 사용 (권장)

#### Step 1: Miniconda 설치
```bash
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
bash Miniconda3-latest-Linux-x86_64.sh
```

#### Step 2: 환경 생성
```bash
conda create -n sizedet python=3.7.12
conda activate sizedet
```

#### Step 3: CUDA Toolkit 설치
```bash
conda install cudatoolkit=11.1.1 cudatoolkit-dev=11.1.1 -c conda-forge
```

#### Step 4: PyTorch 설치
```bash
# 중요: 반드시 이 버전 조합을 사용해야 합니다
conda install pytorch==1.9.0 torchvision==0.10.0 cudatoolkit=11.1 -c pytorch
```

#### Step 5: GCC 10 설치 (mmcv-full 컴파일에 필요)
```bash
sudo apt install gcc-10 g++-10
sudo update-alternatives --install /usr/bin/gcc gcc /usr/bin/gcc-10 10
sudo update-alternatives --install /usr/bin/g++ g++ /usr/bin/g++-10 10
```

#### Step 6: mmcv-full 설치
```bash
# 소스에서 컴파일 - 시간이 걸립니다!
pip install mmcv-full==1.3.9 -f https://download.openmmlab.com/mmcv/dist/cu111/torch1.9.0/index.html
```

#### Step 7: 의존성 패키지 설치
```bash
pip install addict==2.4.0 cython==3.0.12 einops==0.6.1
pip install matplotlib==3.5.3 numpy==1.21.6 scipy==1.7.3
pip install opencv-python==4.13.0.90 shapely==2.0.7
pip install pycocotools==2.0.7 terminaltables==3.1.10
pip install yapf==0.31.0 tqdm==4.67.1
```

#### Step 8: BboxToolkit 설치
```bash
cd BboxToolkit
pip install -e .
cd ..
```

#### Step 9: SizeDet (mmdet) 설치
```bash
pip install -e .
```

#### Step 10: 설치 확인
```python
import torch
print(torch.__version__)           # 1.9.0
print(torch.cuda.is_available())   # True
print(torch.version.cuda)          # 11.1

import mmcv
print(mmcv.__version__)            # 1.3.9

import mmdet
print(mmdet.__version__)           # 2.2.0+unknown
```

### 환경 세부사항
- **Python**: 3.7.12
- **PyTorch**: 1.9.0
- **CUDA**: 11.1
- **mmcv-full**: 1.3.9
- **GCC**: 10.5.0 (mmcv 컴파일에 필수)

> **주의**: 버전 매칭이 매우 중요합니다. PyTorch나 CUDA 버전을 변경하면 mmcv-full을 재컴파일해야 합니다.

## 데이터셋 준비

### DOTA 데이터셋
DOTA v2.0 데이터셋을 다운로드하고 다음과 같이 구성합니다:
```
data/
└── dota/
    ├── train/
    ├── val/
    └── test/
```

데이터 전처리 도구는 [BboxToolkit](BboxToolkit/)을 참고하세요.

## 사용 방법

### 사전학습 가중치 다운로드
GRA-ResNet50 사전학습 가중치 다운로드:
- [Google Drive](https://drive.google.com/file/d/15wGWyPJPQF0ORV8LcPp5BWOtl7rW8ht5/view?usp=sharing)
- 프로젝트 루트에 `checkpoint-model.pth`로 저장

### 학습
```bash
# DOTA v2.0에서 decoupled bbox head로 학습
python tools/train.py configs/obb/gra/gra_orcnn_r50fpn1x_ss_dota20.py
```

### 테스트
```bash
python tools/test.py configs/obb/gra/gra_orcnn_r50fpn1x_ss_dota20.py \
    YOUR_CHECKPOINT_PATH \
    --format-only \
    --options save_dir=YOUR_SAVE_DIR
```

### 설정 파일
- **Decoupled Head (본 연구)**: [gra_orcnn_r50fpn1x_ss_dota20.py](configs/obb/gra/gra_orcnn_r50fpn1x_ss_dota20.py)
  - LayerNorm이 적용된 `OBBDecoupBBoxHead` 사용
  - 중심점과 크기 예측을 위한 분리된 브랜치
- **Baseline**: [gra_orcnn_r50fpn1x_ss_dota10.py](configs/obb/gra/gra_orcnn_r50fpn1x_ss_dota10.py)
  - GRA backbone을 사용한 표준 Oriented R-CNN - **DOTA v1.0 기준**


## Acknowledgement

본 코드는 다음의 우수한 저장소들을 기반으로 개발되었습니다:

### 기반 구현
- **[GRA (Group-wise Rotating and Attention)](https://arxiv.org/pdf/2403.11127)** (ECCV 2024)
  - 저자: Jiangshan Wang, Yifan Pu, Yizeng Han, Jiayi Guo, Yiru Wang, Xiu Li, Gao Huang
  - 회전 객체 탐지를 위한 backbone으로 사용

- **[MMDetection](https://github.com/open-mmlab/mmdetection)**
  - Open MMLab Detection Toolbox
  - 기본 프레임워크 제공

- **[ARC](https://github.com/LeapLabTHU/ARC)**
  - 회전 객체 탐지 유틸리티 및 설정

### 참고사항
본 저장소는 GRA 저장소에서 추출한 **수정 및 최소화 버전**으로, 크기 예측 실험에 필요한 핵심 구성요소만 포함하고 있습니다. 본 연구의 핵심 기여는 decoupled bbox head 설계 등등이며, GRA backbone 자체가 아닙니다.

## 인용

본 코드를 연구에 사용하시는 경우, 원본 GRA 논문을 인용해주세요:

```bibtex
@article{wang2024gra,
  title={GRA: Detecting Oriented Objects through Group-wise Rotating and Attention},
  author={Wang, Jiangshan and Pu, Yifan and Han, Yizeng and Guo, Jiayi and Wang, Yiru and Li, Xiu and Huang, Gao},
  journal={arXiv preprint arXiv:2403.11127},
  year={2024}
}
```

## 라이선스

본 프로젝트는 원본 MMDetection 저장소의 Apache License 2.0을 따릅니다. 자세한 내용은 [LICENSE](LICENSE)를 참고하세요.

## 문의

본 연구에 관한 질문이나 이슈가 있으시면:
- 본 저장소에 issue를 등록해주세요
- 본 연구는 진행 중이며 결과는 예비적(preliminary)입니다
