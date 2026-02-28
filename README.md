<div align="center">

# 🎯 SizeDet

**크기 예측을 통한 회전 객체 탐지 성능 개선 연구**

[![Python](https://img.shields.io/badge/Python-3.7.12-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-1.9.0-red.svg)](https://pytorch.org/)
[![CUDA](https://img.shields.io/badge/CUDA-11.1-green.svg)](https://developer.nvidia.com/cuda-toolkit)
[![License](https://img.shields.io/badge/License-Apache%202.0-yellow.svg)](LICENSE)

*회전 객체 탐지에서 크기 예측 정확도가 detection 성능에 미치는 영향을 탐구합니다*

[**설치 가이드**](docs/INSTALL.md) | [**환경 상세**](docs/ENVIRONMENT.md) | [**설정 파일**](configs/obb/gra/)

</div>

---

## 📌 개요

**SizeDet**은 회전 객체 탐지(Oriented Object Detection, OBB)에서 **크기(width, height) 예측 오차를 줄이는 것**이 최종 탐지 성능에 어떤 영향을 미치는지 검증하는 연구입니다.

### 핵심 질문

> *"객체의 크기 정보를 활용할 수 있다면 더 정교한 localization이 가능하지 않을까?"*

### 사용 데이터셋

- **DOTA v2.0** (Dataset for Object deTection in Aerial images)
- 위성/항공 이미지 기반 회전 객체 탐지 벤치마크

---

## 🚀 Quick Start

```bash
# 전체 설치 가이드: docs/INSTALL.md 참고
python tools/train.py configs/obb/gra/gra_orcnn_r50fpn1x_ss_dota20.py
```

> 📚 **설치 가이드**: [docs/INSTALL.md](docs/INSTALL.md) | **환경 정보**: [docs/ENVIRONMENT.md](docs/ENVIRONMENT.md)

---

## 🔬 연구 배경

### 초기 동기
- **목표**: DOTA 데이터셋에서 tiny object detection 성능 개선
- **가설**: 객체의 통상적인 크기 정보를 활용하면 더 정교한 bbox 예측이 가능할 것

### 현재 연구 초점
Tiny object detection보다는 **localization 정확도와 detection 성능의 근본적인 관계**를 검증하는 데 집중하고 있습니다.

---

## 💡 방법론

### 시도한 접근법

#### 1️⃣ Area + Ratio Loss 추가
**구현**: [gsd_loss.py](mmdet/models/losses/gsd_loss.py)

- **Area Loss**: Ground truth와 예측 bbox의 면적 오차 최소화
- **Ratio Loss**: Width/height 비율 정보 활용 (동일 면적 다른 형태 구분)

#### 2️⃣ Decoupled FC Layer
**구현**: [obb_decoup_convfc_bbox_head.py](mmdet/models/roi_heads/bbox_heads/obb/obb_decoup_convfc_bbox_head.py)

- **분리 전략**: 중심점(cx, cy, θ)과 크기(w, h) 예측 브랜치 분리
- **목적**: Oriented R-CNN 2nd stage에서 예측 간 간섭(interference) 감소

#### 3️⃣ Layer Normalization
- 분리된 브랜치에 LayerNorm 적용
- Feature distribution 안정화

---

## 📊 실험 결과

### 정량적 결과

| 방법 | 크기 오차↓ | 중심점 오차↓ | mAP↑ | 상태 |
|------|----------|------------|------|------|
| **1. Area + Ratio Loss** | - | - | - | ❌ 효과 없음 |
| **2. FC Layer 분리** | - | - | - | ❌ 효과 없음 |
| **3. 분리 + LayerNorm** | ✅ 감소 | ✅ 감소 | ⚠️ 미개선 | 🤔 역설적 |

### 🔍 역설적 발견

**방법 3 (Decoupled + LayerNorm)**에서:

```
✅ 평균 크기 오차 (픽셀):     감소
✅ 평균 중심점 오차 (픽셀):   감소
❌ mAP (Mean Average Precision): 개선 없음 (일부 하락)
```

### ❓ 핵심 연구 질문

> **왜 더 나은 localization이 더 나은 detection 성능으로 이어지지 않는가?**

가능한 가설:
1. **IoU threshold 문제**: 향상된 localization이 COCO mAP의 IoU threshold range에서 충분히 반영되지 않음
2. **Classification 간섭**: Bbox regression 개선이 classification 성능에 부정적 영향
3. **Train-Test 불일치**: 학습 중 개선이 테스트 시 일반화되지 않음

> 🚧 **연구 진행 중**: 추가 분석 및 실험이 필요합니다.

---

## ⚙️ 설치

**전체 설치 가이드**: [docs/INSTALL.md](docs/INSTALL.md)

| 패키지 | 버전 |
|--------|------|
| Python | 3.7.12 |
| PyTorch | 1.9.0 + CUDA 11.1 |
| mmcv-full | 1.3.9 |

---

## 📁 데이터셋 준비

### DOTA v2.0 다운로드 및 구성

```bash
# 디렉토리 구조
data/
└── dota/
    ├── train/
    │   ├── images/
    │   └── annfiles/
    ├── val/
    │   ├── images/
    │   └── annfiles/
    └── test/
        └── images/
```

**다운로드**: [DOTA 공식 웹사이트](https://captain-whu.github.io/DOTA/dataset.html)

**전처리**: [BboxToolkit](BboxToolkit/) 사용
```bash
cd BboxToolkit
python tools/img_split.py --config configs/dota2.json
```

---

## 🎓 사용 방법

### 1. 사전학습 가중치 다운로드

```bash
# GRA-ResNet50 백본 다운로드
wget https://drive.google.com/uc?id=15wGWyPJPQF0ORV8LcPp5BWOtl7rW8ht5 -O checkpoint-model.pth
```

또는 [Google Drive 링크](https://drive.google.com/file/d/15wGWyPJPQF0ORV8LcPp5BWOtl7rW8ht5/view?usp=sharing)에서 수동 다운로드

### 2. 학습

```bash
# DOTA v2.0 - Decoupled Head (본 연구)
python tools/train.py configs/obb/gra/gra_orcnn_r50fpn1x_ss_dota20.py

# 멀티 GPU 학습 (4 GPUs)
bash tools/dist_train.sh configs/obb/gra/gra_orcnn_r50fpn1x_ss_dota20.py 4
```

### 3. 테스트 및 평가

```bash
# 테스트 결과 생성
python tools/test.py \
    configs/obb/gra/gra_orcnn_r50fpn1x_ss_dota20.py \
    work_dirs/gra_orcnn_r50fpn1x_ss_dota20/latest.pth \
    --format-only \
    --options save_dir=results/dota20
```

### 4. 설정 파일

| 설정 파일 | 설명 | 데이터셋 |
|----------|------|---------|
| [gra_orcnn_r50fpn1x_ss_dota20.py](configs/obb/gra/gra_orcnn_r50fpn1x_ss_dota20.py) | **Decoupled Head** (본 연구)<br>LayerNorm + 분리된 브랜치 | DOTA v2.0 |


---

## 🙏 Acknowledgement

본 프로젝트는 다음의 우수한 연구와 오픈소스에 기반합니다:

### 기반 연구 및 코드베이스

- **[GRA](https://arxiv.org/pdf/2403.11127)** - Group-wise Rotating and Attention (ECCV 2024)
  - *Authors*: Jiangshan Wang, Yifan Pu, Yizeng Han, Jiayi Guo, Yiru Wang, Xiu Li, Gao Huang
  - 본 프로젝트의 backbone으로 사용

- **[MMDetection](https://github.com/open-mmlab/mmdetection)** - Open MMLab Detection Toolbox
  - 기본 객체 탐지 프레임워크

- **[ARC](https://github.com/LeapLabTHU/ARC)** - Adaptive Rotated Convolution
  - 회전 객체 탐지 유틸리티

> **📝 Note**: 본 저장소는 GRA의 **수정 및 최소화 버전**으로, 크기 예측 실험에 필요한 핵심 구성요소만 포함합니다.
> 본 연구의 기여는 **Decoupled BBox Head 설계**이며, GRA backbone 자체가 아닙니다.

---

## 📖 Citation

본 코드를 연구에 사용하시는 경우, 원본 GRA 논문을 인용해주세요:

```bibtex
@article{wang2024gra,
  title={GRA: Detecting Oriented Objects through Group-wise Rotating and Attention},
  author={Wang, Jiangshan and Pu, Yifan and Han, Yizeng and Guo, Jiayi and Wang, Yiru and Li, Xiu and Huang, Gao},
  journal={arXiv preprint arXiv:2403.11127},
  year={2024}
}
```

---

## 📄 License

본 프로젝트는 **Apache License 2.0**을 따릅니다 (원본 MMDetection과 동일).

자세한 내용은 [LICENSE](LICENSE) 파일을 참고하세요.

---

## 💬 Contact & Issues

**질문이나 이슈가 있으시면**:
- GitHub Issues에 등록해주세요
- 본 연구는 **진행 중**이며, 결과는 **preliminary**입니다

---

<div align="center">

**⭐ Star this repo if you find it helpful!**

Made with ❤️ for better oriented object detection

</div>
