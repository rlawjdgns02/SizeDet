# 환경 정보

> 본 문서는 SizeDet 프로젝트의 검증된 환경 정보를 기록합니다.

---

## 검증된 버전 조합

| 패키지 | 버전 |
|--------|------|
| Python | 3.7.12 |
| PyTorch | 1.9.0 |
| torchvision | 0.10.0 |
| CUDA | 11.1 |
| cuDNN | 8.0.5 |
| mmcv-full | 1.3.9 |
| GCC | 10.x |

---

## 시스템 환경

| 항목 | 값 |
|------|-----|
| OS | Ubuntu 24.04 LTS |
| GPU Driver | NVIDIA 590.48.01 |

---

## 주요 패키지 버전

### 딥러닝 프레임워크

| 패키지 | 버전 |
|--------|------|
| pytorch | 1.9.0 |
| torchvision | 0.10.0 |
| mmcv-full | 1.3.9 |
| mmdet | 2.2.0+unknown (editable) |
| BboxToolkit | 1.1 (editable) |

### 과학 계산

| 패키지 | 버전 |
|--------|------|
| numpy | 1.21.6 |
| scipy | 1.7.3 |
| opencv-python | 4.13.0.90 |
| shapely | 2.0.7 |

---

## 환경 백업

```bash
# Conda 환경 내보내기
conda env export > environment.yml

# pip 패키지 목록
pip freeze > requirements_freeze.txt
```

---

## 설치 가이드

[INSTALL.md](INSTALL.md)
