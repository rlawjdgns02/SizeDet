# 설치 가이드

## 사전 요구사항

- **OS**: Ubuntu 20.04 / 22.04 / 24.04
- **GPU**: NVIDIA GPU (최소 11GB VRAM)
- **Disk**: 20GB 이상 여유 공간

---

## 설치 순서

### Step 1: Miniconda 설치

```bash
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
bash Miniconda3-latest-Linux-x86_64.sh
source ~/.bashrc
```

### Step 2: Conda 환경 생성

```bash
conda create -n sizedet python=3.7.12
conda activate sizedet
```

### Step 3: CUDA Toolkit 설치

```bash
conda install cudatoolkit=11.1.1 cudatoolkit-dev=11.1.1 -c conda-forge
```

### Step 4: PyTorch 설치

```bash
conda install pytorch==1.9.0 torchvision==0.10.0 cudatoolkit=11.1 -c pytorch
```

### Step 5: GCC 10 설치 (Ubuntu 24.04만 해당)

```bash
sudo apt update
sudo apt install gcc-10 g++-10 -y
sudo update-alternatives --install /usr/bin/gcc gcc /usr/bin/gcc-10 10
sudo update-alternatives --install /usr/bin/g++ g++ /usr/bin/g++-10 10
```

> Ubuntu 20.04/22.04 사용자는 이 단계를 건너뛰어도 됩니다.

### Step 6: mmcv-full 설치

```bash
pip install mmcv-full==1.3.9 -f https://download.openmmlab.com/mmcv/dist/cu111/torch1.9.0/index.html
```

> 소스 컴파일로 10-20분 소요됩니다.

### Step 7: 의존성 패키지 설치

```bash
cd /path/to/sizedet
pip install -r requirements.txt
```

### Step 8: BboxToolkit 설치

```bash
cd BboxToolkit
pip install -e .
cd ..
```

### Step 9: SizeDet 설치

```bash
pip install -e .
```

---

## 설치 확인

```bash
python -c "
import torch
print(f'PyTorch: {torch.__version__}')
print(f'CUDA: {torch.cuda.is_available()}')

import mmcv
print(f'MMCV: {mmcv.__version__}')

import mmdet
print(f'MMDet: {mmdet.__version__}')
"
```

**예상 출력:**
```
PyTorch: 1.9.0
CUDA: True
MMCV: 1.3.9
MMDet: 2.2.0+unknown
```

---

## 문제 해결

### mmcv-full 설치 실패

```bash
# GCC 버전 확인
gcc --version

# Ubuntu 24.04: gcc-10으로 변경
sudo update-alternatives --config gcc  # gcc-10 선택
sudo update-alternatives --config g++  # g++-10 선택

# 재시도
pip install mmcv-full==1.3.9 -f https://download.openmmlab.com/mmcv/dist/cu111/torch1.9.0/index.html
```

### ImportError 발생

```bash
pip uninstall mmcv-full mmdet -y
pip install mmcv-full==1.3.9 -f https://download.openmmlab.com/mmcv/dist/cu111/torch1.9.0/index.html
pip install -e .
```

---

## 참고

- 환경 상세 정보: [ENVIRONMENT.md](ENVIRONMENT.md)
