<img width="6279" height="2019" alt="figure1" src="https://github.com/user-attachments/assets/659cf325-e124-4203-8cfa-e4a078faed29" />
# VoRA: Volterra Low-Rank Adaptation Using Nonlinear Interactions for Image Restoration

<p align="center">
  Parameter-efficient fine-tuning with parallel linear and low-rank quadratic adaptation branches.
</p>

<p align="center">
  선형 LoRA 분기와 저랭크 2차 상호작용 분기를 결합한 이미지 복원용 파라미터 효율적 미세조정 기법입니다.
</p>

<p align="center">
  <a href="#overview">English</a> · <a href="#한국어-소개">한국어</a> ·
  <a href="#results">Results</a> · <a href="#quick-start">Quick Start</a>
</p>

---

## Overview

VoRA (Volterra Low-Rank Adaptation) extends LoRA with a parallel **low-rank quadratic branch** for parameter-efficient image restoration. Standard LoRA applies a linear low-rank residual to a frozen pretrained projection. VoRA preserves that linear path and adds an element-wise quadratic interaction in a compact latent space, allowing the adapter to represent nonlinear and interaction-dependent restoration residuals without constructing a full input-space quadratic kernel.

The repository supports deraining, desnowing, denoising, deblurring, dehazing, and composite-degradation experiments across multiple restoration backbones.

### Highlights

- **Linear + quadratic adaptation:** combines the standard LoRA residual with a structured second-order residual.
- **Parameter-efficient design:** freezes pretrained backbone weights and trains only adapter parameters.
- **Parameter-matched evaluation:** VoRA with linear and quadratic ranks of 4 matches the 294,912 trainable parameters of LoRA rank 8.
- **Backbone generalization:** supports SwinIR, Uformer, HAT, AdaIR, DFPIR Restormer, and MambaIRv2 integrations.
- **Interaction modeling:** supports composite degradation and degradation-intensity experiments.

## Architecture

<img width="6279" height="2019" alt="figure1" src="https://github.com/user-attachments/assets/986d1b7c-01be-45d1-80fb-750c546dbb73" />

**LoRA and VoRA.** LoRA adds a linear low-rank residual branch to a frozen pretrained projection. VoRA retains this linear branch and adds a quadratic branch that projects the input into a low-rank latent space, constructs element-wise quadratic features, and projects them back to the output space.

### Adapter formulation

Given an input feature $x$ and a frozen pretrained weight $W_0$, the base response is

```math
h_0 = W_0 x.
```

The linear LoRA residual is

```math
\Delta h_l = s_l B_l A_l x.
```

The quadratic branch first produces a low-rank latent feature and applies an element-wise square:

```math
z = A_q x, \qquad q = z \odot z.
```

The quadratic residual and final VoRA output are

```math
\Delta h_q = s_q B_q q,
```

```math
h = h_0 + \Delta h_l + \Delta h_q.
```

For linear rank $r_l$ and quadratic rank $r_q$, the number of trainable adapter parameters is

```math
N_{\mathrm{VoRA}} = (r_l + r_q)(d_{\mathrm{in}} + d_{\mathrm{out}}).
```

Setting $r_l=r_q=r/2$ gives VoRA the same parameter count as LoRA rank $r$. The quadratic branch has computational complexity $\mathcal{O}(r_q(d_{\mathrm{in}} + d_{\mathrm{out}}))$ for each input feature vector.

### SwinIR integration

<p align="center">
  <img width="700" height="620" alt="VoRA SwinIR insertion" src="https://github.com/user-attachments/assets/344eac49-accd-4d3a-9ba4-d084ec1ded3e" />
</p>

VoRA is attached to selected frozen projection layers in a SwinIR restoration Transformer block: the fused QKV projection, attention output projection, and MLP projection. The frozen response and VoRA residual are computed from the same input and combined by residual addition.

## Results

Results are reported as **PSNR / SSIM**. The values below are taken from the accompanying manuscript configuration.

### Main image restoration comparison

| Method | Rain100H | CSD | GoPro | RESIDE-6K | Trainable parameters | Ratio |
|---|---:|---:|---:|---:|---:|---:|
| Full fine-tuning | **27.43 / 0.9551** | 25.23 / 0.8952 | **27.80 / 0.9372** | **24.23 / 0.9192** | 3,137,427 | 100% |
| Frozen | 8.71 / 0.3617 | 12.82 / 0.3519 | 13.41 / 0.5533 | 10.63 / 0.3900 | 0 | 0% |
| LoRA | 22.91 / 0.8840 | 24.69 / 0.8917 | 27.17 / 0.9306 | 23.34 / 0.8983 | 147,456 | 4.7% |
| **VoRA** | **24.40 / 0.9141** | **25.65 / 0.9148** | **27.30 / 0.9309** | **23.84 / 0.9080** | 294,912 | 9.4% |

### Parameter-matched comparison

LoRA rank 8 and VoRA with $r_l=r_q=4$ both use 294,912 trainable parameters.

| Task | Dataset | LoRA (r=8) | VoRA ($r_l=r_q=4$) | Delta PSNR | Delta SSIM |
|---|---|---:|---:|---:|---:|
| Deraining | Rain100H | 23.07 / 0.8873 | **24.27 / 0.9123** | +1.2008 | +0.0250 |
| Desnowing | CSD | 24.47 / 0.8916 | **25.74 / 0.9158** | +1.2767 | +0.0241 |
| Denoising | SIDD | 32.96 / 0.8445 | **33.23 / 0.8476** | +0.2761 | +0.0031 |
| Deblurring | GoPro | 27.21 / 0.9305 | **27.36 / 0.9310** | +0.1485 | +0.0005 |
| Dehazing | RESIDE-6K | 23.11 / 0.8980 | **23.53 / 0.9045** | +0.4197 | +0.0066 |
| **Average** | – | 26.16 / 0.8904 | **26.83 / 0.9022** | **+0.6644** | **+0.0118** |

### Generalization across restoration backbones

| Backbone | LoRA (r=8) | VoRA ($r_l=r_q=4$) | Delta PSNR | Delta SSIM |
|---|---:|---:|---:|---:|
| SwinIR | 23.07 / 0.8873 | **24.25 / 0.9097** | +1.1775 | +0.0224 |
| Uformer | 23.78 / 0.8983 | **24.91 / 0.9188** | +1.1297 | +0.0205 |
| HAT | 22.88 / 0.8954 | **24.99 / 0.9181** | +2.1044 | +0.0227 |
| AdaIR | 25.10 / 0.9198 | **26.77 / 0.9443** | +1.6752 | +0.0245 |
| Restormer | 25.14 / 0.9199 | **26.51 / 0.9395** | +1.3656 | +0.0196 |

### Composite degradation restoration

| Composite degradation | LoRA (r=8) | VoRA ($r_l=r_q=4$) | Delta PSNR | Delta SSIM |
|---|---:|---:|---:|---:|
| Rain + Haze | 22.89 / 0.8867 | **24.71 / 0.9147** | +1.82 | +0.0280 |
| Rain + Blur | 21.11 / 0.8437 | **22.29 / 0.8716** | +1.19 | +0.0279 |
| Blur + Noise | 26.24 / 0.9254 | **26.36 / 0.9265** | +0.12 | +0.0011 |
| Haze + Noise | 21.95 / 0.8625 | **22.83 / 0.8834** | +0.89 | +0.0209 |

## Supported datasets and backbones

### Datasets

| Task | Dataset | CLI name | Link |
|---|---|---|---|
| Deraining | Rain100H | `rain100h` | [Download](https://github.com/nnUyi/DerainZoo/blob/master/DerainDatasets.md) |
| Desnowing | CSD | `csd` | [Download](https://github.com/weitingchen83/ICCV2021-Single-Image-Desnowing-HDCWNet) |
| Denoising | SIDD | `sidd` | [Download](https://abdokamel.github.io/sidd/) |
| Deblurring | GoPro | `gopro` | [Download](https://seungjunnah.github.io/Datasets/gopro.html) |
| Dehazing | RESIDE-6K | `reside6k` | [Download](https://www.kaggle.com/datasets/kmljts/reside-6k) |

Dataset files and pretrained backbone weights are not included in this repository.

### Backbones

- SwinIR Lite and official SwinIR
- Uformer
- HAT
- AdaIR
- DFPIR Restormer
- MambaIRv2

## Quick start

### 1. Installation

```bash
git clone https://github.com/degull/VORA.git
cd VORA

python -m venv .venv
# Windows: .venv\Scripts\activate
# macOS/Linux: source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

### 2. Adapter sanity check

```bash
python test.py
```

This checks output shapes, branch outputs, and trainable parameter counts for the LoRA and VoRA linear adapters.

### 3. Prepare data

The shared data root must follow this layout:

```text
<data-root>/
├── rain100h/{train,test}/{input,target}/
├── csd/{train,test}/{input,target}/
├── gopro/{train,test}/{input,target}/
├── reside6k/{train,test}/{input,target}/
└── sidd/{train,test}/{input,target}/
```

Each input image must have a target image with the same filename. See [the dataset guide](docs/DATASETS.md) for checkpoint handling.

### 4. Smoke-test training

```bash
python train.py \
  --dataset rain100h \
  --data-root /path/to/data \
  --method vora_v1 \
  --backbone swinir_lite \
  --rank 4 \
  --target all \
  --steps 10
```

The default sample limits (16 training pairs and 4 validation pairs) are intended for quick checks. For a full dataset run, pass `--max-train-samples 0 --max-val-samples 0`.

### 5. Parameter-matched LoRA and VoRA runs

```bash
# LoRA: rank 8
python train.py --dataset rain100h --data-root /path/to/data \
  --method lora --backbone swinir_official --swinir-size base \
  --rank 8 --target all --max-train-samples 0 --max-val-samples 0

# VoRA: rank 4 for both linear and quadratic branches
python train.py --dataset rain100h --data-root /path/to/data \
  --method vora_v1 --backbone swinir_official --swinir-size base \
  --rank 4 --target all --max-train-samples 0 --max-val-samples 0
```

Checkpoints are saved under `checkpoints/`, and CSV logs are written under `outputs/logs/`. Use `--base-checkpoint PATH` to load pretrained weights before adapters are inserted. Use `--resume PATH` or `--auto-resume` to continue a run.

## Repository structure

```text
VORA/
├── models/                 # LoRA, VoRA, Volterra, and restoration backbones
├── datasets/               # Paired and composite-degradation data loaders
├── engine/                 # Training, evaluation, and metrics
├── losses/                 # Reconstruction and perceptual losses
├── utils/                  # Adapter replacement, checkpoints, logging, visualization
├── docs/                   # Dataset and experiment notes
├── external/               # Local-only third-party repositories
├── checkpoints/            # Downloaded and trained weights
├── outputs/                # Logs, figures, and restored images
├── train.py                # Shared restoration training entry point
└── test.py                 # Adapter sanity check
```

---

## 한국어 소개

VoRA(Volterra Low-Rank Adaptation)는 기존 LoRA의 선형 저랭크 잔차 분기에 **저랭크 2차 분기**를 병렬로 추가한 이미지 복원용 PEFT 기법입니다. 사전학습 백본의 가중치는 고정한 채 어댑터만 학습하며, 입력 전체 차원에서 완전한 2차 커널을 만들지 않고 압축된 잠재 공간에서 원소별 제곱 연산을 수행합니다.

### 핵심 아이디어

- LoRA의 선형 적응 능력을 유지하면서 2차 특징 상호작용을 명시적으로 모델링합니다.
- 입력을 $r_q$차원의 잠재 공간으로 투영한 뒤 $z \odot z$를 계산하므로 완전한 2차 전개보다 효율적입니다.
- LoRA rank 8과 VoRA의 두 분기 rank 4를 비교하면 학습 파라미터 수가 동일합니다.
- Rain100H, CSD, SIDD, GoPro, RESIDE-6K 및 복합 열화 환경을 지원합니다.
- SwinIR뿐 아니라 Uformer, HAT, AdaIR, Restormer에서도 사용할 수 있습니다.

### SwinIR 적용 위치

VoRA는 SwinIR 복원 Transformer 블록의 fused QKV projection, attention output projection, MLP projection에 삽입됩니다. 고정된 projection 출력과 VoRA 잔차는 동일한 입력 특징으로부터 계산되고 원소별 덧셈으로 결합됩니다.

### 지원 작업

- Rain100H 비 제거
- CSD 눈 제거
- SIDD 노이즈 제거
- GoPro 모션 디블러링
- RESIDE-6K 안개 제거
- Rain + Haze, Rain + Blur, Blur + Noise, Haze + Noise 복합 열화 복원
