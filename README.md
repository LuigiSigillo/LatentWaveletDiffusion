<h2 align="center"> <a href="https://openreview.net/forum?id=5og80LMVxG">[ICLR 2026] Latent Wavelet Diffusion For Ultra-High-Resolution Image Synthesis</a></h2>

<h3 align="center"><a href="https://luigisigillo.github.io/LWD-page">Project Page 🚀</a></h3>

<div align="center"><img src="assets/teaser.jpg" width="90%"></div>

<h5 align="center">If you like our project, please give us a star ⭐ on GitHub for the latest updates.</h5>

<h5 align="center">
    
![](https://api.visitorbadge.io/api/VisitorHit?user=luigisigillo&repo=latentwaveletdifffusion&countColor=%237B1E7A)

[![ICLR 2026](https://img.shields.io/badge/ICLR-2026-blue.svg)](https://openreview.net/forum?id=5og80LMVxG)
[![arXiv](https://img.shields.io/badge/arXiv-2506.00433-b31b1b.svg)](https://arxiv.org/abs/2506.00433)
[![Project Page](https://img.shields.io/badge/Project-Page-green.svg)](https://luigisigillo.github.io/LWD-page)
[![License](https://img.shields.io/badge/License-Apache%202.0-yellow.svg)](LICENSE)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.1+-ee4c2c.svg)](https://pytorch.org/)
![visitor badge](https://visitor-badge.laobi.icu/badge?page_id=LuigiSigillo.LatentWaveletDiffusion&format=true)

**[Luigi Sigillo](https://luigisigillo.github.io/)** · [Shengfeng He](https://shengfenghe.github.io/) · [Danilo Comminiello](https://danilocomminiello.site.uniroma1.it/)

</h5>

---

## 📰 News

* **[2026.04.25]** Code is now available! Welcome to **watch** 👀 this repository for the latest updates.
* **[2026.01.26]** 🔥🔥🔥 Paper accepted at **ICLR 2026**! See you in Rio!
* **[2025.09.15]** Preprint available on arXiv and submitted to **ICLR 2026**!

---

## 😮 Highlights

### ⚡ Zero Inference Overhead — Plug-and-Play into Any Diffusion Model

LWD is a **training-only** modification. It requires **no architectural changes** and adds **zero additional cost at inference time** — an LWD-enhanced model has the exact same parameter count and identical inference speed as its baseline counterpart. This makes it a practical, low-risk improvement for any existing latent diffusion pipeline (FLUX, SD3, Sana, PixArt-Σ, and beyond).

### 🌊 Signal-Driven Supervision: No Learned Attention Required

Instead of learning which regions matter, LWD computes a **Discrete Wavelet Transform (DWT)** of the latent tensor and reads structural importance directly from the signal. The aggregated energy of the high-frequency subbands (LH, HL, HH) produces an interpretable, parameter-free saliency map — no extra modules, no extra training, just signal processing.

### 🔥 Consistent SOTA Improvements at 2K and 4K Resolution

LWD consistently improves FID, LPIPS, and perceptual quality across multiple strong baselines at both 2K and 4K resolution. As a bonus, the frequency-aware supervision **accelerates convergence**: LWD-enhanced models reach the quality of their baselines in only **10–50% of the original training iterations**.

---

## 🚀 Main Results

<div align="center"><img src="assets/results4kzoom_mix_page.jpg" width="90%"></div>

<p align="center"><em>4K image generation with LWD paired with different architectures. LWD consistently sharpens textures and fine details without over-sharpening or introducing artifacts.</em></p>

Evaluated on two ultra-resolution benchmarks:
- **Aesthetic-4K** — A curated 4K benchmark with GPT-4o-generated captions and high visual quality.
- **LAION-High-Res** — A filtered subset of LAION-5B with 50K × 2K and 20K × 4K image-caption pairs.

---

## ✨ Takeaway Functions

The entire LWD training strategy reduces to **two functions** that can be dropped into any diffusion training script.

### `compute_wavelet_attention`

Computes a spatial saliency map from a latent tensor by aggregating the energy of the DWT high-frequency subbands:

```python
import torch.nn.functional as F
from pytorch_wavelets import DWTForward

def compute_wavelet_attention(latents, dwt):
    """
    Compute a wavelet-based saliency map from latent codes.

    Args:
        latents:  (B, C, H, W) latent tensor
        dwt:      DWTForward instance (J=1, wave='haar')

    Returns:
        A:  (B, H, W) saliency map in [0, 1]
            Higher values = more high-frequency content (textures, edges)
    """
    _, highs = dwt(latents)                          # highs[0]: (B, C, 3, H/2, W/2)
    lh, hl, hh = highs[0].unbind(dim=2)             # directional subbands

    # Aggregate energy across channels
    energy = (lh.pow(2) + hl.pow(2) + hh.pow(2)).mean(dim=1)  # (B, H/2, W/2)

    # Upsample back to latent resolution
    energy = F.interpolate(energy.unsqueeze(1), size=latents.shape[-2:],
                           mode='bilinear', align_corners=False).squeeze(1)

    # Min-max normalize per sample
    mn = energy.flatten(1).min(1).values[:, None, None]
    mx = energy.flatten(1).max(1).values[:, None, None]
    return (energy - mn) / (mx - mn + 1e-8)
```

- **DWT** decomposes the latent into LL (approximation) and LH, HL, HH (directional details).
- **Energy** of the three detail subbands acts as a proxy for local structural complexity.
- **Normalization** is per-sample, making the map data-adaptive across the batch.

---

### `get_mask_batch`

Given the saliency map and the current diffusion timestep, generates the binary supervision mask:

```python
def get_mask_batch(A, l, T, timesteps):
    """
    Compute a time-dependent binary mask for spatially adaptive loss.

    Formula:  M_t(i,j) = 1  if  T * (A[i,j] + l) >= t,  else 0

    High-saliency regions stay supervised across MORE timesteps.
    All regions receive at least l*T steps of supervision (lower-bound guarantee).

    Args:
        A:          (B, H, W) saliency map in [0, 1]
        l:          float in (0, 1) — lower bound (recommended: 0.3)
        T:          int — total diffusion timesteps (e.g. 1000)
        timesteps:  (B,) current timesteps

    Returns:
        M:  (B, 1, H, W) binary mask
    """
    threshold = T * (A + l)                        # (B, H, W)
    t = timesteps[:, None, None].float()           # (B, 1, 1)
    return (threshold >= t).float().unsqueeze(1)   # (B, 1, H, W)
```

- **`l = 0.3`** is the value used in all paper experiments.
- Low-saliency regions still receive at least `l × T = 300` supervision steps.
- High-saliency regions receive up to `T` steps — more focused, richer training signal.

---

### 🧐 How to Use Them in Practice

```python
from pytorch_wavelets import DWTForward

# ── Setup (once, before training loop) ──────────────────────────────────────
dwt = DWTForward(J=1, wave="haar").to(device)

# ── Inside training loop ─────────────────────────────────────────────────────
# noisy_latents: (B, C, H, W)  — noisy input to the diffusion model
# model_pred:    (B, C, H, W)  — model output (predicted velocity / noise)
# target:        (B, C, H, W)  — ground truth target
# timesteps:     (B,)          — current diffusion timesteps
# weighting:     (B, 1, 1, 1)  — optional SNR weighting

# 1. Compute wavelet saliency map
A = compute_wavelet_attention(noisy_latents, dwt)           # (B, H, W)

# 2. Compute time-dependent binary mask
M = get_mask_batch(A, l=0.3, T=1000, timesteps=timesteps)  # (B, 1, H, W)

# 3. Apply masked loss
masked_diff = M * (model_pred - target)
loss = (weighting * masked_diff.pow(2)).mean()
```

The full masked flow-matching objective from the paper:

$$\mathcal{L}_{\text{masked}} = \left\| M_t \odot \left[ (\epsilon - z_0) - v_\Theta(z_t, t, y) \right] \right\|_2^2$$

---

## 🛠️ Installation

**Prerequisites:** Python 3.12+, CUDA 11.8+, PyTorch 2.1+, GPU with 32GB+ VRAM (48GB+ for 4K)

```bash
# Clone the repository
git clone https://github.com/LuigiSigillo/LatentWaveletDiffusion.git
cd LatentWaveletDiffusion

# Create conda environment
conda create -n lwd python=3.12
conda activate lwd

# Install dependencies
pip install -r requirements.txt

# Install PyTorch Wavelets
pip install -e src/pytorch_wavelets/
```

---

## 🔄 Pipeline Overview

The complete LWD pipeline consists of 5 stages:

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐     ┌─────────────────┐    ┌────────────────┐
│  1. VAE         │    │  2. Cache       │    │  3. Train       │     │  4. Evaluation  │    │  5. Inference  │
│  Fine-tuning    │───▶│  Latents &      │───▶│  with Wavelet  │───▶│                 │───▶│                │
│  (Optional)     │    │  Embeddings     │    │  Masking        │     │                 │    │                │
└─────────────────┘    └─────────────────┘    └─────────────────┘     └─────────────────┘    └────────────────┘
```

<details>
<summary><b>Step 1: VAE Fine-tuning (Optional)</b></summary>

Fine-tune the VAE with a multi-resolution scale-consistency objective (L2 reconstruction + KL regularization + LPIPS perceptual loss + scale-consistency term). This is the **first and prerequisite stage** of LWD: without it, the DWT-based wavelet saliency maps would pick up compression artifacts rather than meaningful structure. The fine-tuned VAE regularizes the latent spectrum so that high-frequency energy reliably corresponds to textures and edges.

```bash
cd src/vae_SE_finetuning

python vae_finetune_diffusability.py \
    --pretrained_model_name_or_path="black-forest-labs/FLUX.1-dev" \
    --subfolder="vae" \
    --data_dir="/path/to/training/images" \
    --output_dir="./ckpt/vae_SE" \
    --image_size=512 \
    --batch_size=8 \
    --learning_rate=1e-5 \
    --max_train_steps=60000 \
    --mixed_precision="bf16" \
    --lpips_weight=0.05 \
    --regularization_alpha=0.1 \
    --with_tracking \
    --checkpointing_steps=5000
```

**Key arguments:**
| Argument | Description |
|----------|-------------|
| `--lpips_weight` | Weight for LPIPS perceptual loss (default: 0.05) |
| `--regularization_alpha` | Multi-scale consistency loss weight (default: 0.1) |
| `--freeze_encoder` | Optionally freeze encoder, train only decoder |

</details>

### Step 2: Cache Latents & Embeddings

Pre-compute VAE latents and text embeddings to avoid OOM during training.

#### 2a. Cache Text Embeddings (T5 + CLIP)

```bash
# Edit src/batch_scripts/cache_prompt_embeds.sh:
export NUM_WORKERS=2
export DATA_DIR="/path/to/your/dataset"
export MODEL_NAME="black-forest-labs/FLUX.1-dev"

bash src/batch_scripts/cache_prompt_embeds.sh
```

#### 2b. Cache VAE Latents

```bash
# Edit src/batch_scripts/cache_latent_codes.sh:
export NUM_WORKERS=4
export DATA_DIR="/path/to/your/dataset"
export OUTPUT_DIR="/path/to/latents"
export MODEL_NAME="black-forest-labs/FLUX.1-dev"  # or your fine-tuned VAE
export RESOLUTION=2048  # target resolution

bash src/batch_scripts/cache_latent_codes.sh
```

**Data Format:**
```
dataset/
├── image_0.jpg
├── image_0.json          # {"prompt": "...", "generated_prompt": "..."}
├── image_0_latent_code.safetensors      # added after caching
├── image_0_prompt_embed.safetensors     # added after caching
├── image_0_generated_prompt_embed.safetensors
└── ...
```

### Step 3: Training with Wavelet Masking

Train the diffusion model with our wavelet-based frequency-adaptive loss.

<p align="center">
  <img src='assets/arch.png' width='100%' />
</p>

```bash
# Edit src/batch_scripts/train_2k.sh:
export MODEL_NAME="black-forest-labs/FLUX.1-dev"
export DATA_DIR="/path/to/dataset"
export LATENT_CODE_DIR="/path/to/cached/latents"
export OUTPUT_DIR="/path/to/checkpoints"

bash src/batch_scripts/train_2k.sh
```

**Key training arguments:**

| Argument | Description | Default |
|----------|-------------|---------|
| `--wavelet_attention` | **Enable wavelet-based loss masking** | `True` |
| `--wav_att_l_mask` | Masking lower bound `l` (see ablation) | `0.3` |
| `--latent_code_dir` | Directory with cached latents | None |
| `--train_batch_size` | Batch size per GPU | 1 |
| `--max_train_steps` | Total training iterations | 2000 |
| `--gradient_checkpointing` | Enable to save VRAM | `False` |
| `--real_prompt_ratio` | Ratio of original vs VLM prompts | 0.2 |

### Step 4: Evaluation

We provide evaluation scripts for both 2K and 4K resolutions.

| Metric | Type | Description |
|--------|------|-------------|
| **FID** | Full-reference | Fréchet Inception Distance |
| **LPIPS** | Full-reference | Learned Perceptual Image Patch Similarity |
| **MAN-IQA** | No-reference | Multi-scale Attention Network IQA |
| **QualiCLIP** | No-reference | CLIP-based quality assessment |
| **HPSv2** | Text-image | Human Preference Score v2 |
| **PickScore** | Text-image | Preference learning score |

<details>
<summary><b>4a. Evaluate 2K Models (eval_v2.py)</b></summary>

```bash
# Generate images for HPDv2 test set
python src/eval_v2.py \
    --generate_hpdv2_testset \
    --json_file="/path/to/HPDv2/test.json" \
    --output_dir="/path/to/output" \
    --checkpoint_path="/path/to/checkpoint-2000" \
    --cache_dir=$HF_HOME \
    --height=2048 --width=2048 --seed=42

# Generate images for DPG benchmark
python src/eval_v2.py \
    --generate_dpg_testset \
    --prompt_folder_dpg="/path/to/dpg_bench/prompts" \
    --output_dir="/path/to/output" \
    --checkpoint_path="/path/to/checkpoint-2000" \
    --cache_dir=$HF_HOME

# Calculate metrics
python src/eval_v2.py \
    --calculate_metrics \
    --generated_folder="/path/to/generated/images" \
    --reference_folder="/path/to/reference/images" \
    --cache_dir=$HF_HOME
```

| Argument | Description |
|----------|-------------|
| `--generate_hpdv2_testset` | Generate images for HPDv2 benchmark |
| `--generate_dpg_testset` | Generate images for DPG benchmark |
| `--calculate_metrics` | Compute FID, PickScore, QualiCLIP, MAN-IQA, HPSv2 |
| `--checkpoint_path` | Path to trained checkpoint |
| `--height`, `--width` | Resolution (default: 2048×2048) |

</details>

<details>
<summary><b>4b. Evaluate 4K Models (eval_4k.py)</b></summary>

```bash
# Generate and evaluate 4K images (uses all available GPUs)
python src/eval_4k.py \
    --checkpoint_path="/path/to/4k/checkpoint/adapter_weights.safetensors" \
    --generate \
    --height=4096 --width=4096 --seed=8888 \
    --cache_dir=$HF_HOME

# Generate for a specific style only
python src/eval_4k.py \
    --checkpoint_path="/path/to/4k/checkpoint/adapter_weights.safetensors" \
    --generate --style="anime" \
    --height=4096 --width=4096 \
    --cache_dir=$HF_HOME
```

**Supported styles:** `anime`, `concept-art`, `paintings`, `photo`

</details>

### Step 5: Inference

For running inference, please use the Jupyter notebooks provided in the [`src/inference_nb/`](src/inference_nb/) folder.

---

## 🔌 Applying to Other Diffusion Models

LWD is **model-agnostic** and integrates into any latent diffusion training pipeline in 4 steps.

### Step-by-Step Integration

**1. Copy required files:**
```bash
cp src/new_wav_attn_maps.py /path/to/your/project/
pip install -e src/pytorch_wavelets/
```

**2. Add arguments to your training script:**
```python
parser.add_argument("--wavelet_attention", action="store_true",
    help="Enable wavelet-based frequency-adaptive loss masking")
parser.add_argument("--wav_att_l_mask", type=float, default=0.3,
    help="Lower bound for wavelet attention mask (default: 0.3)")
```

**3. Initialize DWT before the training loop:**
```python
if args.wavelet_attention:
    from new_wav_attn_maps import compute_wavelet_attention, get_mask_batch
    from pytorch_wavelets import DWTForward
    dwt = DWTForward(J=1, wave="haar").to(accelerator.device)
```

**4. Replace the loss computation:**
```python
if args.wavelet_attention:
    A, _ = compute_wavelet_attention(noisy_latents, dwt)       # (B, H, W)
    M, _ = get_mask_batch(A, l=args.wav_att_l_mask,
                          T=scheduler.num_train_timesteps,
                          timesteps=timesteps)                  # (B, 1, H, W)
    loss = (weighting * (M * (model_pred - target)).pow(2)).mean()
else:
    loss = (weighting * (model_pred - target).pow(2)).mean()
```

### Architecture-Specific Examples

<details>
<summary><b>FLUX.1 (Flow Matching) — This Repository</b></summary>

```python
if args.wavelet_attention:
    from new_wav_attn_maps import compute_wavelet_attention, get_mask_batch
    from pytorch_wavelets import DWTForward
    dwt = DWTForward(J=1, wave="haar").to(device)

for batch in dataloader:
    target = noise - model_input  # flow matching target

    if args.wavelet_attention:
        A, _ = compute_wavelet_attention(noisy_model_input, dwt)
        M, _ = get_mask_batch(A, l=args.wav_att_l_mask,
                              T=scheduler.num_train_timesteps,
                              timesteps=timesteps)
        loss = (weighting * (M * (model_pred - target)).pow(2)).mean()
    else:
        loss = (weighting * (model_pred - target) ** 2).mean()
```

**Reference:** [train_2k.py:L890-L901](src/train_2k.py#L890-L901)

</details>

<details>
<summary><b>Sana / PixArt-Sigma (DiT)</b></summary>

```python
# Before training loop
dwt = DWTForward(J=1, wave="haar").to(accelerator.device) if args.wavelet_attention else None

# In gaussian_diffusion.py — training_losses()
if model_kwargs.get("dwt") is not None:
    from .new_wav_attn_maps import compute_wavelet_attention, get_mask_batch
    A = compute_wavelet_attention(x_t, model_kwargs["dwt"])
    M = get_mask_batch(A, l=0.3, T=self.num_timesteps, timesteps=t)
    terms["mse"] = mean_flat((M * (output - target)).pow(2))
else:
    terms["mse"] = mean_flat(loss)
```

</details>

<details>
<summary><b>Stable Diffusion 3 (Diffusion4K)</b></summary>

```python
for batch in dataloader:
    target = model_input  # SD3 flow matching

    if args.wavelet_attention:
        A = compute_wavelet_attention(noisy_model_input, dwt)
        M = get_mask_batch(A, l=args.wav_att_l_mask,
                           T=noise_scheduler.config.num_train_timesteps,
                           timesteps=timesteps)
        loss = (weighting.float() * (M * (model_pred - target)).pow(2)).mean()
    else:
        loss = (weighting.float() * (model_pred - target).float() ** 2).mean()
```

</details>

### ✅ Integration Checklist

- [ ] Copy `new_wav_attn_maps.py` and install `pytorch_wavelets`
- [ ] Add `--wavelet_attention` and `--wav_att_l_mask` arguments
- [ ] Initialize `DWTForward` before the training loop
- [ ] Replace loss computation with wavelet-masked version
- [ ] Start with `l=0.3` and tune if needed
- [ ] Monitor: FID should improve; high-frequency texture metrics (GLCM) should stay stable or improve

---

## 📖 Citation

If you find this work useful, please cite our ICLR 2026 paper:

```bibtex
@inproceedings{sigillo2026latent,
title={Latent Wavelet Diffusion For Ultra High-Resolution Image Synthesis},
author={Luigi Sigillo and Shengfeng He and Danilo Comminiello},
booktitle={The Fourteenth International Conference on Learning Representations},
year={2026},
url={https://openreview.net/forum?id=5og80LMVxG}
}
```

This work builds upon URAE as a baseline. Please also consider citing:

```bibtex
@article{yu2025urae,
  title={Ultra-Resolution Adaptation with Ease},
  author={Yu, Ruonan and Liu, Songhua and Tan, Zhenxiong and Wang, Xinchao},
  journal={arXiv preprint arXiv:2503.16322},
  year={2025}
}
```

---

## 🙏 Acknowledgements

We thank the authors and contributors of:

- **[URAE](https://github.com/Huage001/URAE)** — High-resolution adaptation baseline
- **[FLUX](https://blackforestlabs.ai/)** — State-of-the-art flow-matching diffusion backbone
- **[pytorch_wavelets](https://github.com/fbcotter/pytorch_wavelets)** — Efficient DWT implementation
- **[Hugging Face Diffusers](https://github.com/huggingface/diffusers)** — Training infrastructure
- **[patch_conv](https://github.com/mit-han-lab/patch_conv)** — Memory-efficient VAE operations

Special thanks to the ICLR 2026 reviewers for their valuable feedback.

---

## 🌟 Star History

<a href="https://star-history.com/#LuigiSigillo/LatentWaveletDiffusion&Date">
  <picture>
    <source media="(prefers-color-scheme: dark)" srcset="https://api.star-history.com/svg?repos=LuigiSigillo/LatentWaveletDiffusion&type=Date&theme=dark" />
    <source media="(prefers-color-scheme: light)" srcset="https://api.star-history.com/svg?repos=LuigiSigillo/LatentWaveletDiffusion&type=Date" />
    <img alt="Star History Chart" src="https://api.star-history.com/svg?repos=LuigiSigillo/LatentWaveletDiffusion&type=Date" />
  </picture>
</a>

---

## 📄 License

This project is licensed under the Apache License 2.0 — see the [LICENSE](LICENSE) file for details.
