"""
Gradio demo for Latent Wavelet Diffusion (LWD) image generation.

Supports both 2K and 4K resolution generation using FLUX + URAE adapters.
Run from the src/ directory:
    python gradio_app.py
"""

import os
import sys
import gc
import time

import torch
import gradio as gr
from huggingface_hub import hf_hub_download
from safetensors.torch import load_file, save_file

# Ensure the src directory is on the path for local imports
SRC_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SRC_DIR)

from pipeline_flux import FluxPipeline
from transformer_flux import FluxTransformer2DModel
from attention_processor import FluxAttnAdaptationProcessor2_0

try:
    from patch_conv import convert_model
    PATCH_CONV_AVAILABLE = True
except ImportError:
    PATCH_CONV_AVAILABLE = False

# ---------------------------------------------------------------------------
# Paths & checkpoint registry
# ---------------------------------------------------------------------------
CKPT_DIR = os.path.join(SRC_DIR, "ckpt")
BFL_REPO = "black-forest-labs/FLUX.1-dev"

# Sentinel value meaning "download from HuggingFace (Huage001/URAE)"
_HF = "__HF__"

# Official HF checkpoints always listed first (paths resolved at load time).
# Local checkpoints are only shown when the file actually exists on disk.
CHECKPOINTS_2K = {
    "URAE 2K (Official)": _HF,                                  # always available via HF
    "URAE 2K + Wav Att": "/mnt/media/luigi/LWD Material/Checkpoints/URAE/2K/l_003_URAE_VAE_NOOO_SE_WAV_ATT_LAION_2048/checkpoint-6000/pytorch_lora_weights.safetensors",
    "URAE 2K + VAE-SE + Wav Att": "/mnt/media/luigi/LWD Material/Checkpoints/URAE/2K/URAE_VAE_SE_WAV_ATT_LAION/checkpoint-2000/pytorch_lora_weights.safetensors",
    "URAE 2K": "/mnt/media/luigi/LWD Material/Checkpoints/URAE/2K/URAE_original_trained_by_me/checkpoint-2000/pytorch_lora_weights.safetensors",
}

CHECKPOINTS_4K = {
    "URAE 4K (Official)": _HF,                                  # always available via HF
    "URAE 4K + Wav Att": "/mnt/media/luigi/LWD Material/Checkpoints/URAE/4K/l_03_4K_URAE_VAE_SE_WAV_ATT_LAION_4096/checkpoint-2000/adapter_weights.safetensors",
    "URAE 4K + VAE-SE + Wav Att": "/mnt/media/luigi/LWD Material/Checkpoints/URAE/4K/4K_URAE_VAE_SE_WAV_ATT_LAION_4096/checkpoint-2000/adapter_weights.safetensors",
    "URAE 4K": "/mnt/media/luigi/LWD Material/Checkpoints/URAE/4K/4K_urae_original/ckpt_xxx/urae_4k_adapter.safetensors",
}


def _build_checkpoint_choices(resolution: str) -> list[str]:
    """Return checkpoint names available for a given resolution.

    HF-sentinel entries are always included.  Local entries are included only
    when their file exists on disk.  A "Custom" option is always appended last.
    """
    registry = CHECKPOINTS_2K if resolution == "2K" else CHECKPOINTS_4K
    choices = [
        name
        for name, path in registry.items()
        if path == _HF or os.path.exists(path)
    ]
    choices.append("Custom")
    return choices

# ---------------------------------------------------------------------------
# Global pipeline cache
# ---------------------------------------------------------------------------
_current_pipe = None
_current_config = None  # (resolution, checkpoint_path, cpu_offload) tuple


def _flush_pipeline():
    """Release the current pipeline and free GPU memory."""
    global _current_pipe, _current_config
    if _current_pipe is not None:
        del _current_pipe
        _current_pipe = None
        _current_config = None
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


# ---------------------------------------------------------------------------
# GPU memory helper
# ---------------------------------------------------------------------------
def _get_vram_info() -> str:
    if not torch.cuda.is_available():
        return "CUDA not available"
    allocated = torch.cuda.memory_allocated() / 1024**3
    reserved = torch.cuda.memory_reserved() / 1024**3
    total = torch.cuda.get_device_properties(0).total_memory / 1024**3
    name = torch.cuda.get_device_properties(0).name
    return f"{name} | Used: {allocated:.1f} GB / Total: {total:.1f} GB (reserved: {reserved:.1f} GB)"


# ---------------------------------------------------------------------------
# Pipeline builders
# ---------------------------------------------------------------------------
def _ensure_urae_2k_adapter() -> str:
    """Return path to the official URAE 2K adapter, downloading it if needed."""
    path = os.path.join(CKPT_DIR, "urae_2k_adapter.safetensors")
    if not os.path.exists(path):
        os.makedirs(CKPT_DIR, exist_ok=True)
        hf_hub_download(
            repo_id="Huage001/URAE",
            filename="urae_2k_adapter.safetensors",
            local_dir=CKPT_DIR,
        )
    return path


def _ensure_urae_4k_adapter() -> str:
    """Return path to the official URAE 4K adapter, downloading it if needed."""
    path = os.path.join(CKPT_DIR, "urae_4k_adapter.safetensors")
    if not os.path.exists(path):
        os.makedirs(CKPT_DIR, exist_ok=True)
        hf_hub_download(
            repo_id="Huage001/URAE",
            filename="urae_4k_adapter.safetensors",
            local_dir=CKPT_DIR,
        )
    return path


def _resolve_checkpoint_path(resolution: str, checkpoint_name: str, custom_path: str) -> str:
    """Turn a checkpoint name into an actual filesystem path.

    - "Custom"          → the user-supplied path
    - name with __HF__  → download from HuggingFace on demand
    - anything else     → the registered local path
    """
    if checkpoint_name == "Custom":
        return custom_path.strip()
    registry = CHECKPOINTS_2K if resolution == "2K" else CHECKPOINTS_4K
    raw = registry.get(checkpoint_name, "")
    if raw == _HF:
        return _ensure_urae_2k_adapter() if resolution == "2K" else _ensure_urae_4k_adapter()
    return raw


def _svd_decompose_blocks(blocks, rank, device, has_to_out=True):
    """Run SVD decomposition on transformer blocks for 4K adapter setup."""
    for idx in range(len(blocks)):
        for proj_name in ["to_q", "to_k", "to_v"]:
            attr = getattr(blocks[idx].attn, proj_name)
            matrix_w = attr.weight.data.to(device)
            u, s, v = torch.linalg.svd(matrix_w)
            attr.weight.data = (
                u[:, :-rank] @ torch.diag(s[:-rank]) @ v[:-rank, :]
            ).to("cpu")
            proc = blocks[idx].attn.processor
            getattr(proc, f"{proj_name}_b").weight.data = (
                u[:, -rank:] @ torch.diag(torch.sqrt(s[-rank:]))
            ).to("cpu")
            getattr(proc, f"{proj_name}_a").weight.data = (
                torch.diag(torch.sqrt(s[-rank:])) @ v[-rank:, :]
            ).to("cpu")

        if has_to_out:
            matrix_w = blocks[idx].attn.to_out[0].weight.data.to(device)
            u, s, v = torch.linalg.svd(matrix_w)
            blocks[idx].attn.to_out[0].weight.data = (
                u[:, :-rank] @ torch.diag(s[:-rank]) @ v[:-rank, :]
            ).to("cpu")
            blocks[idx].attn.processor.to_out_b.weight.data = (
                u[:, -rank:] @ torch.diag(torch.sqrt(s[-rank:]))
            ).to("cpu")
            blocks[idx].attn.processor.to_out_a.weight.data = (
                torch.diag(torch.sqrt(s[-rank:])) @ v[-rank:, :]
            ).to("cpu")


def _load_2k_pipeline(checkpoint_path: str, cpu_offload: bool):
    """Build the 2K generation pipeline (base FLUX + LoRA adapter)."""
    dtype = torch.bfloat16
    device = torch.device("cuda")

    transformer = FluxTransformer2DModel.from_pretrained(
        BFL_REPO, subfolder="transformer", torch_dtype=dtype
    )
    pipe = FluxPipeline.from_pretrained(
        BFL_REPO, transformer=transformer, torch_dtype=dtype
    )
    pipe.scheduler.config.use_dynamic_shifting = False
    pipe.scheduler.config.time_shift = 10

    pipe.load_lora_weights(checkpoint_path)

    if cpu_offload:
        pipe.enable_model_cpu_offload()
    else:
        pipe = pipe.to(device)

    return pipe


def _load_4k_pipeline(checkpoint_path: str):
    """Build the 4K generation pipeline (FLUX + 2K LoRA fused + SVD + 4K adapter).
    
    4K always uses CPU offload to handle VRAM requirements.
    """
    dtype = torch.bfloat16
    device = torch.device("cuda")
    rank = 16

    transformer = FluxTransformer2DModel.from_pretrained(
        BFL_REPO, subfolder="transformer"
    )
    pipe = FluxPipeline.from_pretrained(
        BFL_REPO, transformer=transformer, torch_dtype=dtype
    )
    pipe.scheduler.config.use_dynamic_shifting = False
    pipe.scheduler.config.time_shift = 10
    pipe.enable_model_cpu_offload()

    # Step 1: Load and fuse the 2K LoRA as the base
    urae_2k_path = _ensure_urae_2k_adapter()
    pipe.load_lora_weights(urae_2k_path)
    pipe.fuse_lora()

    # Step 2: Set up adaptation attention processors
    attn_processors = {}
    for k in pipe.transformer.attn_processors.keys():
        attn_processors[k] = FluxAttnAdaptationProcessor2_0(
            rank=rank, to_out="single" not in k
        )
    pipe.transformer.set_attn_processor(attn_processors)

    # Step 3: SVD decomposition (or load from cache)
    cache_path = os.path.join(CKPT_DIR, "_urae_4k_adapter_dev.safetensors")
    if os.path.exists(cache_path):
        pipe.transformer.to(dtype=dtype)
        pipe.transformer.load_state_dict(load_file(cache_path), strict=False)
    else:
        with torch.no_grad():
            _svd_decompose_blocks(
                pipe.transformer.transformer_blocks, rank, device, has_to_out=True
            )
            _svd_decompose_blocks(
                pipe.transformer.single_transformer_blocks, rank, device, has_to_out=False
            )
        pipe.transformer.to(dtype=dtype)
        os.makedirs(CKPT_DIR, exist_ok=True)
        state_dict = pipe.transformer.state_dict()
        attn_state_dict = {k: v for k, v in state_dict.items() if "base_layer" in k}
        save_file(attn_state_dict, cache_path)

    # Step 4: Load the 4K adapter weights
    state_dict = load_file(checkpoint_path)
    _, unexpected = pipe.transformer.load_state_dict(state_dict, strict=False)
    if unexpected:
        print(f"Warning: unexpected keys when loading 4K adapter: {unexpected}")

    # Step 5: Convert VAE for memory-efficient 4K decoding
    if PATCH_CONV_AVAILABLE:
        pipe.vae = convert_model(pipe.vae, splits=4)
    else:
        pipe.vae.enable_tiling()

    return pipe


# ---------------------------------------------------------------------------
# Public helpers called from Gradio
# ---------------------------------------------------------------------------
def get_gpu_status():
    """Return GPU VRAM info string."""
    return _get_vram_info()


def load_model(resolution, checkpoint_name, custom_checkpoint_path, cpu_offload):
    """Load (or re-use) the pipeline for the chosen configuration."""
    global _current_pipe, _current_config

    try:
        ckpt_path = _resolve_checkpoint_path(resolution, checkpoint_name, custom_checkpoint_path)
    except Exception as e:
        return f"❌ Error resolving checkpoint: {e}", _get_vram_info()

    if not ckpt_path:
        return "⚠️ No checkpoint path specified.", _get_vram_info()
    if not os.path.exists(ckpt_path):
        return f"❌ Checkpoint not found:\n{ckpt_path}", _get_vram_info()

    # For 4K resolution, cpu_offload is always forced on
    effective_cpu_offload = True if resolution == "4K" else cpu_offload
    config_key = (resolution, ckpt_path, effective_cpu_offload)

    if _current_config == config_key:
        return f"✅ Model already loaded ({resolution} — {checkpoint_name}).", _get_vram_info()

    _flush_pipeline()

    try:
        t0 = time.time()
        if resolution == "2K":
            _current_pipe = _load_2k_pipeline(ckpt_path, effective_cpu_offload)
        else:
            _current_pipe = _load_4k_pipeline(ckpt_path)
        _current_config = config_key
        elapsed = time.time() - t0
        return (
            f"✅ Model loaded in {elapsed:.1f}s\n{resolution} — {checkpoint_name}",
            _get_vram_info(),
        )
    except Exception as e:
        _flush_pipeline()
        return f"❌ Error loading model:\n{e}", _get_vram_info()


def generate(
    prompt,
    resolution,
    checkpoint_name,
    custom_checkpoint_path,
    cpu_offload,
    width,
    height,
    guidance_scale,
    num_inference_steps,
    seed,
    ntk_factor,
    max_sequence_length,
    proportional_attention,
    progress=gr.Progress(track_tqdm=False),
):
    """Generate an image with the currently loaded pipeline."""
    global _current_pipe, _current_config

    if not prompt or not prompt.strip():
        raise gr.Error("Please enter a prompt before generating.")

    # Resolve checkpoint path (HF entries are downloaded on demand)
    try:
        ckpt_path = _resolve_checkpoint_path(resolution, checkpoint_name, custom_checkpoint_path)
    except Exception as e:
        raise gr.Error(f"Could not resolve checkpoint: {e}")

    effective_cpu_offload = True if resolution == "4K" else cpu_offload
    config_key = (resolution, ckpt_path, effective_cpu_offload)

    # Auto-load if not loaded or config changed
    if _current_config != config_key:
        progress(0, desc="Loading model…")
        status, _ = load_model(resolution, checkpoint_name, custom_checkpoint_path, cpu_offload)
        if _current_pipe is None:
            raise gr.Error(status)

    generator = torch.manual_seed(int(seed)) if int(seed) >= 0 else None
    total_steps = int(num_inference_steps)

    # ── Step-timing state ────────────────────────────────────────────────
    _timing = {"step": 0, "t_start": None, "step_times": []}

    def _step_callback(pipe, step_index: int, timestep, callback_kwargs: dict):
        """Called by diffusers after every denoising step.

        Measures actual wall-clock time per step, computes a rolling mean,
        and updates the Gradio progress bar with an accurate ETA.
        """
        now = time.time()
        if _timing["t_start"] is not None:
            _timing["step_times"].append(now - _timing["t_start"])
        _timing["t_start"] = now
        _timing["step"] = step_index + 1

        done = step_index + 1  # steps completed
        remaining = total_steps - done

        if _timing["step_times"]:
            avg_dt = sum(_timing["step_times"]) / len(_timing["step_times"])
            eta_s = avg_dt * remaining
            if eta_s >= 60:
                eta_str = f"{eta_s / 60:.1f} min remaining"
            else:
                eta_str = f"{eta_s:.0f}s remaining"
            it_s = f"{1.0 / avg_dt:.2f} it/s" if avg_dt > 0 else ""
            desc = f"Step {done}/{total_steps} — {it_s} — {eta_str}"
        else:
            desc = f"Step {done}/{total_steps}…"

        progress(done / total_steps, desc=desc)
        return callback_kwargs

    progress(0, desc=f"Starting {total_steps} denoising steps…")

    image = _current_pipe(
        prompt,
        height=int(height),
        width=int(width),
        guidance_scale=float(guidance_scale),
        num_inference_steps=total_steps,
        max_sequence_length=int(max_sequence_length),
        generator=generator,
        ntk_factor=float(ntk_factor),
        proportional_attention=bool(proportional_attention),
        callback_on_step_end=_step_callback,
    ).images[0]

    progress(1.0, desc="Done ✓")
    return image, _get_vram_info()


# ---------------------------------------------------------------------------
# Gradio UI callbacks
# ---------------------------------------------------------------------------
def on_resolution_change(resolution):
    """Update checkpoint dropdown, width/height sliders and CPU offload visibility."""
    choices = _build_checkpoint_choices(resolution)
    if resolution == "2K":
        w, h, max_dim = 2048, 2048, 2560
        cpu_offload_info = gr.update(visible=True)
    else:
        w, h, max_dim = 4096, 4096, 4096
        cpu_offload_info = gr.update(visible=False)

    return (
        gr.update(choices=choices, value=choices[0]),   # checkpoint_name
        gr.update(value=w, maximum=max_dim),             # width
        gr.update(value=h, maximum=max_dim),             # height
        cpu_offload_info,                                # cpu_offload row
    )


def on_checkpoint_change(checkpoint_name):
    """Show/hide the custom path textbox."""
    return gr.update(visible=(checkpoint_name == "Custom"))


# ---------------------------------------------------------------------------
# Custom CSS
# ---------------------------------------------------------------------------
CUSTOM_CSS = """
/* ── Global font & background ─────────────────────────────────────────── */
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

body, .gradio-container {
    font-family: 'Inter', sans-serif !important;
    background: #0f1117 !important;
}

/* ── Header ───────────────────────────────────────────────────────────── */
.lwd-header {
    background: linear-gradient(135deg, #1a1f2e 0%, #16213e 40%, #0f3460 100%);
    border-radius: 16px;
    padding: 28px 36px;
    margin-bottom: 20px;
    border: 1px solid rgba(99, 179, 237, 0.15);
    box-shadow: 0 8px 32px rgba(0, 0, 0, 0.4);
}
.lwd-header h1 {
    font-size: 2rem !important;
    font-weight: 700 !important;
    background: linear-gradient(90deg, #63b3ed, #9f7aea, #f6ad55);
    -webkit-background-clip: text !important;
    -webkit-text-fill-color: transparent !important;
    background-clip: text !important;
    margin: 0 0 8px 0 !important;
    line-height: 1.2 !important;
}
.lwd-header p {
    color: #a0aec0 !important;
    font-size: 0.95rem !important;
    margin: 0 !important;
    line-height: 1.5 !important;
}

/* ── Panel cards ─────────────────────────────────────────────────────── */
.panel-card {
    background: #1a1f2e !important;
    border-radius: 14px !important;
    border: 1px solid rgba(255,255,255,0.07) !important;
    padding: 20px !important;
    box-shadow: 0 4px 24px rgba(0,0,0,0.3) !important;
}

/* ── Section labels ──────────────────────────────────────────────────── */
.section-label {
    font-size: 0.72rem !important;
    font-weight: 600 !important;
    letter-spacing: 0.12em !important;
    text-transform: uppercase !important;
    color: #63b3ed !important;
    margin-bottom: 10px !important;
    display: block;
}

/* ── Resolution tabs ─────────────────────────────────────────────────── */
.resolution-tabs .tab-nav button {
    font-weight: 600 !important;
    font-size: 0.9rem !important;
    border-radius: 8px 8px 0 0 !important;
}

/* ── Status box ──────────────────────────────────────────────────────── */
.status-box textarea, .status-box input {
    font-family: 'Inter', monospace !important;
    font-size: 0.82rem !important;
    background: #0d1117 !important;
    border: 1px solid rgba(255,255,255,0.08) !important;
    border-radius: 8px !important;
    color: #68d391 !important;
    resize: none !important;
}
.status-box.error textarea {
    color: #fc8181 !important;
}

/* ── Vram badge ──────────────────────────────────────────────────────── */
.vram-info textarea, .vram-info input {
    font-size: 0.78rem !important;
    background: #0d1117 !important;
    border: 1px solid rgba(255,255,255,0.06) !important;
    border-radius: 6px !important;
    color: #718096 !important;
    resize: none !important;
}

/* ── Generate button ─────────────────────────────────────────────────── */
#generate-btn {
    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%) !important;
    border: none !important;
    border-radius: 10px !important;
    font-size: 1rem !important;
    font-weight: 600 !important;
    padding: 14px !important;
    color: white !important;
    transition: all 0.2s ease !important;
    box-shadow: 0 4px 15px rgba(102, 126, 234, 0.3) !important;
}
#generate-btn:hover {
    transform: translateY(-1px) !important;
    box-shadow: 0 6px 20px rgba(102, 126, 234, 0.45) !important;
}
#generate-btn:active {
    transform: translateY(0) !important;
}

/* ── Load model button ───────────────────────────────────────────────── */
#load-btn {
    background: #2d3748 !important;
    border: 1px solid rgba(99, 179, 237, 0.3) !important;
    border-radius: 8px !important;
    color: #63b3ed !important;
    font-weight: 500 !important;
    transition: all 0.2s ease !important;
}
#load-btn:hover {
    background: #3a4a60 !important;
    border-color: rgba(99, 179, 237, 0.6) !important;
}

/* ── Sliders ─────────────────────────────────────────────────────────── */
.gradio-slider input[type=range] {
    accent-color: #7c6ff7 !important;
}

/* ── Output image ────────────────────────────────────────────────────── */
#output-image {
    border-radius: 14px !important;
    overflow: hidden !important;
    border: 1px solid rgba(255,255,255,0.08) !important;
}
#output-image img {
    border-radius: 12px !important;
    width: 100% !important;
    object-fit: contain !important;
}

/* ── Prompt textarea ─────────────────────────────────────────────────── */
#prompt-box textarea {
    background: #0d1117 !important;
    border: 1px solid rgba(255,255,255,0.1) !important;
    border-radius: 10px !important;
    font-size: 0.9rem !important;
    line-height: 1.6 !important;
    color: #e2e8f0 !important;
    transition: border-color 0.2s !important;
}
#prompt-box textarea:focus {
    border-color: rgba(99, 179, 237, 0.5) !important;
    outline: none !important;
}

/* ── Accordion ───────────────────────────────────────────────────────── */
.gradio-accordion {
    background: #1a1f2e !important;
    border: 1px solid rgba(255,255,255,0.06) !important;
    border-radius: 10px !important;
}
.gradio-accordion .label-wrap {
    font-weight: 500 !important;
    color: #a0aec0 !important;
}

/* ── Divider ─────────────────────────────────────────────────────────── */
.divider {
    height: 1px;
    background: rgba(255,255,255,0.06);
    margin: 16px 0;
}

/* Remove footer */
footer { display: none !important; }

/* Dark scrollbar */
::-webkit-scrollbar { width: 6px; }
::-webkit-scrollbar-track { background: #0d1117; }
::-webkit-scrollbar-thumb { background: #2d3748; border-radius: 3px; }
"""


# ---------------------------------------------------------------------------
# Build the Gradio app
# ---------------------------------------------------------------------------
def build_app():
    initial_2k_choices = _build_checkpoint_choices("2K")
    initial_4k_choices = _build_checkpoint_choices("4K")

    with gr.Blocks(
        title="Latent Wavelet Diffusion",
    ) as demo:

        # ── Header ────────────────────────────────────────────────────────
        gr.HTML("""
        <div class="lwd-header">
          <h1>⚡ Latent Wavelet Diffusion</h1>
          <p>High-resolution image generation with FLUX + URAE adapters and wavelet attention.
             Supports <strong>2K</strong> (LoRA-based) and <strong>4K</strong> (SVD-based) generation modes.</p>
        </div>
        """)

        with gr.Row(equal_height=False):

            # ── Left column: controls ──────────────────────────────────────
            with gr.Column(scale=4, min_width=380):

                # ── Model Configuration ────────────────────────────────────
                with gr.Group():
                    gr.HTML('<span class="section-label">🔧 Model Configuration</span>')

                    resolution = gr.Radio(
                        choices=["2K", "4K"],
                        value="2K",
                        label="Resolution Mode",
                        info="2K uses LoRA adapter · 4K uses SVD decomposition + always CPU-offloaded",
                        elem_classes=["resolution-tabs"],
                    )

                    checkpoint_name = gr.Dropdown(
                        choices=initial_2k_choices,
                        value=initial_2k_choices[0],
                        label="Checkpoint",
                        info="Select a pre-trained checkpoint to load",
                    )

                    custom_path = gr.Textbox(
                        label="Custom checkpoint path",
                        visible=False,
                        placeholder="/path/to/weights.safetensors",
                        info="Absolute path to a .safetensors file",
                    )

                    with gr.Row(visible=True) as cpu_offload_row:
                        cpu_offload = gr.Checkbox(
                            label="CPU Offload  (saves VRAM, ~2× slower)",
                            value=False,
                            info="Only applies to 2K mode — 4K always uses offload",
                        )

                with gr.Row():
                    load_btn = gr.Button("⬇  Load Model", elem_id="load-btn", scale=1)
                    refresh_vram_btn = gr.Button("🔄", scale=0, min_width=48, variant="secondary")

                load_status = gr.Textbox(
                    label="Status",
                    interactive=False,
                    value="No model loaded. Click 'Load Model' or hit Generate.",
                    lines=2,
                    max_lines=4,
                    elem_classes=["status-box"],
                )

                vram_info = gr.Textbox(
                    label="GPU",
                    interactive=False,
                    value=_get_vram_info(),
                    lines=1,
                    max_lines=2,
                    elem_classes=["vram-info"],
                )

                gr.HTML('<div class="divider"></div>')

                # ── Prompt ─────────────────────────────────────────────────
                gr.HTML('<span class="section-label">✍️ Prompt</span>')

                prompt = gr.Textbox(
                    label="",
                    lines=4,
                    max_lines=8,
                    placeholder="Describe the image you want to generate…",
                    value=(
                        "A serene woman in a flowing azure dress, gracefully perched "
                        "on a sunlit cliff overlooking a tranquil sea, her hair gently "
                        "tousled by the breeze. The scene is infused with a sense of "
                        "peace, evoking a dreamlike atmosphere, reminiscent of "
                        "Impressionist paintings."
                    ),
                    elem_id="prompt-box",
                )

                gr.HTML('<div class="divider"></div>')

                # ── Image Dimensions ───────────────────────────────────────
                gr.HTML('<span class="section-label">📐 Dimensions</span>')

                with gr.Row():
                    width = gr.Slider(
                        minimum=512, maximum=2560, step=64,
                        value=2048, label="Width (px)",
                    )
                    height = gr.Slider(
                        minimum=512, maximum=2560, step=64,
                        value=2048, label="Height (px)",
                    )

                gr.HTML('<div class="divider"></div>')

                # ── Advanced parameters ────────────────────────────────────
                with gr.Accordion("⚙️  Advanced Parameters", open=False):
                    with gr.Row():
                        guidance_scale = gr.Slider(
                            minimum=0.0, maximum=20.0, step=0.1, value=3.5,
                            label="Guidance Scale",
                            info="CFG strength — 3.5 is a good default for FLUX",
                        )
                        num_inference_steps = gr.Slider(
                            minimum=1, maximum=60, step=1, value=28,
                            label="Inference Steps",
                            info="More steps = higher quality but slower",
                        )
                    with gr.Row():
                        seed = gr.Number(
                            value=8888,
                            label="Seed  (−1 = random)",
                            precision=0,
                            minimum=-1,
                        )
                        ntk_factor = gr.Slider(
                            minimum=1.0, maximum=50.0, step=0.5, value=10.0,
                            label="NTK Factor",
                            info="RoPE NTK scaling for high-res positional encoding",
                        )
                    with gr.Row():
                        max_sequence_length = gr.Slider(
                            minimum=64, maximum=512, step=64, value=512,
                            label="Max Sequence Length",
                            info="T5 text encoder token budget",
                        )
                        proportional_attention = gr.Checkbox(
                            label="Proportional Attention",
                            value=True,
                            info="Scale attention by sequence length ratio",
                        )

                generate_btn = gr.Button(
                    "✨  Generate Image",
                    variant="primary",
                    elem_id="generate-btn",
                    size="lg",
                )

            # ── Right column: output ───────────────────────────────────────
            with gr.Column(scale=5, min_width=520):

                gr.HTML('<span class="section-label">🖼️ Output</span>')

                output_image = gr.Image(
                    label="",
                    type="pil",
                    elem_id="output-image",
                    height=700,
                )

                gr.HTML("""
                <div style="margin-top:10px; padding:12px 16px;
                            background:#1a1f2e; border-radius:10px;
                            border:1px solid rgba(255,255,255,0.06);">
                  <p style="margin:0; color:#718096; font-size:0.8rem; line-height:1.6;">
                    <strong style="color:#a0aec0;">💡 Tips:</strong>
                    4K generation can take <strong style="color:#f6ad55;">`15+ minutes</strong> on a single GPU.
                    Use <em>CPU Offload</em> for 2K if you run into OOM errors.
                    The first 4K load is slow (SVD decomposition is cached afterwards).
                  </p>
                </div>
                """)

        # ── Event wiring ───────────────────────────────────────────────────
        resolution.change(
            fn=on_resolution_change,
            inputs=[resolution],
            outputs=[checkpoint_name, width, height, cpu_offload_row],
        )

        checkpoint_name.change(
            fn=on_checkpoint_change,
            inputs=[checkpoint_name],
            outputs=[custom_path],
        )

        load_btn.click(
            fn=load_model,
            inputs=[resolution, checkpoint_name, custom_path, cpu_offload],
            outputs=[load_status, vram_info],
        )

        refresh_vram_btn.click(
            fn=get_gpu_status,
            inputs=[],
            outputs=[vram_info],
        )

        generate_btn.click(
            fn=generate,
            inputs=[
                prompt,
                resolution,
                checkpoint_name,
                custom_path,
                cpu_offload,
                width,
                height,
                guidance_scale,
                num_inference_steps,
                seed,
                ntk_factor,
                max_sequence_length,
                proportional_attention,
            ],
            outputs=[output_image, vram_info],
        )

    return demo


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    demo = build_app()
    demo.queue(max_size=2)
    demo.launch(
        share=False,
        server_name="0.0.0.0",
        server_port=7860,
        css=CUSTOM_CSS,
        theme=gr.themes.Base(
            primary_hue=gr.themes.colors.purple,
            neutral_hue=gr.themes.colors.slate,
            font=gr.themes.GoogleFont("Inter"),
        )
    )
