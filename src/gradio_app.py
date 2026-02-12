"""
Gradio demo for Latent Wavelet Diffusion (LWD) image generation.

Supports both 2K and 4K resolution generation using FLUX + URAE adapters.
Run from the src/ directory:
    python gradio_app.py
"""

import os
import sys
import gc

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

CHECKPOINTS_2K = {
    # "URAE 2K (Official)": os.path.join(CKPT_DIR, "urae_2k_adapter.safetensors"),
    "URAE 2K + VAE-SE + Wavelet Att": "/mnt/media/luigi/LWD Material/Checkpoints/URAE/2K/URAE_VAE_SE_WAV_ATT_LAION/checkpoint-2000/pytorch_lora_weights.safetensors",
    "URAE 2K (Retrained)": "/mnt/media/luigi/LWD Material/Checkpoints/URAE/2K/URAE_original_trained_by_me/checkpoint-2000/pytorch_lora_weights.safetensors",
    "URAE 2K + Wav Att (No SE-VAE)": "/mnt/media/luigi/LWD Material/Checkpoints/URAE/2K/l_003_URAE_VAE_NOOO_SE_WAV_ATT_LAION_2048/checkpoint-6000/pytorch_lora_weights.safetensors",
}

CHECKPOINTS_4K = {
    # "URAE 4K (Official)": os.path.join(CKPT_DIR, "urae_4k_adapter.safetensors"),
    "URAE 4K + VAE-SE + Wavelet Att": "/mnt/media/luigi/LWD Material/Checkpoints/URAE/4K/4K_URAE_VAE_SE_WAV_ATT_LAION_4096/checkpoint-2000/adapter_weights.safetensors",
    "URAE 4K (Retrained)": "/mnt/media/luigi/LWD Material/Checkpoints/URAE/4K/4K_urae_original/ckpt_xxx/urae_4k_adapter.safetensors",
    "URAE 4K + Wav Att (l=0.3)": "/mnt/media/luigi/LWD Material/Checkpoints/URAE/4K/l_03_4K_URAE_VAE_SE_WAV_ATT_LAION_4096/checkpoint-2000/adapter_weights.safetensors",
}

# ---------------------------------------------------------------------------
# Global pipeline cache
# ---------------------------------------------------------------------------
_current_pipe = None
_current_config = None  # (resolution, checkpoint_path) tuple


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
# Pipeline builders
# ---------------------------------------------------------------------------
def _ensure_urae_2k_adapter():
    """Download the official URAE 2K adapter if missing."""
    path = os.path.join(CKPT_DIR, "urae_2k_adapter.safetensors")
    if not os.path.exists(path):
        os.makedirs(CKPT_DIR, exist_ok=True)
        hf_hub_download(
            repo_id="Huage001/URAE",
            filename="urae_2k_adapter.safetensors",
            local_dir=CKPT_DIR,
        )
    return path


def _ensure_urae_4k_adapter():
    """Download the official URAE 4K adapter if missing."""
    path = os.path.join(CKPT_DIR, "urae_4k_adapter.safetensors")
    if not os.path.exists(path):
        os.makedirs(CKPT_DIR, exist_ok=True)
        hf_hub_download(
            repo_id="Huage001/URAE",
            filename="urae_4k_adapter.safetensors",
            local_dir=CKPT_DIR,
        )
    return path


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


def _load_4k_pipeline(checkpoint_path: str):
    """Build the 4K generation pipeline (FLUX + 2K LoRA fused + SVD + 4K adapter)."""
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
        # Cache the decomposed weights for next time
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
def load_model(resolution, checkpoint_name, custom_checkpoint_path, cpu_offload):
    """Load (or re-use) the pipeline for the chosen configuration."""
    global _current_pipe, _current_config

    # Resolve checkpoint path
    if checkpoint_name == "Custom":
        ckpt_path = custom_checkpoint_path.strip()
    else:
        registry = CHECKPOINTS_2K if resolution == "2K" else CHECKPOINTS_4K
        ckpt_path = registry.get(checkpoint_name, "")

    if not ckpt_path:
        return "No checkpoint path specified."
    if not os.path.exists(ckpt_path):
        return f"Checkpoint not found: {ckpt_path}"

    config_key = (resolution, ckpt_path, cpu_offload)
    if _current_config == config_key:
        return f"Model already loaded ({resolution} — {checkpoint_name})."

    _flush_pipeline()

    try:
        if resolution == "2K":
            _current_pipe = _load_2k_pipeline(ckpt_path, cpu_offload)
        else:
            _current_pipe = _load_4k_pipeline(ckpt_path)
        _current_config = config_key
        return f"Model loaded successfully ({resolution} — {checkpoint_name})."
    except Exception as e:
        _flush_pipeline()
        return f"Error loading model: {e}"


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
):
    """Generate an image with the currently loaded pipeline."""
    global _current_pipe, _current_config

    # Auto-load if not loaded yet or config changed
    ckpt_path = custom_checkpoint_path.strip() if checkpoint_name == "Custom" else (
        CHECKPOINTS_2K if resolution == "2K" else CHECKPOINTS_4K
    ).get(checkpoint_name, "")
    config_key = (resolution, ckpt_path, cpu_offload)

    if _current_config != config_key:
        status = load_model(resolution, checkpoint_name, custom_checkpoint_path, cpu_offload)
        if _current_pipe is None:
            raise gr.Error(status)

    generator = torch.manual_seed(seed) if seed >= 0 else None

    image = _current_pipe(
        prompt,
        height=int(height),
        width=int(width),
        guidance_scale=guidance_scale,
        num_inference_steps=int(num_inference_steps),
        max_sequence_length=int(max_sequence_length),
        generator=generator,
        ntk_factor=ntk_factor,
        proportional_attention=proportional_attention,
    ).images[0]

    return image


# ---------------------------------------------------------------------------
# Gradio UI callbacks
# ---------------------------------------------------------------------------
def on_resolution_change(resolution):
    """Update checkpoint dropdown and default width/height when resolution changes."""
    if resolution == "2K":
        choices = list(CHECKPOINTS_2K.keys()) + ["Custom"]
        w, h = 2048, 2048
    else:
        choices = list(CHECKPOINTS_4K.keys()) + ["Custom"]
        w, h = 4096, 4096
    return (
        gr.update(choices=choices, value=choices[0]),
        gr.update(value=w),
        gr.update(value=h),
    )


def on_checkpoint_change(checkpoint_name):
    """Show/hide the custom path textbox."""
    return gr.update(visible=(checkpoint_name == "Custom"))


# ---------------------------------------------------------------------------
# Build the Gradio app
# ---------------------------------------------------------------------------
def build_app():
    with gr.Blocks(
        title="Latent Wavelet Diffusion Demo",
        css="footer {display: none !important}",
    ) as demo:
        gr.Markdown(
            "# Latent Wavelet Diffusion — Image Generation Demo\n"
            "Generate **2K** or **4K** images using FLUX + URAE adapters with wavelet attention."
        )

        with gr.Row():
            # ---- Left column: controls ----
            with gr.Column(scale=1):
                # -- Model selection --
                gr.Markdown("### Model Configuration")
                resolution = gr.Radio(
                    choices=["2K", "4K"],
                    value="2K",
                    label="Resolution",
                )
                checkpoint_name = gr.Dropdown(
                    choices=list(CHECKPOINTS_2K.keys()) + ["Custom"],
                    value=list(CHECKPOINTS_2K.keys())[0],
                    label="Checkpoint",
                )
                custom_path = gr.Textbox(
                    label="Custom checkpoint path (.safetensors)",
                    visible=False,
                    placeholder="/path/to/weights.safetensors",
                )
                cpu_offload = gr.Checkbox(
                    label="Enable CPU offload (saves VRAM, slower)",
                    value=False,
                )
                load_btn = gr.Button("Load Model", variant="secondary")
                load_status = gr.Textbox(
                    label="Status", interactive=False, value="No model loaded."
                )

                gr.Markdown("---")

                # -- Prompt --
                gr.Markdown("### Generation")
                prompt = gr.Textbox(
                    label="Prompt",
                    lines=4,
                    value=(
                        "A serene woman in a flowing azure dress, gracefully perched "
                        "on a sunlit cliff overlooking a tranquil sea, her hair gently "
                        "tousled by the breeze. The scene is infused with a sense of "
                        "peace, evoking a dreamlike atmosphere, reminiscent of "
                        "Impressionist paintings."
                    ),
                )

                with gr.Row():
                    width = gr.Slider(
                        minimum=512, maximum=4096, step=64, value=2048, label="Width"
                    )
                    height = gr.Slider(
                        minimum=512, maximum=4096, step=64, value=2048, label="Height"
                    )

                # -- Hyperparameters --
                with gr.Accordion("Hyperparameters", open=False):
                    guidance_scale = gr.Slider(
                        minimum=0.0, maximum=20.0, step=0.1, value=3.5,
                        label="Guidance Scale",
                    )
                    num_inference_steps = gr.Slider(
                        minimum=1, maximum=100, step=1, value=28,
                        label="Inference Steps",
                    )
                    seed = gr.Number(
                        value=8888,
                        label="Seed (-1 for random)",
                        precision=0,
                    )
                    ntk_factor = gr.Slider(
                        minimum=1.0, maximum=50.0, step=0.5, value=10.0,
                        label="NTK Factor",
                    )
                    max_sequence_length = gr.Slider(
                        minimum=64, maximum=512, step=64, value=512,
                        label="Max Sequence Length",
                    )
                    proportional_attention = gr.Checkbox(
                        label="Proportional Attention",
                        value=True,
                    )

                generate_btn = gr.Button("Generate", variant="primary")

            # ---- Right column: output ----
            with gr.Column(scale=1):
                output_image = gr.Image(label="Generated Image", type="pil")

        # ---- Event wiring ----
        resolution.change(
            fn=on_resolution_change,
            inputs=[resolution],
            outputs=[checkpoint_name, width, height],
        )
        checkpoint_name.change(
            fn=on_checkpoint_change,
            inputs=[checkpoint_name],
            outputs=[custom_path],
        )
        load_btn.click(
            fn=load_model,
            inputs=[resolution, checkpoint_name, custom_path, cpu_offload],
            outputs=[load_status],
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
            outputs=[output_image],
        )

    return demo


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    demo = build_app()
    demo.queue()
    demo.launch(share=False, server_name="0.0.0.0", server_port=7860)
