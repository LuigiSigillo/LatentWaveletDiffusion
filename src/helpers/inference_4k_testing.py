import os
import torch
from huggingface_hub import hf_hub_download
from pipeline_flux import FluxPipeline
from transformer_flux import FluxTransformer2DModel
from attention_processor import FluxAttnAdaptationProcessor2_0
from safetensors.torch import load_file, save_file
from patch_conv import convert_model


bfl_repo="black-forest-labs/FLUX.1-dev"
device = torch.device('cuda')
dtype = torch.bfloat16
transformer = FluxTransformer2DModel.from_pretrained(bfl_repo, subfolder="transformer")
pipe = FluxPipeline.from_pretrained(bfl_repo, transformer=transformer, torch_dtype=dtype)
pipe.scheduler.config.use_dynamic_shifting = False
pipe.scheduler.config.time_shift = 10
pipe.enable_model_cpu_offload()

if not os.path.exists('ckpt/urae_2k_adapter.safetensors'):
    hf_hub_download(repo_id="Huage001/URAE", filename='urae_2k_adapter.safetensors', local_dir='ckpt', local_dir_use_symlinks=False)
pipe.load_lora_weights("ckpt/urae_2k_adapter.safetensors")
pipe.fuse_lora()

# ckpt_path = "/mnt/media/luigi/LWD Material/Checkpoints/URAE/4K/4K_URAE_VAE_SE_WAV_ATT_LAION_4096/checkpoint-2000/adapter_weights.safetensors"
# pipe.load_lora_weights(ckpt_path)
# pipe.fuse_lora()

# * Substitute original attention processors
rank = 16
attn_processors = {}
for k in pipe.transformer.attn_processors.keys():
    attn_processors[k] = FluxAttnAdaptationProcessor2_0(rank=rank, to_out='single' not in k)
pipe.transformer.set_attn_processor(attn_processors)

# * If no cached major components, compute them via SVD and save them to cache_path
# * If you don't want to save cached major components, simply set `cache_path = None`
# cache_path = 'ckpt/_urae_4k_adapter_dev.safetensors'
cache_path= "/mnt/media/luigi/LWD Material/Checkpoints/URAE/4K/4K_URAE_VAE_SE_WAV_ATT_LAION_4096/checkpoint-2000/adapter_weights.safetensors"
if cache_path is not None and os.path.exists(cache_path):
    pipe.transformer.to(dtype=dtype)
    pipe.transformer.load_state_dict(load_file(cache_path), strict=False)
else:
    with torch.no_grad():
        for idx in range(len(pipe.transformer.transformer_blocks)):
            matrix_w = pipe.transformer.transformer_blocks[idx].attn.to_q.weight.data.to(device)
            matrix_u, matrix_s, matrix_v = torch.linalg.svd(matrix_w)
            pipe.transformer.transformer_blocks[idx].attn.to_q.weight.data = (
                matrix_u[:, :-rank] @ torch.diag(matrix_s[:-rank]) @ matrix_v[:-rank, :]
            ).to('cpu')
            pipe.transformer.transformer_blocks[idx].attn.processor.to_q_b.weight.data = (
                matrix_u[:, -rank:] @ torch.diag(torch.sqrt(matrix_s[-rank:]))
            ).to('cpu')
            pipe.transformer.transformer_blocks[idx].attn.processor.to_q_a.weight.data = (
                torch.diag(torch.sqrt(matrix_s[-rank:])) @ matrix_v[-rank:, :]
            ).to('cpu')

            matrix_w = pipe.transformer.transformer_blocks[idx].attn.to_k.weight.data.to(device)
            matrix_u, matrix_s, matrix_v = torch.linalg.svd(matrix_w)
            pipe.transformer.transformer_blocks[idx].attn.to_k.weight.data = (
                matrix_u[:, :-rank] @ torch.diag(matrix_s[:-rank]) @ matrix_v[:-rank, :]
            ).to('cpu')
            pipe.transformer.transformer_blocks[idx].attn.processor.to_k_b.weight.data = (
                matrix_u[:, -rank:] @ torch.diag(torch.sqrt(matrix_s[-rank:]))
            ).to('cpu')
            pipe.transformer.transformer_blocks[idx].attn.processor.to_k_a.weight.data = (
                torch.diag(torch.sqrt(matrix_s[-rank:])) @ matrix_v[-rank:, :]
            ).to('cpu')

            matrix_w = pipe.transformer.transformer_blocks[idx].attn.to_v.weight.data.to(device)
            matrix_u, matrix_s, matrix_v = torch.linalg.svd(matrix_w)
            pipe.transformer.transformer_blocks[idx].attn.to_v.weight.data = (
                matrix_u[:, :-rank] @ torch.diag(matrix_s[:-rank]) @ matrix_v[:-rank, :]
            ).to('cpu')
            pipe.transformer.transformer_blocks[idx].attn.processor.to_v_b.weight.data = (
                matrix_u[:, -rank:] @ torch.diag(torch.sqrt(matrix_s[-rank:]))
            ).to('cpu')
            pipe.transformer.transformer_blocks[idx].attn.processor.to_v_a.weight.data = (
                torch.diag(torch.sqrt(matrix_s[-rank:])) @ matrix_v[-rank:, :]
            ).to('cpu')

            matrix_w = pipe.transformer.transformer_blocks[idx].attn.to_out[0].weight.data.to(device)
            matrix_u, matrix_s, matrix_v = torch.linalg.svd(matrix_w)
            pipe.transformer.transformer_blocks[idx].attn.to_out[0].weight.data = (
                matrix_u[:, :-rank] @ torch.diag(matrix_s[:-rank]) @ matrix_v[:-rank, :]
            ).to('cpu')
            pipe.transformer.transformer_blocks[idx].attn.processor.to_out_b.weight.data = (
                matrix_u[:, -rank:] @ torch.diag(torch.sqrt(matrix_s[-rank:]))
            ).to('cpu')
            pipe.transformer.transformer_blocks[idx].attn.processor.to_out_a.weight.data = (
                torch.diag(torch.sqrt(matrix_s[-rank:])) @ matrix_v[-rank:, :]
            ).to('cpu')
        for idx in range(len(pipe.transformer.single_transformer_blocks)):
            matrix_w = pipe.transformer.single_transformer_blocks[idx].attn.to_q.weight.data.to(device)
            matrix_u, matrix_s, matrix_v = torch.linalg.svd(matrix_w)
            pipe.transformer.single_transformer_blocks[idx].attn.to_q.weight.data = (
                matrix_u[:, :-rank] @ torch.diag(matrix_s[:-rank]) @ matrix_v[:-rank, :]
            ).to('cpu')
            pipe.transformer.single_transformer_blocks[idx].attn.processor.to_q_b.weight.data = (
                matrix_u[:, -rank:] @ torch.diag(torch.sqrt(matrix_s[-rank:]))
            ).to('cpu')
            pipe.transformer.single_transformer_blocks[idx].attn.processor.to_q_a.weight.data = (
                torch.diag(torch.sqrt(matrix_s[-rank:])) @ matrix_v[-rank:, :]
            ).to('cpu')

            matrix_w = pipe.transformer.single_transformer_blocks[idx].attn.to_k.weight.data.to(device)
            matrix_u, matrix_s, matrix_v = torch.linalg.svd(matrix_w)
            pipe.transformer.single_transformer_blocks[idx].attn.to_k.weight.data = (
                matrix_u[:, :-rank] @ torch.diag(matrix_s[:-rank]) @ matrix_v[:-rank, :]
            ).to('cpu')
            pipe.transformer.single_transformer_blocks[idx].attn.processor.to_k_b.weight.data = (
                matrix_u[:, -rank:] @ torch.diag(torch.sqrt(matrix_s[-rank:]))
            ).to('cpu')
            pipe.transformer.single_transformer_blocks[idx].attn.processor.to_k_a.weight.data = (
                torch.diag(torch.sqrt(matrix_s[-rank:])) @ matrix_v[-rank:, :]
            ).to('cpu')

            matrix_w = pipe.transformer.single_transformer_blocks[idx].attn.to_v.weight.data.to(device)
            matrix_u, matrix_s, matrix_v = torch.linalg.svd(matrix_w)
            pipe.transformer.single_transformer_blocks[idx].attn.to_v.weight.data = (
                matrix_u[:, :-rank] @ torch.diag(matrix_s[:-rank]) @ matrix_v[:-rank, :]
            ).to('cpu')
            pipe.transformer.single_transformer_blocks[idx].attn.processor.to_v_b.weight.data = (
                matrix_u[:, -rank:] @ torch.diag(torch.sqrt(matrix_s[-rank:]))
            ).to('cpu')
            pipe.transformer.single_transformer_blocks[idx].attn.processor.to_v_a.weight.data = (
                torch.diag(torch.sqrt(matrix_s[-rank:])) @ matrix_v[-rank:, :]
            ).to('cpu')
    pipe.transformer.to(dtype=dtype)
    if cache_path is not None:
        state_dict = pipe.transformer.state_dict()
        attn_state_dict = {}
        for k in state_dict.keys():
            if 'base_layer' in k:
                attn_state_dict[k] = state_dict[k]
        save_file(attn_state_dict, cache_path)

# * Download pre-trained 4k adapter
if not os.path.exists('ckpt/urae_4k_adapter.safetensors'):
    hf_hub_download(repo_id="Huage001/URAE", filename='urae_4k_adapter.safetensors', local_dir='ckpt', local_dir_use_symlinks=False)

# * Optionally, you can convert the minor-component adapter into a LoRA for easier use

# %%
lora_conversion = True
if lora_conversion and not os.path.exists('ckpt/urae_4k_adapter_lora_conversion_dev.safetensors'):
    cur = pipe.transformer.state_dict()
    tgt = load_file('ckpt/urae_4k_adapter.safetensors')
    ref = load_file('ckpt/urae_2k_adapter.safetensors')
    new_ckpt = {}
    for k in tgt.keys():
        if 'to_k_a' in k:
            k_ = 'transformer.' + k.replace('.processor.to_k_a', '.to_k.lora_A')
        elif 'to_k_b' in k:
            k_ = 'transformer.' + k.replace('.processor.to_k_b', '.to_k.lora_B')
        elif 'to_q_a' in k:
            k_ = 'transformer.' + k.replace('.processor.to_q_a', '.to_q.lora_A')
        elif 'to_q_b' in k:
            k_ = 'transformer.' + k.replace('.processor.to_q_b', '.to_q.lora_B')
        elif 'to_v_a' in k:
            k_ = 'transformer.' + k.replace('.processor.to_v_a', '.to_v.lora_A')
        elif 'to_v_b' in k:
            k_ = 'transformer.' + k.replace('.processor.to_v_b', '.to_v.lora_B')
        elif 'to_out_a' in k:
            k_ = 'transformer.' + k.replace('.processor.to_out_a', '.to_out.0.lora_A')
        elif 'to_out_b' in k:
            k_ = 'transformer.' + k.replace('.processor.to_out_b', '.to_out.0.lora_B')
        else:
            print(k)
            assert False
        if '_a.' in k and '_b.' not in k:
            new_ckpt[k_] = torch.cat([-cur[k], tgt[k], ref[k_]], dim=0)
        elif '_b.' in k and '_a.' not in k:
            new_ckpt[k_] = torch.cat([cur[k], tgt[k], ref[k_]], dim=1)
        else:
            print(k)
            assert False
    save_file(new_ckpt, 'ckpt/urae_4k_adapter_lora_conversion_dev.safetensors')

# * Load state_dict of 4k adapter
m, u = pipe.transformer.load_state_dict(load_file('ckpt/urae_4k_adapter.safetensors'), strict=False)
assert len(u) == 0

# * Use patch-wise convolution for VAE to avoid OOM error when decoding
# * If still OOM, try replacing the following line with `pipe.vae.enable_tiling()`
pipe.vae = convert_model(pipe.vae, splits=4)

# * Everything ready. Let's generate!
# * 4K generation using FLUX-1.dev can take a while, e.g., 5min on H100.

prompt = "A serene woman in a flowing azure dress, gracefully perched on a sunlit cliff overlooking a tranquil sea, her hair gently tousled by the breeze. The scene is infused with a sense of peace, evoking a dreamlike atmosphere, reminiscent of Impressionist paintings."
height = 4096
width = 4096
image = pipe(
    prompt,
    height=height,
    width=width,
    guidance_scale=3.5,
    num_inference_steps=28,
    max_sequence_length=512,
    generator=torch.manual_seed(8888),
    ntk_factor=10,
    proportional_attention=True
).images[0]
image


