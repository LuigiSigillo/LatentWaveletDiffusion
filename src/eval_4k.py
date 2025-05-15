import os
import torch
from huggingface_hub import hf_hub_download
from pipeline_flux import FluxPipeline
from transformer_flux import FluxTransformer2DModel
from attention_processor import FluxAttnAdaptationProcessor2_0
from safetensors.torch import load_file, save_file
from patch_conv import convert_model
import hpsv2
from tqdm import tqdm
import multiprocessing
import argparse


def load_model_4k(device_str, 
                  cache_dir, 
                  ckpt_lora_weights_2k = "/leonardo_scratch/fast/IscrC_UniMod/luigi/HighResolutionWav/src/ckpt/urae_2k_adapter.safetensors",
                  ckpt_path_4k = "/leonardo_scratch/fast/IscrC_UniMod/luigi/HighResolutionWav/src/ckpt/4K_URAE_VAE_SE_WAV_ATT_LAION_4096/checkpoint-2000/adapter_weights.safetensors",
                  cache_path_adapter='/leonardo_scratch/fast/IscrC_UniMod/luigi/HighResolutionWav/src/ckpt/_urae_4k_adapter_dev.safetensors',
                  ):
    bfl_repo="black-forest-labs/FLUX.1-dev"
    device = torch.device(device_str)
    dtype = torch.bfloat16
    transformer = FluxTransformer2DModel.from_pretrained(bfl_repo, subfolder="transformer", cache_dir=cache_dir)
    pipe = FluxPipeline.from_pretrained(bfl_repo, transformer=transformer, torch_dtype=dtype, cache_dir=cache_dir)
    pipe.scheduler.config.use_dynamic_shifting = False
    pipe.scheduler.config.time_shift = 10
    pipe.enable_model_cpu_offload()


    # if not os.path.exists('ckpt/urae_2k_adapter.safetensors'):
    #     hf_hub_download(repo_id="Huage001/URAE", filename='urae_2k_adapter.safetensors', local_dir='ckpt', local_dir_use_symlinks=False)
    pipe.load_lora_weights(ckpt_lora_weights_2k)
    pipe.fuse_lora()

    rank = 16
    attn_processors = {}
    for k in pipe.transformer.attn_processors.keys():
        attn_processors[k] = FluxAttnAdaptationProcessor2_0(rank=rank, to_out='single' not in k)
    pipe.transformer.set_attn_processor(attn_processors)


    if cache_path_adapter is not None and os.path.exists(cache_path_adapter):
        pipe.transformer.to(dtype=dtype)
        pipe.transformer.load_state_dict(load_file(cache_path_adapter), strict=False)
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
        if cache_path_adapter is not None:
            state_dict = pipe.transformer.state_dict()
            attn_state_dict = {}
            for k in state_dict.keys():
                if 'base_layer' in k:
                    attn_state_dict[k] = state_dict[k]
            save_file(attn_state_dict, cache_path_adapter)


    
    # With this code
    # state_dict = load_file(ckpt_path, device="cpu")  # Explicitly load to CPU first
    state_dict = load_file(ckpt_path_4k)
    m, u = pipe.transformer.load_state_dict(state_dict, strict=False)
    assert len(u) == 0

    pipe.vae = convert_model(pipe.vae, splits=4)
    return pipe

def generate_images(root_path_proj, checkpoint_path,device_str, height=4096, width=4096, seed=8888, cache_dir=None):
    
    cache_path_adapter = '/leonardo_scratch/fast/IscrC_UniMod/luigi/HighResolutionWav/src/ckpt/_urae_4k_adapter_dev.safetensors'
    # cache_path_adapter = None
    pipe = load_model_4k(device_str,
                         cache_dir=cache_dir,
                         ckpt_lora_weights_2k="/leonardo_scratch/fast/IscrC_UniMod/luigi/HighResolutionWav/src/ckpt/urae_2k_adapter.safetensors",
                         ckpt_path_4k=checkpoint_path,
                         cache_path_adapter=cache_path_adapter,
                        )

    # Get benchmark prompts (<style> = all, anime, concept-art, paintings, photo)
    all_prompts = hpsv2.benchmark_prompts('all') 
    gen_seed = torch.manual_seed(seed=seed)
    #take name by the checkpoint name
    name_exp = checkpoint_path.split("/")[-2]
    print(name_exp)
    # Iterate over the benchmark prompts to generate images
    for style, prompts in tqdm(all_prompts.items(), total=len(all_prompts), desc="Generating images"):
        # Create the directory if it doesn't exist
        output_dir = os.path.join(root_path_proj,"src","output",name_exp, style)
        print(output_dir)
        os.makedirs(output_dir, exist_ok=True)
        for idx, prompt in tqdm(enumerate(prompts), total=len(prompts), desc=f"Generating {style} images"):
            if os.path.join(root_path_proj,"src","output",name_exp, style, f"{idx:05d}.jpg") in os.listdir(os.path.join(root_path_proj,"src","output",name_exp, style)):
                print(f"Image {idx:05d}.jpg already exists. Skipping.")
                continue
            # image = TextToImageModel(prompt)
            image = pipe(
                prompt,
                height=height,
                width=width,
                guidance_scale=3.5,
                num_inference_steps=28,
                max_sequence_length=512,
                generator=gen_seed,
                ntk_factor=10,
                proportional_attention=True
            ).images[0]
            # TextToImageModel is the model you want to evaluate
            image.save(os.path.join(root_path_proj,"src","output",name_exp, style, f"{idx:05d}.jpg")) 
            # <image_path> is the folder path to store generated images, as the input of hpsv2.evaluate().

    hps2 = hpsv2.evaluate("output/"+checkpoint_path.split("/")[-2]) 
    return hps2


def generate_images_for_style(style, prompts, device_id, root_path_proj, checkpoint_path, height, width, seed, cache_dir):
    # os.environ["CUDA_VISIBLE_DEVICES"] = str(device_id)  # Assign specific GPU
    torch.cuda.set_device(device_id)
    device = torch.device("cuda:"+str(device_id))
    print(f"Using device: {device} and loading the pipe")
    pipe = load_model_4k(device,
                         cache_dir=cache_dir,
                         ckpt_lora_weights_2k="/leonardo_scratch/fast/IscrC_UniMod/luigi/HighResolutionWav/src/ckpt/urae_2k_adapter.safetensors",
                         ckpt_path_4k=checkpoint_path,
                         cache_path_adapter='/leonardo_scratch/fast/IscrC_UniMod/luigi/HighResolutionWav/src/ckpt/_urae_4k_adapter_dev.safetensors',
                        )
    # pipe.to(device)
    # Set up output directory
    name_exp = checkpoint_path.split("/")[-3]
    output_dir = os.path.join(root_path_proj, "src", "output", name_exp, style)
    os.makedirs(output_dir, exist_ok=True)

    # Generate images for the given style
    gen_seed = torch.manual_seed(seed=seed)
    print(f"STARTING images for style: {style} on device {device_id}")
    for idx, prompt in tqdm(enumerate(prompts), total=len(prompts), desc=f"Generating {style} images, with device {device_id}"):
        if os.path.join(f"{idx:05d}.jpg") in os.listdir(os.path.join(root_path_proj, "src", "output", name_exp, style)):
            print(f"Image {idx:05d}.jpg already exists. Skipping.")
            continue
        image = pipe(
            prompt,
            height=height,
            width=width,
            guidance_scale=3.5,
            num_inference_steps=28,
            max_sequence_length=512,
            generator=gen_seed,
            ntk_factor=10,
            proportional_attention=True
        ).images[0]
        image.save(os.path.join(output_dir, f"{idx:05d}.jpg"))
        del image
        torch.cuda.empty_cache()  # Clear GPU memory
        import gc
        gc.collect()  # Clear Python garbage collector

def parallel_generate_images(root_path_proj, checkpoint_path, device_ids, height=2048, width=2048, seed=888, cache_dir=None):
    # Get benchmark prompts
    all_prompts = hpsv2.benchmark_prompts('all')
    styles = list(all_prompts.keys())
    # Check if only one GPU is available
    if len(device_ids) == 1:
        print("Only one GPU detected. Running processes sequentially.")
        for style in styles:
            prompts = all_prompts[style]
            device_id = device_ids[0]  # Use the single available GPU
            generate_images_for_style(
                style, prompts, device_id, root_path_proj, checkpoint_path, height, width, seed, cache_dir
            )
    else:
        # Create a process for each style, assigning GPUs in a round-robin fashion
        processes = []
        print(f"Using {len(device_ids)} GPUs for parallel processing.")
        for i, style in enumerate(styles):
            prompts = all_prompts[style]
            device_id = device_ids[i] 
            p = multiprocessing.Process(
                target=generate_images_for_style,
                args=(style, prompts, device_id, root_path_proj, checkpoint_path, height, width, seed, cache_dir)
            )
            processes.append(p)
            p.start()

        # Wait for all processes to complete
        for p in processes:
            p.join()


def set_hf_cache_dir(cache_dir):
    if cache_dir:
        os.environ["HF_HOME"] = cache_dir
        #also the torch cache when downloading the model
        os.environ["TORCH_HOME"] = cache_dir
        print(f"Set Huggingface and torch cache to: {cache_dir}")

if __name__ == "__main__":
    multiprocessing.set_start_method("spawn")
    parser = argparse.ArgumentParser(description="Image generation and evaluation script")
    
    # General arguments
    parser.add_argument('--checkpoint_path', type=str, required=True,
                       help='Path to the checkpoint')
    parser.add_argument('--device', type=str, default='cuda',
                       help='Device to use (cuda or cpu)')
    
    # Generation options
    parser.add_argument('--generate', action='store_true',
                       help='Generate images using the checkpoint')
    parser.add_argument('--height', type=int, default=4096,
                       help='Height of generated images')
    parser.add_argument('--width', type=int, default=4096,
                       help='Width of generated images')
    parser.add_argument('--seed', type=int, default=8888,
                       help='Random seed for generation')
    parser.add_argument('--cache_dir', type=str, default=None,
                       help='/leonardo_scratch/large/userexternal/lsigillo')
    parser.add_argument('--style', type=str, default=None,
                        help='Specify a style to generate (e.g., anime, concept-art, paintings, photo)')

    args = parser.parse_args()
    
    set_hf_cache_dir(args.cache_dir)

    root_path_proj = os.path.abspath(os.getcwd())
    # checkpoint_path = "/leonardo_scratch/fast/IscrC_UniMod/luigi/HighResolutionWav/src/ckpt/4K_URAE_VAE_SE_WAV_ATT_LAION_4096/checkpoint-2000/adapter_weights.safetensors"
    
# Get available GPU IDs
    num_gpus = torch.cuda.device_count()
    if num_gpus == 0:
        print("No GPUs available. Exiting.")
        exit(1)
    device_ids = list(range(num_gpus))  # Generate GPU IDs [0, 1, ..., num_gpus-1]
    print(f"Available GPUs: {device_ids}")
    
    # Check if a specific style is provided
    if args.style:
        print(f"Generating images for style: {args.style}")
        all_prompts = hpsv2.benchmark_prompts('all')
        if args.style not in all_prompts:
            print(f"Error: Style '{args.style}' not found in available styles: {list(all_prompts.keys())}")
            exit(1)
        
        # Use the first available GPU for the specified style
        generate_images_for_style(
            args.style,
            all_prompts[args.style],
            device_ids[0],  # Use the first GPU
            root_path_proj,
            args.checkpoint_path,
            args.height,
            args.width,
            args.seed,
            args.cache_dir
        )
    else:
        # Run parallel generation for all styles
        parallel_generate_images(
            root_path_proj,
            args.checkpoint_path,
            device_ids,  # Pass the dynamically detected GPU IDs
            height=args.height,
            width=args.width,
            seed=args.seed,
            cache_dir=args.cache_dir
        )
    
    # prompt = "A serene woman in a flowing azure dress, gracefully perched on a sunlit cliff overlooking a tranquil sea, her hair gently tousled by the breeze. The scene is infused with a sense of peace, evoking a dreamlike atmosphere, reminiscent of Impressionist paintings."
    # height = 4096
    # width = 4096
    # image = pipe(
    #     prompt,
    #     height=height,
    #     width=width,
    #     guidance_scale=3.5,
    #     num_inference_steps=28,
    #     max_sequence_length=512,
    #     generator=torch.manual_seed(8888),
    #     ntk_factor=10,
    #     proportional_attention=True
    # ).images[0]
    # image.save("output.jpg")