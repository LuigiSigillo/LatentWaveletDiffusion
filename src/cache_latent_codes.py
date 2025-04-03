import argparse
import os
import numpy as np
from PIL import Image
import math
from safetensors.torch import save_file
import torch
import torch.nn as nn
import tqdm
import cv2
from diffusers import AutoencoderKL
from patch_conv import PatchConv2d


def parse_args(input_args=None):
    parser = argparse.ArgumentParser(description="Simple example of a training script.")
    parser.add_argument(
        "--pretrained_model_name_or_path",
        type=str,
        default=None,
        required=True,
        help="Path to pretrained model or model identifier from huggingface.co/models.",
    )
    parser.add_argument(
        "--revision",
        type=str,
        default=None,
        required=False,
        help="Revision of pretrained model identifier from huggingface.co/models.",
    )
    parser.add_argument(
        "--variant",
        type=str,
        default=None,
        help="Variant of the model files of the pretrained model identifier from huggingface.co/models, 'e.g.' fp16",
    )
    parser.add_argument(
        "--data_root",
        type=str,
        default=None
    )
    parser.add_argument(
        "--cache_dir",
        type=str,
        default=None,
        help="The directory where the downloaded models and datasets will be stored.",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="flux-lora",
        help="The output directory where the model predictions and checkpoints will be written.",
    )
    parser.add_argument(
        "--mixed_precision",
        type=str,
        default=None,
        choices=["no", "fp16", "bf16"],
        help=(
            "Whether to use mixed precision. Choose between fp16 and bf16 (bfloat16). Bf16 requires PyTorch >="
            " 1.10.and an Nvidia Ampere GPU.  Default to the value of accelerate config of the current system or the"
            " flag passed with the `accelerate.launch` command. Use this argument to override the accelerate config."
        ),
    )
    parser.add_argument("--local_rank", type=int, default=0, help="For distributed training: local_rank")
    parser.add_argument(
        "--resolution",
        type=int,
        default=None,
        help="Image resolution for resizing. If None, the original resolution will be used.",
    )
    parser.add_argument(
        "--num_workers",
        type=int,
        default=1,
        help="Number of workers",
    )

    if input_args is not None:
        args = parser.parse_args(input_args)
    else:
        args = parser.parse_args()

    args.local_rank = int(os.environ.get("LOCAL_RANK", 0))
    
    return args


def resize(img, base_size=4096):
    all_sizes = np.array([
        [base_size, base_size],
        [base_size, base_size * 3 // 4],
        [base_size, base_size // 2],
        [base_size, base_size // 4]
    ])
    width, height = img.size
    if width < height:
        size = all_sizes[np.argmin(np.abs(all_sizes[:, 0] / all_sizes[:, 1] - height / width))][::-1]
    else:
        size = all_sizes[np.argmin(np.abs(all_sizes[:, 0] / all_sizes[:, 1] - width / height))]
    return img.resize((size[0], size[1]))


def convert_model(model: nn.Module, splits: int = 4, sequential: bool=True) -> nn.Module:
    """
    Convert the convolutions in the model to PatchConv2d.
    """
    if isinstance(model, PatchConv2d):
        return model
    elif isinstance(model, nn.Conv2d) and model.kernel_size[0] > 1 and model.kernel_size[1] > 1:
        return PatchConv2d(splits=splits, conv2d=model)
    else:
        for name, module in model.named_modules():
            if isinstance(module, (nn.Conv2d, PatchConv2d)):
                continue
            if 'downsamplers' in name:
                continue
            for subname, submodule in module.named_children():
                if isinstance(submodule, nn.Conv2d) and submodule.kernel_size[0] > 1 and submodule.kernel_size[1] > 1:
                    setattr(module, subname, PatchConv2d(splits=splits, sequential=sequential, conv2d=submodule))
        return model


def main(args):
    if torch.cuda.is_available():
        torch.cuda.set_device(args.local_rank)
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    if args.mixed_precision == 'fp16':
        dtype = torch.float16
    elif args.mixed_precision == 'bf16':
        dtype = torch.bfloat16
    else:
        dtype = torch.float32

    vae = AutoencoderKL.from_pretrained(
        args.pretrained_model_name_or_path,
        subfolder="vae",
        revision=args.revision,
        variant=args.variant,
    ).to(device, dtype)
    
    if args.resolution is not None and args.resolution > 3072:
        vae = convert_model(vae, splits=4)
    
    all_info = sorted([item for item in os.listdir(args.data_root) if item.endswith('.jpg')])

    os.makedirs(args.output_dir, exist_ok=True)

    work_load = math.ceil(len(all_info) / args.num_workers)
    for idx in tqdm.tqdm(range(work_load * args.local_rank, min(work_load * (args.local_rank + 1), len(all_info)))):
        output_path = os.path.join(args.output_dir, f"{all_info[idx][:all_info[idx].rfind('.')]}_latent_code.safetensors")
        img = cv2.cvtColor(cv2.imread(os.path.join(args.data_root, all_info[idx])), cv2.COLOR_BGR2RGB)
        img = Image.fromarray(img)
        if args.resolution is not None:
            img = resize(img, args.resolution)
        img = torch.from_numpy((np.array(img) / 127.5) - 1)
        img = img.permute(2, 0, 1)
        with torch.no_grad():
            img = img.unsqueeze(0)
            data = vae.encode(img.to(device, vae.dtype)).latent_dist
            mean = data.mean[0].cpu().data
            std = data.std[0].cpu().data
        save_file(
            {'mean': mean, 'std': std},
            output_path
        )


if __name__ == '__main__':
    main(parse_args())