import torch
torch.backends.cuda.matmul.allow_tf32 = True

from models import DiT_models
from download import find_model
from diffusion import create_diffusion
from diffusers.models import AutoencoderKL
from torchvision import transforms
from tqdm import tqdm
import os
from PIL import Image
import numpy as np
import math
import argparse
import clip
from Image_path import ImagePaths
from torch.utils.data import DataLoader

def center_crop_arr(pil_image, image_size):
    while min(*pil_image.size) >= 2 * image_size:
        pil_image = pil_image.resize(
            tuple(x // 2 for x in pil_image.size), resample=Image.BOX
        )
    scale = image_size / min(*pil_image.size)
    pil_image = pil_image.resize(
        tuple(round(x * scale) for x in pil_image.size), resample=Image.BICUBIC
    )
    arr = np.array(pil_image)
    crop_y = (arr.shape[0] - image_size) // 2
    crop_x = (arr.shape[1] - image_size) // 2
    return Image.fromarray(arr[crop_y: crop_y + image_size, crop_x: crop_x + image_size])

def crop_and_resize(pil_image, image_size):
    return center_crop_arr(pil_image, image_size)

def create_npz_from_sample_folder(sample_dir, num=50_000):
    samples = []
    for i in tqdm(range(num), desc="Building .npz file from samples"):
        sample_pil = Image.open(f"{sample_dir}/{i:06d}.png")
        sample_np = np.asarray(sample_pil).astype(np.uint8)
        samples.append(sample_np)
    samples = np.stack(samples)
    assert samples.shape == (num, samples.shape[1], samples.shape[2], 3)
    npz_path = f"{sample_dir}.npz"
    np.savez(npz_path, arr_0=samples)
    print(f"Saved .npz file to {npz_path} [shape={samples.shape}].")
    return npz_path

def main(args):
    torch.set_grad_enabled(False)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Load model
    latent_size = args.image_size // 8
    model = DiT_models[args.model](
        input_size=latent_size,
        num_classes=args.num_classes
    ).to(device)
    
    state_dict = find_model(args.ckpt or f"DiT-XL-2-{args.image_size}x{args.image_size}.pt")
    model.load_state_dict(state_dict)
    model.eval()

    diffusion = create_diffusion(str(args.num_sampling_steps))
    vae = AutoencoderKL.from_pretrained(f"stabilityai/sd-vae-ft-{args.vae}").to(device)
    
    assert args.cfg_scale >= 1.0, "In almost all cases, cfg_scale should be >= 1.0"
    using_cfg = args.cfg_scale > 1.0
    
    # Create folder to save samples
    model_string_name = args.model.replace("/", "-")
    ckpt_string_name = os.path.basename(args.ckpt).replace(".pt", "") if args.ckpt else "pretrained"
    folder_name = f"{model_string_name}-{ckpt_string_name}-size-{args.image_size}-vae-{args.vae}-cfg-{args.cfg_scale}-seed-{args.global_seed}"
    sample_folder_dir = f"{args.sample_dir}/{folder_name}"
    os.makedirs(sample_folder_dir, exist_ok=True)
    print(f"Saving .png samples at {sample_folder_dir}")

    transform = transforms.Compose([
        transforms.Lambda(lambda pil_image: crop_and_resize(pil_image, args.image_size)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5], inplace=True)
    ])

    dataset = ImagePaths(args.data_path, transform=transform)
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=True
    )

    clip_model, preprocess = clip.load("ViT-B/32", device=device)
    
    prompt = "a picture"  # Null guidance
    text = clip.tokenize([prompt] * args.batch_size).to(device)
    prompt_features = clip_model.encode_text(text)
    prompt_features = prompt_features.float()

    total_samples = 0
    pbar = tqdm(total=args.num_fid_samples)

    while total_samples < args.num_fid_samples:
        for x, y in loader:
            z = torch.randn(args.batch_size, 4, latent_size, latent_size, device=device)
            y = y.to(device)

            if using_cfg:
                y = clip_model.encode_text(y)
                y = y.float()
                y = torch.cat([y, prompt_features], 0)
                z = torch.cat([z, z], 0)
                model_kwargs = dict(y=y, cfg_scale=args.cfg_scale)
                sample_fn = model.forward_with_cfg
            else:
                y = clip_model.encode_text(y)
                y = y.float()
                model_kwargs = dict(y=y)
                sample_fn = model.forward

            # Sample images
            samples = diffusion.p_sample_loop(
                sample_fn, z.shape, z, clip_denoised=False, model_kwargs=model_kwargs, progress=False, device=device
            )

            if using_cfg:
                samples, _ = samples.chunk(2, dim=0)

            samples = vae.decode(samples / 0.23126199671607967).sample
            samples = torch.clamp(127.5 * samples + 128.0, 0, 255).permute(0, 2, 3, 1).to("cpu", dtype=torch.uint8).numpy()

            # Save samples to disk as individual .png files
            for i, sample in enumerate(samples):
                index = total_samples + i
                Image.fromarray(sample).save(f"{sample_folder_dir}/{str(index)}.png")
                Image.fromarray(sample).save(f"{sample_folder_dir}/{index:06d}.png")

            total_samples += samples.shape[0]
            pbar.update(samples.shape[0])

            if total_samples >= args.num_fid_samples:
                break

    pbar.close()
    create_npz_from_sample_folder(sample_folder_dir, args.num_fid_samples)
    print("Done.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-path", type=str, default='.\\pair_list_cub.pkl') #Path to the .pkl file which stores text descriptions for CUB images
    parser.add_argument("--model", type=str, choices=list(DiT_models.keys()), default="DiT-L/2")
    parser.add_argument("--vae", type=str, choices=["ema", "mse"], default="ema")
    parser.add_argument("--sample-dir", type=str, default="samples")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--num-fid-samples", type=int, default=5000)
    parser.add_argument("--image-size", type=int, choices=[256, 512], default=256)
    parser.add_argument("--num-classes", type=int, default=1000)
    parser.add_argument("--cfg-scale", type=float, default=1.5)
    parser.add_argument("--num-sampling-steps", type=int, default=250)
    parser.add_argument("--global-seed", type=int, default=100)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--ckpt", type=str, default='')
    args = parser.parse_args()
    main(args)