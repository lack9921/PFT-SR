#!/usr/bin/env python3
"""
PFT-SR LoViF Restoration - Testing Script
Usage: python test_restoration.py [checkpoint_path] [input_dir] [output_dir]

Examples:
  python test_restoration.py --checkpoint experiments/PFT_LoViF_AllTracks/models/net_g_latest.pth
  python test_restoration.py --checkpoint experiments/PFT_LoViF_AllTracks/models/net_g_67000.pth \
                             --input /datasets_host/LoViF_val/Haze/LQ \
                             --output test_results
"""

import os
import sys
import argparse
import time
import torch
import numpy as np
from PIL import Image
from os import path as osp

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from basicsr.archs.pft_arch import PFT
from basicsr.utils import img2tensor, tensor2img


def parse_args():
    parser = argparse.ArgumentParser(description='PFT-SR LoViF Restoration Testing')
    parser.add_argument('--checkpoint', type=str, required=True,
                        help='Path to model checkpoint (.pth)')
    parser.add_argument('--input', type=str, default=None,
                        help='Input image or directory of images')
    parser.add_argument('--output', type=str, default='./test_results',
                        help='Output directory for restored images')
    parser.add_argument('--device', type=str, default='cuda',
                        help='Device to run inference on (cuda/cpu)')
    return parser.parse_args()


def build_model(checkpoint_path, device):
    """Build the PFT restoration model and load weights."""
    # Network config matching the training config
    net_config = {
        'type': 'PFT',
        'upscale': 1,
        'in_chans': 3,
        'img_size': 64,
        'embed_dim': 90,
        'depths': [6, 6, 6, 6, 6, 6],
        'num_heads': 6,
        'num_topk': [1024, 1024, 1024, 1024, 1024, 1024,
                     256, 256, 256, 256, 256, 256,
                     128, 128, 128, 128, 128, 128,
                     64, 64, 64, 64, 64, 64,
                     32, 32, 32, 32, 32, 32,
                     16, 16, 16, 16, 16, 16],
        'window_size': 16,
        'convffn_kernel_size': 5,
        'img_range': 1.0,
        'mlp_ratio': 2.0,
        'upsampler': '',
        'resi_connection': '1conv',
        'use_checkpoint': True,
    }

    model = PFT(**net_config)
    model.eval()
    model.to(device)

    # Load checkpoint
    if osp.exists(checkpoint_path):
        state = torch.load(checkpoint_path, map_location=device)
        if 'params' in state:
            model.load_state_dict(state['params'], strict=True)
            print(f'Loaded params from {checkpoint_path}')
        elif 'params_ema' in state:
            model.load_state_dict(state['params_ema'], strict=True)
            print(f'Loaded params_ema from {checkpoint_path}')
        else:
            model.load_state_dict(state, strict=True)
            print(f'Loaded state dict from {checkpoint_path}')
    else:
        raise FileNotFoundError(f'Checkpoint not found: {checkpoint_path}')

    return model


def process_image(model, img_tensor, device='cuda'):
    """Run restoration on a single image tensor."""
    img_tensor = img_tensor.unsqueeze(0).to(device)  # add batch dim

    with torch.no_grad():
        output = model(img_tensor)

    return output.squeeze(0)  # remove batch dim


def process_folder(model, input_dir, output_dir, device='cuda'):
    """Restore all images in a directory."""
    os.makedirs(output_dir, exist_ok=True)

    supported = ('.png', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff')
    files = [f for f in os.listdir(input_dir)
             if f.lower().endswith(supported)]
    files.sort()

    print(f'Found {len(files)} images in {input_dir}')
    total_time = 0

    for idx, fname in enumerate(files):
        fpath = osp.join(input_dir, fname)

        # Load image
        img = Image.open(fpath).convert('RGB')
        img_tensor = img2tensor(img, bgr2rgb=False, float32=True)

        # Run inference
        start = time.time()
        out_tensor = process_image(model, img_tensor, device)
        total_time += time.time() - start

        # Save output
        out_img = tensor2img(out_tensor, min_max=(0, 1))
        out_path = osp.join(output_dir, fname)
        Image.fromarray(out_img).save(out_path)

        if (idx + 1) % 10 == 0:
            print(f'  [{idx+1}/{len(files)}] processed, avg time: {total_time/(idx+1):.3f}s')

    print(f'Done! Average time: {total_time/len(files):.3f}s per image')
    print(f'Results saved to: {output_dir}')


def main():
    args = parse_args()

    # Check for checkpoint
    if not osp.exists(args.checkpoint):
        print(f'[ERROR] Checkpoint not found: {args.checkpoint}')
        print('Available checkpoints:')
        exp_dir = osp.join(osp.dirname(__file__), 'experiments', 'PFT_LoViF_AllTracks', 'models')
        if osp.exists(exp_dir):
            for f in sorted(os.listdir(exp_dir)):
                print(f'  {osp.join(exp_dir, f)}')
        sys.exit(1)

    # Build model
    device = args.device if torch.cuda.is_available() else 'cpu'
    print(f'Using device: {device}')
    model = build_model(args.checkpoint, device)

    # Process
    if args.input:
        if osp.isfile(args.input):
            # Single image
            img = Image.open(args.input).convert('RGB')
            img_tensor = img2tensor(img, bgr2rgb=False, float32=True)
            out_tensor = process_image(model, img_tensor, device)
            out_img = tensor2img(out_tensor, min_max=(0, 1))
            os.makedirs(args.output, exist_ok=True)
            out_path = osp.join(args.output, osp.basename(args.input))
            Image.fromarray(out_img).save(out_path)
            print(f'Saved result to: {out_path}')
        elif osp.isdir(args.input):
            # Directory of images
            process_folder(model, args.input, args.output, device)
        else:
            print(f'[ERROR] Invalid input path: {args.input}')
            sys.exit(1)
    else:
        # No input specified - just load model and verify
        print('Model loaded successfully!')
        print(f'Parameters: {sum(p.numel() for p in model.parameters()):,}')
        print('Use --input to specify images for restoration.')


if __name__ == '__main__':
    main()
