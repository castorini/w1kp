import argparse
import asyncio
import json
import random
from pathlib import Path

import PIL.Image
import pandas as pd
import sys

import torch
from matplotlib import pyplot as plt
from tqdm.asyncio import tqdm
import aioboto3

from w1kp import DinoV2DistanceMeasure


async def amain():
    def plot_images(im1, im2):
        plt.clf()
        fig, ax = plt.subplots(1, 2, figsize=(15, 7.5))

        # Set titles
        ax[0].set_title('Image A')
        ax[1].set_title('Image B')

        # Set bold
        for axi in ax:
            axi.set_title(axi.get_title(), fontweight='bold')

        # Make font larger
        for axi in ax:
            axi.title.set_fontsize(24)

        ax[0].set_axis_off()
        ax[1].set_axis_off()
        ax[0].imshow(im1)
        ax[1].imshow(im2)

        return fig, ax

    parser = argparse.ArgumentParser()
    parser.add_argument('--model-path', type=Path, required=True)
    parser.add_argument('--input-folder', '-i', required=True, type=Path)
    parser.add_argument('--output-path', '-o', type=Path, default='magnitude-outputs')
    parser.add_argument('--model', type=str, default='dalle3', choices=['dalle3', 'sdxl', 'sd2', 'imagen', 'midjourney'])
    parser.add_argument('--attn-prob', type=float, default=0.2)
    parser.add_argument('--limit', type=int, default=300)
    args = parser.parse_args()

    measure = DinoV2DistanceMeasure()
    measure.load_state_dict(torch.load(args.model_path))
    data_rows = []
    args.output_path.mkdir(exist_ok=True, parents=True)
    num = 0

    for path in (args.input_folder / args.model).iterdir():
        if num >= args.limit:
            break

        if not path.is_dir():
            continue

        seeds = list(path.iterdir())
        id = path.name

        if len(seeds) < 2:
            continue

        image_a = seeds[0] / 'image.png'
        image_b = seeds[1] / 'image.png'
        image_a = PIL.Image.open(image_a)
        image_b = PIL.Image.open(image_b)
        fig, ax = plot_images(image_a, image_b)
        plt.savefig(args.output_path / f'{args.model}-{id}-true.jpg', bbox_inches='tight')
        plt.close(fig)

        distance = measure.measure('unused', image_a, image_b)
        data_rows.append(dict(distance=distance, image_url=f'{args.model}-{id}-true.jpg'))
        print(distance, path)

        if random.random() < args.attn_prob:
            fig, ax = plot_images(image_b, image_b)
            plt.savefig(args.output_path / f'{args.model}-{id}-fake.jpg', bbox_inches='tight')
            plt.close(fig)
            data_rows.append(dict(distance=0, image_url=f'{args.model}-{id}-fake.jpg'))

        num += 1
        image_a.close()
        image_b.close()

    df = pd.DataFrame(data_rows).sample(frac=1.0)
    df.to_csv(args.output_path / 'magnitude.csv', index=False)


def main():
    asyncio.run(amain())


if __name__ == '__main__':
    main()