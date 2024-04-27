import argparse
import asyncio
import itertools
import math

import numpy as np
import pandas as pd
import torch
import torch.utils.data as tud
from tqdm import tqdm, trange

from w1kp import GenerationExperiment, LPIPSDistanceMeasure, HitBatch, CLIPDistanceMeasure, ViTDistanceMeasure, \
    DinoV2DistanceMeasure, LPIPSCollator
from w1kp.model.distance import DISTSDistanceMeasure


async def amain():
    choices = ['lpips-alex', 'lpips-vgg', 'lpips-squeeze', 'clip', 'vit', 'oracle', 'dino-v2', 'dists',
               'stlpips-alex', 'stlpips-vgg']

    parser = argparse.ArgumentParser()
    parser.add_argument('--input-file', '-i', type=str, required=True)
    parser.add_argument('--input-image-folder', '-iif', type=str, required=True)
    parser.add_argument('--train-pct', '-l', type=int, default=90)
    parser.add_argument('--method', type=str, choices=choices, default='clip')
    parser.add_argument('--batch-size', '-bsz', type=int, default=16)
    parser.add_argument('--lr', '-lr', type=float, default=3e-5)
    parser.add_argument('--num-epochs', '-ne', type=int, default=5)
    parser.add_argument('--eval-only', '-eo', action='store_true')
    parser.add_argument('--weights-path', type=str)
    parser.add_argument('--loss-type', type=str, choices=['bce', 'bce-rank'], default='bce')
    args = parser.parse_args()

    batch = HitBatch.from_csv(args.input_file, approved_only=True, remove_attention_checks=True)

    match args.method:
        case 'lpips-alex':
            measure = LPIPSDistanceMeasure(network='alex')
        case 'lpips-vgg':
            measure = LPIPSDistanceMeasure(network='vgg')
        case 'lpips-squeeze':
            measure = LPIPSDistanceMeasure(network='squeeze')
        case 'stlpips-alex':
            measure = LPIPSDistanceMeasure(network='alex', shift_tolerant=True)
        case 'stlpips-vgg':
            measure = LPIPSDistanceMeasure(network='vgg', shift_tolerant=True)
        case 'clip':
            measure = CLIPDistanceMeasure()
        case 'vit':
            measure = ViTDistanceMeasure()
        case 'dino-v2':
            measure = DinoV2DistanceMeasure()
        case 'dists':
            measure = DISTSDistanceMeasure(args.weights_path)
        case _:
            measure = None

    measure.set_loss_type(args.loss_type)
    train_batch, test_batch = batch.split(args.train_pct)

    if not args.eval_only and not args.num_epochs == 0:
        train_ds = train_batch.to_lpips_dataset(args.input_image_folder)
        collator = LPIPSCollator(measure.processor)
        collator.train()

        if 'lpips' in args.method:
            measure.train()

        optimizer = torch.optim.Adam(measure.get_trainable_parameters(), lr=args.lr)
        train_dl = tud.DataLoader(train_ds, batch_size=args.batch_size, collate_fn=collator, num_workers=16, shuffle=True)

        measure.cuda()

        for _ in trange(args.num_epochs, position=0, desc='Epochs'):
            with tqdm(train_dl, position=1, desc='Batches', leave=False) as pbar:
                for batch in pbar:
                    optimizer.zero_grad(set_to_none=True)
                    loss = measure(
                        image1=batch['image1'].to(measure.device),
                        image2=batch['image2'].to(measure.device),
                        ref_image=batch['ref_image'].to(measure.device),
                        ground_truths=batch['judgement'].to(measure.device)
                    )

                    loss.backward()
                    optimizer.step()
                    tqdm.write(f'Loss: {loss.item():.4f}')
                    pbar.set_postfix(loss=f'{loss.item():.4f}')

    tot_acc = 0
    tot_score = 0
    tot_num = 0
    pbar = tqdm(test_batch.iter_group_by_seed())

    if measure is not None:
        measure.eval()

    for hits in pbar:
        hit = hits[0]
        exp = hit.load_comparison_experiment(args.input_image_folder)
        s1, s2, ref = exp.load_exp_images()

        choices = [hit.choice for hit in hits]
        a_pct = choices.count('a') / len(choices)

        if args.method == 'oracle':
            m1 = 1 - round(a_pct)
            m2 = 1 - m1
        else:
            m1 = measure.measure(exp.load_prompt(), s1, ref)
            m2 = measure.measure(exp.load_prompt(), s2, ref)

        tot_acc += 0 if m1 < m2 and a_pct < 0.5 or m1 > m2 and a_pct > 0.5 else 1
        tot_score += a_pct if m1 < m2 else 1 - a_pct
        tot_num += 1

        pbar.set_postfix(score=f'{tot_score / tot_num:.4f}', acc=f'{tot_acc / tot_num:.4f}')


def main():
    asyncio.run(amain())


if __name__ == '__main__':
    main()
