import argparse
import asyncio

import torch
import torch.utils.data as tud
from tqdm import tqdm, trange
from transformers import BatchEncoding

from w1kp import LPIPSDistanceMeasure, HitBatch, CLIPDistanceMeasure, ViTDistanceMeasure, \
    DinoV2DistanceMeasure, LPIPSCollator, StratifiedIDSampler
from w1kp.model.distance import DISTSDistanceMeasure, GroupViTDistanceMeasure
from w1kp.utils import apply_ema


"""
Some notes:

0.2 wdecay and 2 epochs works well for Dino-v2 small (71.3)
0.05 wdecay and 2 epochs works well for CLIP without text features
"""


async def amain():
    choices = ['lpips-alex', 'lpips-vgg', 'lpips-squeeze', 'clip', 'vit', 'oracle', 'dino-v2', 'dists',
               'stlpips-alex', 'stlpips-vgg', 'group-vit']

    parser = argparse.ArgumentParser()
    parser.add_argument('--input-files', '-i', type=str, nargs='+', required=True)
    parser.add_argument('--input-image-folder', '-iif', type=str, required=True)
    parser.add_argument('--train-pct', '-l', type=int, default=90)
    parser.add_argument('--method', type=str, choices=choices, default='clip')
    parser.add_argument('--batch-size', '-bsz', type=int, default=16)
    parser.add_argument('--lr', '-lr', type=float, default=3e-5)
    parser.add_argument('--lr-scheduler', type=str, default='cosine', choices=['linear', 'cosine', 'none'])
    parser.add_argument('--num-epochs', '-ne', type=int, default=1)
    parser.add_argument('--eval-only', '-eo', action='store_true')
    parser.add_argument('--weights-path', type=str)
    parser.add_argument('--loss-type', type=str, choices=['bce', 'bce-rank', 'bce-rank-coper'], default='bce')
    parser.add_argument('--negative-sampling-pct', type=int, default=0)
    parser.add_argument('--sampler', type=str, default='stratified', choices=['stratified', 'default'])
    parser.add_argument('--no-low-confidence', action='store_false', dest='low_confidence')
    parser.add_argument('--save-path', type=str, default='model.pt')
    parser.add_argument('--use-default-featurizer', action='store_true', dest='default_featurizer')
    parser.add_argument('--use-text', action='store_true')
    parser.add_argument('--weight-decay', '-wd', type=float)
    parser.add_argument('--distance-type', '-dt', type=str, default='cosine', choices=['cosine', 'l2', 'dot'])
    parser.add_argument('--macro-size', type=int, default=3)
    args = parser.parse_args()

    opt_kwargs = dict(weight_decay=args.weight_decay) if args.weight_decay is not None else dict()
    opt_kwargs['lr'] = args.lr
    dt = args.distance_type

    match args.method:
        case 'lpips-alex':
            measure = LPIPSDistanceMeasure(network='alex', distance_type=dt)
        case 'lpips-vgg':
            measure = LPIPSDistanceMeasure(network='vgg', distance_type=dt)
        case 'lpips-squeeze':
            measure = LPIPSDistanceMeasure(network='squeeze', distance_type=dt)
        case 'stlpips-alex':
            measure = LPIPSDistanceMeasure(network='alex', shift_tolerant=True, distance_type=dt)
        case 'stlpips-vgg':
            measure = LPIPSDistanceMeasure(network='vgg', shift_tolerant=True, distance_type=dt)
        case 'clip':
            measure = CLIPDistanceMeasure(default_featurizer=args.default_featurizer, use_text=args.use_text, distance_type=dt)
        case 'group-vit':
            measure = GroupViTDistanceMeasure(default_featurizer=args.default_featurizer, use_text=args.use_text, distance_type=dt)
        case 'vit':
            measure = ViTDistanceMeasure(distance_type=dt)
        case 'dino-v2':
            measure = DinoV2DistanceMeasure(distance_type=dt)
        case 'dists':
            measure = DISTSDistanceMeasure(args.weights_path, distance_type=dt)
        case _:
            measure = None

    if measure is not None:
        measure.set_loss_type(args.loss_type)

    batch = HitBatch.from_csv(*args.input_files, approved_only=True, remove_attention_checks=True)
    train_batch, test_batch = batch.split(args.train_pct)

    if not args.eval_only and not args.num_epochs == 0:
        train_ds = train_batch.to_lpips_dataset(
            args.input_image_folder,
            negative_sampling_pct=args.negative_sampling_pct,
            include_low_confidence=args.low_confidence
        )

        collator = LPIPSCollator(measure.processor, tokenizer=measure.tokenizer)
        collator.train()

        if 'lpips' in args.method:
            measure.train()

        optimizer = torch.optim.AdamW(measure.get_trainable_parameters(), **opt_kwargs)

        if args.sampler == 'default':
            train_dl = tud.DataLoader(train_ds, collate_fn=collator, num_workers=16, batch_size=args.batch_size, shuffle=True)
        else:
            train_dl = tud.DataLoader(train_ds, collate_fn=collator, num_workers=16, batch_sampler=StratifiedIDSampler(train_ds, macro_size=args.macro_size))

        match args.lr_scheduler:
            case 'linear':
                scheduler = torch.optim.lr_scheduler.LinearLR(
                    optimizer,
                    start_factor=1.0,
                    end_factor=0.01,
                    total_iters=args.num_epochs * len(train_dl)
                )
            case 'cosine':
                scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                    optimizer,
                    T_max=args.num_epochs * len(train_dl)
                )
            case _:
                scheduler = None

        measure.cuda()

        for _ in trange(args.num_epochs, position=0, desc='Epochs'):
            with tqdm(train_dl, position=1, desc='Batches', leave=False) as pbar:
                for batch in pbar:
                    optimizer.zero_grad(set_to_none=True)
                    prompt_kwargs = {}

                    if batch['prompt'] and isinstance(batch['prompt'], BatchEncoding):
                        prompt_kwargs = dict(prompt={k: v.to(measure.device) for k, v in batch['prompt'].items() if isinstance(v, torch.Tensor)})

                    loss = measure(
                        image1=batch['image1'].to(measure.device),
                        image2=batch['image2'].to(measure.device),
                        ref_image=batch['ref_image'].to(measure.device),
                        ground_truths=batch['judgement'].to(measure.device),
                        **prompt_kwargs
                    )

                    loss.backward()
                    optimizer.step()

                    if scheduler is not None:
                        scheduler.step()

                    tqdm.write(f'Loss: {loss.item():.4f}')
                    pbar.set_postfix(loss=f'{loss.item():.4f}')

    tot_acc = 0
    tot_score = 0
    tot_num = 0
    pbar = tqdm(test_batch.iter_group_by_seed())

    if measure is not None:
        torch.save(measure.state_dict(), args.save_path)
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
