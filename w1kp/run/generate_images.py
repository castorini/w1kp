import argparse
import asyncio
import math

from tqdm import tqdm
from tqdm.asyncio import tqdm_asyncio

from w1kp import PromptDataset, AzureOpenAIImageGenerator, GenerationExperiment, StableDiffusionXLImageGenerator, \
    StableDiffusion2ImageGenerator, ImagineApiMidjourneyGenerator


async def amain():
    models = ['sdxl', 'sd2', 'dalle3', 'imagen', 'midjourney']

    parser = argparse.ArgumentParser()
    parser.add_argument('--output-folder', '-o', type=str, default='output')
    parser.add_argument('--azure-keys-config', '-ac', type=str, default='azure-keys.json')
    parser.add_argument('--num-images-per-seed', '-nps', type=int, default=10)
    parser.add_argument('--num-prompts', '-np', type=int, default=100000)
    parser.add_argument('--model', type=str, default='dalle3', choices=models)
    parser.add_argument('--midjourney-api-key', type=str)
    parser.add_argument('--type', '-t', type=str, default='diffusiondb', choices=['diffusiondb', 'stdin', 'cli'])
    parser.add_argument('--filter-guidance', '-fg', type=float, default=7.0)
    parser.add_argument('--prompts', '-p', type=str, nargs='+')
    parser.add_argument('--regenerate', '-r', action='store_true')
    parser.add_argument('--skip-num', '-sn', type=int, default=0, help='The number of prompts to skip from the beginning')
    args = parser.parse_args()

    match args.type:
        case 'diffusiondb':
            prompt_dataset = PromptDataset.from_diffusiondb(filter_guidance=args.filter_guidance)
        case 'stdin':
            prompt_dataset = PromptDataset.from_stdin()
        case 'cli':
            prompt_dataset = PromptDataset(args.prompts)

    match args.model:
        case 'dalle3':
            image_gens = AzureOpenAIImageGenerator.parse_from_path(args.azure_keys_config)
        case 'sdxl':
            image_gens = [StableDiffusionXLImageGenerator()]
        case 'sd2':
            image_gens = [StableDiffusion2ImageGenerator()]
        case 'midjourney':
            if args.midjourney_api_key is None:
                raise ValueError('MidJourney API key required (--midjourney-api-key) for model midjourney')

            image_gens = [ImagineApiMidjourneyGenerator(api_key=args.midjourney_api_key) for _ in range(3)]
        case _:
            raise ValueError('Model not implemented')

    num_images_per_seed = math.ceil(args.num_images_per_seed / image_gens[0].num_multiple)

    for ds_idx, (prompt, image) in enumerate(tqdm(prompt_dataset, position=1, desc='Generating images')):
        if ds_idx < args.skip_num:
            continue

        if args.skip_num + ds_idx >= args.num_prompts:
            break

        skip = False

        for seed in range(num_images_per_seed):
            seed = str(seed)
            exp = GenerationExperiment(
                prompt,
                model_name=args.model,
                id=str(ds_idx),
                seed=seed,
                root_folder=args.output_folder
            )

            if exp.get_path('image.png').exists() and not args.regenerate:
                print(f'Skipping {ds_idx}')
                skip = True
                break

        if skip:
            continue

        print(f'Generating prompt {ds_idx}: {prompt}')
        coroutines = []

        for seed in range(num_images_per_seed):
            coroutines.append(image_gens[seed % len(image_gens)].generate_image(prompt, seed=seed))

        outputs = await tqdm_asyncio.gather(*coroutines, desc='Generating images', position=2)

        for seed, ret in zip(range(num_images_per_seed), outputs):
            if ret is None:
                continue

            if isinstance(ret, dict):
                ret = [ret]

            for idx, r in enumerate(ret):
                sub_seed_idx = seed * image_gens[0].num_multiple + idx
                gen_prompt = r['revised_prompt']
                gen_image = r['image']

                exp = GenerationExperiment(
                    gen_prompt,
                    seed=str(sub_seed_idx),
                    model_name=args.model,
                    image=gen_image,
                    id=str(ds_idx),
                    root_folder=args.output_folder
                )

                exp.save(overwrite=True)


def main():
    asyncio.run(amain())


if __name__ == '__main__':
    main()
