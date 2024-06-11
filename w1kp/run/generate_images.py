import argparse
import asyncio
import math
from pathlib import Path

from tqdm import tqdm
from tqdm.asyncio import tqdm_asyncio

from w1kp import PromptDataset, AzureOpenAIImageGenerator, GenerationExperiment, StableDiffusionXLImageGenerator, \
    StableDiffusion2ImageGenerator, ImagineApiMidjourneyGenerator, GoogleImagenImageGenerator, AsyncSDXLClient


async def amain():
    models = ['sdxl', 'sd2', 'dalle3', 'imagen', 'midjourney', 'async-sdxl']

    parser = argparse.ArgumentParser()
    parser.add_argument('--output-folder', '-o', type=str, default='output')
    parser.add_argument('--azure-keys-config', '-ac', type=str, default='azure-keys.json')
    parser.add_argument('--num-images-per-seed', '-nps', type=int, default=10)
    parser.add_argument('--num-prompts', '-np', type=int, default=100000)
    parser.add_argument('--model', type=str, default='dalle3', choices=models)
    parser.add_argument('--midjourney-api-key', type=str)
    parser.add_argument('--imagen-project-id', type=str)
    parser.add_argument('--type', '-t', type=str, default='diffusiondb', choices=['diffusiondb', 'stdin', 'cli', 'folder', 'wordnet'])
    parser.add_argument('--load-folder', '-lf', type=str)
    parser.add_argument('--filter-guidance', '-fg', type=float, default=7.0)
    parser.add_argument('--prompts', '-p', type=str, nargs='+')
    parser.add_argument('--regenerate', '-r', action='store_true')
    parser.add_argument('--regenerate-only-if-present', '-ro', action='store_true')
    parser.add_argument('--skip-num', '-sn', type=int, default=0, help='The number of prompts to skip from the beginning')
    parser.add_argument('--start-id', '-sid', type=int, default=0)
    parser.add_argument('--async-sdxl-urls', '-asu', type=str, nargs='+')
    parser.add_argument('--start-seed', '-ss', type=int, default=0)
    parser.add_argument('--guidance-scale', '-gs', type=float, default=None)
    args = parser.parse_args()

    match args.type:
        case 'diffusiondb':
            prompt_dataset = PromptDataset.from_diffusiondb(filter_guidance=args.filter_guidance)
        case 'stdin':
            prompt_dataset = PromptDataset.from_stdin()
        case 'cli':
            prompt_dataset = PromptDataset(args.prompts)
        case 'wordnet':
            prompt_dataset = PromptDataset.from_stdin()
            prompt_dataset = prompt_dataset.filter_wordnet()
        case 'folder':
            folder = args.load_folder if args.load_folder else args.output_folder
            folder = Path(folder)
            prompts = []

            for filepath in folder.glob('**/prompt.txt'):
                prompt = filepath.read_text().strip()
                prompts.append((int(filepath.parent.parent.name), prompt))

            prompts = [x[1] for x in sorted(prompts, key=lambda x: x[0])]
            prompts = list(dict.fromkeys(prompts))
            prompt_dataset = PromptDataset(prompts)

    match args.model:
        case 'dalle3':
            image_gens = AzureOpenAIImageGenerator.parse_from_path(args.azure_keys_config)
        case 'sdxl':
            image_gens = [StableDiffusionXLImageGenerator()]
        case 'async-sdxl':
            if args.async_sdxl_urls is None:
                raise ValueError('Async SDXL URLs required (--async-sdxl-urls) for model async-sdxl')

            image_gens = [AsyncSDXLClient(url) for url in args.async_sdxl_urls]
        case 'sd2':
            image_gens = [StableDiffusion2ImageGenerator()]
        case 'midjourney':
            if args.midjourney_api_key is None:
                raise ValueError('MidJourney API key required (--midjourney-api-key) for model midjourney')

            image_gens = [ImagineApiMidjourneyGenerator(api_key=args.midjourney_api_key) for _ in range(3)]
        case 'imagen':
            if args.imagen_project_id is None:
                raise ValueError('Imagen project ID required (--imagen-project-id) for model imagen')

            image_gens = [GoogleImagenImageGenerator(project_id=args.imagen_project_id)]
        case _:
            raise ValueError('Model not implemented')

    num_images_per_seed = math.ceil(args.num_images_per_seed / image_gens[0].num_multiple)

    for ds_idx, (prompt, image) in enumerate(tqdm(prompt_dataset, position=1, desc='Generating images')):
        if ds_idx < args.skip_num:
            continue

        if args.skip_num + ds_idx >= args.num_prompts:
            break

        skip = False
        exists = False

        for seed in range(args.start_seed, args.start_seed + num_images_per_seed):
            seed = str(seed)
            exp = GenerationExperiment(
                prompt,
                model_name=args.model,
                id=str(ds_idx + args.start_id),
                seed=seed,
                root_folder=args.output_folder
            )

            if exp.get_path('image.png').parent.parent.exists():
                exists = True

            if exp.get_path('image.png').exists() and (not args.regenerate and not args.regenerate_only_if_present):
                print(f'Skipping {ds_idx}')
                skip = True
                break

        if skip or (not exists and args.regenerate_only_if_present):
            continue

        print(f'Generating prompt {ds_idx}: {prompt}')
        coroutines = []
        gen_kwargs = dict(guidance_scale=args.guidance_scale) if args.guidance_scale is not None else {}

        for seed in range(args.start_seed, args.start_seed + num_images_per_seed):
            coroutines.append(image_gens[seed % len(image_gens)].generate_image(prompt, seed=seed, **gen_kwargs))

        outputs = await tqdm_asyncio.gather(*coroutines, desc='Generating images', position=2)

        for seed, ret in zip(range(args.start_seed, args.start_seed + num_images_per_seed), outputs):
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
                    id=str(ds_idx + args.start_id),
                    root_folder=args.output_folder
                )

                if args.guidance_scale is not None:
                    exp.metadata = dict(guidance_scale=args.guidance_scale)

                exp.save(overwrite=True)
                gen_image.close()


def main():
    asyncio.run(amain())


if __name__ == '__main__':
    main()
