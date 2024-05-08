__all__ = ['PromptDataset', 'LPIPSDataset', 'LPIPSCollator', 'StratifiedIDSampler']

import random
from collections import defaultdict
from pathlib import Path
from typing import List, Tuple, Dict

import PIL.Image
import nltk
import pandas as pd
import torch
from datasets import load_dataset
from nltk.corpus import wordnet as wn
import numpy as np
from PIL import Image
import torch.utils.data as tud
from transformers import AutoTokenizer, CLIPTokenizer

from .experiment import Comparison2AFCExperiment
from ..utils import load_image


class PromptDataset(tud.Dataset):
    def __init__(self, prompts: List[str], images: List[Image.Image] = None):
        self.prompts = prompts
        self.images = images

    def __getitem__(self, item: int) -> Tuple[str, Image.Image | None]:
        return self.prompts[item], self.images[item] if self.images is not None else None

    def __len__(self) -> int:
        return len(self.prompts)

    @classmethod
    def from_dataset(
            cls,
            dataset_name: str,
            split: str,
            prompts_column: str = 'prompt',
            images_column: str = None,
            filter_guidance: float = None,
    ) -> 'PromptDataset':
        dataset = load_dataset(dataset_name, split, version='0.9.1')['train']
        prompts = dataset[prompts_column]
        images = dataset[images_column] if images_column is not None else None

        if filter_guidance is not None:
            ds_idxs = np.arange(len(dataset))[np.array(dataset['cfg']) == filter_guidance]
            prompts = [prompts[idx] for idx in ds_idxs]
            images = [images[idx] for idx in ds_idxs] if images is not None else None

        return cls(prompts, images)

    @classmethod
    def from_diffusiondb(cls, split: str = '2m_random_100k', **kwargs) -> 'PromptDataset':
        return cls.from_dataset('poloclub/diffusiondb', split, 'prompt', **kwargs)

    def filter_wordnet(self) -> 'PromptDataset':
        nltk.download('wordnet')
        prompts = []

        for candidate in self.prompts:
            candidate = candidate.lower().strip()
            synsets = wn.synsets(candidate)

            if not synsets:
                continue

            if all(x.pos() != 'n' for x in synsets):
                continue

            prompts.append(candidate)

        return PromptDataset(prompts)

    @classmethod
    def from_stdin(cls) -> 'PromptDataset':
        prompts = []

        with open(0) as f:
            for line in f:
                prompts.append(line.strip())

        return cls(prompts)


class LPIPSCollator:
    def __init__(self, processor, tokenizer: CLIPTokenizer = None):
        self.processor = processor
        self.training = False
        self.tokenizer = tokenizer

    def train(self):
        self.training = True

    def eval(self):
        self.training = False

    def __call__(self, batch: List[Dict[str, Image.Image | float | str | None]]) -> Dict[str, List[Image.Image | float | str | None]]:
        images1 = [item['image1'] for item in batch]
        images2 = [item['image2'] for item in batch]
        ref_images = [item['ref_image'] for item in batch]
        judgement = [item['judgement'] for item in batch]
        prompts = [item['prompt'] for item in batch]

        inputs1 = self.processor(images=images1, return_tensors='pt')
        inputs2 = self.processor(images=images2, return_tensors='pt')
        ref_inputs = self.processor(images=ref_images, return_tensors='pt')
        judgements = torch.Tensor(judgement)

        if self.tokenizer:
            prompts = self.tokenizer(prompts, return_tensors='pt', padding=True, truncation=True)

        if self.training:
            # Randomly add equal amounts of noise to the images
            if random.random() < -0.5:  # disabled
                noise = torch.randn_like(inputs1['pixel_values']) * 0.01
                inputs1['pixel_values'] += noise
                inputs2['pixel_values'] += noise
                ref_inputs['pixel_values'] += noise

            # Randomly flip all the images horizontally
            if random.random() < -0.5:  # disabled
                inputs1['pixel_values'] = torch.flip(inputs1['pixel_values'], dims=[3])
                inputs2['pixel_values'] = torch.flip(inputs2['pixel_values'], dims=[3])
                ref_inputs['pixel_values'] = torch.flip(ref_inputs['pixel_values'], dims=[3])

        return dict(
            image1=inputs1['pixel_values'],
            image2=inputs2['pixel_values'],
            ref_image=ref_inputs['pixel_values'],
            judgement=judgements,
            prompt=prompts
        )


class StratifiedIDSampler(tud.BatchSampler):
    def __init__(self, dataset: 'LPIPSDataset', macro_size: int = 3):
        self.dataset = dataset
        self.id_df = pd.DataFrame({'id': dataset.ids, 'model': dataset.models})
        self.id_groups = [group.reset_index()['index'].tolist() for _, group in self.id_df.groupby(['model', 'id'])]
        self.macro_size = macro_size

    def __iter__(self):
        available_ids = set(range(len(self.id_groups)))

        while available_ids:
            id_sample = random.sample(available_ids, min(self.macro_size, len(available_ids)))
            available_ids -= set(id_sample)

            yield [x for i in id_sample for x in self.id_groups[i]]

    def __len__(self):
        return len(self.id_groups) // self.macro_size


class LPIPSDataset(tud.Dataset):
    def __init__(
            self,
            images1: List[Path],
            images2: List[Path],
            ref_images: List[Path],
            judgements: List[float],
            prompts: List[str] = None,
            ids: List[str] = None,
            models: List[str] = None,
    ):
        self.images1 = images1
        self.images2 = images2
        self.ref_images = ref_images
        self.judgements = judgements
        self.prompts = prompts
        self.ids = ids
        self.models = models

    def split(self, train_pct: int) -> Tuple['LPIPSDataset', 'LPIPSDataset']:
        old_state = random.getstate()
        random.seed(0)

        data1 = defaultdict(list)
        data2 = defaultdict(list)

        for i in range(len(self)):
            data = data1 if 100 * random.random() < train_pct else data2
            data['images1'].append(self.images1[i])
            data['images2'].append(self.images2[i])
            data['ref_images'].append(self.ref_images[i])
            data['judgements'].append(self.judgements[i])

            if self.prompts:
                data['prompts'].append(self.prompts[i])

            if self.ids:
                data['ids'].append(self.ids[i])

            if self.models:
                data['models'].append(self.models[i])

        random.setstate(old_state)

        return LPIPSDataset(**data1), LPIPSDataset(**data2)

    def __getitem__(self, item: int) -> Dict[str, Image.Image | float | str | None]:
        im1, im2, ref_im = self.images1[item], self.images2[item], self.ref_images[item]
        judgement = self.judgements[item]
        prompt = self.prompts[item] if self.prompts is not None else None

        ims = []

        for im in (im1, im2, ref_im):
            if im.with_suffix('.npy').exists():
                im_ = np.load(im.with_suffix('.npy'))
                im_ = PIL.Image.fromarray(im_)
            else:
                im_ = load_image(str(im))

            ims.append(im_)

        return dict(image1=ims[0], image2=ims[1], ref_image=ims[2], judgement=judgement, prompt=prompt)

    def __len__(self) -> int:
        return len(self.images1)

    def __iadd__(self, other: 'LPIPSDataset') -> 'LPIPSDataset':
        self.images1.extend(other.images1)
        self.images2.extend(other.images2)
        self.ref_images.extend(other.ref_images)
        self.judgements.extend(other.judgements)

        if self.prompts is not None and other.prompts is not None:
            self.prompts.extend(other.prompts)

        if self.ids is not None and other.ids is not None:
            self.ids.extend(other.ids)

        if self.models is not None and other.models is not None:
            self.models.extend(other.models)

        return self

    def __add__(self, other: 'LPIPSDataset') -> 'LPIPSDataset':
        return LPIPSDataset(
            self.images1 + other.images1,
            self.images2 + other.images2,
            self.ref_images + other.ref_images,
            self.judgements + other.judgements,
            self.prompts + other.prompts if self.prompts is not None and other.prompts is not None else None,
            self.ids + other.ids if self.ids is not None and other.ids is not None else None,
            self.models + other.models if self.models is not None and other.models is not None else None
        )

    @classmethod
    def from_experiments(cls, experiments: List[Comparison2AFCExperiment]):
        images1 = [exp.get_gen_path(exp.seed1, 'image.png') for exp in experiments]
        images2 = [exp.get_gen_path(exp.seed2, 'image.png', is_id2=True) for exp in experiments]
        ref_images = [exp.get_gen_path(exp.ref_seed, 'image.png') for exp in experiments]
        judgements = [exp.judgement for exp in experiments]
        prompts = [exp.load_prompt() for exp in experiments]
        ids = [exp.id for exp in experiments]
        models = [exp.model_name for exp in experiments]

        return cls(images1, images2, ref_images, judgements, prompts, ids, models)

    @classmethod
    def from_folder(cls, path: str, resize: int = 64) -> 'LPIPSDataset':
        """
        Load a dataset from an LPIPS-standardized folder structure in the following format:

        .. code-block::bash

            2afc/train/cnn <- this should be the path argument
            ├── judge/
            │   ├── 000000.npy
            │   ├── 000001.npy
            │   └── ...
            ├── p0/
            │   ├── 000000.png
            │   ├── 000001.png
            │   └── ...
            ├── p1/
            │   ├── 000000.png
            │   ├── 000001.png
            │   └── ...
            └── ref/
            │   ├── 000000.png
            │   ├── 000001.png
            │   └── ...
            └── prompts.txt  # optional
        """
        def image_open(p: Path) -> Image.Image:
            with Image.open(p) as im:
                return im.resize((resize, resize)).copy()

        path = Path(path)

        images1 = list(sorted(path.glob('p0/*.png')))
        images2 = list(sorted(path.glob('p1/*.png')))
        ref_images = list(sorted(path.glob('ref/*.png')))
        judgements = [float(np.load(p)[0]) for p in sorted(path.glob('judge/*.npy'))]

        prompts = None

        if (path / 'prompts.txt').exists():
            prompts = (path / 'prompts.txt').read_text().splitlines()

        return cls(images1, images2, ref_images, judgements, prompts)
