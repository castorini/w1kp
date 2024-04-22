__all__ = ['PromptDataset', 'LPIPSDataset']

from pathlib import Path
from typing import List, Tuple, Dict

from datasets import load_dataset
import joblib
import numpy as np
from PIL import Image
import torch.utils.data as tud
from tqdm import tqdm


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


class LPIPSDataset(tud.Dataset):
    def __init__(
            self,
            images1: List[Image.Image],
            images2: List[Image.Image],
            ref_images: List[Image.Image],
            judgements: List[float],
            prompts: List[str] = None,
    ):
        self.images1 = images1
        self.images2 = images2
        self.ref_images = ref_images
        self.judgements = judgements
        self.prompts = prompts

    def __getitem__(self, item: int) -> Dict[str, Image.Image | float | str | None]:
        im1, im2, ref_im = self.images1[item], self.images2[item], self.ref_images[item]
        judgement = self.judgements[item]
        prompt = self.prompts[item] if self.prompts is not None else None

        return dict(image1=im1, image2=im2, ref_image=ref_im, judgement=judgement, prompt=prompt)

    def __len__(self) -> int:
        return len(self.images1)

    def __iadd__(self, other: 'LPIPSDataset') -> 'LPIPSDataset':
        self.images1.extend(other.images1)
        self.images2.extend(other.images2)
        self.ref_images.extend(other.ref_images)
        self.judgements.extend(other.judgements)

        if self.prompts is not None and other.prompts is not None:
            self.prompts.extend(other.prompts)

        return self

    def __add__(self, other: 'LPIPSDataset') -> 'LPIPSDataset':
        return LPIPSDataset(
            self.images1 + other.images1,
            self.images2 + other.images2,
            self.ref_images + other.ref_images,
            self.judgements + other.judgements,
            self.prompts + other.prompts if self.prompts is not None and other.prompts is not None else None
        )

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

        images1 = joblib.Parallel(n_jobs=joblib.cpu_count())(joblib.delayed(image_open)(p) for p in tqdm(list(sorted(path.glob('p0/*.png')))))
        images2 = joblib.Parallel(n_jobs=joblib.cpu_count())(joblib.delayed(image_open)(p) for p in tqdm(list(sorted(path.glob('p1/*.png')))))
        ref_images = joblib.Parallel(n_jobs=joblib.cpu_count())(joblib.delayed(image_open)(p) for p in tqdm(list(sorted(path.glob('ref/*.png')))))
        judgements = [float(np.load(p)[0]) for p in sorted(path.glob('judge/*.npy'))]

        prompts = None

        if (path / 'prompts.txt').exists():
            prompts = (path / 'prompts.txt').read_text().splitlines()

        return cls(images1, images2, ref_images, judgements, prompts)
