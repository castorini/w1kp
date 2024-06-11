__all__ = ['GenerationExperiment', 'Comparison2AFCExperiment', 'synthesize_2afc_experiments']

import json
from contextlib import closing
from dataclasses import dataclass
import itertools
from pathlib import Path
import random
from typing import List, Tuple
import uuid

import numpy as np
from PIL.Image import Image
import PIL.Image
from matplotlib import pyplot as plt
from pydantic import BaseModel


@dataclass
class GenerationExperiment:
    """Serializable representation of a single image generation experiment."""
    prompt: str
    model_name: str = 'UNSPECIFIED'
    id: str = ''
    seed: str = None
    root_folder: Path = None
    image: None | Image = None
    metadata: dict = None

    def __post_init__(self):
        if self.seed is None:
            self.seed = str(uuid.uuid4())

    def get_path(self, filename: str = '.') -> Path:
        return Path(self.root_folder) / self.model_name / self.id / self.seed / filename

    def load_image(self) -> PIL.Image.Image:
        # Lazy loading
        if self.image is None:
            with PIL.Image.open(str(self.get_path('image.png'))) as image:
                self.image = image.copy()

        return self.image

    @classmethod
    def from_folder(cls, folder: Path | str, model_name: str = 'UNSPECIFIED', seed: str = None) -> 'GenerationExperiment':
        try:
            folder = Path(folder)
            prompt = (folder / 'prompt.txt').read_text().strip()
        except:
            try:
                folder = Path(folder) / model_name / seed
                prompt = (folder / 'prompt.txt').read_text().strip()
            except:
                raise

        try:
            metadata = json.loads((folder / 'metadata.json').read_text())
        except FileNotFoundError:
            metadata = None

        seed = seed or folder.name

        return cls(prompt, id=folder.parent.name, root_folder=folder.parent.parent.parent, model_name=model_name, seed=seed, metadata=metadata)

    @classmethod
    def get_ids(cls, folder: Path | str, model_name: str = 'UNSPECIFIED') -> List[str]:
        folder = Path(folder) / model_name
        return [f.name for f in folder.iterdir() if f.is_dir()]

    @classmethod
    def iter_by_seed(cls, folder: Path | str, id: str, model_name: str = 'UNSPECIFIED') -> List['GenerationExperiment']:
        experiments = []
        folder = Path(folder) / model_name / id

        for f_seed in folder.iterdir():
            if not f_seed.is_dir():
                continue

            seed = f_seed.name

            if (f_seed / 'image.png').exists():
                yield cls.from_folder(f_seed, model_name=model_name, seed=seed)

        return experiments

    @classmethod
    def iter_by_id(cls, folder: Path | str, model_name: str = 'UNSPECIFIED') -> List[List['GenerationExperiment']]:
        folder = Path(folder) / model_name

        for f_id in folder.iterdir():
            if not f_id.is_dir():
                continue

            ret = yield list(cls.iter_by_seed(folder.parent, f_id.name, model_name=model_name))

            if ret:
                yield  # close the generator
                return

    @classmethod
    def iter_all(cls, folder: Path | str) -> List['GenerationExperiment']:
        folder = Path(folder)

        for f_model in folder.iterdir():
            if not f_model.is_dir():
                continue

            model_name = f_model.name

            for exp in cls.iter_by_id(folder, model_name=model_name):
                yield exp

    def save(self, root_folder: Path | str = None, overwrite: bool = False):
        if not self.id:
            self.id = uuid.uuid4().hex

        self.root_folder = root_folder or self.root_folder
        self.root_folder = Path(self.root_folder)

        root_folder = self.get_path()
        root_folder.mkdir(exist_ok=overwrite, parents=True)
        (root_folder / 'prompt.txt').write_text(self.prompt)

        if self.metadata is not None:
            (root_folder / 'metadata.json').write_text(json.dumps(self.metadata))

        if self.image is not None:
            self.image.save(root_folder / 'image.png')


def _to_2afc_exp(
        exp1: GenerationExperiment,
        exp2: GenerationExperiment,
        ref_exp: GenerationExperiment,
        root_folder: Path | str,
        attention_check: bool = False,
) -> 'Comparison2AFCExperiment':
    if random.random() < 0.5:
        seed1, seed2 = exp1.seed, exp2.seed
        attn_gt = exp1.seed
    else:
        seed1, seed2 = exp2.seed, exp1.seed
        attn_gt = exp2.seed

    if attention_check:
        ref_seed = seed1
        kw = dict(ground_truth=attn_gt)
    else:
        ref_seed = ref_exp.seed
        kw = {}

    return Comparison2AFCExperiment(
        ref_seed=ref_seed, seed1=seed1, seed2=seed2, id=exp1.id, root_folder=root_folder, model_name=exp1.model_name, **kw
    )


def synthesize_2afc_experiments(
        experiments: List[GenerationExperiment],
        num_samples: int = 20,
        strategy: str = 'random',
        attention_check: bool = False
) -> List['Comparison2AFCExperiment']:
    """
    Synthesize 2AFC experiments from a list of generation experiments.

    Args:
        experiments: List of generation experiments.
        num_samples: Number of 2AFC experiments to synthesize.
        strategy: Strategy to use for synthesizing 2AFC experiments. Options are 'random' and 'pairwise'.
        attention_check: Whether to sample attention check experiments, e.g., ref_id is the same as one of id1 or id2.

    Returns:
        List of synthesized 2AFC experiments.
    """
    assert all(exp.root_folder == experiments[0].root_folder for exp in experiments), 'All experiments must have the same root folder'
    assert all(exp.model_name == experiments[0].model_name for exp in experiments), 'All experiments must have the same model name'

    root_folder = experiments[0].root_folder
    chosen_pairs = set()

    if strategy == 'random':
        for _ in range(num_samples):
            is_duplicate = True

            while is_duplicate:
                exp1, exp2, ref_exp = random.sample(experiments, 3)
                exp = _to_2afc_exp(exp1, exp2, ref_exp, root_folder, attention_check)

                if (exp.seed1, exp.seed2, exp.ref_seed) not in chosen_pairs:
                    is_duplicate = False
                    chosen_pairs.add((exp.seed1, exp.seed2, exp.ref_seed))

                    yield exp
    elif strategy == 'pairwise':
        for exp1, exp2, ref_exp in itertools.combinations(experiments, 3):
            exp = _to_2afc_exp(exp1, exp2, ref_exp, root_folder, attention_check)
            chosen_pairs.add((exp.id1, exp.id2, exp.ref_id))

            yield exp
    else:
        raise ValueError(f'Invalid strategy: {strategy}')


class Comparison2AFCExperiment(BaseModel):
    """Serializable representation of a single 2AFC experiment."""
    ref_seed: str
    seed1: str
    seed2: str
    id: str
    id2: str | None = None
    root_folder: Path | None = None
    model_name: str = 'UNSPECIFIED'
    ground_truth: str = None  # only specified if the ground truth is known
    judgement: float | None = None

    def __repr__(self):
        return f'2AFC<<{self.id1} vs. {self.id2} (ref: {self.ref_id}; gt: {self.ground_truth})>>'

    def load_prompt(self) -> str:
        try:
            return (self.root_folder / self.model_name / self.id / self.seed1 / 'prompt.txt').read_text()
        except UnicodeDecodeError:
            return ''

    def model_post_init(self, *args, **kwargs):
        if self.root_folder is not None:
            self.root_folder = Path(self.root_folder)

    def get_gen_path(self, seed: str, filename: str = '.', is_id2: bool = False) -> Path:
        id = self.id2 if (is_id2 and self.id2 is not None) else self.id
        return Path(self.root_folder) / self.model_name / id / seed / filename

    def get_comparison_path(self) -> Path:
        seed1, seed2, ref_seed = self.seed1, self.seed2, self.ref_seed

        return Path(self.root_folder) / self.model_name / self.id / f'comparison-{seed1}-{seed2}-{ref_seed}.jpg'

    def load_comparison_image(self) -> PIL.Image.Image:
        return PIL.Image.open(str(self.get_comparison_path()))

    def create_comparison_image(self):
        plt.clf()
        fig, ax = plt.subplots(1, 3, figsize=(15, 5))

        # Set titles
        ax[0].set_title('Image A')
        ax[1].set_title('REFERENCE')
        ax[1].title.set_color('red')
        ax[2].set_title('Image B')

        # Set bold
        for axi in ax:
            axi.set_title(axi.get_title(), fontweight='bold')

        # Make font larger
        for axi in ax:
            axi.title.set_fontsize(24)

        ax[0].set_axis_off()
        ax[1].set_axis_off()
        ax[2].set_axis_off()

        im1, im2, ref_im = self.load_exp_images()

        try:
            ax[0].imshow(im1)
            ax[1].imshow(ref_im)
            ax[2].imshow(im2)
        finally:
            im1.close()
            im2.close()
            ref_im.close()

        plt.savefig(self.get_comparison_path(), bbox_inches='tight')
        plt.close(fig)

    def load_exp_images(self) -> Tuple[PIL.Image.Image, PIL.Image.Image, PIL.Image.Image]:
        ref_image = PIL.Image.open(str(self.get_gen_path(self.ref_seed, 'image.png')))
        image1 = PIL.Image.open(str(self.get_gen_path(self.seed1, 'image.png')))
        image2 = PIL.Image.open(str(self.get_gen_path(self.seed2, 'image.png', is_id2=True)))

        return image1, image2, ref_image

    @classmethod
    def from_folder(cls, path: Path | str) -> 'List[Comparison2AFCExperiment]':
        path = Path(path) / '2afc.jsonl'
        lines = path.read_text().splitlines()
        experiments = []

        for line in lines:
            exp = Comparison2AFCExperiment.parse_raw(line)
            exp.root_folder = path.parent
            experiments.append(exp)

        return experiments

    def save(self):
        old_root = self.root_folder

        with (self.root_folder / '2afc.jsonl').open('a') as f:
            try:
                self.root_folder = None
                print(self.model_dump_json(exclude_none=True), file=f)
            finally:
                self.root_folder = old_root
