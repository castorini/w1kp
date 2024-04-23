__all__ = ['GenerationExperiment', 'Comparison2AFCExperiment', 'synthesize_2afc_experiments']

from dataclasses import dataclass
import itertools
from pathlib import Path
import random
from typing import List
import uuid

from PIL.Image import Image
import PIL.Image
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

    def __post_init__(self):
        if self.seed is None:
            self.seed = str(uuid.uuid4())

    def get_path(self, filename: str = '.') -> Path:
        return Path(self.root_folder) / self.model_name / self.id / self.seed / filename

    def load_image(self) -> PIL.Image.Image:
        # Lazy loading
        if self.image is None:
            self.image = PIL.Image.open(str(self.get_path('image.png')))

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

        seed = seed or folder.name

        return cls(prompt, id=folder.parent.name, root_folder=folder.parent.parent.parent, model_name=model_name, seed=seed)

    @classmethod
    def load_all_seeds(cls, folder: Path | str, id: str, model_name: str = 'UNSPECIFIED') -> List['GenerationExperiment']:
        experiments = []
        folder = Path(folder) / model_name / id

        for f_seed in folder.iterdir():
            if not f_seed.is_dir():
                continue

            seed = f_seed.name

            if (f_seed / 'image.png').exists():
                experiments.append(cls.from_folder(f_seed, model_name=model_name, seed=seed))

        return experiments

    @classmethod
    def load_all_ids(cls, folder: Path | str, model_name: str = 'UNSPECIFIED') -> List['GenerationExperiment']:
        experiments = []
        folder = Path(folder) / model_name

        for f_id in folder.iterdir():
            if not f_id.is_dir():
                continue

            experiments.extend(cls.load_all_seeds(folder, f_id.name, model_name=model_name))

        return experiments

    @classmethod
    def iter_by_id(cls, folder: Path | str, model_name: str = 'UNSPECIFIED') -> List['GenerationExperiment']:
        folder = Path(folder) / model_name

        for f_id in folder.iterdir():
            if not f_id.is_dir():
                continue

            yield cls.load_all_seeds(folder.parent, f_id.name, model_name=model_name)

    @classmethod
    def load_all(cls, folder: Path | str) -> List['GenerationExperiment']:
        experiments = []
        folder = Path(folder)

        for f_model in folder.iterdir():
            if not f_model.is_dir():
                continue

            model_name = f_model.name
            experiments.extend(cls.load_all_ids(folder, model_name=model_name))

        return experiments

    def save(self, root_folder: Path | str = None, overwrite: bool = False):
        if not self.id:
            self.id = uuid.uuid4().hex

        self.root_folder = root_folder or self.root_folder
        self.root_folder = Path(self.root_folder)

        root_folder = self.get_path()
        root_folder.mkdir(exist_ok=overwrite, parents=True)
        (root_folder / 'prompt.txt').write_text(self.prompt)

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
        id1, id2 = exp1.id, exp2.id
        attn_gt = exp1.id
    else:
        id1, id2 = exp2.id, exp1.id
        attn_gt = exp2.id

    if attention_check:
        ref_id = id1
        kw = dict(ground_truth=attn_gt)
    else:
        ref_id = ref_exp.id
        kw = {}

    return Comparison2AFCExperiment(
        ref_id=ref_id, id1=id1, id2=id2, seed=exp1.seed, root_folder=root_folder, model_name=exp1.model_name, **kw
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

                if (exp.id1, exp.id2, exp.ref_id) not in chosen_pairs:
                    is_duplicate = False
                    chosen_pairs.add((exp.id1, exp.id2, exp.ref_id))

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
    ref_id: str
    id1: str
    id2: str
    seed: str
    root_folder: Path
    model_name: str = 'UNSPECIFIED'
    id: str = None
    ground_truth: str = None  # only specified if the ground truth is known

    def model_post_init(self):
        self.root_folder = Path(self.root_folder)

        if self.id is None:
            self.id = str(uuid.uuid4())

    def get_gen_path(self, id: str, filename: str = '.') -> Path:
        return Path(self.root_folder) / self.model_name / id / self.seed / filename

    def load_images(self):
        # Lazy loading
        self.ref_image = PIL.Image.open(str(self.get_gen_path(self.ref_id, 'image.png')))
        self.image1 = PIL.Image.open(str(self.get_gen_path(self.id1, 'image.png')))
        self.image2 = PIL.Image.open(str(self.get_gen_path(self.id2, 'image.png')))

    @classmethod
    def from_file(cls, path: Path | str) -> 'List[Comparison2AFCExperiment]':
        lines = path.read_text().splitlines()
        experiments = []

        for line in lines:
            exp = Comparison2AFCExperiment.parse_raw(line)
            exp.root_folder = path.parent
            experiments.append(exp)

        return experiments

    @staticmethod
    def save_all(self, experiments: 'List[Comparison2AFCExperiment]', path: Path | str):
        if not self.id:
            self.id = uuid.uuid4().hex

        path = Path(path)

        with path.open('w') as f:
            for exp in experiments:
                print(exp.json(), file=f)
