from dataclasses import dataclass
from pathlib import Path
from typing import List
import uuid

from PIL.Image import Image
import PIL.Image

__all__ = ['GenerationExperiment']


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


@dataclass
class Comparison2AFCExperiment:
    """Serializable representation of a single 2AFC experiment."""
    prompt: str
    ref_id: str
    id1: str
    id2: str
    seed: str
    root_folder: Path
    model_name: str = 'UNSPECIFIED'
    ref_image: None | Image = None
    image1: None | Image = None
    image2: None | Image = None

    def get_path(self, id: str, filename: str = '.') -> Path:
        return Path(self.root_folder) / self.model_name / id / self.seed / filename

    def load_images(self):
        # Lazy loading
        self.ref_image = PIL.Image.open(str(self.get_path(self.ref_id, 'image.png')))
        self.image1 = PIL.Image.open(str(self.get_path(self.id1, 'image.png')))
        self.image2 = PIL.Image.open(str(self.get_path(self.id2, 'image.png')))

    @classmethod
    def from_folder(cls, folder: Path | str, model_name: str = 'UNSPECIFIED', seed: str = None) -> 'Comparison2AFCExperiment':
        try:
            folder = Path(folder)
            prompt = (folder / 'prompt.txt').read_text().strip()
        except:
            try:
                folder = Path(folder) / model_name / seed
                prompt = (folder / 'prompt.txt').read_text().strip()
            except:
                raise

        seed = seed or folder.parent.name

        return cls(prompt, id=folder.name, root_folder=folder, model_name=model_name, seed=seed)

    def save(self, root_folder: Path | str = None, overwrite: bool = False):
        if not self.id:
            self.id = uuid.uuid4().hex

        self.root_folder = root_folder or self.root_folder
        self.root_folder = Path(self.root_folder)

        root_folder = self.get
