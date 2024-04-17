from dataclasses import dataclass
from pathlib import Path
from typing import List
import uuid

from PIL.Image import Image
import PIL.Image

__all__ = ['Experiment']


@dataclass
class Experiment:
    """Serializable representation of a single image generation experiment."""
    prompt: str
    id: str = ''
    root_folder: Path = None
    image: None | Image = None

    def load_image(self):
        # Lazy loading
        self.image = PIL.Image.open(str(Path(self.root_folder) / 'image.png'))

    @classmethod
    def from_folder(cls, folder: Path | str) -> 'Experiment':
        folder = Path(folder)
        prompt = (folder / 'prompt.txt').read_text().strip()

        return cls(prompt, id=folder.name, root_folder=folder)

    @classmethod
    def load_all(cls, folder: Path | str) -> List['Experiment']:
        folder = Path(folder)
        return [cls.from_folder(f) for f in folder.iterdir() if f.is_dir()]

    def save(self, root_folder: Path | str, overwrite: bool = False):
        if not self.id:
            self.id = uuid.uuid4().hex

        root_folder = Path(root_folder) / self.id
        root_folder.mkdir(exist_ok=overwrite, parents=True)
        (root_folder / 'prompt.txt').write_text(self.prompt)

        if self.image is not None:
            self.image.save(root_folder / 'image.png')
