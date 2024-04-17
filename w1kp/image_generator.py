import json
import logging
import multiprocessing as mp
from io import BytesIO
from pathlib import Path
import time
from typing import List

from diffusers import DiffusionPipeline, StableDiffusionPipeline, DPMSolverMultistepScheduler
import nltk
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
import editdistance
import requests
import torch

from PIL import Image
from tqdm import trange


__all__ = ['ImageGenerator', 'AzureOpenAIImageGenerator', 'ImageGeneratorPool', 'OPINION_COLOR_PROMPT',
           'StableDiffusionXLImageGenerator', 'StableDiffusion2ImageGenerator', 'OPINION_COLOR_PROMPT_SD']


nltk.download('punkt')
nltk.download('stopwords')
stop_words = set(stopwords.words('english'))
stop_words.update({'.', ',', '!', '?', '(', ')', '[', ']', '{', '}', '<', '>', ':', ';', '"', "'s"})
OPINION_COLOR_PROMPT = 'Legend: red is negative, green is positive. A single color describing this word: "{word}"'
OPINION_COLOR_PROMPT_SD = 'Legend: red = bad, green = good. Title: "{word}", monochrome.'


class ImageGeneratorPool:
    """
    Generates a batch of images from multiple image generators in parallel using multiprocessing. If the generator does
    not support multiprocessing (:py:meth:`.ImageGenerator.is_multiprocessable`), it falls back to single-threading.

    Examples:
        >>> image_generators = AzureOpenAIImageGenerator.parse_from_path('path/to/image/config.json')
        >>> pool = ImageGeneratorPool(image_generators)
        >>> images = pool.generate_all(['prompt 1', 'prompt 2', 'prompt 3'], quality='standard')
    """
    def __init__(self, image_generators: List['ImageGenerator'], delay_secs: float = 0.1, suppress_errors: bool = True):
        self.suppress_errors = suppress_errors
        self.mp_model_queue = mp.Queue()
        self.st_model_queue = mp.Queue()

        for image_generator in image_generators:
            if image_generator.is_multiprocessable():
                self.mp_model_queue.put(image_generator)
            else:
                self.st_model_queue.put(image_generator)

        self.input_queue = mp.Queue()
        self.output_queue = mp.Queue()
        self.processes = [mp.Process(target=self._run) for _ in range(len(image_generators))]
        self._current_idx = 0
        self.delay_secs = delay_secs
        self.has_mp_models = not self.mp_model_queue.empty()
        self.has_st_models = not self.st_model_queue.empty()

        for process in self.processes:
            process.start()

    def generate_all(self, inputs: List[str], **kwargs):
        for prompt in inputs:
            self._enqueue(prompt, **kwargs)

        values = [self._dequeue() for _ in trange(len(inputs))]
        values.sort(key=lambda x: x[0])
        values = [value[1] for value in values]

        return values

    def _enqueue(self, prompt: str, **kwargs) -> int:
        self.input_queue.put((self._current_idx, prompt, kwargs))
        self._current_idx += 1

        return self._current_idx - 1

    def _dequeue(self):
        return self.output_queue.get()

    def _run(self):
        while True:
            if self.has_mp_models:
                if self.mp_model_queue.empty() and self.has_st_models:
                    model = self.st_model_queue.get()
                else:
                    model = self.mp_model_queue.get()
            else:
                model = self.st_model_queue.get()

            idx, prompt, kwargs = self.input_queue.get()

            try:
                image = model.generate_image(prompt, **kwargs)
                self.output_queue.put((idx, image))
            except:
                import traceback
                traceback.print_exc()

                if not self.suppress_errors:
                    raise

                self.input_queue.put((idx, prompt, kwargs))  # try again
            finally:
                self.mp_model_queue.put(model)  # put back in queue

            time.sleep(self.delay_secs)


class ImageGenerator:
    def is_multiprocessable(self) -> bool:
        return True

    def generate_image(self, prompt: str, **kwargs) -> Image:
        """
        Generates an image from a prompt using Azure's OpenAI API.

        Args:
            prompt: The prompt to generate an image from.
            **kwargs: Additional arguments to pass to the API

        Returns:
            The generated image as a Pillow Image.
        """
        raise NotImplementedError()


class StableDiffusion2ImageGenerator(ImageGenerator):
    def __init__(self):
        model_id = 'stabilityai/stable-diffusion-2-1'

        # Use the DPMSolverMultistepScheduler (DPM-Solver++) scheduler here instead
        pipe = StableDiffusionPipeline.from_pretrained(model_id, torch_dtype=torch.float16)
        pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)
        self.pipe = pipe.to('cuda')

    def is_multiprocessable(self) -> bool:
        return False

    def generate_image(self, prompt: str, **kwargs) -> Image:
        return self.pipe(prompt=prompt, **kwargs).images[0]


class StableDiffusionXLImageGenerator(ImageGenerator):
    def __init__(self):
        pipe = DiffusionPipeline.from_pretrained(
            'stabilityai/stable-diffusion-xl-base-1.0',
            torch_dtype=torch.float16,
            use_safetensors=True,
            variant='fp16'
        )
        pipe.to('cuda')
        self.pipe = pipe

    def is_multiprocessable(self) -> bool:
        return False

    def generate_image(self, prompt: str, **kwargs) -> Image:
        return self.pipe(prompt=prompt, add_watermarker=False, **kwargs).images[0]


def clean_prompt(prompt: str) -> str:
    prompt = prompt.lower().strip()
    ps = PorterStemmer()
    prompt = ' '.join(ps.stem(word) for word in nltk.tokenize.word_tokenize(prompt) if word not in stop_words)

    return prompt


class AzureOpenAIImageGenerator(ImageGenerator):
    def __init__(self, api_base=None, api_key=None, deployment_name=None, api_version='2024-02-01'):
        self.api_base = api_base
        self.api_key = api_key
        self.deployment_name = deployment_name
        self.api_version = api_version
        self.url = f'{self.api_base}/openai/deployments/{self.deployment_name}/images/generations?api-version={self.api_version}'

    @classmethod
    def parse_from_path(cls, path: Path | str) -> List['AzureOpenAIImageGenerator']:
        """
        Parses a list of AzureOpenAIImageGenerator objects from a path. The path should be a file in the format

        .. code-block::json
            [
                {
                    "api_base": "https://api.cognitive.microsoft.com",
                    "api_key": "REDACTED",
                    "deployment_name": "REDACTED",
                    "api_version": "2023-12-01-preview",
                }, ...
            ]

        Args:
            path: The path to parse from.

        Returns:
            A list of AzureOpenAIImageGenerator objects.
        """
        path = Path(path)

        return [cls(**d) for d in json.loads(path.read_text())]

    def generate_image(self, prompt: str, allow_revised_prompt: bool = True, **kwargs) -> Image:
        """
        Generates an image from a prompt using Azure's OpenAI API.

        Args:
            prompt: The prompt to generate an image from.
            allow_revised_prompt: Whether to allow prompt revision.
            **kwargs: Additional arguments to pass to the API. Valid keys are `'quality'` and `'style'`. Quality can be
                one of `'hd'` or `'standard'`, and style one of `'vivid'` or `'natural'`.

        Returns:
            The generated image as a Pillow Image.

        Raises:
            requests.exceptions.Timeout: If the request times out.
            KeyError: If the returned JSON does not contain the appropriate keys.
        """
        return_dict = requests.post(
            self.url,
            headers={'api-key': self.api_key, 'Content-Type': 'application/json'},
            json=dict(prompt=prompt, size='1024x1024', n=1, **kwargs),
            timeout=60,
        ).json()

        try:
            if not allow_revised_prompt:
                match return_dict['data'][0]:
                    case {'revised_prompt': revised_prompt}:
                        print('Revision: ', revised_prompt)
                        prompt = clean_prompt(prompt)
                        revised_prompt = clean_prompt(revised_prompt)

                        if editdistance.eval(revised_prompt, prompt) >= 3:
                            logging.error(f'Prompt was revised to "{revised_prompt}"')
                            return None

            time.sleep(1)  # allow time for saving the image

            return Image.open(BytesIO(requests.get(return_dict['data'][0]['url'], timeout=60).content))
        except KeyError:
            match return_dict:
                case {'error': {'code': 'tooManyRequests'}}:
                    logging.error('Too many requests!')
                    raise

            logging.error(str(return_dict))

            return None
