__all__ = ['ImageGenerator', 'AzureOpenAIImageGenerator', 'StableDiffusionXLImageGenerator',
           'StableDiffusion2ImageGenerator', 'ImagineApiMidjourneyGenerator', 'GoogleImagenImageGenerator',
           'AsyncSDXLClient', 'AsyncSDXLServer']

import asyncio
import base64
import json
import logging
import random
from io import BytesIO
from pathlib import Path
import time
from typing import List, Dict, Any

import aiohttp
import numpy as np
import zmq
import zmq.asyncio
from diffusers import DiffusionPipeline, StableDiffusionPipeline, DPMSolverMultistepScheduler
from PIL import Image
import torch
import vertexai
from vertexai.preview.vision_models import ImageGenerationModel

from ..utils import set_seed


class ImageGenerator:
    @property
    def is_multiple(self) -> bool:
        return self.num_multiple > 1

    @property
    def num_multiple(self) -> int:
        return 1

    async def generate_image(self, prompt: str, **kwargs) -> List[Dict[str, Any]] | Dict[str, Any]:
        """
        Generates an image from a prompt using Azure's OpenAI API.

        Args:
            prompt: The prompt to generate an image from.
            **kwargs: Additional arguments to pass to the API

        Returns:
            A (possibly list of) dictionary with the key `'image'` containing the generated image as a Pillow Image
            and `'revised_prompt'` the revised prompt. If `is_multiple` is True, the return value will be a list.
        """
        raise NotImplementedError()


class ImagineApiMidjourneyGenerator(ImageGenerator):
    def __init__(self, api_base='https://cl.imagineapi.dev', api_key=None):
        self.api_base = api_base
        self.api_key = api_key
        self.url = f'{self.api_base}/items/images/'

    @property
    def num_multiple(self) -> int:
        return 4

    async def generate_image(self, prompt: str, allow_revised_prompt: bool = True, **kwargs) -> List[Dict[str, Any]]:
        seed = kwargs.get('seed', 0)
        headers = {'Authorization': f'Bearer {self.api_key}', 'Content-Type': 'application/json'}

        try:
            async with aiohttp.ClientSession() as session:
                async with session.post(self.url, headers=headers, json=dict(prompt=prompt + f' --seed {seed}')) as response:
                    response = await response.json()
                    finished = False

                    while not finished:
                        async with session.get(f"{self.url}{response['data']['id']}", headers=headers) as response_data:
                            response_data = await response_data.json()

                            if response_data['data']['status'] in {'completed', 'failed'}:
                                finished = True
                            else:
                                await asyncio.sleep(5)

                    if response_data['data']['status'] == 'failed':
                        return None
                    else:
                        images = []

                        for image_url in response_data['data']['upscaled_urls']:
                            async with aiohttp.ClientSession() as session, session.get(image_url, timeout=60) as r:
                                content = await r.read()
                                images.append(dict(image=Image.open(BytesIO(content)), revised_prompt=prompt))

                        return images
        except:
            import traceback
            traceback.print_exc()

            return None


class StableDiffusion2ImageGenerator(ImageGenerator):
    def __init__(self):
        model_id = 'stabilityai/stable-diffusion-2-1'

        # Use the DPMSolverMultistepScheduler (DPM-Solver++) scheduler here instead
        pipe = StableDiffusionPipeline.from_pretrained(model_id, torch_dtype=torch.float16)
        pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)
        self.pipe = pipe.to('cuda')

    async def generate_image(self, prompt: str, **kwargs) -> List[Dict[str, Any]] | Dict[str, Any]:
        set_seed(kwargs.get('seed', 0))
        return dict(image=self.pipe(prompt=prompt, num_inference_steps=30, guidance_scale=5.0, **kwargs).images[0], revised_prompt=prompt)


class StableDiffusionXLImageGenerator(ImageGenerator):
    def __init__(self, device_idx: int = 0):
        pipe = DiffusionPipeline.from_pretrained(
            'stabilityai/stable-diffusion-xl-base-1.0',
            torch_dtype=torch.float16,
            use_safetensors=True,
            variant='fp16'
        )
        pipe.to(f'cuda:{device_idx}')
        self.pipe = pipe

    async def generate_image(self, prompt: str, **kwargs) -> List[Dict[str, Any]] | Dict[str, Any]:
        set_seed(kwargs.get('seed', 0))
        image = self.pipe(prompt=prompt, num_inference_steps=30, add_watermarker=False, **kwargs).images[0]

        return dict(image=image, revised_prompt=prompt)


class AsyncSDXLServer:
    # Server counterpart to AsyncSDXLClient; runs in a ZeroMQ loop
    def __init__(self, zero_mq_url: str, generator: StableDiffusionXLImageGenerator):
        self.zero_mq_url = zero_mq_url
        self.generator = generator

    async def run(self):
        context = zmq.asyncio.Context()
        socket = context.socket(zmq.REP)
        socket.bind(self.zero_mq_url)

        while True:
            message = await socket.recv_json()
            prompt = message['prompt']
            seed = message.get('seed', 0)
            guidance_scale = message.get('guidance_scale')
            kwargs = dict(guidance_scale=guidance_scale) if guidance_scale is not None else {}

            image = (await self.generator.generate_image(prompt, seed=seed, **kwargs))['image']
            buffer = BytesIO()
            image.save(buffer, format='PNG')
            image_str = base64.b64encode(buffer.getvalue()).decode('utf-8')
            response = dict(image=image_str)

            socket.send_json(response)


class AsyncSDXLClient(ImageGenerator):
    def __init__(self, zero_mq_url: str):
        self.zero_mq_url = zero_mq_url
        self.semaphore = asyncio.Semaphore(3)  # Don't exceed 3 at a time

    async def generate_image(self, prompt: str, **kwargs) -> List[Dict[str, Any]] | Dict[str, Any]:
        async with self.semaphore:
            context = zmq.asyncio.Context()
            socket = context.socket(zmq.REQ)
            socket.connect(self.zero_mq_url)

            socket.send_json(dict(prompt=prompt, **kwargs))
            response = await socket.recv_json()
            socket.close()

        image = Image.open(BytesIO(base64.b64decode(response['image'])))

        return dict(image=image, revised_prompt=prompt)


class GoogleImagenImageGenerator(ImageGenerator):
    def __init__(self, project_id: str):
        # TODO(developer): Update and un-comment below lines
        vertexai.init(project=project_id, location='us-central1')
        self.model = ImageGenerationModel.from_pretrained('imagegeneration@006')
        self.sem = asyncio.Semaphore(5)

    @property
    def num_multiple(self) -> int:
        return 4

    async def generate_image(self, prompt: str, **kwargs) -> List[Dict[str, Any]] | Dict[str, Any]:
        async with self.sem:
            try:
                images = self.model.generate_images(
                    prompt=prompt,
                    number_of_images=self.num_multiple,
                    language='en',
                    add_watermark=False,
                    seed=kwargs.get('seed', 0),
                    aspect_ratio='1:1',
                    safety_filter_level='block_some',
                    person_generation='allow_adult',
                ).images

                if not images:
                    return None

                return [dict(image=image._pil_image, revised_prompt=prompt) for image in images]
            except:
                return None


class AzureOpenAIImageGenerator(ImageGenerator):
    def __init__(self, api_base=None, api_key=None, deployment_name=None, api_version='2024-02-01', num_parallel=2):
        self.api_base = api_base
        self.api_key = api_key
        self.deployment_name = deployment_name
        self.api_version = api_version
        self.url = f'{self.api_base}/openai/deployments/{self.deployment_name}/images/generations?api-version={self.api_version}'
        self.semaphore = asyncio.Semaphore(num_parallel)

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

    async def generate_image(self, prompt: str, **kwargs) -> List[Dict[str, Any]] | Dict[str, Any]:
        """
        Generates an image from a prompt using Azure's OpenAI API.

        Args:
            prompt: The prompt to generate an image from.
            **kwargs: Additional arguments to pass to the API. Valid keys are `'quality'`, `'style'`, and `'seed'`.
                Quality can be one of `'hd'` or `'standard'`, and style one of `'vivid'` or `'natural'`.

        Returns:
            The generated image as a Pillow Image.

        Raises:
            asyncio.exceptions.TimeoutError: If the request times out.
            KeyError: If the returned JSON does not contain the appropriate keys.
        """
        if 'quality' not in kwargs:
            kwargs['quality'] = 'hd'

        if 'style' not in kwargs:
            kwargs['style'] = 'vivid'

        while True:
            try:
                prompt_fix = 'I NEED to test how the tool works with extremely simple prompts. DO NOT add any detail, just use it AS-IS: '

                async with self.semaphore, aiohttp.ClientSession() as session:
                    async with session.post(
                        self.url,
                        headers={'api-key': self.api_key, 'Content-Type': 'application/json'},
                        json=dict(prompt=prompt_fix + prompt, size='1024x1024', n=1, **kwargs),
                        timeout=60,
                    ) as response:
                        return_dict = await response.json()

                revised_prompt = prompt
                time.sleep(1)  # allow time for saving the image

                async with aiohttp.ClientSession() as session, session.get(return_dict['data'][0]['url'], timeout=60) as r:
                    content = await r.read()

                return dict(
                    image=Image.open(BytesIO(content)),
                    revised_prompt=revised_prompt
                )
            except KeyError:
                match return_dict:
                    case {'error': {'code': '429'}}:
                        continue  # retry if rate limited

                logging.error(str(return_dict))

                return None
            except asyncio.exceptions.TimeoutError:
                continue
