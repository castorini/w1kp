__all__ = ['RGBColorFeatureExtractor', 'PairwiseDistanceMeasure', 'LPIPSDistanceMeasure', 'ListwiseDistanceMeasure']

import itertools
from typing import List

import lpips
import numpy as np
import torch
from PIL.Image import Image


class ListwiseDistanceMeasure:
    def __call__(self, prompt: str, images: List[Image], **kwargs) -> float:
        raise NotImplementedError


class PairsListwiseDistanceMeasure(ListwiseDistanceMeasure):
    def __init__(self, pairwise_measure: 'PairwiseDistanceMeasure', num_sample: int = None):
        self.pairwise_measure = pairwise_measure
        self.num_sample = num_sample

    def __call__(self, prompt: str, images: List[Image], debug: bool = False) -> float:
        n = len(images)
        total_distance = 0
        distances = []
        combs = np.array(list(itertools.combinations(range(n), 2)))

        if self.num_sample is not None:
            num_sample = min(self.num_sample, len(combs))
            combs = combs[np.random.choice(np.arange(len(combs)), num_sample, replace=False)]

        for i, j in combs:
            d = self.pairwise_measure(prompt, images[i], images[j])
            total_distance += d

            if debug:
                distances.append(((i, j), d))

        if debug:
            distances = sorted(distances, key=lambda x: x[1])

            for ij, dist in distances[:3]:
                print('min', ij, dist)

            for ij, dist in distances[-3:]:
                print('max', ij, dist)

        return total_distance / len(combs)


class PairwiseDistanceMeasure:
    def __call__(self, prompt: str, image1: Image, image2: Image) -> float:
        raise NotImplementedError

    def to_listwise(self, num_sample: int = None) -> ListwiseDistanceMeasure:
        return PairsListwiseDistanceMeasure(self, num_sample=num_sample)


class LPIPSDistanceMeasure(PairwiseDistanceMeasure):
    def __init__(self, network: str = 'alex'):
        self.lpips = lpips.LPIPS(net=network)
        self.device = 'cpu'

        if torch.cuda.is_available():
            self.lpips = self.lpips.cuda()
            self.device = 'cuda'

    def __call__(self, prompt: str, image1: Image, image2: Image) -> float:
        # Downsample to 64x64
        image1 = image1.resize((64, 64))
        image2 = image2.resize((64, 64))

        # Convert to tensor
        image1 = torch.tensor(np.array(image1)).permute(2, 0, 1).float() / 255
        image2 = torch.tensor(np.array(image2)).permute(2, 0, 1).float() / 255

        # Normalize to [-1, 1]
        image1 = image1 * 2 - 1
        image2 = image2 * 2 - 1

        # Tensor to GPU
        image1 = image1.unsqueeze(0)
        image2 = image2.unsqueeze(0)
        image1 = image1.to(self.device)
        image2 = image2.to(self.device)

        return self.lpips(image1, image2).item()


class RGBColorFeatureExtractor:
    def __init__(self, image: Image):
        self.image = image

    def mean_area(self, channel: str, hue_tolerances: List[int] = [15, 30, 40], saturation_threshold: int = 10) -> float:
        match channel:
            case 'r' | 'red':
                channel_hue_center = 0
                channel = 0
            case 'g' | 'green':
                channel_hue_center = 120
                channel = 1
            case 'b' | 'blue':
                channel_hue_center = 220
                channel = 2
            case _:
                raise ValueError(f'Invalid channel {channel}')

        pixels = np.array(self.image.getdata())

        # RGB to HSL
        pixels = pixels / 255
        cmax = np.max(pixels, axis=1)
        cmin = np.min(pixels, axis=1)
        delta = cmax - cmin

        # Hue
        hue = np.zeros(len(pixels))
        hue[cmax == cmin] = 0
        hue[cmax == pixels[:, 0]] = 60 * (((pixels[cmax == pixels[:, 0], 1] - pixels[cmax == pixels[:, 0], 2]) / delta[cmax == pixels[:, 0]]) % 6)
        hue[cmax == pixels[:, 1]] = 60 * (((pixels[cmax == pixels[:, 1], 2] - pixels[cmax == pixels[:, 1], 0]) / delta[cmax == pixels[:, 1]]) + 2)
        hue[cmax == pixels[:, 2]] = 60 * (((pixels[cmax == pixels[:, 2], 0] - pixels[cmax == pixels[:, 2], 1]) / delta[cmax == pixels[:, 2]]) + 4)

        # Saturation
        saturation = np.zeros(len(pixels))
        saturation[cmax != 0] = delta[cmax != 0] / cmax[cmax != 0]

        mask = np.zeros(len(pixels))

        if channel == 0:
            mask[np.abs(((hue + 180) % 360) - (channel_hue_center + 180)) <= hue_tolerances[channel]] = 1
        else:
            mask[np.abs(hue - channel_hue_center) <= hue_tolerances[channel]] = 1

        mask[saturation < saturation_threshold / 100] = 0

        return np.mean(mask)
