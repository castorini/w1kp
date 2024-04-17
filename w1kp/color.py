from typing import List

import numpy as np
from PIL.Image import Image

__all__ = ['RGBColorFeatureExtractor']


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
