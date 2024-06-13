# <img src="icon-banner.svg" height="32" style="position: relative; margin-top: 15px;"/>: An Image Variability Metric
[![Website](https://img.shields.io/badge/Website-online-green.svg)](http://w1kp.com) [![Citation](https://img.shields.io/badge/Citation-arXiv-orange.svg)](https://gist.github.com/daemon/1d1351f23aacef1ef6edef1a4afd9784) [![PyPi version](https://badgen.net/pypi/v/w1kp?color=blue)](https://pypi.org/project/w1kp) [![Downloads](https://static.pepy.tech/badge/w1kp)](https://pepy.tech/project/w1kp)

<p align="center">
  <img src="https://github.com/castorini/w1kp/assets/6188572/4f4c2ad2-2716-43aa-9119-41b4c7d85465"/>
</p>

As proposed in [our paper](https://arxiv.org/abs/2406.08482), the "Words of a Thousand Pictures" metric (W1KP) measures perceptual variability for sets of images in text-to-image generation, bootstrapped from existing perceptual distances such as DreamSim.

## Getting Started

### Installation
1. Install [PyTorch](https://pytorch.org) for your Python 3.10+ environment.

2. Install W1KP: `pip install w1kp`

3. Download the [calibration data file](cdf-xy.pt).

4. You're done!

### Sample Library Usage

We recommend $\text{DreamSim}_{\ell_2}$, the best-performing perceptual distance backbone in our paper.
```python
import asyncio

import torch
from w1kp import StableDiffusionXLImageGenerator, DreamSimDistanceMeasure, query_inverted_cdf


async def amain():
  # Generate 10 SDXL images for a prompt
  prompt = 'cat'
  images = []
  image_gen = StableDiffusionXLImageGenerator()

  for seed in range(10):
    ret = await image_gen.generate_image(prompt, seed=seed)
    images.append(ret['image'])

  # Compute and normalize the W1KP score
  dreamsim_l2 = DreamSimDistanceMeasure().to_listwise()
  w1kp_score = dreamsim_l2.measure(images)
  cdf_x, cdf_y = torch.load('cdf-xy.pt')  # download this data file from the repo

  dist = dreamsim_l2.measure(prompt, images)
  dist = query_inverted_cdf(cdf_x, cdf_y, dist)  # normalize to U[0, 1]
  w1kp_score = 1 - dist  # invert for the W1KP score

  for im in images:
    im.show()

  print(f'The W1KP score for the images are {w1kp_score}')
  

if __name__ == '__main__':
  asyncio.run(amain())
```

## Citation
```
@article{tang2024w1kp,
  title={Words Worth a Thousand Pictures: Measuring and Understanding Perceptual Variability in Text-to-Image Generation},
  author={Tang, Raphael and Zhang, Xinyu and Xu, Lixinyu and Lu, Yao and Li, Wenyan and Stenetorp, Pontus and Lin, Jimmy and Ture, Ferhan},
  journal={arXiv:2210.04885},
  year={2024}
}
```
