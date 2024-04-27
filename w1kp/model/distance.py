__all__ = ['RGBColorFeatureExtractor', 'PairwiseDistanceMeasure', 'LPIPSDistanceMeasure', 'ListwiseDistanceMeasure',
           'CLIPDistanceMeasure', 'ViTDistanceMeasure', 'DinoV2DistanceMeasure']

import itertools
from typing import List, Dict, Tuple

import lpips
import numpy as np
import torch
from DISTS_pytorch import DISTS
from PIL.Image import Image
from stlpips_pytorch import stlpips
from torch import nn
from transformers import CLIPModel, CLIPProcessor, ViTImageProcessor, ViTModel, AutoImageProcessor, AutoModel, \
    Dinov2Model


class ListwiseDistanceMeasure:
    def measure(self, prompt: str, images: List[Image], **kwargs) -> float:
        raise NotImplementedError


class PairsListwiseDistanceMeasure(ListwiseDistanceMeasure):
    def __init__(self, pairwise_measure: 'PairwiseDistanceMeasure', num_sample: int = None):
        self.pairwise_measure = pairwise_measure
        self.num_sample = num_sample

    def measure(self, prompt: str, images: List[Image], debug: bool = False) -> float:
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
    def measure(self, prompt: str, image1: Image, image2: Image) -> float:
        raise NotImplementedError

    def to_listwise(self, num_sample: int = None) -> ListwiseDistanceMeasure:
        return PairsListwiseDistanceMeasure(self, num_sample=num_sample)


class TrainablePairwiseMeasure(nn.Module, PairwiseDistanceMeasure):
    def __init__(
            self,
            model: nn.Module,
            processor: AutoImageProcessor,
            use_dynamo: bool = True
    ):
        super().__init__()
        self.model = model
        self.processor = processor
        self.device = 'cpu'
        self.loss_type = 'bce'
        self.rank_module = stlpips.BCERankingLoss()

        if torch.cuda.is_available():
            torch.set_float32_matmul_precision('high')
            self.model = self.model.cuda()
            self.device = 'cuda'

            if use_dynamo:
                self.model: nn.Module  = torch.compile(self.model)

    def set_loss_type(self, loss_type: str):
        self.loss_type = loss_type

        if loss_type == 'bce-rank':
            for n, p in self.rank_module.named_parameters():
                p.requires_grad = True

    def compute_loss(self, encoded_scores: torch.Tensor, ground_truths: torch.Tensor) -> torch.Tensor:
        if self.loss_type == 'bce':
            return torch.nn.functional.binary_cross_entropy_with_logits(encoded_scores, ground_truths.round())
        elif self.loss_type == 'bce-rank':
            encoded_scores = (encoded_scores[0].unsqueeze(1).unsqueeze(1).unsqueeze(1),
                              encoded_scores[1].unsqueeze(1).unsqueeze(1).unsqueeze(1))
            ground_truths = ground_truths.unsqueeze(1).unsqueeze(1).unsqueeze(1)

            return self.rank_module(*encoded_scores, 2 * ground_truths - 1)

    def forward(
            self,
            prompt: str = None,
            image1: torch.Tensor = None,
            image2: torch.Tensor = None,
            ref_image: torch.Tensor = None,
            ground_truths: torch.Tensor = None,
    ):
        encoded_scores = self.encode_scores(prompt, image1, image2, ref_image)
        return self.compute_loss(encoded_scores, ground_truths)

    def measure(self, prompt: str, image1: Image, image2: Image) -> float:
        with torch.no_grad():
            inputs = self.processor(images=[image1, image2], return_tensors='pt')
            inputs['pixel_values'] = inputs['pixel_values'].to(self.device)
            outputs = self.model(**inputs, return_dict=True)

            return 1 - torch.nn.functional.cosine_similarity(
                outputs.pooler_output[0],
                outputs.pooler_output[1],
                dim=-1
            ).mean().item()

    def encode_scores(
            self,
            prompt: str,
            image_tensor1: torch.Tensor,
            image_tensor2: torch.Tensor,
            ref_tensor: torch.Tensor,
    ) -> torch.Tensor | Tuple[torch.Tensor, torch.Tensor]:
        outputs1 = self.model(pixel_values=image_tensor1, return_dict=True, output_hidden_states=True)
        outputs2 = self.model(pixel_values=image_tensor2, return_dict=True, output_hidden_states=True)
        outputs_ref = self.model(pixel_values=ref_tensor, return_dict=True, output_hidden_states=True)

        features1 = outputs1['pooler_output']
        features2 = outputs2['pooler_output']
        features_ref = outputs_ref['pooler_output']

        scores1 = torch.cosine_similarity(features1, features_ref, dim=-1)
        scores2 = torch.cosine_similarity(features2, features_ref, dim=-1)

        if self.loss_type == 'bce':
            return self.a * (scores1 - scores2) + self.b
        else:
            return scores1, scores2

    def get_trainable_parameters(self):
        return [p for p in self.parameters() if p.requires_grad]


class DinoV2DistanceMeasure(TrainablePairwiseMeasure):
    def __init__(self, model: str = 'facebook/dinov2-small'):
        # On Midjourney, DinoV2-small does best with one epoch of training (77.86 last two layers)
        super().__init__(
            Dinov2Model.from_pretrained(model),
            AutoImageProcessor.from_pretrained(model),
            use_dynamo=False,
        )

        trainable_names = {
            f'encoder.layer.{len(self.model.encoder.layer) - 1}.',
            f'encoder.layer.{len(self.model.encoder.layer) - 2}.',
            'layernorm',
        }

        for name, param in self.model.named_parameters():
            if any(substr in name for substr in trainable_names):
                continue

            param.requires_grad = False

        self.a = nn.Parameter(torch.ones(1), requires_grad=True)
        self.b = nn.Parameter(torch.zeros(1), requires_grad=True)


class ViTDistanceMeasure(TrainablePairwiseMeasure):
    def __init__(self, model: str = 'google/vit-base-patch32-224-in21k'):
        super().__init__(
            ViTModel.from_pretrained(model),
            ViTImageProcessor.from_pretrained(model),
            use_dynamo=False,
        )
        num_layers = len(self.model.encoder.layer)

        trainable_names = {
            'layernorm',
            'pooler',
            f'encoder.layer.{num_layers - 1}.',
        }

        for name, param in self.model.named_parameters():
            if any(substr in name for substr in trainable_names):
                continue

            param.requires_grad = False

        self.a = nn.Parameter(torch.ones(1), requires_grad=True)
        self.b = nn.Parameter(torch.zeros(1), requires_grad=True)


class CLIPDistanceMeasure(TrainablePairwiseMeasure):
    def __init__(self, model: str = 'openai/clip-vit-base-patch32'):
        super().__init__(CLIPModel.from_pretrained(model), CLIPProcessor.from_pretrained(model))
        num_layers = len(self.model.vision_model.encoder.layers)
        trainable_names = {
            'post_layernorm',
            f'vision_model.encoder.layers.{num_layers - 1}.',
        }

        for name, param in self.model.named_parameters():
            if any(substr in name for substr in trainable_names):
                continue

            param.requires_grad = False

        self.a = nn.Parameter(torch.ones(1), requires_grad=True)
        self.b = nn.Parameter(torch.zeros(1), requires_grad=True)
        self.eval()

    def encode_scores(
            self,
            prompt: str,
            image_tensor1: torch.Tensor,
            image_tensor2: torch.Tensor,
            ref_tensor: torch.Tensor,
    ) -> torch.Tensor | Tuple[torch.Tensor, torch.Tensor]:
        vision_outputs1 = self.model.vision_model(
            pixel_values=image_tensor1,
            output_attentions=None,
            output_hidden_states=False,
            return_dict=True,
        )

        vision_outputs2 = self.model.vision_model(
            pixel_values=image_tensor2,
            output_attentions=None,
            output_hidden_states=False,
            return_dict=True,
        )

        vision_outputs_ref = self.model.vision_model(
            pixel_values=ref_tensor,
            output_attentions=None,
            output_hidden_states=False,
            return_dict=True,
        )

        features1 = vision_outputs1['pooler_output']
        features2 = vision_outputs2['pooler_output']
        features_ref = vision_outputs_ref['pooler_output']

        scores1 = torch.cosine_similarity(features1, features_ref, dim=-1)
        scores2 = torch.cosine_similarity(features2, features_ref, dim=-1)

        if self.loss_type == 'bce':
            return self.a * (scores1 - scores2) + self.b
        else:
            return scores1, scores2

    def get_vision_features(self, pixel_values: torch.FloatTensor | None = None):
        vision_outputs = self.model.vision_model(
            pixel_values=pixel_values,
            output_attentions=None,
            output_hidden_states=True,
            return_dict=True,
        )

        return vision_outputs['pooler_output']

    def measure(self, prompt: str, image1: Image, image2: Image) -> float:
        with torch.no_grad():
            inputs = self.processor(text=[prompt], images=[image1, image2], return_tensors='pt', padding=True)
            inputs['pixel_values'] = inputs['pixel_values'].to(self.device)
            inputs['input_ids'] = inputs['input_ids'].to(self.device)
            inputs['attention_mask'] = inputs['attention_mask'].to(self.device)

            outputs = self.get_vision_features(pixel_values=inputs['pixel_values'])
            features = torch.nn.functional.cosine_similarity(outputs[0], outputs[1], dim=-1)

            return 1 - features.item()


class LPIPSProcessor:
    def __call__(self, images: List[Image], **kwargs) -> Dict[str, torch.Tensor]:
        images = [image.resize((64, 64)) for image in images]  # best on 64x64
        images = [torch.tensor(np.array(image)).permute(2, 0, 1).float() / 255 for image in images]
        images = [image * 2 - 1 for image in images]
        images = [image.unsqueeze(0) for image in images]

        return dict(pixel_values=torch.cat(images))


class STLPIPSProcessor:
    def __call__(self, images: List[Image], **kwargs) -> Dict[str, torch.Tensor]:
        images = [image.resize((256, 256)) for image in images]  # best on 256x256
        images = [torch.tensor(np.array(image)).permute(2, 0, 1).float() / 255 for image in images]
        images = [image * 2 - 1 for image in images]
        images = [image.unsqueeze(0) for image in images]

        return dict(pixel_values=torch.cat(images))


class DISTSProcessor:
    def __call__(self, images: List[Image], **kwargs) -> Dict[str, torch.Tensor]:
        images = [image.resize((256, 256)) for image in images]  # best on 256x256
        images = [torch.tensor(np.array(image)).permute(2, 0, 1).float() / 255 for image in images]
        # <-- Note the lack of normalization here compared to LPIPS
        images = [image.unsqueeze(0) for image in images]

        return dict(pixel_values=torch.cat(images))


class DISTSDistanceMeasure(TrainablePairwiseMeasure):
    def __init__(self, weights_path: str):
        super().__init__(DISTS(weights_path), DISTSProcessor())

    def measure(self, prompt: str, image1: Image, image2: Image) -> float:
        with torch.no_grad():
            images = self.processor([image1, image2])['pixel_values']
            image1, image2 = images
            image1 = image1.to(self.device).unsqueeze(0)
            image2 = image2.to(self.device).unsqueeze(0)

            return self.model(image1, image2).item()


class LPIPSDistanceMeasure(TrainablePairwiseMeasure):
    def __init__(self, network: str = 'alex', ft_dnn_only: bool = True, shift_tolerant: bool = False):
        if shift_tolerant:
            model = stlpips.LPIPS(net=network, variant='shift_tolerant')
            processor = STLPIPSProcessor()
        else:
            model = lpips.LPIPS(net=network)
            processor = LPIPSProcessor()

        super().__init__(model, processor, use_dynamo=False)
        self.a = nn.Parameter(torch.ones(1), requires_grad=True)
        self.b = nn.Parameter(torch.zeros(1), requires_grad=True)

        for name, param in self.model.named_parameters():
            param.requires_grad = 'lin' in name or not ft_dnn_only

    def encode_scores(
            self,
            prompt: str,
            image_tensor1: torch.Tensor,
            image_tensor2: torch.Tensor,
            ref_tensor: torch.Tensor,
    ) -> torch.Tensor | Tuple[torch.Tensor, torch.Tensor]:
        s1 = self.model(image_tensor2, ref_tensor).squeeze()
        s2 = self.model(image_tensor1, ref_tensor).squeeze()

        if self.loss_type == 'bce':
            return self.a * (s1 - s2) + self.b
        else:
            return s1, s2

    def measure(self, prompt: str, image1: Image, image2: Image) -> float:
        with torch.no_grad():
            images = self.processor([image1, image2])['pixel_values']
            image1, image2 = images
            image1 = image1.to(self.device)
            image2 = image2.to(self.device)

            return self.model(image1, image2).item()


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
