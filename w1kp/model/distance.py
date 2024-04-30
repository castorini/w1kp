__all__ = ['RGBColorFeatureExtractor', 'PairwiseDistanceMeasure', 'LPIPSDistanceMeasure', 'ListwiseDistanceMeasure',
           'CLIPDistanceMeasure', 'ViTDistanceMeasure', 'DinoV2DistanceMeasure', 'GroupViTDistanceMeasure']

import itertools
from typing import List, Dict, Tuple

import lpips
import numpy as np
import torch
from DISTS_pytorch import DISTS
from PIL.Image import Image
from stlpips_pytorch import stlpips
from torch import nn
from torch.nn.functional import binary_cross_entropy_with_logits as bce_loss
from transformers import CLIPModel, CLIPProcessor, ViTImageProcessor, ViTModel, AutoImageProcessor, \
    Dinov2Model, AutoTokenizer, CLIPTokenizer, GroupViTModel, AutoProcessor


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
            d = self.pairwise_measure.measure(prompt, images[i], images[j])
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
            use_dynamo: bool = True,
            tokenizer: AutoTokenizer = None,
            use_text: bool = False,
            distance_type: str = 'cosine',
    ):
        super().__init__()
        self.model = model
        self.processor = processor
        self.device = 'cpu'
        self.loss_type = 'bce'
        self.rank_module = stlpips.BCERankingLoss()
        self.coper_w = nn.Parameter(torch.ones(self.vision_hidden_size), requires_grad=False)
        self.tokenizer = tokenizer
        self.use_text = use_text
        self.distance_type = distance_type
        self.register_buffer('max_l2_norm', torch.Tensor([1.0]))

        if torch.cuda.is_available():
            torch.set_float32_matmul_precision('high')
            self.model = self.model.cuda()
            self.device = 'cuda'

            if use_dynamo:
                self.model: nn.Module = torch.compile(self.model)

    def get_image_features(self, pixel_values: torch.FloatTensor | None = None, **kwargs):
        raise NotImplementedError

    def get_text_features(self, **prompt):
        raise NotImplementedError

    @property
    def vision_num_layers(self) -> int:
        return 1

    @property
    def vision_hidden_size(self) -> int:
        return 384

    def set_loss_type(self, loss_type: str):
        self.loss_type = loss_type

        if loss_type == 'bce-rank' or loss_type == 'bce-rank-coper':
            for n, p in self.rank_module.named_parameters():
                p.requires_grad = True

        if loss_type == 'bce-rank-coper':
            self.coper_w.requires_grad = True

            for name, param in self.model.named_parameters():
                param.requires_grad = False

    def compute_loss(self, encoded_scores: torch.Tensor | Tuple[torch.Tensor], ground_truths: torch.Tensor) -> torch.Tensor:
        match self.loss_type:
            case 'bce':
                return bce_loss(encoded_scores, ground_truths.round())
            case 'bce-rank' | 'bce-rank-coper':
                encoded_scores = (encoded_scores[0].unsqueeze(1).unsqueeze(1).unsqueeze(1),
                                  encoded_scores[1].unsqueeze(1).unsqueeze(1).unsqueeze(1))
                ground_truths = ground_truths.unsqueeze(1).unsqueeze(1).unsqueeze(1)

                return self.rank_module(*encoded_scores, 2 * ground_truths - 1)

    def forward(
            self,
            prompt: torch.Tensor = None,
            image1: torch.Tensor = None,
            image2: torch.Tensor = None,
            ref_image: torch.Tensor = None,
            ground_truths: torch.Tensor = None,
    ):
        encoded_scores = self.encode_scores(prompt, image1, image2, ref_image)
        return self.compute_loss(encoded_scores, ground_truths)

    def distance_function(self, x, y, dim: int = -1):
        match self.distance_type:
            case 'cosine':
                sim = torch.nn.functional.cosine_similarity(x, y, dim=dim).clamp(-0.1, 1)
                return 1 - sim
            case 'dot':
                return (-(x * y).sum(dim=dim))  # unnormalized, do not use
            case 'l2':
                norm = (x - y).norm(p=2, dim=dim)
                self.max_l2_norm = max(self.max_l2_norm, norm.max().item())

                return norm / self.max_l2_norm  # normalize to [0, 1]

    def measure(self, prompt: str, image1: Image, image2: Image) -> float:
        with torch.no_grad():
            inputs = self.processor(images=[image1, image2], return_tensors='pt')
            inputs = {k: v.to(self.device) for k, v in inputs.items() if isinstance(v, torch.Tensor)}

            outputs = self.get_image_features(pixel_values=inputs['pixel_values'], return_dict=True)
            features1, features2 = outputs['features'][0], outputs['features'][1]

            if self.use_text and self.tokenizer:
                prompt = self.tokenizer([prompt], return_tensors='pt', padding=True, truncation=True)
                prompt = {k: v.to(self.device) for k, v in prompt.items()}
                text_features = self.get_text_features(**prompt)

                min_length = min(text_features.size(1), features1.size(0))
                features1 = features1[:min_length]
                features2 = features2[:min_length]
                text_features = text_features[:, min_length]
                features1 += text_features.squeeze()
                features2 += text_features.squeeze()

            if self.loss_type == 'bce' or self.loss_type == 'bce-rank':
                dist = self.distance_function(features1, features2).item()

                if self.distance_type == 'cosine':
                    dist = dist / 1.1  # normalize to [0, 1]

                return dist
            else:
                diffs = []

                for outputs in outputs['hidden_states']:
                    hid1 = outputs[0].unsqueeze(0)
                    hid2 = outputs[1].unsqueeze(0)
                    l, c = hid1.size(1) - 1, hid1.size(2)
                    l = int(l ** 0.5)
                    h1 = hid1[:, 1:].view(hid1.size(0), l, l, c)
                    h2 = hid2[:, 1:].view(hid2.size(0), l, l, c)

                    # Unit normalize channel-wise
                    h1 = h1 / (h1.abs().max(dim=-1, keepdim=True)[0] + 1e-6)
                    h2 = h2 / (h2.abs().max(dim=-1, keepdim=True)[0] + 1e-6)

                    h1 = h1 * self.coper_w.unsqueeze(0).unsqueeze(0).unsqueeze(0)
                    h2 = h2 * self.coper_w.unsqueeze(0).unsqueeze(0).unsqueeze(0)

                    # Subtract
                    diff = (h1 - h2).norm(p=2, dim=-1)
                    diff = diff.mean(-1).mean(-1)
                    diffs.append(diff)

                return torch.stack(diffs, 0).mean().item()

    def encode_scores(
            self,
            prompt: Dict[str, torch.Tensor],
            image_tensor1: torch.Tensor,
            image_tensor2: torch.Tensor,
            ref_tensor: torch.Tensor,
    ) -> torch.Tensor | Tuple[torch.Tensor, torch.Tensor]:
        outputs1 = self.get_image_features(image_tensor1)
        outputs2 = self.get_image_features(image_tensor2)
        outputs_ref = self.get_image_features(ref_tensor)

        if self.loss_type == 'bce' or self.loss_type == 'bce-rank':
            features1 = outputs1['features']
            features2 = outputs2['features']
            features_ref = outputs_ref['features']

            if self.use_text and self.tokenizer:
                text_features = self.get_text_features(**prompt)

                min_length = min(text_features.size(1), features1.size(1))
                features1 = features1[:, :min_length]
                features2 = features2[:, :min_length]
                features_ref = features_ref[:, :min_length]
                text_features = text_features[:, :min_length]

                features1 += text_features
                features2 += text_features
                features_ref += text_features

            scores1 = -self.distance_function(features1, features_ref, dim=-1)
            scores2 = -self.distance_function(features2, features_ref, dim=-1)
        else:  # CoPer
            diffs1 = []
            diffs2 = []

            for hid1, hid2, hid_ref in zip(
                    outputs1['hidden_states'],
                    outputs2['hidden_states'],
                    outputs_ref['hidden_states']
            ):
                l, c = hid1.size(1) - 1, hid1.size(2)
                l = int(l ** 0.5)
                h1 = hid1[:, 1:].view(hid1.size(0), l, l, c)
                h2 = hid2[:, 1:].view(hid2.size(0), l, l, c)
                h_ref = hid_ref[:, 1:].view(hid_ref.size(0), l, l, c)

                # Unit normalize channel-wise
                h1 = h1 / (h1.abs().max(dim=-1, keepdim=True)[0] + 1e-6)
                h2 = h2 / (h2.abs().max(dim=-1, keepdim=True)[0] + 1e-6)
                h_ref = h_ref / (h_ref.abs().max(dim=-1, keepdim=True)[0] + 1e-6)

                h1 = h1 * self.coper_w.unsqueeze(0).unsqueeze(0).unsqueeze(0)
                h2 = h2 * self.coper_w.unsqueeze(0).unsqueeze(0).unsqueeze(0)
                h_ref = h_ref * self.coper_w.unsqueeze(0).unsqueeze(0).unsqueeze(0)

                # Subtract
                diff1 = (h1 - h_ref).norm(p=2, dim=-1)
                diff1 = diff1.mean(-1).mean(-1)
                diffs1.append(diff1)

                diff2 = (h2 - h_ref).norm(p=2, dim=-1)
                diff2 = diff2.mean(-1).mean(-1)
                diffs2.append(diff2)

            scores1 = torch.stack(diffs1, 0).mean(0)
            scores2 = torch.stack(diffs2, 0).mean(0)

        if self.loss_type == 'bce':
            return self.a * (scores1 - scores2)
        else:
            return scores1, scores2

    def get_trainable_parameters(self):
        return [p for p in self.parameters() if p.requires_grad]


class DinoV2DistanceMeasure(TrainablePairwiseMeasure):
    def __init__(self, model: str = 'facebook/dinov2-small', **kwargs):
        # In pilot experiments, DinoV2-small does best with two epochs of training and a weight decay of 0.2
        super().__init__(
            Dinov2Model.from_pretrained(model),
            AutoImageProcessor.from_pretrained(model),
            use_dynamo=False,
            **kwargs
        )

        trainable_names = {
            f'encoder.layer.{len(self.model.encoder.layer) - 1}.',
            'layernorm',
        }

        for name, param in self.model.named_parameters():
            if any(substr in name for substr in trainable_names):
                continue

            param.requires_grad = False

        self.a = nn.Parameter(torch.ones(1), requires_grad=True)
        self.b = nn.Parameter(torch.zeros(1), requires_grad=True)

    def get_image_features(self, pixel_values: torch.FloatTensor | None = None, **kwargs):
        ret = self.model(pixel_values=pixel_values, return_dict=True, output_hidden_states=True)
        return dict(features=ret['pooler_output'], hidden_states=ret['hidden_states'])

    @property
    def vision_num_layers(self) -> int:
        return len(self.model.encoder.layer)

    @property
    def vision_hidden_size(self) -> int:
        return self.model.config.hidden_size


class ViTDistanceMeasure(TrainablePairwiseMeasure):
    def __init__(self, model: str = 'google/vit-base-patch32-224-in21k', **kwargs):
        super().__init__(
            ViTModel.from_pretrained(model),
            ViTImageProcessor.from_pretrained(model),
            use_dynamo=False,
            **kwargs
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

    def get_image_features(self, pixel_values: torch.FloatTensor | None = None, **kwargs):
        ret = self.model(pixel_values=pixel_values, return_dict=True, output_hidden_states=True)
        return dict(features=ret['pooler_output'], hidden_states=ret['hidden_states'])

    @property
    def vision_num_layers(self) -> int:
        return len(self.model.encoder.layer)

    @property
    def vision_hidden_size(self) -> int:
        return self.model.config.hidden_size


class CLIPDistanceMeasure(TrainablePairwiseMeasure):
    def __init__(
            self,
            model: str = 'laion/CLIP-ViT-B-32-laion2B-s34B-b79K',
            default_featurizer: bool = False,
            use_text: bool = False,
            **kwargs
    ):
        super().__init__(
            CLIPModel.from_pretrained(model),
            CLIPProcessor.from_pretrained(model),
            tokenizer=CLIPTokenizer.from_pretrained(model),
            use_text=use_text,
            **kwargs
        )

        num_layers = len(self.model.vision_model.encoder.layers)
        trainable_names = {
            'post_layernorm',
            f'vision_model.encoder.layers.{num_layers - 1}.',
            f'text_model.encoder.layers.{num_layers - 1}.',
            'final_layer_norm',
            'visual_projection',
            'text_projection',
        }

        for name, param in self.model.named_parameters():
            if any(substr in name for substr in trainable_names):
                continue

            param.requires_grad = False

        self.a = nn.Parameter(torch.ones(1), requires_grad=True)
        self.b = nn.Parameter(torch.zeros(1), requires_grad=True)
        self.eval()
        self.default_featurizer = default_featurizer

    @property
    def vision_num_layers(self) -> int:
        return len(self.model.vision_model.encoder.layers)

    @property
    def vision_hidden_size(self) -> int:
        return self.model.vision_model.config.hidden_size

    def get_image_features(self, pixel_values: torch.FloatTensor | None = None, **kwargs):
        if self.default_featurizer:
            out = self.model.get_image_features(pixel_values=pixel_values, output_hidden_states=True, return_dict=True)

            return dict(features=out)
        else:
            ret = self.model.vision_model(
                pixel_values=pixel_values,
                output_attentions=None,
                output_hidden_states=True,
                return_dict=True,
            )  # this yields slightly better quality
            out = ret['pooler_output']

            return dict(features=out, hidden_states=ret['hidden_states'])

    def get_text_features(self, **prompt):
        if self.default_featurizer:
            return self.model.get_text_features(**prompt)
        else:
            text_outputs = self.model.text_model(**prompt)
            pooled_output = text_outputs[1]

            return pooled_output


class GroupViTDistanceMeasure(TrainablePairwiseMeasure):
    def __init__(
            self,
            model: str = 'nvidia/groupvit-gcc-yfcc',
            default_featurizer: bool = False,
            use_text: bool = False,
            **kwargs
    ):
        super().__init__(
            GroupViTModel.from_pretrained(model),
            AutoProcessor.from_pretrained(model),
            tokenizer=CLIPTokenizer.from_pretrained(model),
            use_text=use_text,
            **kwargs
        )

        trainable_names = {
            'vision_model.layernorm',
            f'vision_model.encoder.stages.2.layers.2.',
            f'text_model.encoder.layers.11.',
            'final_layer_norm',
            'visual_projection',
            'text_projection',
        }

        for name, param in self.model.named_parameters():
            if any(substr in name for substr in trainable_names):
                continue

            param.requires_grad = False

        self.a = nn.Parameter(torch.ones(1), requires_grad=True)
        self.b = nn.Parameter(torch.zeros(1), requires_grad=True)
        self.eval()
        self.default_featurizer = default_featurizer

    @property
    def vision_num_layers(self) -> int:
        return 2

    @property
    def vision_hidden_size(self) -> int:
        return self.model.vision_model.config.hidden_size

    def get_image_features(self, pixel_values: torch.FloatTensor | None = None, **kwargs):
        if self.default_featurizer:
            out = self.model.get_image_features(pixel_values=pixel_values, output_hidden_states=True, return_dict=True)

            return dict(features=out)
        else:
            ret = self.model.vision_model(
                pixel_values=pixel_values,
                output_attentions=None,
                output_hidden_states=True,
                return_dict=True,
            )  # this yields slightly better quality
            out = ret['pooler_output']

            return dict(features=out, hidden_states=ret['hidden_states'])

    def get_text_features(self, **prompt):
        if self.default_featurizer:
            return self.model.get_text_features(**prompt)
        else:
            text_outputs = self.model.text_model(**prompt)
            pooled_output = text_outputs[1]

            return pooled_output


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
    def __init__(self, weights_path: str, **kwargs):
        super().__init__(DISTS(weights_path), DISTSProcessor(), **kwargs)
        self.a = nn.Parameter(torch.ones(1), requires_grad=True)
        self.b = nn.Parameter(torch.zeros(1), requires_grad=True)

        for name, param in self.model.named_parameters():
            param.requires_grad = False

        self.model.alpha.requires_grad = True
        self.model.beta.requires_grad = True

    def encode_scores(
            self,
            prompt: Dict[str, torch.Tensor],
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
            image1 = image1.to(self.device).unsqueeze(0)
            image2 = image2.to(self.device).unsqueeze(0)

            return self.model(image1, image2).item()


class LPIPSDistanceMeasure(TrainablePairwiseMeasure):
    def __init__(self, network: str = 'alex', ft_dnn_only: bool = True, shift_tolerant: bool = False, **kwargs):
        if shift_tolerant:
            model = stlpips.LPIPS(net=network, variant='shift_tolerant')
            processor = STLPIPSProcessor()
        else:
            model = lpips.LPIPS(net=network)
            processor = LPIPSProcessor()

        super().__init__(model, processor, use_dynamo=False, **kwargs)
        self.a = nn.Parameter(torch.ones(1), requires_grad=True)
        self.b = nn.Parameter(torch.zeros(1), requires_grad=True)

        for name, param in self.model.named_parameters():
            param.requires_grad = 'lin' in name or not ft_dnn_only

    def encode_scores(
            self,
            prompt: Dict[str, torch.Tensor],
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
