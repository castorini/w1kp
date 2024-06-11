__all__ = ['PairwiseDistanceMeasure', 'LPIPSDistanceMeasure', 'ListwiseDistanceMeasure',
           'CLIPDistanceMeasure', 'ViTDistanceMeasure', 'DinoV2DistanceMeasure', 'GroupViTDistanceMeasure',
           'L2DistanceMeasure', 'SSIMDistanceMeasure', 'SD2DistanceMeasure', 'SSCDDistanceMeasure', 'DreamSimDistanceMeasure',
           'ListwiseDistanceMeasureServer', 'ListwiseDistanceMeasureClient']

import base64
import itertools
from io import BytesIO
from typing import List, Dict, Tuple

import PIL.Image
import lpips
import numpy as np
import torch
import zmq
from DISTS_pytorch import DISTS
from PIL.Image import Image
from diffusers import StableDiffusionPipeline
from dreamsim import dreamsim
from skimage.metrics import structural_similarity as ssim
from stlpips_pytorch import stlpips
from torch import nn
from torch.nn.functional import binary_cross_entropy_with_logits as bce_loss
from torcheval.metrics import FrechetInceptionDistance
from torchvision import transforms
from transformers import CLIPModel, CLIPProcessor, ViTImageProcessor, ViTModel, AutoImageProcessor, \
    Dinov2Model, AutoTokenizer, CLIPTokenizer, GroupViTModel, AutoProcessor


class ListwiseDistanceMeasure:
    def measure(self, prompt: str, images: List[Image], **kwargs) -> float:
        raise NotImplementedError


class ListwiseDistanceMeasureServer:
    def __init__(self, zero_mq_url: str, measure: ListwiseDistanceMeasure):
        self.zero_mq_url = zero_mq_url
        self.measure = measure

    def run(self):
        context = zmq.Context()
        socket = context.socket(zmq.REP)
        socket.bind(self.zero_mq_url)

        while True:
            message = socket.recv_json()
            prompt = message['prompt']
            reduce = message['reduce']
            images = []
            print('Received', prompt, len(message['images']))

            for image_string in message['images']:
                with PIL.Image.open(BytesIO(base64.b64decode(image_string))) as image:
                    images.append(image.copy())

            result = self.measure.measure(
                prompt,
                images,
                reduce=reduce,
                extract_pairwise_distances=message.get('extract_pairwise_distances', False)
            )
            response = dict(result=result)
            socket.send_json(response)


class ListwiseDistanceMeasureClient(ListwiseDistanceMeasure):
    def __init__(self, zero_mq_url: str):
        self.zero_mq_url = zero_mq_url

    def measure(self, prompt: str, images: List[Image], **kwargs) -> float | Dict[str, float]:
        context = zmq.Context()
        socket = context.socket(zmq.REQ)
        socket.connect(self.zero_mq_url)

        image_strings = []

        for image in images:
            buffer = BytesIO()
            image.save(buffer, format='PNG')
            image_str = base64.b64encode(buffer.getvalue()).decode('utf-8')
            image_strings.append(image_str)

        message = dict(prompt=prompt, images=image_strings, **kwargs)
        socket.send_json(message)
        response = socket.recv_json()
        socket.close()

        return response['result']


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

    def set_loss_type(self, loss_type: str):
        pass


class DeepPairwiseMeasure(nn.Module, PairwiseDistanceMeasure):
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

                dist1 = self.distance_function(features1, text_features, dim=-1).item()
                dist2 = self.distance_function(features2, text_features, dim=-1).item()
                dist3 = self.distance_function(features1, features2, dim=-1).item()

                D = (dist1 + dist2 + dist3) / 3
                D = dist3

                return D / 1.1  # normalize to [0, 1]

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
                # text_features = self.get_text_features(**prompt)  # disable for now

                # score1t = torch.nn.functional.cosine_similarity(features1, text_features, dim=-1)
                # scorert = torch.nn.functional.cosine_similarity(features_ref, text_features, dim=-1)
                score1r = torch.nn.functional.cosine_similarity(features1, features_ref, dim=-1)
                # score2t = torch.nn.functional.cosine_similarity(features2, text_features, dim=-1)
                score2r = torch.nn.functional.cosine_similarity(features2, features_ref, dim=-1)

                # scores1 = torch.stack([score1r, scorert, score1t], 0).mean(0).clamp(-0.1, 1)
                # scores2 = torch.stack([score2r, scorert, score2t], 0).mean(0).clamp(-0.1, 1)
                scores1 = torch.stack([score1r], 0).mean(0).clamp(-0.1, 1)
                scores2 = torch.stack([score2r], 0).mean(0).clamp(-0.1, 1)

                return self.a.exp() * (scores1 - scores2)
            else:
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
            return self.a.exp() * (scores1 - scores2)
        else:
            return scores1, scores2

    def get_trainable_parameters(self):
        return [p for p in self.parameters() if p.requires_grad]


class DinoV2DistanceMeasure(DeepPairwiseMeasure):
    def __init__(self, model: str = 'facebook/dinov2-large', **kwargs):
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

        self.a = nn.Parameter(torch.zeros(1), requires_grad=True)
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


class ViTDistanceMeasure(DeepPairwiseMeasure):
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

        self.a = nn.Parameter(torch.zeros(1), requires_grad=True)
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


class CLIPDistanceMeasure(DeepPairwiseMeasure):
    def __init__(
            self,
            model: str = 'openai/clip-vit-large-patch14-336', # 'openai/clip-vit-base-patch32',  # 'openai/clip-vit-large-patch14-336', # 'openai/clip-vit-base-patch32', # 'laion/CLIP-ViT-B-32-laion2B-s34B-b79K',
            default_featurizer: bool = False,
            use_text: bool = False,
            **kwargs
    ):
        # Apple models are missing preprocessor_config.json
        processor = 'laion/CLIP-ViT-H-14-laion2B-s32B-b79K' if 'apple/' in model else model
        super().__init__(
            CLIPModel.from_pretrained(model),
            CLIPProcessor.from_pretrained(processor),
            tokenizer=CLIPTokenizer.from_pretrained(model),
            use_text=use_text,
            **kwargs
        )

        if 'apple/' in model:
            # Update to match Apple's settings
            self.processor.image_processor.size = dict(shortest_edge=384)
            self.processor.image_processor.crop_size = dict(height=384, width=384)

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

        self.a = nn.Parameter(torch.zeros(1), requires_grad=True)
        self.a = self.model.logit_scale
        self.a.requires_grad = True

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


class GroupViTDistanceMeasure(DeepPairwiseMeasure):
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

        self.a = nn.Parameter(torch.zeros(1), requires_grad=True)
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
        images = [image.resize((384, 384)) for image in images]  # best on 384x384
        images = [torch.tensor(np.array(image)).permute(2, 0, 1).float() / 255 for image in images]
        # <-- Note the lack of normalization here compared to LPIPS
        images = [image.unsqueeze(0) for image in images]

        return dict(pixel_values=torch.cat(images))


class DISTSDistanceMeasure(DeepPairwiseMeasure):
    def __init__(self, weights_path: str, **kwargs):
        super().__init__(DISTS(weights_path), DISTSProcessor(), **kwargs)
        self.a = nn.Parameter(torch.zeros(1), requires_grad=True)
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


class L2DistanceMeasure(DeepPairwiseMeasure):
    def __init__(self, **kwargs):
        super().__init__(nn.Module(), DISTSProcessor(), **kwargs)

    def measure(self, prompt: str, image1: Image, image2: Image) -> float:
        with torch.no_grad():
            images = self.processor([image1, image2])['pixel_values']
            image1, image2 = images
            image1 = image1.to(self.device).unsqueeze(0)
            image2 = image2.to(self.device).unsqueeze(0)

            return (image1 - image2).norm(p=2).item()


class SSCDProcessor:
    def __call__(self, images: List[Image], **kwargs) -> Dict[str, torch.Tensor]:
        normalize = transforms.Normalize(
            mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225],
        )

        small_288 = transforms.Compose([
            transforms.Resize(288),
            transforms.ToTensor(),
            normalize,
        ])

        images = [small_288(image) for image in images]
        images = [image.unsqueeze(0) for image in images]

        return dict(pixel_values=torch.cat(images))


class ListwiseDreamSimDistanceMeasure(ListwiseDistanceMeasure):
    def __init__(self, measure: 'DreamSimDistanceMeasure' = None):
        if measure is None:
            measure = DreamSimDistanceMeasure()

        self.measure_ = measure

    def measure(self, _, images: List[Image], reduce: str = 'mean', **kwargs) -> float | Dict[str, float]:
        """
        Args:
            reduce (str): One of 'mean', 'min', 'u5', 'all', 'reusability'. U5 expected minimum distance among
            5 images, estimated by subsampling for a U-statistic estimator. If 'all' is specified, a dictionary
            containing all three metrics is returned. If 'reusability' is specified, the u5, u10, u25, u50, u100, u150,
            and u200 metrics are computed and returned along with the median minimum indices, which may take a while.
        """
        with torch.no_grad():
            images = [self.measure_.preprocess(image).cuda() for image in images]
            image_tensor = torch.cat(images)
            embeds = self.measure_.model.embed(image_tensor)
            pairwise_l2 = torch.cdist(embeds, embeds, p=2).pow(2)
            pairwise_l2_nonzero = pairwise_l2[torch.where(pairwise_l2 > 1e-4)]  # remove self-distances

            ret = {}

            if kwargs.get('extract_pairwise_distances'):
                ret['pairwise_l2'] = pairwise_l2.cpu().numpy().tolist()

            if reduce == 'mean' or reduce == 'all':
                ret['mean'] = pairwise_l2_nonzero.mean().item()

            if reduce == 'min' or reduce == 'all':
                ret['min'] = pairwise_l2_nonzero.min().item()

            pairwise_l2.fill_diagonal_(1e7)  # set diagonal to a large value to avoid self-distances

            if reduce == 'u5' or reduce == 'all' or reduce == 'reusability':
                sizes = [2, 3, 5, 7, 10, 15, 20, 25, 30, 40, 50, 60, 75, 100, 125, 150, 200, 250, 300] if reduce == 'reusability' else [3, 5]

                for size in sizes:
                    dists = []
                    min_indices = []

                    for _ in range(1000):
                        perm = torch.randperm(len(images))
                        perm_map = {i: j.item() for i, j in enumerate(perm)}
                        perm_size = min(size, len(images))
                        idx = perm[:perm_size]

                        pl2 = pairwise_l2[idx][:, idx]
                        dists.append(pl2.min().item())
                        avg_min_idx = pl2.argmin().item()
                        min_indices.append((pl2.min().item(), (perm_map[avg_min_idx // perm_size], perm_map[avg_min_idx % perm_size])))

                    min_indices.sort(key=lambda x: x[0])
                    avg_min_idx = min_indices[len(min_indices) // 2][1]
                    ret[f'u{size}_avg_min_idx'] = avg_min_idx
                    ret[f'u{size}'] = float(np.mean(dists))

            return ret if reduce == 'all' or reduce == 'reusability' else ret[reduce]


class DreamSimDistanceMeasure(DeepPairwiseMeasure):
    def __init__(self, **kwargs):
        model, self.preprocess = dreamsim(pretrained=True)
        super().__init__(model, None, use_dynamo=False, **kwargs)
        self.distance_type = 'l2'

    def to_listwise(self, num_sample: int = None) -> ListwiseDreamSimDistanceMeasure:
        return ListwiseDreamSimDistanceMeasure(self)

    def measure(self, prompt: str, image1: Image, image2: Image) -> float:
        with torch.no_grad():
            im1 = self.preprocess(image1).cuda()
            im2 = self.preprocess(image2).cuda()

            if self.distance_type == 'l2':
                embed_a = self.model.embed(im1)
                embed_b = self.model.embed(im2)

                return (embed_a - embed_b).norm(p=2).item() ** 2
            else:
                return self.model(im1, im2)  # this is slightly worse for some reason


class SSCDWrapper(nn.Module):
    def __init__(self, weights_path: str):
        super().__init__()
        self.model = torch.jit.load(weights_path)

    def forward(self, pixel_values) -> torch.Tensor:
        return self.model(pixel_values)


class SSCDDistanceMeasure(DeepPairwiseMeasure):
    def __init__(self, weights_path: str, **kwargs):
        super().__init__(SSCDWrapper(weights_path), SSCDProcessor(), use_dynamo=False, **kwargs)

    def get_image_features(self, pixel_values: torch.FloatTensor | None = None, **kwargs):
        return dict(features=self.model(pixel_values))


class SD2DistanceMeasure(PairwiseDistanceMeasure):
    def __init__(self, **kwargs):
        model_id = 'stabilityai/stable-diffusion-2-1'
        self.pipe = StableDiffusionPipeline.from_pretrained(model_id, torch_dtype=torch.float16)
        self.pipe.to('cuda')
        self.processor = self.pipe.image_processor
        self.vae = self.pipe.vae
        self.unet = self.pipe.unet
        self.vae_scale_factor = self.pipe.vae_scale_factor
        self.scheduler = self.pipe.scheduler

    def _encode_vae_image(self, image: Image) -> torch.Tensor:
        height = self.unet.config.sample_size * self.vae_scale_factor
        width = self.unet.config.sample_size * self.vae_scale_factor
        image = self.processor.preprocess(image, height=height, width=width)
        image = image.to(dtype=torch.float32).cuda()
        needs_upcasting = self.vae.dtype == torch.float16 and self.vae.config.force_upcast

        if needs_upcasting:
            self.vae.to(dtype=torch.float32)
            image = image.to(next(iter(self.vae.post_quant_conv.parameters())).dtype)

        image_latents = self.vae.encode(image).latent_dist.sample()
        image_latents = self.vae.config.scaling_factor * image_latents

        # cast back to fp16 if needed
        if needs_upcasting:
            self.vae.to(dtype=torch.float16)

        return image_latents.half() / self.scheduler.init_noise_sigma  # unscale because we don't care about noise

    def measure(self, prompt: str, image1: Image, image2: Image) -> float:
        from daam import trace, cached_nlp

        with trace(self.pipe) as tc:
            latents1 = self._encode_vae_image(image1).half()
            self.pipe(prompt, latents=latents1, num_inference_steps=1)
            heat_map1 = tc.compute_global_heat_map(prompt=prompt)

        with trace(self.pipe) as tc:
            latents2 = self._encode_vae_image(image2).half()
            self.pipe(prompt, latents=latents2, num_inference_steps=1)
            heat_map2 = tc.compute_global_heat_map(prompt=prompt)

        diffs = []

        for token in cached_nlp(prompt):
            # Discard punctuation and determiners
            if token.is_punct or token.is_stop:
                continue

            try:
                hm1 = heat_map1.compute_word_heat_map(token.text).heatmap.cuda()
                hm2 = heat_map2.compute_word_heat_map(token.text).heatmap.cuda()
                diffs.append((hm1 - hm2).norm(p=1).item())
            except ValueError:
                pass

        return np.mean(diffs)


class SSIMProcessor:
    def __call__(self, images: List[Image], **kwargs) -> Dict[str, torch.Tensor]:
        images = [image.resize((384, 384)) for image in images]  # best on 384x384
        images = [torch.tensor(np.array(image)).permute(2, 0, 1) for image in images]
        images = [image.unsqueeze(0) for image in images]

        return dict(pixel_values=torch.cat(images))


class SSIMDistanceMeasure(DeepPairwiseMeasure):
    def __init__(self, **kwargs):
        super().__init__(nn.Module(), SSIMProcessor(), **kwargs)

    def measure(self, prompt: str, image1: Image, image2: Image) -> float:
        with torch.no_grad():
            images = self.processor([image1, image2])['pixel_values']
            image1, image2 = images

            return -ssim(image1.cpu().numpy(), image2.cpu().numpy(), channel_axis=0)


class LPIPSDistanceMeasure(DeepPairwiseMeasure):
    def __init__(
            self,
            network: str = 'alex',
            ft_dnn_only: bool = True,
            shift_tolerant: bool = False,
            pretrained: bool = True,
            **kwargs
    ):
        if shift_tolerant:
            model = stlpips.LPIPS(net=network, variant='shift_tolerant', pretrained=pretrained)
            processor = STLPIPSProcessor()
        else:
            model = lpips.LPIPS(net=network, pretrained=pretrained)
            processor = LPIPSProcessor()

        super().__init__(model, processor, use_dynamo=False, **kwargs)
        self.a = nn.Parameter(torch.zeros(1), requires_grad=True)
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
