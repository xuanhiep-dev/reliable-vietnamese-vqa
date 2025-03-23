from timm.models.registry import register_model
from timm.models.layers import trunc_normal_ as __call_trunc_normal_
from torchscale.component.multiway_network import MutliwayEmbedding
from torchscale.component.embedding import PositionalEmbedding
from torchscale.architecture.encoder import Encoder
from torchscale.architecture.config import EncoderConfig
import torch.nn as nn
import torch.nn.functional as F
import torch
import math
from transformers.utils.generic import ModelOutput
from dataclasses import dataclass
from typing import Optional
from models.language_embedding import BartPhoExtractor
from models.vision_embedding import Blip2EfficientExtractor
from models.selective_models import SelectivePredictor


@dataclass
class ViVQAOutput(ModelOutput):
    loss: Optional[torch.FloatTensor] = None
    logits: torch.FloatTensor = None


def trunc_normal_(tensor, mean=0., std=1.):
    __call_trunc_normal_(tensor, mean=mean, std=std, a=-std, b=std)


def _get_base_config(**kwargs):
    return EncoderConfig(
        multiway=True,
        layernorm_embedding=False,
        normalize_output=True,
        no_output_layer=True,
        drop_path_rate=kwargs["drop_path_rate"],
        encoder_embed_dim=kwargs["encoder_embed_dim"],
        encoder_attention_heads=kwargs["encoder_attention_heads_layers"],
        encoder_ffn_embed_dim=int(
            kwargs["encoder_embed_dim"] * kwargs["mlp_ratio"]),
        encoder_layers=kwargs["encoder_layers"],
    )


class Pooler(nn.Module):
    def __init__(self, input_features, output_features, norm_layer):
        super().__init__()
        self.norm = norm_layer(input_features)
        self.dense = nn.Linear(input_features, output_features)
        self.activation = nn.Tanh()

    def forward(self, x):
        cls_rep = x[:, 0, :]
        cls_rep = self.norm(cls_rep)
        pooled_output = self.dense(cls_rep)
        pooled_output = self.activation(pooled_output)
        return pooled_output


class ViVQABEiT3(nn.Module):
    def __init__(self, args, **kwargs):
        super().__init__()
        self.args = args
        assert args.multiway
        assert not args.share_encoder_input_output_embed

        self.text_embed = BartPhoExtractor()
        self.vision_embed = Blip2EfficientExtractor()
        for param in self.vision_embed.parameters():
            param.requires_grad = False

        self.linear = nn.Linear(1024, 768)

        # being consistent with Fairseq, which starts from 2 for position embedding
        num_position_embeddings = 64
        embed_positions = MutliwayEmbedding(
            modules=[
                PositionalEmbedding(
                    num_position_embeddings + 2, args.encoder_embed_dim),
                PositionalEmbedding(args.max_source_positions,
                                    args.encoder_embed_dim),
            ],
            dim=1,
        )
        self.encoder = Encoder(
            args,
            embed_tokens=None,
            embed_positions=embed_positions,
            output_projection=None,
            is_encoder_decoder=False,
        )

    def forward(self, textual_tokens, visual_tokens, text_padding_position):
        x1 = self.vision_embed(visual_tokens)
        multiway_split_position = x1.size(1)

        attention_mask = 1 - text_padding_position
        x2 = self.text_embed(textual_tokens, attention_mask)
        x2 = self.linear(x2)

        x = torch.cat([x1, x2], dim=1)

        encoder_padding_mask = torch.cat(
            [
                torch.zeros(x1.shape[:-1]).to(x1.device).bool(),
                text_padding_position,
            ],
            dim=1,
        )

        encoder_out = self.encoder(
            src_tokens=None,
            encoder_padding_mask=encoder_padding_mask,
            token_embeddings=x,
            multiway_split_position=multiway_split_position
        )
        encoder_out["multiway_split_position"] = multiway_split_position

        return encoder_out


class BEiT3Wrapper(nn.Module):
    def __init__(self, args, **kwargs):
        super().__init__()
        self.args = args
        self.beit3 = ViVQABEiT3(args, **kwargs)
        # self.apply(self._init_weights)

    def fix_init_weight(self):
        def rescale(param, layer_id):
            param.div_(math.sqrt(2.0 * layer_id))

        for layer_id, layer in enumerate(self.blocks):
            rescale(layer.attn.proj.weight.data, layer_id + 1)
            rescale(layer.mlp.fc2.weight.data, layer_id + 1)

    def get_num_layers(self):
        return self.beit3.encoder.num_layers

    @torch.jit.ignore
    def no_weight_decay(self):
        return {'pos_embed', 'cls_token', 'beit3.encoder.embed_positions.A.weight', 'beit3.vision_embed.cls_token', 'logit_scale'}

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def compute_loss(self, logits, labels=None):
        loss = None
        if labels is not None:
            loss = F.cross_entropy(logits, labels)

        return ViVQAOutput(loss=loss, logits=logits)


class BEiT3ForVietnameseVisualQuestionAnswering(BEiT3Wrapper):
    def __init__(self, args,
                 norm_layer=nn.LayerNorm,
                 use_selector=False,
                 **kwargs):
        super(BEiT3ForVietnameseVisualQuestionAnswering,
              self).__init__(args=args, **kwargs)

        self.use_selector = use_selector
        embed_dim = args.encoder_embed_dim
        self.pooler = Pooler(
            input_features=embed_dim,
            output_features=embed_dim,
            norm_layer=norm_layer,
        )
        self.pooler.apply(self._init_weights)
        self.head = nn.Sequential(
            nn.Linear(embed_dim, embed_dim * 2),
            norm_layer(embed_dim * 2),
            nn.GELU(),
            nn.Linear(embed_dim * 2, kwargs["classes"]),
        )
        self.head.apply(self._init_weights)

    def forward(self, image, question, padding_mask, labels=None):
        question = question.squeeze(dim=1)
        padding_mask = padding_mask.squeeze(dim=1)

        self._image = image
        self._question = question
        self._padding_mask = padding_mask

        outputs = self.beit3(
            textual_tokens=question,
            visual_tokens=image,
            text_padding_position=padding_mask
        )
        x = outputs["encoder_out"]
        cls_rep = self.pooler(x)
        logits = self.head(cls_rep)

        if not self.use_selector:
            return self.compute_loss(logits, labels)
        return logits


class ViVQABEiT3Selective(BEiT3ForVietnameseVisualQuestionAnswering):
    def __init__(self, args, norm_layer=nn.LayerNorm, **kwargs):
        super().__init__(args, norm_layer=norm_layer, **kwargs)
        self._init_selector()

    def _init_selector(self, **kwargs):
        self.selector = SelectivePredictor(
            kwargs["type"], **kwargs["params"])

    def get_optimizer_parameters(self, **kwargs):
        params = []

        selector_params = {"params": list(self.selector.parameters())}

        if not kwargs["freeze_vqa"]:
            params.append(
                {"params": list(set(self.parameters()) - set(self.selector.parameters()))})
        else:
            params.append(selector_params)

        if "sel_lr" in kwargs:
            selector_params["lr"] = kwargs["sel_lr"]

        return params

    def forward(self, image, question, padding_mask, labels=None, **kwargs):
        outputs = super().forward(image, question, padding_mask, labels, **kwargs)

        encoder_out = outputs.encoder_out
        multiway_split_position = encoder_out["multiway_split_position"]
        multimodal_emb = encoder_out["encoder_out"]
        image_emb = encoder_out["encoder_out"][:,
                                               :multiway_split_position, :]
        text_emb = encoder_out["encoder_out"][:,
                                              multiway_split_position:, :]

        selector_output = self.selector(
            outputs.detach(),
            image_emb.detach(),
            text_emb.detach(),
            multimodal_emb.detach(),
        )
        logits = selector_output["confidences"]

        return self.compute_loss(logits, labels)


@register_model
def avivqa_model(pretrained=False, **kwargs):
    args = _get_base_config(**kwargs["vqa"])

    use_selector = kwargs["use_selector"]
    if use_selector:
        model = ViVQABEiT3Selective(args, **kwargs["selector"])
    else:
        model = BEiT3ForVietnameseVisualQuestionAnswering(
            args, use_selector=use_selector, **kwargs["vqa"])

    return model
