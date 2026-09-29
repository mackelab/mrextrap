"""3D UNets used by MRExtrap: the diffusion denoiser and the single-step mean/variance prior.

Ported verbatim from the research code (``models/diffusion.py``): submodule names, attribute names
and defaults are unchanged so released weights load without key changes. Only code unreachable
from ``UNetModel`` / ``UNetModelMeanVarSingleStep`` was dropped. Based on the latent-diffusion
UNet (https://link.springer.com/chapter/10.1007/978-3-031-18576-2_12).
"""

import math
from abc import abstractmethod
from dataclasses import dataclass
from inspect import isfunction

import torch
import torch as th
import torch.nn.functional as F
from diffusers.configuration_utils import ConfigMixin, register_to_config
from diffusers.models.modeling_utils import ModelMixin
from diffusers.utils import BaseOutput
from einops import einsum, rearrange, repeat
from torch import nn


def exists(val):
    return val is not None


def default(val, d):
    if exists(val):
        return val
    return d() if isfunction(d) else d


# feedforward
class GEGLU(nn.Module):
    def __init__(self, dim_in, dim_out):
        super().__init__()
        self.proj = nn.Linear(dim_in, dim_out * 2)

    def forward(self, x):
        x, gate = self.proj(x).chunk(2, dim=-1)
        return x * F.gelu(gate)


class FeedForward(nn.Module):
    def __init__(self, dim, dim_out=None, mult=4, glu=False, dropout=0.0):
        super().__init__()
        inner_dim = int(dim * mult)
        dim_out = default(dim_out, dim)
        project_in = (
            nn.Sequential(nn.Linear(dim, inner_dim), nn.GELU())
            if not glu
            else GEGLU(dim, inner_dim)
        )

        self.net = nn.Sequential(project_in, nn.Dropout(dropout), nn.Linear(inner_dim, dim_out))

    def forward(self, x):
        return self.net(x)


def zero_module(module):
    """
    Zero out the parameters of a module and return it.
    """
    for p in module.parameters():
        p.detach().zero_()
    return module


def Normalize(in_channels):
    return torch.nn.GroupNorm(num_groups=32, num_channels=in_channels, eps=1e-6, affine=True)


class CrossAttention(nn.Module):
    def __init__(self, query_dim, context_dim=None, heads=8, dim_head=64, dropout=0.0):
        super().__init__()
        inner_dim = dim_head * heads
        context_dim = default(context_dim, query_dim)

        self.scale = dim_head**-0.5
        self.heads = heads

        self.to_q = nn.Linear(query_dim, inner_dim, bias=False)
        self.to_k = nn.Linear(context_dim, inner_dim, bias=False)
        self.to_v = nn.Linear(context_dim, inner_dim, bias=False)

        self.to_out = nn.Sequential(nn.Linear(inner_dim, query_dim), nn.Dropout(dropout))

    def forward(self, x, context=None, mask=None):
        h = self.heads

        q = self.to_q(x)
        context = default(context, x)
        k = self.to_k(context)
        v = self.to_v(context)

        q, k, v = (rearrange(t, "b n (h d) -> (b h) n d", h=h) for t in (q, k, v))

        sim = einsum(q, k, "b i d, b j d -> b i j") * self.scale

        if exists(mask):
            mask = rearrange(mask, "b ... -> b (...)")
            max_neg_value = -torch.finfo(sim.dtype).max
            mask = repeat(mask, "b j -> (b h) () j", h=h)
            sim.masked_fill_(~mask, max_neg_value)

        # attention, what we cannot get enough of
        attn = sim.softmax(dim=-1)

        out = einsum(attn, v, "b i j, b j d -> b i d")
        out = rearrange(out, "(b h) n d -> b n (h d)", h=h)
        return self.to_out(out)


class BasicTransformerBlock(nn.Module):
    def __init__(self, dim, n_heads, d_head, dropout=0.0, context_dim=None, gated_ff=True):
        super().__init__()
        self.attn1 = CrossAttention(
            query_dim=dim, heads=n_heads, dim_head=d_head, dropout=dropout
        )  # is a self-attention
        self.ff = FeedForward(dim, dropout=dropout, glu=gated_ff)
        self.attn2 = CrossAttention(
            query_dim=dim,
            context_dim=context_dim,
            heads=n_heads,
            dim_head=d_head,
            dropout=dropout,
        )  # is self-attn if context is none
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)
        self.norm3 = nn.LayerNorm(dim)

    def forward(self, x, context=None):
        x = self.attn1(self.norm1(x)) + x
        x = self.attn2(self.norm2(x), context=context) + x
        x = self.ff(self.norm3(x)) + x
        return x


class SpatialTransformer(nn.Module):
    """
    Transformer block for image-like data.
    First, project the input (aka embedding)
    and reshape to b, t, d.
    Then apply standard transformer action.
    Finally, reshape to image
    """

    def __init__(self, in_channels, n_heads, d_head, depth=1, dropout=0.0, context_dim=None):
        super().__init__()
        self.in_channels = in_channels
        inner_dim = n_heads * d_head
        self.norm = Normalize(in_channels)

        self.proj_in = nn.Conv3d(in_channels, inner_dim, kernel_size=1, stride=1, padding=0)

        self.transformer_blocks = nn.ModuleList(
            [
                BasicTransformerBlock(
                    inner_dim, n_heads, d_head, dropout=dropout, context_dim=context_dim
                )
                for d in range(depth)
            ]
        )

        self.proj_out = zero_module(
            nn.Conv3d(inner_dim, in_channels, kernel_size=1, stride=1, padding=0)
        )

    def forward(self, x, context=None):
        # note: if no context is given, cross-attention defaults to self-attention
        _b, _c, h, w, d = x.shape
        x_in = x
        x = self.norm(x)
        x = self.proj_in(x)
        x = rearrange(x, "b c h w d -> b (h w d) c")
        for block in self.transformer_blocks:
            x = block(x, context=context)
        x = rearrange(x, "b (h w d) c -> b c h w d", h=h, w=w, d=d)
        x = self.proj_out(x)
        return x + x_in


def timestep_embedding(timesteps, dim, max_period=10000, repeat_only=False):
    """
    Create sinusoidal timestep embeddings.
    :param timesteps: a 1-D Tensor of N indices, one per batch element.
                      These may be fractional.
    :param dim: the dimension of the output.
    :param max_period: controls the minimum frequency of the embeddings.
    :return: an [N x dim] Tensor of positional embeddings.
    """
    if not repeat_only:
        half = dim // 2
        freqs = torch.exp(
            -math.log(max_period) * torch.arange(start=0, end=half, dtype=torch.float32) / half
        ).to(device=timesteps.device)
        args = timesteps[:, None].float() * freqs[None]
        embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        if dim % 2:
            embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
    else:
        embedding = repeat(timesteps, "b -> b d", d=dim)
    return embedding


class TimestepBlock(nn.Module):
    @abstractmethod
    def forward(self, x, emb):
        """
        Apply the module to `x` given `emb` timestep embeddings.
        """


class TimestepEmbedSequential(nn.Sequential, TimestepBlock):
    """
    A sequential module that passes timestep embeddings to the children that
    support it as an extra input.
    """

    def forward(self, x, emb, context=None):
        for layer in self:
            if isinstance(layer, TimestepBlock):
                x = layer(x, emb)
            elif isinstance(layer, SpatialTransformer):
                x = layer(x, context)
            else:
                x = layer(x)
        return x


class QKVAttentionLegacy(nn.Module):
    """
    A module which performs QKV attention. Matches legacy QKVAttention + input/ouput heads shaping
    """

    def __init__(self, n_heads):
        super().__init__()
        self.n_heads = n_heads

    def forward(self, qkv):
        """
        Apply QKV attention.
        :param qkv: an [N x (H * 3 * C) x T] tensor of Qs, Ks, and Vs.
        :return: an [N x (H * C) x T] tensor after attention.
        """
        bs, width, length = qkv.shape
        assert width % (3 * self.n_heads) == 0
        ch = width // (3 * self.n_heads)
        q, k, v = qkv.reshape(bs * self.n_heads, ch * 3, length).split(ch, dim=1)
        scale = 1 / math.sqrt(math.sqrt(ch))
        weight = th.einsum(
            "bct,bcs->bts", q * scale, k * scale
        )  # More stable with f16 than dividing afterwards
        weight = th.softmax(weight.float(), dim=-1).type(weight.dtype)
        a = th.einsum("bts,bcs->bct", weight, v)
        return a.reshape(bs, -1, length)


class AttentionBlock(nn.Module):
    """
    An attention block that allows spatial positions to attend to each other.
    Originally ported from here, but adapted to the N-d case.
    https://github.com/hojonathanho/diffusion/blob/1e0dceb3b3495bbe19116a5e1b3596cd0706c543/diffusion_tf/models/unet.py#L66.
    """

    def __init__(
        self,
        channels,
        num_heads=1,
        num_head_channels=-1,
        use_checkpoint=False,
    ):
        super().__init__()
        self.channels = channels
        if num_head_channels == -1:
            self.num_heads = num_heads
        else:
            assert channels % num_head_channels == 0, (
                f"q,k,v channels {channels} is not divisible by num_head_channels {num_head_channels}"
            )
            self.num_heads = channels // num_head_channels
        self.use_checkpoint = use_checkpoint
        self.norm = Normalize(channels)
        self.qkv = nn.Conv1d(channels, channels * 3, 1)
        self.attention = QKVAttentionLegacy(self.num_heads)

        self.proj_out = zero_module(nn.Conv1d(channels, channels, 1))

    def forward(self, x):
        return self._forward(
            x,
        )

    def _forward(self, x):
        b, c, *spatial = x.shape
        x = x.reshape(b, c, -1)
        qkv = self.qkv(self.norm(x))
        h = self.attention(qkv)
        h = self.proj_out(h)
        return (x + h).reshape(b, c, *spatial)


class Downsample(nn.Module):
    """
    A downsampling layer with an optional convolution.

    :param channels: channels in the inputs and outputs.
    :param use_conv: a bool determining if a convolution is applied.
    """

    def __init__(self, channels, use_conv, out_channels=None, padding=1):
        super().__init__()
        self.channels = channels
        self.out_channels = out_channels or channels
        self.use_conv = use_conv
        if use_conv:
            self.op = nn.Conv3d(self.channels, self.out_channels, 3, stride=2, padding=padding)
        else:
            assert self.channels == self.out_channels
            self.op = nn.AvgPool3d(kernel_size=2, stride=2)

    def forward(self, x):
        assert x.shape[1] == self.channels
        return self.op(x)


class Upsample(nn.Module):
    """
    An upsampling layer with an optional convolution.
    :param channels: channels in the inputs and outputs.
    :param use_conv: a bool determining if a convolution is applied.
    """

    def __init__(self, channels, use_conv, out_channels=None, padding=1):
        super().__init__()
        self.channels = channels
        self.out_channels = out_channels or channels
        self.use_conv = use_conv
        if use_conv:
            self.conv = nn.Conv3d(self.channels, self.out_channels, 3, padding=padding)

    def forward(self, x):
        assert x.shape[1] == self.channels
        x = F.interpolate(x, scale_factor=2, mode="nearest")
        if self.use_conv:
            x = self.conv(x)
        return x


class ResBlock(TimestepBlock):
    """
    A residual block that can optionally change the number of channels.
    :param channels: the number of input channels.
    :param emb_channels: the number of timestep embedding channels.
    :param dropout: the rate of dropout.
    :param out_channels: if specified, the number of out channels.
    :param use_conv: if True and out_channels is specified, use a spatial
        convolution instead of a smaller 1x1 convolution to change the
        channels in the skip connection.
    :param up: if True, use this block for upsampling.
    :param down: if True, use this block for downsampling.
    """

    def __init__(
        self,
        channels,
        emb_channels,
        dropout,
        out_channels=None,
        use_conv=False,
        use_scale_shift_norm=False,
        up=False,
        down=False,
    ):
        super().__init__()
        self.channels = channels
        self.emb_channels = emb_channels
        self.dropout = dropout
        self.out_channels = out_channels or channels
        self.use_conv = use_conv
        self.use_scale_shift_norm = use_scale_shift_norm

        self.in_layers = nn.Sequential(
            Normalize(channels),
            nn.SiLU(),
            nn.Conv3d(channels, self.out_channels, 3, padding=1),
        )

        self.updown = up or down

        if up:
            self.h_upd = Upsample(channels, False)
            self.x_upd = Upsample(channels, False)
        elif down:
            self.h_upd = Downsample(channels, False)
            self.x_upd = Downsample(channels, False)
        else:
            self.h_upd = self.x_upd = nn.Identity()

        self.emb_layers = nn.Sequential(
            nn.SiLU(),
            nn.Linear(
                emb_channels,
                2 * self.out_channels if use_scale_shift_norm else self.out_channels,
            ),
        )
        self.out_layers = nn.Sequential(
            Normalize(self.out_channels),
            nn.SiLU(),
            nn.Dropout(p=dropout),
            zero_module(nn.Conv3d(self.out_channels, self.out_channels, 3, padding=1)),
        )

        if self.out_channels == channels:
            self.skip_connection = nn.Identity()
        elif use_conv:
            self.skip_connection = nn.Conv3d(channels, self.out_channels, 3, padding=1)
        else:
            self.skip_connection = nn.Conv3d(channels, self.out_channels, 1)

    def forward(self, x, emb):
        return self._forward(x, emb)

    def _forward(self, x, emb):
        if self.updown:
            in_rest, in_conv = self.in_layers[:-1], self.in_layers[-1]
            h = in_rest(x)
            h = self.h_upd(h)
            x = self.x_upd(x)
            h = in_conv(h)
        else:
            h = self.in_layers(x)
        emb_out = self.emb_layers(emb).type(h.dtype)
        while len(emb_out.shape) < len(h.shape):
            emb_out = emb_out[..., None]
        if self.use_scale_shift_norm:
            out_norm, out_rest = self.out_layers[0], self.out_layers[1:]
            scale, shift = th.chunk(emb_out, 2, dim=1)
            h = out_norm(h) * (1 + scale) + shift
            h = out_rest(h)
        else:
            h = h + emb_out
            h = self.out_layers(h)
        return self.skip_connection(x) + h


class UNetModel(ModelMixin, ConfigMixin):
    @register_to_config
    def __init__(
        self,
        image_size,
        in_channels,
        model_channels,
        out_channels,
        num_res_blocks,
        attention_resolutions,
        dropout=0,
        channel_mult=(1, 2, 4, 8),
        conv_resample=True,
        num_classes=None,
        num_heads=1,
        num_head_channels=-1,
        num_heads_upsample=-1,
        use_scale_shift_norm=False,
        resblock_updown=False,
        use_spatial_transformer=False,  # custom transformer support
        transformer_depth=1,  # custom transformer support
        context_dim=None,  # custom transformer support
        n_embed=None,  # custom support for prediction of discrete ids into codebook of first stage vq model
        concat_emb=False,  # if true, concat age embedding instead of adding it
        emb_bias=True,
        single_step=False,  # if true, this model is used for single step prediction
    ):
        super().__init__()

        if use_spatial_transformer:
            assert context_dim is not None, (
                "Fool!! You forgot to include the dimension of your cross-attention conditioning..."
            )

        if context_dim is not None:
            assert use_spatial_transformer, (
                "Fool!! You forgot to use the spatial transformer for your cross-attention conditioning..."
            )

        if num_heads_upsample == -1:
            num_heads_upsample = num_heads

        self.image_size = image_size
        self.sample_size = image_size[1:]  # for diffusers pipeline
        self.in_channels = in_channels
        self.model_channels = model_channels
        self.out_channels = out_channels
        self.num_res_blocks = num_res_blocks
        self.attention_resolutions = attention_resolutions
        self.dropout = dropout
        self.channel_mult = channel_mult
        self.conv_resample = conv_resample
        self.num_classes = num_classes
        self.num_heads = num_heads
        self.num_head_channels = num_head_channels
        self.num_heads_upsample = num_heads_upsample
        self.predict_codebook_ids = n_embed is not None
        self.single_step = single_step

        time_embed_dim = model_channels * 4
        # NOTE(paper): time_embed stays None in single-step mode (no timestep embedding, no keys).
        # Kept as an attribute for multi-gpu training, so that all params get backprop'd.
        self.time_embed = None
        if not self.single_step:
            self.time_embed = nn.Sequential(
                nn.Linear(model_channels, time_embed_dim),
                nn.SiLU(),
                nn.Linear(time_embed_dim, time_embed_dim),
            )

        if self.num_classes is not None:
            # disease label in one-hot form, the linear layer is used to project the one-hot label
            # to the same dimension as the time embedding
            # for a label of the form [0,0,...0] (all zeros), the label embedding
            # is all zeros as well (no information) because of zero bias
            # so unconditional generation is possible and the unconditional model
            # can simply be finetuned as the disease-conditional model
            self.label_emb = nn.Linear(
                num_classes, time_embed_dim, bias=(emb_bias)
            )  # for age, use bias

        self.input_blocks = nn.ModuleList(
            [TimestepEmbedSequential(nn.Conv3d(in_channels, model_channels, 3, padding=1))]
        )
        self._feature_size = model_channels
        input_block_chans = [model_channels]
        ch = model_channels
        ds = 1

        embed_dim = time_embed_dim if not single_step else 0
        if single_step:
            embed_dim = time_embed_dim
        else:
            embed_dim += (time_embed_dim) if concat_emb else 0
        self.embed_dim = embed_dim
        self.concat_emb = concat_emb

        for level, mult in enumerate(channel_mult):
            for _ in range(num_res_blocks):
                layers = [
                    ResBlock(
                        ch,
                        embed_dim,
                        dropout,
                        out_channels=mult * model_channels,
                        use_scale_shift_norm=use_scale_shift_norm,
                    )
                ]
                ch = mult * model_channels
                if ds in attention_resolutions:
                    dim_head = ch // num_heads
                    layers.append(
                        AttentionBlock(
                            ch,
                            num_heads=num_heads,
                            num_head_channels=num_head_channels,
                        )
                        if not use_spatial_transformer
                        else SpatialTransformer(
                            ch,
                            num_heads,
                            dim_head,
                            depth=transformer_depth,
                            context_dim=context_dim,
                        )
                    )
                self.input_blocks.append(TimestepEmbedSequential(*layers))
                self._feature_size += ch
                input_block_chans.append(ch)
            if level != len(channel_mult) - 1:
                out_ch = ch
                self.input_blocks.append(
                    TimestepEmbedSequential(
                        ResBlock(
                            ch,
                            embed_dim,
                            dropout,
                            out_channels=out_ch,
                            use_scale_shift_norm=use_scale_shift_norm,
                            down=True,
                        )
                        if resblock_updown
                        else Downsample(ch, conv_resample, out_channels=out_ch)
                    )
                )
                ch = out_ch
                input_block_chans.append(ch)
                ds *= 2
                self._feature_size += ch

        dim_head = ch // num_heads
        self.middle_block = TimestepEmbedSequential(
            ResBlock(
                ch,
                embed_dim,
                dropout,
                use_scale_shift_norm=use_scale_shift_norm,
            ),
            (
                AttentionBlock(
                    ch,
                    num_heads=num_heads,
                    num_head_channels=num_head_channels,
                )
                if not use_spatial_transformer
                else SpatialTransformer(
                    ch,
                    num_heads,
                    dim_head,
                    depth=transformer_depth,
                    context_dim=context_dim,
                )
            ),
            ResBlock(
                ch,
                embed_dim,
                dropout,
                use_scale_shift_norm=use_scale_shift_norm,
            ),
        )
        self._feature_size += ch

        self.output_blocks = nn.ModuleList([])
        for level, mult in list(enumerate(channel_mult))[::-1]:
            for i in range(num_res_blocks + 1):
                ich = input_block_chans.pop()
                layers = [
                    ResBlock(
                        ch + ich,
                        embed_dim,
                        dropout,
                        out_channels=model_channels * mult,
                        use_scale_shift_norm=use_scale_shift_norm,
                    )
                ]
                ch = model_channels * mult
                if ds in attention_resolutions:
                    dim_head = ch // num_heads
                    layers.append(
                        AttentionBlock(
                            ch,
                            num_heads=num_heads_upsample,
                            num_head_channels=num_head_channels,
                        )
                        if not use_spatial_transformer
                        else SpatialTransformer(
                            ch,
                            num_heads,
                            dim_head,
                            depth=transformer_depth,
                            context_dim=context_dim,
                        )
                    )
                if level and i == num_res_blocks:
                    out_ch = ch
                    layers.append(
                        ResBlock(
                            ch,
                            embed_dim,
                            dropout,
                            out_channels=out_ch,
                            use_scale_shift_norm=use_scale_shift_norm,
                            up=True,
                        )
                        if resblock_updown
                        else Upsample(ch, conv_resample, out_channels=out_ch)
                    )
                    ds //= 2
                self.output_blocks.append(TimestepEmbedSequential(*layers))
                self._feature_size += ch

        self.out = nn.Sequential(
            Normalize(ch),
            nn.SiLU(),
            zero_module(nn.Conv3d(model_channels, out_channels, 3, padding=1)),
        )
        if self.predict_codebook_ids:
            self.id_predictor = nn.Sequential(
                Normalize(ch),
                nn.Conv3d(model_channels, n_embed, 1),
            )

    def forward(self, x, timestep=None, context=None, y=None, **kwargs):
        """
        Apply the model to an input batch.
        :param x: an [N x C x ...] Tensor of inputs.
        :param timesteps: a 1-D batch of timesteps.
        :param context: conditioning plugged in via crossattn
        :param y: an [N] Tensor of labels, if class-conditional.
        :return: an [N x C x ...] Tensor of outputs.
        """

        if y is not None:
            assert self.num_classes is not None, (
                "must specify y only if the model is class-conditional"
            )

        emb = 0
        if not self.single_step:  # if the unet is NOT stand-alone and used for denoising diffusion
            assert timestep is not None, "need to implement no-timestep usage"
            if not torch.is_tensor(timestep):
                timestep = torch.tensor([timestep], dtype=torch.long, device=x.device)
            elif torch.is_tensor(timestep) and len(timestep.shape) == 0:
                timestep = timestep[None].to(x.device)

            t_emb = timestep_embedding(timestep, self.model_channels, repeat_only=False)
            emb = self.time_embed(t_emb)

        hs = []

        if self.num_classes is not None and y is not None:
            assert y.shape[0] == x.shape[0]
            if self.concat_emb:
                if emb.shape[0] != x.shape[0]:
                    emb = emb.expand(x.shape[0], -1)
                emb = torch.cat([emb, self.label_emb(y)], dim=1)
            else:
                emb = emb + self.label_emb(y)

        h = x
        for module in self.input_blocks:
            h = module(h, emb, context)
            hs.append(h)

        h = self.middle_block(h, emb, context)

        for module in self.output_blocks:
            h1 = hs.pop()
            h = torch.cat([h, h1], dim=1)
            h = module(h, emb, context)

        if self.predict_codebook_ids:
            return UNet3DOutput(sample=self.id_predictor(h))

        else:
            return UNet3DOutput(sample=self.out(h))


## unet output


@dataclass
class UNet3DOutput(BaseOutput):
    """
    Args:
        sample (`torch.FloatTensor` of shape `(batch_size, num_channels, height, width)`):
            Hidden states output. Output of last layer of model.
    """

    sample: torch.FloatTensor


# NOTE(paper): not decorated with @dataclass in the original either; BaseOutput then behaves as
# a plain ordered dict with attribute access, which is all the callers use.
class UNet3DMeanVarOutput(BaseOutput):
    """
    Args:
        sample (`torch.FloatTensor` of shape `(batch_size, num_channels, height, width)`):
            Hidden states output. Output of last layer of model.
    """

    sample_mean: torch.FloatTensor
    sample_logvar: torch.FloatTensor


## normalization for embedding
class Normalization(nn.Module):
    def __init__(self, mus, sigmas):
        super().__init__()
        self.mus = nn.Parameter(mus, requires_grad=False)
        self.sigmas = nn.Parameter(sigmas, requires_grad=False)

    def forward(self, x, denormalize=False):
        if denormalize:
            return x * self.sigmas + self.mus
        return (x - self.mus) / self.sigmas


class UNetModelMeanVarSingleStep(ModelMixin, ConfigMixin):
    """UNet model that predicts mean and variance in a single forward pass.
    Based on the diffusion UNet architecture but without timestep embeddings.
    """

    @register_to_config
    def __init__(
        self,
        image_size,
        in_channels,
        model_channels,
        out_channels,
        num_res_blocks,
        attention_resolutions,
        dropout=0,
        channel_mult=(1, 2, 4, 8),
        conv_resample=True,
        num_classes=None,
        num_heads=1,
        num_head_channels=-1,
        num_heads_upsample=-1,
        use_scale_shift_norm=False,
        resblock_updown=False,
        use_spatial_transformer=False,
        transformer_depth=1,
        context_dim=None,
        emb_bias=True,
    ):
        super().__init__()

        if use_spatial_transformer:
            assert context_dim is not None, "context_dim required for spatial transformer"

        if num_heads_upsample == -1:
            num_heads_upsample = num_heads

        self.image_size = image_size
        self.in_channels = in_channels
        self.model_channels = model_channels
        self.out_channels = out_channels
        self.num_res_blocks = num_res_blocks
        self.attention_resolutions = attention_resolutions
        self.dropout = dropout
        self.channel_mult = channel_mult
        self.conv_resample = conv_resample
        self.num_classes = num_classes
        self.num_heads = num_heads
        self.num_head_channels = num_head_channels
        self.num_heads_upsample = num_heads_upsample

        # embedding dimension for age conditioning
        emb_dim = model_channels * 4

        if self.num_classes is not None:
            self.label_emb = nn.Linear(num_classes, emb_dim, bias=emb_bias)

        # input conv
        self.input_blocks = nn.ModuleList(
            [TimestepEmbedSequential(nn.Conv3d(in_channels, model_channels, 3, padding=1))]
        )
        self._feature_size = model_channels
        input_block_chans = [model_channels]
        ch = model_channels
        ds = 1

        # For conditioning, we only need the label embedding dimension
        embed_dim = emb_dim if self.num_classes is not None else 0

        # Build encoder part
        for level, mult in enumerate(channel_mult):
            for _ in range(num_res_blocks):
                layers = [
                    ResBlock(
                        ch,
                        embed_dim,
                        dropout,
                        out_channels=mult * model_channels,
                        use_scale_shift_norm=use_scale_shift_norm,
                    )
                ]
                ch = mult * model_channels
                if ds in attention_resolutions:
                    dim_head = ch // num_heads
                    layers.append(
                        AttentionBlock(ch, num_heads=num_heads, num_head_channels=num_head_channels)
                        if not use_spatial_transformer
                        else SpatialTransformer(
                            ch,
                            num_heads,
                            dim_head,
                            depth=transformer_depth,
                            context_dim=context_dim,
                        )
                    )
                self.input_blocks.append(TimestepEmbedSequential(*layers))
                self._feature_size += ch
                input_block_chans.append(ch)
            if level != len(channel_mult) - 1:
                out_ch = ch
                self.input_blocks.append(
                    TimestepEmbedSequential(
                        ResBlock(
                            ch,
                            embed_dim,
                            dropout,
                            out_channels=out_ch,
                            use_scale_shift_norm=use_scale_shift_norm,
                            down=True,
                        )
                        if resblock_updown
                        else Downsample(ch, conv_resample, out_channels=out_ch)
                    )
                )
                ch = out_ch
                input_block_chans.append(ch)
                ds *= 2
                self._feature_size += ch

        # Build bottleneck
        dim_head = ch // num_heads
        self.middle_block = TimestepEmbedSequential(
            ResBlock(ch, embed_dim, dropout, use_scale_shift_norm=use_scale_shift_norm),
            AttentionBlock(ch, num_heads=num_heads, num_head_channels=num_head_channels)
            if not use_spatial_transformer
            else SpatialTransformer(
                ch,
                num_heads,
                dim_head,
                depth=transformer_depth,
                context_dim=context_dim,
            ),
            ResBlock(ch, embed_dim, dropout, use_scale_shift_norm=use_scale_shift_norm),
        )
        self._feature_size += ch

        # Build decoder part with shared layers
        self.output_blocks = nn.ModuleList([])
        for level, mult in list(enumerate(channel_mult))[::-1]:
            # Only build shared decoder blocks until the last level
            if level != 0:  # Not the final level
                for i in range(num_res_blocks + 1):
                    ich = input_block_chans.pop()
                    layers = [
                        ResBlock(
                            ch + ich,
                            embed_dim,
                            dropout,
                            out_channels=model_channels * mult,
                            use_scale_shift_norm=use_scale_shift_norm,
                        )
                    ]
                    ch = model_channels * mult
                    if ds in attention_resolutions:
                        dim_head = ch // num_heads
                        layers.append(
                            AttentionBlock(
                                ch,
                                num_heads=num_heads_upsample,
                                num_head_channels=num_head_channels,
                            )
                            if not use_spatial_transformer
                            else SpatialTransformer(
                                ch,
                                num_heads,
                                dim_head,
                                depth=transformer_depth,
                                context_dim=context_dim,
                            )
                        )
                    if i == num_res_blocks:
                        out_ch = ch
                        layers.append(
                            ResBlock(
                                ch,
                                embed_dim,
                                dropout,
                                out_channels=out_ch,
                                use_scale_shift_norm=use_scale_shift_norm,
                                up=True,
                            )
                            if resblock_updown
                            else Upsample(ch, conv_resample, out_channels=out_ch)
                        )
                        ds //= 2
                    self.output_blocks.append(TimestepEmbedSequential(*layers))
                    self._feature_size += ch

        # Separate final decoder blocks for mean and variance
        ich = input_block_chans.pop()

        # Mean prediction path with same channel progression as UNetModel
        mean_layers = [
            ResBlock(
                ch + ich,
                embed_dim,
                dropout,
                out_channels=model_channels,  # Goes to 64 channels first
                use_scale_shift_norm=use_scale_shift_norm,
            ),
            Normalize(model_channels),
            nn.SiLU(),
            ResBlock(
                model_channels,
                embed_dim,
                dropout,
                out_channels=model_channels,
                use_scale_shift_norm=use_scale_shift_norm,
            ),
            Normalize(model_channels),
            nn.SiLU(),
            zero_module(nn.Conv3d(model_channels, out_channels, 3, padding=1)),
        ]
        self.mean_block = TimestepEmbedSequential(*mean_layers)

        # Variance prediction path with same channel progression
        var_layers = [
            ResBlock(
                ch + ich,
                embed_dim,
                dropout,
                out_channels=model_channels,  # Goes to 64 channels first
                use_scale_shift_norm=use_scale_shift_norm,
            ),
            Normalize(model_channels),
            nn.SiLU(),
            ResBlock(
                model_channels,
                embed_dim,
                dropout,
                out_channels=model_channels,
                use_scale_shift_norm=use_scale_shift_norm,
            ),
            Normalize(model_channels),
            nn.SiLU(),
            zero_module(nn.Conv3d(model_channels, out_channels, 3, padding=1)),
        ]
        self.var_block = TimestepEmbedSequential(*var_layers)

    def forward(self, x, y=None, context=None):
        """
        Apply the model to an input batch.
        Args:
            x: Input tensor [B, C, ...]
            y: Conditioning label (e.g. age) [B, num_classes]
            context: Optional context for attention
        Returns:
            UNet3DOutput with mean and log variance predictions
        """
        # Get conditioning embedding if provided
        if y is not None:
            assert self.num_classes is not None
            emb = self.label_emb(y)
        else:
            emb = 0

        # Encoder
        hs = []
        h = x
        for module in self.input_blocks:
            h = module(h, emb, context)
            hs.append(h)

        # Bottleneck
        h = self.middle_block(h, emb, context)

        # Shared decoder layers
        for module in self.output_blocks:
            h = torch.cat([h, hs.pop()], dim=1)
            h = module(h, emb, context)

        # Final skip connection
        h = torch.cat([h, hs.pop()], dim=1)

        # Separate paths for mean and variance
        mean = self.mean_block(h, emb, context)
        log_var = self.var_block(h, emb, context)

        return UNet3DMeanVarOutput(sample_mean=mean, sample_logvar=log_var)
