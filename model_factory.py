"""Factory functions to build the MambaVLA model to match the standalone repo."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, List, Dict, Any

import torch

from mambavla_model import MambaVLA
from policy.flowmatching import ActionFLowMatching
from policy.policy import MambaVLAPolicy
from backbones.multi_img_obs_encoder import MultiImageObsEncoder
from backbones.resnet.resnets import ResNetEncoder
from backbones.clip.clip_lang_encoder import LangClip
from mamba.mamba import MixerModel
from utils.networks.transformers.transformer_blocks import TransformerEncoder


def create_mamba_backbone(
    embed_dim: int = 256,
    n_layer: int = 5,
    d_intermediate: int = 256,
    device: str = 'cuda',
    ssm_cfg: Optional[Dict] = None,
    attn_layer_idx: Optional[List] = None,
    attn_cfg: Optional[Dict] = None,
    rms_norm: bool = True,
    fused_add_norm: bool = False,
    residual_in_fp32: bool = False,
):
    if ssm_cfg is None:
        ssm_cfg = {
            "layer": "Mamba1",
            "d_state": 64,
            "d_conv": 4,
            "expand": 2,
        }
    if attn_layer_idx is None:
        attn_layer_idx = []
    if attn_cfg is None:
        attn_cfg = {}

    return MixerModel(
        d_model=embed_dim,
        n_layer=n_layer,
        d_intermediate=d_intermediate,
        ssm_cfg=ssm_cfg,
        attn_layer_idx=attn_layer_idx,
        attn_cfg=attn_cfg,
        rms_norm=rms_norm,
        initializer_cfg=None,
        fused_add_norm=fused_add_norm,
        residual_in_fp32=residual_in_fp32,
        device=device,
        dtype=torch.float32,
    )


def create_transformer_backbone(
    embed_dim: int,
    n_layer: int,
    n_heads: int,
    attn_pdrop: float,
    resid_pdrop: float,
    mlp_pdrop: float,
    block_size: int,
    bias: bool = False,
    use_rot_embed: bool = False,
    rotary_xpos: bool = False,
):
    return TransformerEncoder(
        embed_dim=embed_dim,
        n_heads=n_heads,
        attn_pdrop=attn_pdrop,
        resid_pdrop=resid_pdrop,
        mlp_pdrop=mlp_pdrop,
        n_layers=n_layer,
        block_size=block_size,
        bias=bias,
        use_rot_embed=use_rot_embed,
        rotary_xpos=rotary_xpos,
    )


@dataclass
class OptimizerConfig:
    transformer_weight_decay: float = 0.05
    obs_encoder_weight_decay: float = 0.05
    learning_rate: float = 1e-4
    betas: List[float] | None = None


@dataclass
class LRSchedulerConfig:
    init_lr: float = 1e-4
    init_lr_scale: float = 0.1
    final_lr_scale: float = 1e-6
    total_steps: int = 50000
    phase_ratio: str = "(0.02, 0.08, 0.9)"
    lr: float = 1e-4


def _extract_camera_names_from_dataloader(dataloader) -> List[str]:
    if hasattr(dataloader, 'camera_names'):
        return dataloader.camera_names
    if hasattr(dataloader, 'get_camera_names'):
        return dataloader.get_camera_names()
    raise ValueError(
        "dataloader must have a 'camera_names' attribute or 'get_camera_names()' method"
    )


def _infer_lang_emb_dim_from_dataset(dataset) -> Optional[int]:
    if hasattr(dataset, 'data_embs') and len(dataset.data_embs) > 0:
        sample_emb = dataset.data_embs[0]
        if isinstance(sample_emb, torch.Tensor):
            return sample_emb.shape[-1]
        if hasattr(sample_emb, 'shape'):
            return sample_emb.shape[-1]
    return None


def create_mambavla_model(
    dataloader=None,
    camera_names: Optional[List[str]] = None,
    latent_dim: int = 256,
    action_dim: int = 7,
    lang_emb_dim: int = 512,
    embed_dim: int = 256,
    obs_tok_len: Optional[int] = None,
    action_seq_len: int = 5,
    perception_seq_len: int = 1,
    state_dim: int = 7,
    device: str = 'cuda',
    n_layer: int = 5,
    d_intermediate: int = 256,
    sampling_steps: int = 4,
    transformer_weight_decay: float = 0.05,
    obs_encoder_weight_decay: float = 0.05,
    learning_rate: float = 1e-4,
    betas: Optional[List[float]] = None,
    use_language_encoder: bool = True,
    freeze_language_encoder: bool = True,
    clip_model_name: str = 'ViT-B/32',
    model_type: str = "mamba",
    transformer_cfg: Optional[Dict] = None,
):
    if camera_names is None:
        if dataloader is None:
            raise ValueError("camera_names must be provided or dataloader must be provided")
        camera_names = _extract_camera_names_from_dataloader(dataloader)

    if obs_tok_len is None:
        obs_tok_len = len(camera_names)

    if dataloader is not None:
        if hasattr(dataloader, 'action_dim'):
            action_dim = dataloader.action_dim
        if hasattr(dataloader, 'state_dim'):
            state_dim = dataloader.state_dim
        inferred_lang = _infer_lang_emb_dim_from_dataset(dataloader)
        if inferred_lang is not None:
            lang_emb_dim = inferred_lang

    if betas is None:
        betas = [0.9, 0.9]

    # Build obs encoder
    shape_meta = {
        "obs": {
            f"{cam}_image": {"shape": [3, 128, 128], "type": "rgb"}
            for cam in camera_names
        }
    }

    resnet_cfg = {
        "_target_": "backbones.resnet.resnets.ResNetEncoder",
        "latent_dim": latent_dim,
        "pretrained": False,
        "freeze_backbone": False,
        "use_mlp": True,
    }

    obs_encoder = MultiImageObsEncoder(
        shape_meta=shape_meta,
        rgb_model=resnet_cfg,
        resize_shape=None,
        crop_shape=None,
        random_crop=False,
        use_group_norm=True,
        share_rgb_model=False,
        imagenet_norm=True,
    ).to(device)

    language_encoder = None
    if use_language_encoder:
        language_encoder = LangClip(
            freeze_backbone=freeze_language_encoder,
            model_name=clip_model_name,
        ).to(device)

    if model_type == "transformer":
        if transformer_cfg is None:
            transformer_cfg = {}
        block_size = 1 + obs_tok_len + action_seq_len + (1 if use_language_encoder else 0)
        encoder = create_transformer_backbone(
            embed_dim=embed_dim,
            n_layer=n_layer,
            n_heads=transformer_cfg.get("n_heads", 8),
            attn_pdrop=transformer_cfg.get("attn_pdrop", 0.1),
            resid_pdrop=transformer_cfg.get("resid_pdrop", 0.1),
            mlp_pdrop=transformer_cfg.get("mlp_pdrop", 0.0),
            block_size=block_size,
            bias=transformer_cfg.get("bias", False),
            use_rot_embed=transformer_cfg.get("use_rot_embed", False),
            rotary_xpos=transformer_cfg.get("rotary_xpos", False),
        ).to(device)
    else:
        encoder = create_mamba_backbone(
            embed_dim=embed_dim,
            n_layer=n_layer,
            d_intermediate=d_intermediate,
            device=device,
        )

    backbone = MambaVLAPolicy(
        encoder=encoder,
        latent_dim=latent_dim,
        action_dim=action_dim,
        lang_emb_dim=lang_emb_dim,
        device=device,
        goal_conditioned=True,
        embed_dim=embed_dim,
        embed_pdrob=0,
        lang_tok_len=1,
        obs_tok_len=obs_tok_len,
        action_seq_len=action_seq_len,
        linear_output=True,
        use_ada_conditioning=False,
        use_pos_emb=True,
    ).to(device)

    flow_model = ActionFLowMatching(
        backbones=backbone,
        ln=False,
        device=device,
    )

    optimizer_cfg = OptimizerConfig(
        transformer_weight_decay=transformer_weight_decay,
        obs_encoder_weight_decay=obs_encoder_weight_decay,
        learning_rate=learning_rate,
        betas=betas,
    )
    lr_scheduler_cfg = LRSchedulerConfig()

    model = MambaVLA(
        model=flow_model,
        obs_encoders=obs_encoder,
        language_encoders=language_encoder,
        optimizer=optimizer_cfg,
        lr_scheduler=lr_scheduler_cfg,
        action_dim=action_dim,
        perception_seq_len=perception_seq_len,
        action_seq_len=action_seq_len,
        cam_names=camera_names,
        use_lr_scheduler=False,
        consider_robot_states=False,
        if_film_condition=False,
        device=device,
        state_dim=state_dim,
        latent_dim=latent_dim,
        sampling_steps=sampling_steps,
    ).to(device)

    return model
