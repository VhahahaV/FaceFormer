import os
import logging
import torch
import numpy as np
from collections import defaultdict
from typing import Dict, Any


def get_logger(log_file: str) -> logging.Logger:
    """Get logger."""
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)

    # Create formatter
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')

    # Create file handler
    file_handler = logging.FileHandler(log_file)
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(formatter)

    # Create console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(formatter)

    # Add handlers to logger
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)

    return logger


def get_average_meter_dict() -> Dict[str, float]:
    """Get average meter dictionary."""
    return defaultdict(float)


def seed_everything(seed: int) -> None:
    """Set random seed."""
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def get_audio_encoder_dim(audio_encoder_repo: str) -> int:
    """Get audio encoder feature dimension."""
    if 'wav2vec2' in audio_encoder_repo:
        return 768
    elif 'wavlm' in audio_encoder_repo:
        return 768
    elif 'hubert' in audio_encoder_repo:
        return 768
    else:
        raise ValueError(f"Unknown audio encoder: {audio_encoder_repo}")


def load_ckpt(model: torch.nn.Module, weight_path: str, re_init_decoder_and_head: bool = False) -> None:
    """Load checkpoint."""
    checkpoint = torch.load(weight_path, map_location='cpu')
    if 'state_dict' in checkpoint:
        state_dict = checkpoint['state_dict']
    else:
        state_dict = checkpoint

    # Filter state dict
    model_state_dict = model.state_dict()
    filtered_state_dict = {}
    for k, v in state_dict.items():
        if k in model_state_dict and v.shape == model_state_dict[k].shape:
            filtered_state_dict[k] = v

    model.load_state_dict(filtered_state_dict, strict=False)
    print(f"Loaded {len(filtered_state_dict)}/{len(state_dict)} parameters from {weight_path}")


def filter_unitalker_state_dict(state_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    """Filter UniTalker state dict for FaceFormer."""
    filtered_state_dict = {}
    for k, v in state_dict.items():
        # Skip UniTalker specific layers
        if 'pca' in k or 'blendshape' in k:
            continue
        filtered_state_dict[k] = v
    return filtered_state_dict
