import os
import json
import numpy as np
from torch.utils import data
from tqdm import tqdm
from transformers import Wav2Vec2Processor
from typing import List, Tuple, Optional, Dict, Any

from .data_item import DataItem, FlameCoeffDataItem


class FaceFormerDataset(data.Dataset):
    """FaceFormer dataset supporting multiple datasets with unified preprocessing."""

    def __init__(self, data_root_list: List[str], json_list: List[str], fps_list: List[int],
                 duplicate_list: List[int] = None, processor: Optional[Wav2Vec2Processor] = None,
                 split: str = 'train'):
        """
        Args:
            data_root_list: List of root directories for each dataset
            json_list: List of JSON config files (relative to data_root)
            fps_list: List of original fps for each dataset
            duplicate_list: List of duplication factors for balancing
            processor: Wav2Vec2 processor for audio
            split: Dataset split ('train', 'val', 'test')
        """
        self.data_root_list = data_root_list
        self.json_list = json_list
        self.fps_list = fps_list
        self.duplicate_list = duplicate_list or [1] * len(json_list)
        self.processor = processor
        self.split = split

        # Load data from all datasets
        self.data_list = []
        for i, (data_root, json_path, fps, duplicate) in enumerate(zip(
            data_root_list, json_list, fps_list, self.duplicate_list)):

            full_json_path = os.path.join(data_root, json_path)
            if os.path.exists(full_json_path):
                print(f"Loading {split} data from {full_json_path} (fps={fps}, duplicate={duplicate})")
                self._load_from_json(full_json_path, data_root, fps, duplicate)

        print(f"Loaded {len(self.data_list)} {split} samples from {len(json_list)} datasets")

    def _load_from_json(self, json_path: str, data_root: str, original_fps: int, duplicate: int):
        """Load data from JSON config file with fps information."""
        with open(json_path, 'r') as f:
            config = json.load(f)

        for subject_key, subject_data in config.items():
            try:
                # Get paths (relative to data_root)
                audio_path = os.path.join(data_root, subject_data['audio_path'])
                flame_path = os.path.join(data_root, subject_data.get('flame_coeff_save_path') or subject_data.get('flame_path'))
                template_path = subject_data.get('template_save_path')
                if template_path:
                    template_path = os.path.join(data_root, template_path)

                # Create data item with fps information
                data_item = DataItem(
                    audio_path=audio_path,
                    flame_path=flame_path,
                    template_path=template_path,
                    subject_id=subject_key,
                    processor=self.processor,
                    original_fps=original_fps,
                    target_fps=25  # 统一重采样到25fps
                )

                if data_item.duration >= 0.5:  # Skip too short samples
                    # 根据duplicate_list重复添加样本
                    for _ in range(duplicate):
                        self.data_list.append(data_item)

            except Exception as e:
                print(f"Error loading {subject_key}: {e}")
                continue

    def __len__(self) -> int:
        return len(self.data_list)

    def __getitem__(self, index: int):
        return self.data_list[index].get_dict()

    def get_identity_num(self) -> int:
        """Get number of unique identities."""
        identities = set()
        for item in self.data_list:
            identities.add(item.subject_id)
        return len(identities)


def get_dataloaders(config: dict) -> Tuple[data.DataLoader, data.DataLoader, data.DataLoader]:
    """Get train, validation, and test dataloaders with multi-dataset support."""

    # Get data config
    data_config = config['DATA']

    # Create processor
    model_config = config['MODEL']
    processor = Wav2Vec2Processor.from_pretrained(model_config['audio_encoder_repo'])

    # Get batch sizes
    train_config = config['TRAIN']
    batch_size = train_config['batch_size']
    batch_size_val = train_config.get('batch_size_val', batch_size)
    workers = train_config.get('workers', 4)

    # Create datasets
    if 'train_json_list' in data_config:
        # New multi-dataset format
        data_root_list = data_config['data_root']
        fps_list = data_config['fps']

        train_dataset = FaceFormerDataset(
            data_root_list=data_root_list,
            json_list=data_config['train_json_list'],
            fps_list=fps_list,
            duplicate_list=data_config.get('duplicate_list', [1] * len(data_config['train_json_list'])),
            processor=processor,
            split='train'
        )

        val_dataset = FaceFormerDataset(
            data_root_list=data_root_list[:len(data_config['val_json_list'])],  # Use same roots
            json_list=data_config['val_json_list'],
            fps_list=fps_list[:len(data_config['val_json_list'])],  # Use corresponding fps
            duplicate_list=[1] * len(data_config['val_json_list']),
            processor=processor,
            split='val'
        )

        test_dataset = FaceFormerDataset(
            data_root_list=data_root_list[:len(data_config['test_json_list'])],  # Use same roots
            json_list=data_config['test_json_list'],
            fps_list=fps_list[:len(data_config['test_json_list'])],  # Use corresponding fps
            duplicate_list=[1] * len(data_config['test_json_list']),
            processor=processor,
            split='test'
        )
    else:
        # Fallback to old format (for compatibility)
        data_root = data_config['data_root']
        data_jsons = data_config['data_jsons']

        train_dataset = FaceFormerDataset(
            data_root_list=[data_root] * len(data_jsons),
            json_list=data_jsons,
            fps_list=[25] * len(data_jsons),  # Default fps
            processor=processor,
            split='train'
        )

        val_dataset = FaceFormerDataset(
            data_root_list=[data_root] * len(data_jsons),
            json_list=data_jsons,
            fps_list=[25] * len(data_jsons),
            processor=processor,
            split='val'
        )

        test_dataset = FaceFormerDataset(
            data_root_list=[data_root] * len(data_jsons),
            json_list=data_jsons,
            fps_list=[25] * len(data_jsons),
            processor=processor,
            split='test'
        )

    # Create dataloaders
    train_loader = data.DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=workers,
        pin_memory=True,
        collate_fn=collate_fn
    )

    val_loader = data.DataLoader(
        val_dataset,
        batch_size=batch_size_val,
        shuffle=False,
        num_workers=workers,
        pin_memory=True,
        collate_fn=collate_fn
    )

    test_loader = data.DataLoader(
        test_dataset,
        batch_size=batch_size_val,
        shuffle=False,
        num_workers=workers,
        pin_memory=True,
        collate_fn=collate_fn
    )

    return train_loader, val_loader, test_loader


def collate_fn(batch):
    """Custom collate function to handle variable length sequences."""
    import torch

    # Find max length in batch
    max_len = max(item['motion_coeff'].shape[0] for item in batch)

    # Pad sequences to max length
    padded_batch = []
    for item in batch:
        padded_item = {}
        for key, value in item.items():
            if isinstance(value, torch.Tensor) and len(value.shape) > 1:
                # Pad sequence dimension
                if value.shape[0] < max_len:
                    padding = torch.zeros(max_len - value.shape[0], *value.shape[1:])
                    if value.dtype == torch.long:
                        padding = padding.long()
                    value = torch.cat([value, padding], dim=0)
            padded_item[key] = value
        padded_batch.append(padded_item)

    # Use default collate for the rest
    return data.default_collate(padded_batch)
