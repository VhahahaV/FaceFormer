import os
import json
import numpy as np
import torch
import librosa
from transformers import Wav2Vec2Processor
from typing import Dict, Any, Optional
from scipy import interpolate


class DataItem:
    """Base data item class with fps detection and resampling."""

    def __init__(self, audio_path: str, flame_path: str, template_path: Optional[str] = None,
                 subject_id: str = "", processor: Optional[Wav2Vec2Processor] = None,
                 original_fps: int = 25, target_fps: int = 25):
        self.audio_path = audio_path
        self.flame_path = flame_path
        self.template_path = template_path
        self.subject_id = subject_id
        self.processor = processor
        self.original_fps = original_fps  # 原始fps (20 for MultiModal200, 25 for others)
        self.target_fps = target_fps     # 目标fps (统一为25)

        # Load data
        self.audio = self._load_audio()
        self.flame_data = self._load_flame()
        self.template = self._load_template()
        self.motion_coeff = self._extract_motion_coeff()

        # 如果需要，重采样到25fps
        if self.original_fps != self.target_fps:
            self._resample_to_target_fps()

        # 注意：序列长度已在_extract_motion_coeff中截断到500帧

    def _resample_to_target_fps(self):
        """重采样FLAME系数到目标fps (25fps)。"""
        if self.original_fps == self.target_fps:
            return

        seq_len = self.motion_coeff.shape[0]
        original_times = np.linspace(0, seq_len / self.original_fps, seq_len)

        # 目标时间点 (25fps)
        target_seq_len = int(seq_len * self.target_fps / self.original_fps)
        target_times = np.linspace(0, seq_len / self.original_fps, target_seq_len)

        # 对每个维度进行插值
        resampled_motion = []
        for dim in range(self.motion_coeff.shape[1]):
            interp_func = interpolate.interp1d(original_times, self.motion_coeff[:, dim],
                                             kind='linear', bounds_error=False,
                                             fill_value='extrapolate')
            resampled_dim = interp_func(target_times)
            resampled_motion.append(resampled_dim)

        self.motion_coeff = np.stack(resampled_motion, axis=1)

        # 同样重采样FLAME数据
        if isinstance(self.flame_data, dict):
            for key in self.flame_data:
                if isinstance(self.flame_data[key], np.ndarray) and len(self.flame_data[key].shape) > 1:
                    seq_len_flame = self.flame_data[key].shape[0]
                    original_times_flame = np.linspace(0, seq_len_flame / self.original_fps, seq_len_flame)
                    target_times_flame = np.linspace(0, seq_len_flame / self.original_fps, target_seq_len)

                    resampled_data = []
                    for dim in range(self.flame_data[key].shape[1]):
                        interp_func = interpolate.interp1d(original_times_flame, self.flame_data[key][:, dim],
                                                         kind='linear', bounds_error=False,
                                                         fill_value='extrapolate')
                        resampled_dim = interp_func(target_times_flame)
                        resampled_data.append(resampled_dim)

                    self.flame_data[key] = np.stack(resampled_data, axis=1)

    def _load_audio(self) -> np.ndarray:
        """Load and process audio."""
        if not os.path.exists(self.audio_path):
            raise FileNotFoundError(f"Audio file not found: {self.audio_path}")

        speech_array, sr = librosa.load(self.audio_path, sr=16000)
        if len(speech_array) > 16000 * 10:  # Max 10 seconds
            speech_array = speech_array[:16000 * 10]

        if self.processor is not None:
            input_values = np.squeeze(self.processor(speech_array, sampling_rate=16000).input_values)
            return input_values
        else:
            return speech_array

    def _load_flame(self) -> Dict[str, np.ndarray]:
        """Load FLAME coefficients."""
        if not os.path.exists(self.flame_path):
            raise FileNotFoundError(f"FLAME file not found: {self.flame_path}")

        return np.load(self.flame_path)

    def _load_template(self) -> Optional[np.ndarray]:
        """Load template (51-dim motion coefficients)."""
        if self.template_path and os.path.exists(self.template_path):
            template_data = np.load(self.template_path)
            if isinstance(template_data, np.ndarray):
                # If template is stored as array, extract motion coefficients
                if template_data.shape[-1] == 287:  # Full FLAME coefficients
                    # Extract first frame as template
                    return self._extract_motion_coeff_from_frame(template_data[0])
                elif template_data.shape[-1] == 51:  # Already motion coefficients
                    return template_data
            else:
                # Handle dict format - use the same logic as motion coefficient extraction
                template_motion = self._extract_motion_coeff_from_dict(template_data)
                return template_motion

        # Default neutral template with small random noise to avoid zero gradients
        template = np.zeros(51)
        template += np.random.normal(0, 0.01, 51)  # Add small noise
        return template

    def _extract_motion_coeff_from_frame(self, flame_frame: np.ndarray) -> np.ndarray:
        """Extract 51-dim motion coefficients from a single FLAME frame."""
        # This is a simplified version - for template, we mainly care about neutral pose
        # We'll use zeros for expression and a neutral jaw pose
        return np.zeros(51)  # [50 expr + 1 jaw_pose]

    def _extract_motion_coeff_from_dict(self, template_data: dict) -> np.ndarray:
        """Extract 51-dim motion coefficients from template dict."""
        # Use the same logic as _extract_motion_coeff but for template
        if 'expcode' in template_data:  # digital_human format
            expr = template_data['expcode'][0] if template_data['expcode'].ndim > 1 else template_data['expcode']  # (50,)
            jaw_pose = template_data['posecode'][3]  # scalar jaw rotation
        elif 'expr' in template_data:  # MEAD_VHAP and MultiModal200 format
            expr = template_data['expr'][0] if template_data['expr'].ndim > 1 else template_data['expr']  # (50,)
            jaw_pose = template_data['jaw_pose'][0] if 'jaw_pose' in template_data else 0.0
        else:
            return np.zeros(51)

        # Concatenate expr and jaw_pose
        return np.concatenate([expr, np.array([jaw_pose])], axis=0)  # (51,)

    def _extract_motion_coeff(self) -> np.ndarray:
        """Extract 51-dim motion coefficients (50 expr + 1 jaw_pose)."""
        flame_data = self.flame_data

        # Handle different FLAME formats
        if 'expcode' in flame_data:  # digital_human format (旧格式)
            expr = flame_data['expcode']  # (N, 50)
            # jaw藏在posecode的第4个位置 (index 3)
            jaw_pose = flame_data['posecode'][:, 3:4]  # jaw rotation (N, 1)
        elif 'expr' in flame_data:  # MEAD_VHAP and MultiModal200 format (新格式)
            expr = flame_data['expr']  # (N, 50)
            if 'jaw_pose' in flame_data:
                # 只取jaw_pose的第一维
                jaw_pose = flame_data['jaw_pose'][:, :1]  # jaw rotation (N, 1)
            else:
                jaw_pose = np.zeros((expr.shape[0], 1))  # Default
        else:
            raise ValueError(f"Unsupported FLAME format in {self.flame_path}")

        # Concatenate expr and jaw_pose -> (N, 51)
        motion_coeff = np.concatenate([expr, jaw_pose], axis=1)

        # Keep full sequence length (matching UniTalker behavior)
        # No truncation applied

        return motion_coeff

    @property
    def duration(self) -> float:
        """Get audio duration."""
        return len(self.audio) / 16000.0

    def get_dict(self) -> Dict[str, Any]:
        """Get data as dictionary."""
        return {
            'audio': torch.FloatTensor(self.audio),
            'motion_coeff': torch.FloatTensor(self.motion_coeff),
            'template': torch.FloatTensor(self.template),
            'subject_id': self.subject_id,
            'audio_path': self.audio_path,
            'flame_path': self.flame_path
        }


class FlameCoeffDataItem(DataItem):
    """FLAME coefficient data item for full coefficient prediction."""

    def _extract_motion_coeff(self) -> np.ndarray:
        """Extract full FLAME coefficients."""
        flame_data = self.flame_data

        # Handle different FLAME formats and create full 287-dim coefficients
        if 'expcode' in flame_data:  # digital_human format
            flame_combined = np.concatenate([
                flame_data['shapecode'],    # (N, 100)
                flame_data['expcode'],      # (N, 50)
                flame_data['posecode'],     # (N, 6)
                flame_data['cam']          # (N, 3)
            ], axis=1)  # (N, 159)

            if 'detailcode' in flame_data and flame_data['detailcode'].shape[1] > 0:
                flame_combined = np.concatenate([
                    flame_combined,
                    flame_data['detailcode']  # (N, 128)
                ], axis=1)  # (N, 287)

        elif 'expr' in flame_data:  # MEAD_VHAP and MultiModal200 format
            seq_len = flame_data['expr'].shape[0]
            shape_coeffs = np.tile(flame_data['shape'], (seq_len, 1))  # (N, 100)
            expr_coeffs = flame_data['expr']  # (N, 50)
            jaw_coeffs = flame_data['jaw_pose']  # (N, 3)

            # Create pose coefficients
            head_rotation = flame_data.get('rotation', np.zeros((seq_len, 3)))  # (N, 3)
            pose_coeffs = np.concatenate([head_rotation, jaw_coeffs], axis=1)  # (N, 6)
            cam_coeffs = np.zeros((seq_len, 3))  # (N, 3)

            flame_combined = np.concatenate([
                shape_coeffs, expr_coeffs, pose_coeffs, cam_coeffs
            ], axis=1)  # (N, 159)

            # Pad to 287 dimensions
            if flame_combined.shape[1] < 287:
                padding = np.zeros((seq_len, 287 - flame_combined.shape[1]))
                flame_combined = np.concatenate([flame_combined, padding], axis=1)

        # Truncate to max frames
        max_frames = 300
        if flame_combined.shape[0] > max_frames:
            flame_combined = flame_combined[:max_frames]

        return flame_combined
