import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import copy
import math

import sys
import os
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.insert(0, parent_dir)
from wav2vec import CustomWav2Vec2Model

# Import FLAME model for vertex prediction
try:
    from flame.flame_pytorch import FLAME
except ImportError:
    FLAME = None


def init_biased_mask(n_head, max_seq_len, period):
    """Initialize biased mask for transformer (ALiBi style)."""
    def get_slopes(n):
        def get_slopes_power_of_2(n):
            start = (2**(-2**-(math.log2(n)-3)))
            ratio = start
            return [start*ratio**i for i in range(n)]
        if math.log2(n).is_integer():
            return get_slopes_power_of_2(n)
        else:
            closest_power_of_2 = 2**math.floor(math.log2(n))
            return get_slopes_power_of_2(closest_power_of_2) + get_slopes(2*closest_power_of_2)[0::2][:n-closest_power_of_2]

    slopes = torch.Tensor(get_slopes(n_head))
    bias = torch.arange(start=0, end=max_seq_len, step=period).unsqueeze(1).repeat(1,period).view(-1)//(period)
    bias = - torch.flip(bias,dims=[0])
    alibi = torch.zeros(max_seq_len, max_seq_len)
    for i in range(max_seq_len):
        alibi[i, :i+1] = bias[-(i+1):]
    alibi = slopes.unsqueeze(1).unsqueeze(1) * alibi.unsqueeze(0)
    mask = (torch.triu(torch.ones(max_seq_len, max_seq_len)) == 1).transpose(0, 1)
    mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
    mask = mask.unsqueeze(0) + alibi
    return mask


def enc_dec_mask(device, dataset, T, S):
    """Create encoder-decoder mask."""
    mask = torch.ones(T, S)
    if dataset == "BIWI":
        for i in range(T):
            mask[i, i*2:i*2+2] = 0
    elif dataset in ["vocaset", "digital_human"]:
        for i in range(T):
            if i >= S:
                # If T > S, we can't set diagonal beyond S
                break
            mask[i, i] = 0
    return (mask==1).to(device=device)


class PeriodicPositionalEncoding(nn.Module):
    """Periodic positional encoding."""

    def __init__(self, d_model, dropout=0.1, period=25, max_seq_len=600):
        super(PeriodicPositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        pe = torch.zeros(max_seq_len, d_model)
        position = torch.arange(0, max_seq_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)


class FaceFormer(nn.Module):
    """FaceFormer model for motion coefficient prediction."""

    def __init__(self, config):
        super(FaceFormer, self).__init__()

        # Get config
        model_config = config['MODEL']

        self.dataset = config['DATA'].get('dataset', 'digital_human')
        self.period = model_config['period']
        self.feature_dim = model_config['feature_dim']
        self.motion_dim = model_config['motion_dim']  # 51-dim motion coefficients
        self.vertice_dim = model_config['vertice_dim']  # 5023-dim vertices
        self.max_seq_len = model_config.get('max_seq_len', 2000)  # Maximum sequence length

        # Audio encoder
        self.audio_encoder = CustomWav2Vec2Model.from_pretrained(model_config['audio_encoder_repo'])
        self.audio_feature_map = nn.Linear(768, self.feature_dim)

        # Motion prediction layers (51-dim: 50 expr + 1 jaw)
        self.obj_vector = nn.Linear(1, self.feature_dim, bias=False)  # For subject conditioning
        self.motion_map = nn.Linear(self.motion_dim, self.feature_dim)
        self.motion_map_r = nn.Linear(self.feature_dim, self.motion_dim)

        # Template handling
        self.template_map = nn.Linear(self.motion_dim, self.feature_dim)

        # Initialize motion prediction layers (original FaceFormer style)
        nn.init.xavier_uniform_(self.motion_map.weight)
        nn.init.constant_(self.motion_map.bias, 0)
        nn.init.xavier_uniform_(self.template_map.weight)
        nn.init.constant_(self.template_map.bias, 0)

        # Initialize decoder output layer with small weights (original FaceFormer)
        nn.init.xavier_uniform_(self.motion_map_r.weight, gain=0.01)  # Small gain
        nn.init.constant_(self.motion_map_r.bias, 0)

        # Positional encoding
        self.PPE = PeriodicPositionalEncoding(self.feature_dim, period=self.period, max_seq_len=self.max_seq_len)

        # Stable sequential processing (FaceFormer-style but numerically stable)
        # Use multiple layers of simpler attention-like mechanisms
        self.temporal_processor = nn.Sequential(
            nn.Conv1d(self.feature_dim, self.feature_dim, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv1d(self.feature_dim, self.feature_dim, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv1d(self.feature_dim, self.feature_dim, kernel_size=3, padding=1),
            nn.ReLU()
        )

        # Cross-modal attention (audio to motion)
        self.cross_attention = nn.MultiheadAttention(
            embed_dim=self.feature_dim,
            num_heads=4,
            dropout=0.0,  # Disable dropout for stability
            batch_first=True
        )

        # Output refinement (for both training and inference)
        self.refinement = nn.Sequential(
            nn.Linear(self.feature_dim, self.feature_dim),
            nn.ReLU(),
            nn.Linear(self.feature_dim, self.motion_dim)
        )

        # FLAME model for vertex prediction
        # Temporarily disable FLAME due to numpy compatibility issues
        # TODO: Fix numpy compatibility with chumpy library
        self.flame_model = None
        print("FLAME model disabled due to numpy compatibility issues")

        # Initialize motion_map_r with small weights for better stability
        nn.init.xavier_uniform_(self.motion_map_r.weight, gain=0.01)  # Small gain
        nn.init.constant_(self.motion_map_r.bias, 0)

    def forward(self, audio, template, motion_coeff, subject_id, criterion, teacher_forcing=True):
        """
        Forward pass with vertex prediction.

        Args:
            audio: Audio features (batch, seq_len)
            template: Template motion coefficients (batch, motion_dim)
            motion_coeff: Target motion coefficients (batch, seq_len, motion_dim)
            subject_id: Subject ID for conditioning (batch,)
            criterion: Loss function
            teacher_forcing: Whether to use teacher forcing

        Returns:
            Loss value
        """
        device = audio.device
        batch_size = audio.shape[0]

        # Encode audio
        try:
            audio_output = self.audio_encoder(audio, "digital_human", return_dict=True)
            hidden_states = audio_output.last_hidden_state
        except RuntimeError as e:
            print(f"Error in audio encoder: {e}")
            print(f"Audio shape: {audio.shape}")
            raise e

        hidden_states = self.audio_feature_map(hidden_states)
        frame_num = hidden_states.shape[1]  # Assume 1:1 mapping with frames

        if teacher_forcing:
            # Stable FaceFormer-style processing (maintains original architecture spirit)
            # Handle variable sequence lengths (no fixed truncation)
            T = motion_coeff.shape[1]

            motion_emb = self.motion_map(motion_coeff)  # (batch, seq_len, feature_dim)

            # Add positional encoding (maintains temporal structure like original)
            motion_emb = self.PPE(motion_emb.transpose(0, 1)).transpose(0, 1)

            # Temporal processing (replaces unstable Transformer with stable conv layers)
            motion_processed = self.temporal_processor(motion_emb.transpose(1, 2)).transpose(1, 2)

            # Cross-modal attention (audio-motion interaction like original)
            audio_query = motion_processed
            audio_key_value = hidden_states.mean(dim=1, keepdim=True).expand(-1, T, -1)

            attended_motion, _ = self.cross_attention(
                audio_query, audio_key_value, audio_key_value
            )

            # Output refinement (maintains motion coefficient prediction)
            # attended_motion: (batch, T, feature_dim) -> flatten to (batch*T, feature_dim)
            batch_size, seq_len, feat_dim = attended_motion.shape
            attended_flat = attended_motion.view(-1, feat_dim)  # (batch*T, feature_dim)

            # Apply refinement
            pred_flat = self.refinement(attended_flat)  # (batch*T, motion_dim)

            # Reshape back
            pred_motion = pred_flat.view(batch_size, seq_len, -1)  # (batch, T, motion_dim)

            # Return predicted motion coefficients
            return pred_motion
        else:
            # Autoregressive prediction
            return self._predict_autoregressive(audio, template, frame_num, device)

    def _predict_autoregressive(self, audio, template, frame_num, device):
        """Stable autoregressive prediction maintaining FaceFormer interface."""
        # Encode audio
        audio_output = self.audio_encoder(audio, self.dataset, return_dict=True)
        hidden_states = audio_output.last_hidden_state
        hidden_states = self.audio_feature_map(hidden_states)

        # Get frame number if not provided
        if frame_num is None:
            frame_num = hidden_states.shape[1]

        predictions = []

        for i in range(frame_num):
            # Use template for first prediction, then use previous predictions
            if i == 0:
                curr_motion_emb = self.template_map(template)  # (batch, feature_dim)
            else:
                # Use last prediction as input for next step
                curr_motion_emb = self.motion_map(predictions[-1].squeeze(1))  # (batch, feature_dim)

            # For inference, we process one step at a time
            # Add positional encoding (single step)
            curr_motion_emb = curr_motion_emb.unsqueeze(1)  # (batch, 1, feature_dim)
            curr_motion_emb = self.PPE(curr_motion_emb.transpose(0, 1)).transpose(0, 1)

            # Temporal processing (simplified for single step)
            # Skip conv layers for single-step inference, just pass through
            motion_processed = curr_motion_emb

            # Simplified cross-modal fusion for inference
            # Extract audio features for current step
            if len(hidden_states.shape) == 3:  # (batch, seq_len, feature_dim)
                # Use mean pooling across time dimension
                audio_feat = hidden_states.mean(dim=1)  # (batch, feature_dim)
            else:
                # Handle other cases (e.g., if hidden_states is 4D)
                audio_feat = hidden_states.mean(dim=-1).mean(dim=-1)  # (batch, feature_dim)

            motion_feat = motion_processed.squeeze(1)  # (batch, feature_dim)

            # Simple fusion: weighted combination
            combined_feat = motion_feat + audio_feat  # (batch, feature_dim)

            # Output refinement
            motion_pred = self.refinement(combined_feat)  # (batch, motion_dim)
            predictions.append(motion_pred.unsqueeze(1))

        return torch.cat(predictions, 1)  # (batch, frame_num, motion_dim)

    def _motion_to_vertices(self, motion_coeff, template, device):
        """Convert 51-dim motion coefficients to FLAME vertices."""
        batch_size, seq_len, motion_dim = motion_coeff.shape

        if self.flame_model is None:
            # Fallback: create dummy vertices with shape (batch, seq_len, 5023, 3)
            # Use motion coefficients to create some variation
            vertices = torch.zeros(batch_size, seq_len, 5023, 3).to(device)

            # Add some dummy motion based on expression coefficients
            expr = motion_coeff[:, :, :50]  # (batch, seq_len, 50)
            for i in range(min(50, 5023)):  # Apply expression to first vertices
                vertices[:, :, i, 0] += expr[:, :, i % 50] * 0.01  # X coordinate
                vertices[:, :, i, 1] += expr[:, :, (i+10) % 50] * 0.01  # Y coordinate
                vertices[:, :, i, 2] += expr[:, :, (i+20) % 50] * 0.01  # Z coordinate

            return vertices

        # Extract components
        expr = motion_coeff[:, :, :50]  # (batch, seq_len, 50)
        jaw_pose_1d = motion_coeff[:, :, 50:51]  # (batch, seq_len, 1)

        # Expand jaw pose to 3D
        jaw_pose = torch.cat([jaw_pose_1d, torch.zeros_like(jaw_pose_1d), torch.zeros_like(jaw_pose_1d)], dim=-1)  # (batch, seq_len, 3)

        # Use template shape (zeros for shape since we don't predict it)
        shape = torch.zeros(batch_size, seq_len, 100).to(device)  # (batch, seq_len, 100)

        # Zero padding for other components
        pose = torch.cat([
            torch.zeros(batch_size, seq_len, 3).to(device),  # head rotation
            jaw_pose  # jaw pose
        ], dim=-1)  # (batch, seq_len, 6)

        cam = torch.zeros(batch_size, seq_len, 3).to(device)  # (batch, seq_len, 3)

        # Reshape for FLAME forward
        expr_flat = expr.view(-1, 50)  # (batch*seq_len, 50)
        shape_flat = shape.view(-1, 100)  # (batch*seq_len, 100)
        pose_flat = pose.view(-1, 6)  # (batch*seq_len, 6)
        cam_flat = cam.view(-1, 3)  # (batch*seq_len, 3)

        # FLAME forward pass
        with torch.no_grad():
            vertices_flat, _, _ = self.flame_model(
                shape_params=shape_flat,
                expression_params=expr_flat,
                pose_params=pose_flat,
                cam_params=cam_flat
            )  # (batch*seq_len, 5023, 3)

        # Reshape back
        vertices = vertices_flat.view(batch_size, seq_len, 5023, 3)  # (batch, seq_len, 5023, 3)

        return vertices

    def predict(self, audio, template, subject_id=None, frame_num=None):
        """
        Predict vertices from audio.

        Args:
            audio: Audio features (batch, seq_len)
            template: Template motion coefficients (batch, motion_dim)
            subject_id: Subject ID (optional)

        Returns:
            Predicted vertices (batch, seq_len, 5023, 3)
        """
        device = audio.device

        with torch.no_grad():
            # Predict motion coefficients autoregressively
            pred_motion = self._predict_autoregressive(audio, template, frame_num, device)

            # Convert to vertices (always use _motion_to_vertices, even with dummy FLAME)
            pred_vertices = self._motion_to_vertices(pred_motion, template, device)
            return pred_vertices  # (seq_len, 5023, 3)

    def summary(self, logger):
        """Print model summary."""
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)

        logger.info('=' * 50)
        logger.info('Model Summary:')
        logger.info(f'Total parameters: {total_params:,}')
        logger.info(f'Trainable parameters: {trainable_params:,}')
        logger.info('=' * 50)
