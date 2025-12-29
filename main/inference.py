#!/usr/bin/env python
# yapf: disable
import os
import torch
import numpy as np
import json
from tqdm import tqdm
from transformers import Wav2Vec2Processor

from dataset.dataset import FaceFormerDataset
from models.faceformer import FaceFormer
from utils.config import get_config
from utils.utils import load_ckpt, get_logger

# yapf: enable


def reconstruct_full_flame(motion_coeff: np.ndarray, template_flame: np.ndarray,
                          dataset_format: str = 'digital_human') -> np.ndarray:
    """
    Reconstruct full FLAME coefficients from 51-dim motion coefficients.

    Args:
        motion_coeff: 51-dim motion coefficients (N, 51) [50 expr + 1 jaw]
        template_flame: Template FLAME coefficients (287,)
        dataset_format: Dataset format ('digital_human', 'mead_vhap', etc.)

    Returns:
        Full FLAME coefficients (N, 287)
    """
    seq_len = motion_coeff.shape[0]
    expr = motion_coeff[:, :50]  # (N, 50)
    jaw_pose = motion_coeff[:, 50:51]  # (N, 1)

    if dataset_format == 'digital_human':
        # digital_human format: shapecode(100) + expcode(50) + posecode(6) + cam(3) + detailcode(128)
        shape_code = template_flame[:100]  # Use template shape
        shape_coeffs = np.tile(shape_code, (seq_len, 1))  # (N, 100)

        # Head rotation (set to zero for now)
        head_rotation = np.zeros((seq_len, 3))  # (N, 3)

        # Jaw pose (extend from 1-dim to 3-dim)
        jaw_pose_full = np.concatenate([head_rotation, jaw_pose.repeat(3, axis=1)], axis=1)  # (N, 6)

        # Camera parameters (set to zero)
        cam = np.zeros((seq_len, 3))  # (N, 3)

        # Detail code (use template)
        detail_code = template_flame[159:] if len(template_flame) > 159 else np.zeros((seq_len, 128))  # (N, 128)

        flame_full = np.concatenate([
            shape_coeffs, expr, jaw_pose_full, cam, detail_code
        ], axis=1)  # (N, 287)

    elif dataset_format in ['mead_vhap', 'multimodal200']:
        # MEAD_VHAP format: shape(100) + expr(50) + rotation(3) + jaw_pose(3) + neck_pose(3) + eyes_pose(6)
        shape_code = template_flame[:100]  # Use template shape
        shape_coeffs = np.tile(shape_code, (seq_len, 1))  # (N, 100)

        # Head rotation (set to zero)
        head_rotation = np.zeros((seq_len, 3))  # (N, 3)

        # Jaw pose (extend from 1-dim to 3-dim)
        jaw_pose_full = jaw_pose.repeat(3, axis=1)  # (N, 3)

        # Neck pose (set to zero)
        neck_pose = np.zeros((seq_len, 3))  # (N, 3)

        # Eyes pose (set to zero)
        eyes_pose = np.zeros((seq_len, 6))  # (N, 6)

        flame_full = np.concatenate([
            shape_coeffs, expr, head_rotation, jaw_pose_full,
            neck_pose, eyes_pose
        ], axis=1)  # (N, 100+50+3+3+3+6=165)

        # Pad to 287 dimensions
        if flame_full.shape[1] < 287:
            padding = np.zeros((seq_len, 287 - flame_full.shape[1]))
            flame_full = np.concatenate([flame_full, padding], axis=1)

    return flame_full


def inference():
    """Run inference on test data and save vertices to evaluation-compatible format."""
    config = get_config()

    # Setup
    test_config = config.get('TEST', {})
    data_config = config['DATA']
    model_config = config['MODEL']
    train_config = config['TRAIN']

    device = train_config.get('device', 'cuda')
    save_path = train_config['save_path']

    # Setup logging
    output_base = test_config.get('test_out_path', './results/metrics')
    os.makedirs(output_base, exist_ok=True)
    logger = get_logger(os.path.join(output_base, 'inference.log'))

    # Load model
    logger.info('Loading model...')
    model = FaceFormer(config)

    # Try to load best_model.pth first, then fallback to epoch_200.pth
    model_path = os.path.join(save_path, 'best_model.pth')
    if not os.path.exists(model_path):
        # Fallback to epoch_200.pth (since training completed with 200 epochs)
        model_path = os.path.join(save_path, 'epoch_200.pth')
        logger.info(f'best_model.pth not found, trying epoch_200.pth...')

    if os.path.exists(model_path):
        load_ckpt(model, model_path)
        logger.info(f'Loaded model from {model_path}')
    else:
        logger.error(f'Model not found at {model_path} or epoch_200.pth')
        return

    # Move model to device (use CPU if GPU memory is low)
    if torch.cuda.is_available() and device == 'cuda':
        try:
            model = model.cuda()
        except RuntimeError:
            print("GPU memory insufficient, using CPU")
            device = 'cpu'
    model.eval()

    # Create processor
    processor = Wav2Vec2Processor.from_pretrained(model_config['audio_encoder_repo'])

    # Create test dataset
    logger.info('Loading test data...')
    if 'test_json_list' in data_config:
        data_root_list = data_config['data_root'][:len(data_config['test_json_list'])]
        fps_list = data_config['fps'][:len(data_config['test_json_list'])]

        test_dataset = FaceFormerDataset(
            data_root_list=data_root_list,
            json_list=data_config['test_json_list'],
            fps_list=fps_list,
            duplicate_list=[1] * len(data_config['test_json_list']),
            processor=processor,
            split='test'
        )
    else:
        # Fallback
        logger.error('test_json_list not found in config')
        return

    logger.info(f'Loaded {len(test_dataset)} test samples')

    # Run inference
    logger.info('Running inference...')
    results_saved = 0

    with torch.no_grad():
        for i, sample in enumerate(tqdm(test_dataset, desc='Inference')):
            try:
                # Prepare input
                audio = sample['audio'].unsqueeze(0).to(device)  # (1, seq_len)
                template = sample['template'].unsqueeze(0).to(device)  # (1, 51)
                motion_coeff = sample['motion_coeff'].unsqueeze(0).to(device)  # (1, seq_len, 51)

                # Get the expected sequence length from the ground truth
                expected_seq_len = motion_coeff.shape[1]

                subject_id = sample['subject_id']

                # Predict 3D vertices (matching UniTalker output format)
                pred_vertices = model.predict(audio, template, frame_num=expected_seq_len)  # (seq_len, 5023, 3)

                if len(pred_vertices.shape) == 4:  # (1, seq_len, 5023, 3)
                    pred_vertices = pred_vertices.squeeze(0)

                pred_vertices = pred_vertices.cpu().numpy()  # (seq_len, 5023, 3)

                # Determine dataset from audio_path and parse subject_id
                audio_path = sample.get('audio_path', '')
                if 'MEAD_VHAP' in audio_path:
                    dataset_part = 'MEAD_VHAP'
                elif 'MultiModal200' in audio_path:
                    dataset_part = 'MultiModal200'
                else:
                    dataset_part = 'unknown'

                # Parse subject_id to extract speaker and emotion
                if isinstance(subject_id, str):
                    # Handle different formats:
                    # MEAD_VHAP: W024_happy -> speaker=W024, emotion=happy
                    # MultiModal200: m_ch022_Surprise1 -> speaker=m_ch022, emotion=Surprise1
                    if '_' in subject_id:
                        parts = subject_id.split('_')
                        if len(parts) >= 2:
                            speaker = parts[0]
                            emotion = '_'.join(parts[1:])  # Handle cases like "Surprise1"
                        else:
                            speaker = subject_id
                            emotion = 'unknown'
                    else:
                        speaker = subject_id
                        emotion = 'unknown'
                else:
                    speaker = str(subject_id)
                    emotion = 'unknown'

                # Create output directory structure matching UniTalker format: DATASET/["speaker", "emotion"]_passionate/
                style_id = f'["{speaker}", "{emotion}"]_passionate'
                output_dir = os.path.join(output_base, dataset_part, style_id)
                os.makedirs(output_dir, exist_ok=True)

                # Save entire sequence as single .npy file (matching UniTalker format)
                output_path = os.path.join(output_dir, "0.npy")
                np.save(output_path, pred_vertices)

                results_saved += 1

            except Exception as e:
                logger.warning(f'Error processing sample {i}: {e}')
                import traceback
                logger.warning(f'Traceback: {traceback.format_exc()}')
                continue

    logger.info(f'Successfully saved {results_saved} test results')
    logger.info(f'Output directory: {output_base}')
    logger.info('Inference completed!')


if __name__ == '__main__':
    inference()

