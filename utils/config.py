import yaml
import argparse
from typing import Dict, Any


def get_parser() -> argparse.Namespace:
    """Get argument parser."""
    parser = argparse.ArgumentParser(description='FaceFormer Training and Inference')

    # Config file
    parser.add_argument('--config', type=str, default='config/faceformer.yaml',
                        help='Path to config file')

    # Override config with command line args
    parser.add_argument('--data_root', type=str, default=None,
                        help='Data root directory')
    parser.add_argument('--data_jsons', type=str, default=None,
                        help='List of dataset JSON files (JSON string or comma-separated)')
    parser.add_argument('--train_subjects', type=str, default=None,
                        help='Training subjects')
    parser.add_argument('--val_subjects', type=str, default=None,
                        help='Validation subjects')
    parser.add_argument('--test_subjects', type=str, default=None,
                        help='Test subjects')
    parser.add_argument('--save_path', type=str, default=None,
                        help='Path to save checkpoints')
    parser.add_argument('--device', type=str, default=None,
                        help='Device to use')
    parser.add_argument('--batch_size', type=int, default=None,
                        help='Batch size')
    parser.add_argument('--lr', type=float, default=None,
                        help='Learning rate')
    parser.add_argument('--epochs', type=int, default=None,
                        help='Number of epochs')

    # Training mode
    parser.add_argument('--mode', type=str, default='train',
                        choices=['train', 'test', 'inference'],
                        help='Running mode')

    args = parser.parse_args()
    return args


def load_config(config_path: str) -> Dict[str, Any]:
    """Load configuration from YAML file."""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def merge_config(args: argparse.Namespace, config: Dict[str, Any]) -> Dict[str, Any]:
    """Merge command line arguments with config file."""
    # Override config with command line arguments
    if args.data_root is not None:
        config['DATA']['data_root'] = args.data_root
    if args.data_jsons is not None:
        # Try to parse as JSON first, then as comma-separated string
        import json
        try:
            config['DATA']['data_jsons'] = json.loads(args.data_jsons)
        except (json.JSONDecodeError, TypeError):
            # Fallback to comma-separated parsing
            config['DATA']['data_jsons'] = [s.strip() for s in args.data_jsons.split(',') if s.strip()]
    if args.train_subjects is not None:
        config['DATA']['train_subjects'] = args.train_subjects
    if args.val_subjects is not None:
        config['DATA']['val_subjects'] = args.val_subjects
    if args.test_subjects is not None:
        config['DATA']['test_subjects'] = args.test_subjects
    if args.save_path is not None:
        config['TRAIN']['save_path'] = args.save_path
    if args.device is not None:
        config['TRAIN']['device'] = args.device
    if args.batch_size is not None:
        config['TRAIN']['batch_size'] = args.batch_size
    if args.lr is not None:
        config['TRAIN']['lr'] = args.lr
    if args.epochs is not None:
        config['TRAIN']['epochs'] = args.epochs

    return config


def get_config() -> Dict[str, Any]:
    """Get final configuration."""
    args = get_parser()
    config = load_config(args.config)
    config = merge_config(args, config)
    return config
