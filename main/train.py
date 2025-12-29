#!/usr/bin/env python
# yapf: disable
import glob
import os
import torch
import torch.optim
import torch.utils.data
from tensorboardX import SummaryWriter

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from dataset.dataset import get_dataloaders
from loss.loss import FaceFormerLoss
from models.faceformer import FaceFormer
from utils.utils import (
    get_average_meter_dict, get_logger, seed_everything, load_ckpt
)
from train_eval_loop import train_epoch, validate_epoch

# yapf: enable


def main():
    from utils.config import get_config
    config = get_config()

    seed_everything(42)

    # Get data loaders
    train_loader, val_loader, test_loader = get_dataloaders(config)

    # Update config with dataset info
    config['identity_num'] = train_loader.dataset.get_identity_num()

    # Create directories
    train_config = config['TRAIN']
    save_path = train_config['save_path']
    os.makedirs(save_path, exist_ok=True)

    # Setup logging
    log_file = os.path.join(save_path, 'train.log')
    logger = get_logger(log_file)
    writer = SummaryWriter(save_path)

    logger.info('=> creating model ...')
    model = FaceFormer(config)

    # Load pretrained weights if specified
    if train_config.get('weight_path'):
        logger.info(f'=> loading model {train_config["weight_path"]} ...')
        load_ckpt(model, train_config['weight_path'])

    # Move model to device
    device = train_config.get('device', 'cuda')
    if torch.cuda.is_available() and device == 'cuda':
        model = model.cuda()
        logger.info('Model moved to GPU')
    else:
        logger.info('Using CPU for training')

    model.summary(logger)

    # Create loss function
    criterion = FaceFormerLoss(config)

    # Create optimizer with better stability settings
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=train_config['lr'],
        betas=(0.9, 0.999),  # Default but explicit
        eps=1e-8,            # Slightly higher for numerical stability
        weight_decay=1e-4    # Add small weight decay for regularization
    )

    # Training loop
    best_loss = float('inf')
    epochs = train_config['epochs']
    save_every = train_config.get('save_every', 25)
    eval_every = train_config.get('eval_every', 1)

    for epoch in range(1, epochs + 1):
        logger.info(f'Epoch {epoch}/{epochs}')

        # Train one epoch
        train_loss = train_epoch(model, train_loader, optimizer, criterion, device, logger)

        # Log training loss
        writer.add_scalar('train/loss', train_loss, epoch)
        logger.info(f'Train Loss: {train_loss:.6f}')

        # Validate
        if epoch % eval_every == 0:
            val_loss = validate_epoch(model, val_loader, criterion, device, logger)
            writer.add_scalar('val/loss', val_loss, epoch)
            logger.info(f'Val Loss: {val_loss:.6f}')

            # Save best model
            if val_loss < best_loss:
                best_loss = val_loss
                torch.save(model.state_dict(), os.path.join(save_path, 'best_model.pth'))
                logger.info('Saved best model')

        # Save checkpoint
        if epoch % save_every == 0:
            torch.save(model.state_dict(), os.path.join(save_path, f'epoch_{epoch}.pth'))
            logger.info(f'Saved checkpoint at epoch {epoch}')

    # Final test
    if train_config.get('do_test', True):
        logger.info('Running final test...')
        test_loss = validate_epoch(model, test_loader, criterion, device, logger)
        logger.info(f'Test Loss: {test_loss:.6f}')

    logger.info('Training completed!')


if __name__ == '__main__':
    main()

