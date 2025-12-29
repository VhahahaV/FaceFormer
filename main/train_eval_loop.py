import torch
from tqdm import tqdm
from typing import Dict, Any


def train_epoch(model: torch.nn.Module, train_loader: torch.utils.data.DataLoader,
                optimizer: torch.optim.Optimizer, criterion: torch.nn.Module,
                device: str, logger) -> float:
    """Train one epoch."""

    model.train()
    total_loss = 0.0
    num_batches = 0

    pbar = tqdm(train_loader, desc='Training')
    for batch_idx, batch in enumerate(pbar):
        # Move data to device
        audio = batch['audio'].to(device)
        motion_coeff = batch['motion_coeff'].to(device)
        template = batch['template'].to(device)

        # Get subject IDs (for future conditioning)
        subject_ids = batch.get('subject_id', ['unknown'] * audio.shape[0])

        # Create dummy subject conditioning (can be improved)
        subject_cond = torch.zeros(audio.shape[0], 1).to(device)

        # Forward pass
        optimizer.zero_grad()
        pred_motion = model(audio, template, motion_coeff, subject_cond, criterion, teacher_forcing=True)
        loss = criterion(pred_motion, motion_coeff)

        # Check for NaN/Inf in loss
        if torch.isnan(loss) or torch.isinf(loss):
            print(f"Warning: NaN/Inf loss detected, skipping batch")
            continue

        # Backward pass
        loss.backward()

        # Check for NaN/Inf gradients and clip
        try:
            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            if torch.isnan(grad_norm) or torch.isinf(grad_norm):
                print(f"Warning: NaN/Inf gradients detected, skipping batch")
                optimizer.zero_grad()
                continue
        except RuntimeError:
            # If gradient clipping fails, skip this batch
            print(f"Warning: Gradient clipping failed, skipping batch")
            optimizer.zero_grad()
            continue

        optimizer.step()

        # Update statistics
        total_loss += loss.item()
        num_batches += 1

        # Update progress bar
        pbar.set_postfix({'loss': f'{loss.item():.6f}'})

    avg_loss = total_loss / num_batches
    return avg_loss


def validate_epoch(model: torch.nn.Module, val_loader: torch.utils.data.DataLoader,
                   criterion: torch.nn.Module, device: str, logger) -> float:
    """Validate one epoch."""

    model.eval()
    total_loss = 0.0
    num_batches = 0

    with torch.no_grad():
        pbar = tqdm(val_loader, desc='Validating')
        for batch_idx, batch in enumerate(pbar):
            # Move data to device
            audio = batch['audio'].to(device)
            motion_coeff = batch['motion_coeff'].to(device)
            template = batch['template'].to(device)

            # Get subject IDs
            subject_ids = batch.get('subject_id', ['unknown'] * audio.shape[0])
            subject_cond = torch.zeros(audio.shape[0], 1).to(device)

            # Forward pass (teacher forcing for validation)
            pred_motion = model(audio, template, motion_coeff, subject_cond, criterion, teacher_forcing=True)
            loss = criterion(pred_motion, motion_coeff)

            # Update statistics
            total_loss += loss.item()
            num_batches += 1

            # Update progress bar
            pbar.set_postfix({'loss': f'{loss.item():.6f}'})

    avg_loss = total_loss / num_batches
    return avg_loss
