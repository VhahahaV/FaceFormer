import torch
import torch.nn as nn


class FaceFormerLoss(nn.Module):
    """FaceFormer loss function for vertex prediction."""

    def __init__(self, config: dict):
        super(FaceFormerLoss, self).__init__()
        self.rec_weight = config['LOSS'].get('rec_weight', 1.0)
        self.mse_loss = nn.MSELoss()

    def forward(self, pred_vertices: torch.Tensor, target_vertices: torch.Tensor) -> torch.Tensor:
        """Forward pass with numerical stability checks.

        Args:
            pred_vertices: Predicted motion coefficients (batch, seq_len, 51)
            target_vertices: Target motion coefficients (batch, seq_len, 51)

        Returns:
            Loss value
        """
        # Ensure inputs are finite (only warn, don't modify)
        if not torch.isfinite(pred_vertices).all():
            print(f"Warning: Non-finite values in predictions")

        if not torch.isfinite(target_vertices).all():
            print(f"Warning: Non-finite values in targets")

        # Compute MSE loss
        rec_loss = self.mse_loss(pred_vertices, target_vertices)

        # Check for NaN/Inf in loss
        if not torch.isfinite(rec_loss):
            print(f"Warning: Non-finite loss detected, returning zero loss")
            return torch.tensor(0.0, device=pred_vertices.device, requires_grad=True)

        total_loss = self.rec_weight * rec_loss
        return total_loss

    def __repr__(self):
        return f"FaceFormerLoss(rec_weight={self.rec_weight})"

