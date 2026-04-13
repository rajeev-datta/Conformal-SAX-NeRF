import torch
import torch.nn as nn

class BatchedPinballLoss(nn.Module):
    def __init__(self, reduction="mean"):
        super().__init__()
        self.reduction = reduction

    def __call__(self, loss_dict, pred: torch.Tensor, target: torch.Tensor, quantiles: torch.Tensor) -> torch.Tensor:
        assert pred.shape == target.shape, f"{pred.shape} vs {target.shape}"
        assert quantiles.ndim == 1, "Quantiles tensor must be 1D"
        assert quantiles.shape[0] == pred.shape[0], "Quantiles tensor must have the same batch size as output"
        assert torch.all((0 < quantiles) & (quantiles < 1)), "Quantiles must be in the range (0, 1)"

        quantiles = quantiles.view([pred.shape[0]] + [1] * (pred.ndim - 1))
        error = pred - target

        loss = torch.where(
            error >= 0,
            (1 - quantiles) * error,
            quantiles * (-error),
        )

        loss_dict["loss"] += loss.mean()
        loss_dict["loss_mse"] = loss.mean()

        # if self.reduction == "sum":
        #     return loss.sum()
        # if self.reduction == "mean":
        #     return loss.mean()
        return loss_dict

def calc_mse_loss(loss, x, y):
    """
    Calculate mse loss.
    """
    # Compute loss
    loss_mse = torch.mean((x-y)**2)
    loss["loss"] += loss_mse
    loss["loss_mse"] = loss_mse
    return loss

def calc_mse_loss_raw(loss, x, y, k = 1):
    """
    Calculate mse loss for raw.
    """
    # Compute loss for raw
    loss_mse_raw = torch.mean((x-y)**2)
    loss["loss"] += k * loss_mse_raw
    loss["loss_mse_raw"] = loss_mse_raw
    return loss

def calc_tv_loss(loss, x, k):
    """
    Calculate total variation loss.
    Args:
        x (n1, n2, n3, 1): 3d density field.
        k: relative weight
    """
    n1, n2, n3 = x.shape
    tv_1 = torch.abs(x[1:,1:,1:]-x[:-1,1:,1:]).sum()
    tv_2 = torch.abs(x[1:,1:,1:]-x[1:,:-1,1:]).sum()
    tv_3 = torch.abs(x[1:,1:,1:]-x[1:,1:,:-1]).sum()
    tv = (tv_1+tv_2+tv_3) / (n1*n2*n3)
    loss["loss"] += tv * k
    loss["loss_tv"] = tv * k
    return loss



