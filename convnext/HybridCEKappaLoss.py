import torch
import torch.nn as nn
import torch.nn.functional as F


class WeightedKappaLoss(nn.Module):
    def __init__(self, num_classes=5, epsilon=1e-7):
        super().__init__()
        self.num_classes = num_classes  # number of ordinal classes (APTOS: 5 grades, 0-4)
        self.epsilon = epsilon          # numerical stability constant for log and division

        i = torch.arange(num_classes).float().unsqueeze(1)
        j = torch.arange(num_classes).float().unsqueeze(0)
        weights = ((i - j) ** 2) / ((num_classes - 1) ** 2)
        self.register_buffer("weights", weights)

    def forward(self, logits, targets):
        probs = F.softmax(logits, dim=1)
        targets_onehot = F.one_hot(targets, self.num_classes).float()

        confusion = targets_onehot.t() @ probs
        true_hist = targets_onehot.sum(dim=0)
        pred_hist = probs.sum(dim=0)

        expected = torch.outer(true_hist, pred_hist)
        expected = expected / (expected.sum() + self.epsilon) * confusion.sum()

        numerator = (self.weights * confusion).sum()
        denominator = (self.weights * expected).sum()

        # ratio form: stays in [0, ~1], mixes cleanly with CE.
        # (the old log form could dive to log(eps) ≈ -16 and overwhelm CE.)
        return numerator / (denominator + self.epsilon)


class HybridCEKappaLoss(nn.Module):
    def __init__(
        self,
        num_classes=5,           # number of ordinal classes
        alpha=1.0,               # CE/kappa mixing weight: 1.0=pure CE, 0.0=pure kappa; overwritten each epoch by get_alpha()
        class_weights=None,      # per-class CE weights (tensor of shape [num_classes]) for handling imbalance; None = uniform
        label_smoothing=0.0,     # smooths one-hot targets in CE; 0.05 is gentle, helps with noisy ordinal labels
    ):
        super().__init__()
        self.alpha = alpha
        self.ce = nn.CrossEntropyLoss(
            weight=class_weights,
            label_smoothing=label_smoothing,
        )
        self.kappa = WeightedKappaLoss(num_classes=num_classes)

    def forward(self, logits, targets):
        ce_loss = self.ce(logits, targets)
        kappa_loss = self.kappa(logits, targets)
        return self.alpha * ce_loss + (1.0 - self.alpha) * kappa_loss


def get_alpha(
    epoch,
    warmup_epochs=5,    # epochs of pure CE before kappa is mixed in; longer = more stable but slower to align with QWK
    total_epochs=30,    # total training epochs; defines the decay horizon
    alpha_final=0.2,    # alpha floor at end of training; 0.2 = 80% kappa dominant late; lower = more kappa-driven, higher = safer
):
    if epoch < warmup_epochs:
        return 1.0
    progress = (epoch - warmup_epochs) / max(1, total_epochs - warmup_epochs)
    return max(alpha_final, 1.0 - progress * (1.0 - alpha_final))
