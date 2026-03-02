"""
Shared evaluation loops for classification (Phases 1 & 2).
"""
import torch
from tqdm import tqdm


def evaluate(model, loader, criterion, device, desc="Evaluating",
             use_amp=False, use_channels_last=False):
    """Standard evaluation: returns (loss, accuracy).

    Args:
        model: PyTorch model in eval mode.
        loader: DataLoader yielding (inputs, targets).
        criterion: Loss function.
        device: torch.device.
        desc: Progress bar description.
        use_amp: Enable automatic mixed precision.
        use_channels_last: Convert inputs to channels_last memory format.
    """
    model.eval()
    total_loss, correct, total = 0.0, 0, 0
    with torch.no_grad():
        for inputs, targets in tqdm(loader, desc=desc, leave=False):
            inputs = inputs.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)
            if use_channels_last:
                try:
                    inputs = inputs.to(memory_format=torch.channels_last)
                except Exception:
                    pass
            with torch.autocast("cuda", enabled=use_amp):
                outputs = model(inputs)
                loss = criterion(outputs, targets)
            total_loss += loss.item() * inputs.size(0)
            _, preds = outputs.max(1)
            correct += preds.eq(targets).sum().item()
            total += targets.size(0)
    return total_loss / total if total else 0.0, correct / total if total else 0.0


def evaluate_imagenet_a(model, loader, criterion, device, class_mapping,
                        desc="ImageNet-A", use_amp=False, use_channels_last=False):
    """Evaluate on ImageNet-A with proper class mapping.

    The model outputs 1000 logits. ImageFolder assigns 0..199 labels
    based on sorted folder names. We map those folder indices to the
    corresponding ImageNet-1K class indices for accuracy computation.
    """
    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for inputs, targets in tqdm(loader, desc=desc, leave=False):
            inputs = inputs.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)
            if use_channels_last:
                try:
                    inputs = inputs.to(memory_format=torch.channels_last)
                except Exception:
                    pass
            with torch.autocast("cuda", enabled=use_amp):
                outputs = model(inputs)
            _, preds = outputs.max(1)
            mapped_targets = torch.tensor(
                [class_mapping[t.item()] for t in targets],
                device=device, dtype=torch.long,
            )
            correct += preds.eq(mapped_targets).sum().item()
            total += targets.size(0)
    acc = correct / total if total else 0.0
    return acc
