"""Model utility functions for channel adaptation and transfer learning."""

from typing import Optional, Tuple, Dict, Any
import torch
import torch.nn as nn
import logging

logger: logging.Logger = logging.getLogger(__name__)


def adapt_input_channels(
    model: nn.Module,
    original_channels: int,
    target_channels: int,
    init_method: str = "average",
) -> nn.Module:
    """
    Adapt the first convolutional layer to accept different number of input channels.

    Args:
        model: PyTorch model
        original_channels: Original number of input channels (e.g., 3 for RGB)
        target_channels: Target number of input channels (e.g., 1 for grayscale)
        init_method: How to initialize new weights ('average', 'random', 'zeros')

    Returns:
        Modified model with adapted first layer
    """
    if original_channels == target_channels:
        logger.info(
            f"No channel adaptation needed ({original_channels} -> {target_channels})"
        )
        return model

    logger.info(
        f"Adapting input channels: {original_channels} -> {target_channels} (method: {init_method})"
    )

    # Find the first convolutional layer
    first_conv: Optional[nn.Module] = None
    first_conv_name: Optional[str] = None

    for name, module in model.named_modules():
        if isinstance(module, nn.Conv2d):
            first_conv = module
            first_conv_name = name
            break

    if first_conv is None:
        raise ValueError("No Conv2d layer found in the model")

    # Get original weights and bias
    original_weight: torch.Tensor = first_conv.weight.data
    original_bias: Optional[torch.Tensor] = (
        first_conv.bias.data if first_conv.bias is not None else None
    )

    # Create new layer with adjusted input channels
    new_conv: nn.Conv2d = nn.Conv2d(
        target_channels,
        first_conv.out_channels,
        kernel_size=first_conv.kernel_size,
        stride=first_conv.stride,
        padding=first_conv.padding,
        dilation=first_conv.dilation,
        groups=first_conv.groups,
        bias=(original_bias is not None),
    )

    # Initialize new weights based on init_method
    with torch.no_grad():
        if init_method == "average":
            # Average across original channels
            if target_channels == 1 and original_channels == 3:
                # RGB to grayscale: average RGB weights
                new_weight: torch.Tensor = original_weight.mean(dim=1, keepdim=True)
                new_conv.weight.copy_(new_weight)
            elif target_channels > original_channels:
                # Repeat channels
                repeats: Tuple = (target_channels // original_channels,) + (1,) * 3
                remainder: int = target_channels % original_channels
                repeated_weight: torch.Tensor = original_weight.repeat(repeats)
                if remainder > 0:
                    remainder_weight: torch.Tensor = original_weight[
                        :, :remainder, :, :
                    ]
                    repeated_weight = torch.cat(
                        [repeated_weight, remainder_weight], dim=1
                    )
                new_conv.weight.copy_(repeated_weight)
            else:
                # Take subset of channels
                new_conv.weight.copy_(original_weight[:, :target_channels, :, :])
        elif init_method == "random":
            # Initialize new dimensions randomly
            new_conv.weight = nn.Parameter(torch.randn_like(new_conv.weight) * 0.01)
        elif init_method == "zeros":
            # Initialize new dimensions with zeros
            new_conv.weight = nn.Parameter(torch.zeros_like(new_conv.weight))
        else:
            raise ValueError(f"Unknown init_method: {init_method}")

        # Copy bias if exists
        if original_bias is not None:
            new_conv.bias = nn.Parameter(original_bias.clone())

    # Replace the layer in the model
    if first_conv_name:
        parts: list = first_conv_name.split(".")
        current: Any = model
        for part in parts[:-1]:
            current = getattr(current, part)
        setattr(current, parts[-1], new_conv)

    logger.info(f"Successfully adapted layer: {first_conv_name}")
    return model


def freeze_backbone(
    model: nn.Module, freeze_backbone: bool = True, freeze_bn: bool = True
) -> nn.Module:
    """
    Freeze/unfreeze backbone layers of a model.

    Args:
        model: PyTorch model
        freeze_backbone: Whether to freeze backbone parameters
        freeze_bn: Whether to freeze batch normalization layers

    Returns:
        Modified model with frozen/unfrozen backbone
    """
    backbone_names: set = {"backbone", "stem", "stage", "layer"}
    bn_names: set = {"BatchNorm", "bn", "batch_norm"}

    for name, param in model.named_parameters():
        # Check if this is a backbone parameter
        is_backbone: bool = any(bn in name.lower() for bn in backbone_names)
        is_bn: bool = any(bn in name for bn in bn_names)

        if freeze_backbone and is_backbone:
            param.requires_grad = False
            logger.debug(f"Frozen: {name}")
        elif freeze_bn and is_bn:
            param.requires_grad = False
            logger.debug(f"Frozen BN: {name}")
        else:
            param.requires_grad = True
            logger.debug(f"Trainable: {name}")

    frozen_count: int = sum(1 for p in model.parameters() if not p.requires_grad)
    total_count: int = sum(1 for _ in model.parameters())
    logger.info(f"Frozen {frozen_count}/{total_count} parameters")

    return model


def unfreeze_model(model: nn.Module) -> nn.Module:
    """
    Unfreeze all parameters in a model.

    Args:
        model: PyTorch model

    Returns:
        Model with all parameters unfrozen
    """
    for param in model.parameters():
        param.requires_grad = True
    logger.info("Unfroze all model parameters")
    return model


def count_parameters(
    model: nn.Module, only_trainable: bool = True
) -> Tuple[int, float]:
    """
    Count the number of parameters in a model.

    Args:
        model: PyTorch model
        only_trainable: Only count trainable parameters

    Returns:
        Tuple of (count, size_in_mb)
    """
    if only_trainable:
        params_list: list = [p for p in model.parameters() if p.requires_grad]
    else:
        params_list = list(model.parameters())

    count: int = sum(p.numel() for p in params_list)
    size_bytes: int = sum(p.numel() * p.element_size() for p in params_list)
    size_mb: float = size_bytes / (1024 * 1024)

    return count, size_mb


def get_optimizer(
    model: nn.Module,
    optimizer_type: str = "adamw",
    lr: float = 1e-4,
    weight_decay: float = 5e-4,
    momentum: float = 0.937,
    **kwargs: Any,
) -> torch.optim.Optimizer:
    """
    Create optimizer for model training.

    Args:
        model: PyTorch model
        optimizer_type: Type of optimizer ('adamw', 'adam', 'sgd')
        lr: Learning rate
        weight_decay: Weight decay
        momentum: Momentum (for SGD)
        **kwargs: Additional optimizer arguments

    Returns:
        Optimizer instance
    """
    trainable_params: list = [p for p in model.parameters() if p.requires_grad]

    if optimizer_type.lower() == "adamw":
        optimizer = torch.optim.AdamW(
            trainable_params, lr=lr, weight_decay=weight_decay, **kwargs
        )
    elif optimizer_type.lower() == "adam":
        optimizer = torch.optim.Adam(
            trainable_params, lr=lr, weight_decay=weight_decay, **kwargs
        )
    elif optimizer_type.lower() == "sgd":
        optimizer = torch.optim.SGD(
            trainable_params,
            lr=lr,
            momentum=momentum,
            weight_decay=weight_decay,
            **kwargs,
        )
    else:
        raise ValueError(f"Unknown optimizer type: {optimizer_type}")

    logger.info(
        f"Created {optimizer_type.upper()} optimizer with {len(trainable_params)} parameters"
    )
    return optimizer


def get_scheduler(
    optimizer: torch.optim.Optimizer,
    scheduler_type: str = "cosine",
    epochs: int = 100,
    min_lr: float = 1e-6,
    warmup_epochs: int = 5,
    **kwargs: Any,
) -> torch.optim.lr_scheduler._LRScheduler:
    """
    Create learning rate scheduler.

    Args:
        optimizer: Optimizer instance
        scheduler_type: Type of scheduler ('cosine', 'step', 'onecycle')
        epochs: Total number of training epochs
        min_lr: Minimum learning rate
        warmup_epochs: Number of warmup epochs
        **kwargs: Additional scheduler arguments

    Returns:
        Learning rate scheduler
    """
    if scheduler_type.lower() == "cosine":
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=epochs - warmup_epochs, eta_min=min_lr, **kwargs
        )
    elif scheduler_type.lower() == "step":
        scheduler = torch.optim.lr_scheduler.StepLR(
            optimizer, step_size=epochs // 3, gamma=0.1, **kwargs
        )
    elif scheduler_type.lower() == "onecycle":
        scheduler = torch.optim.lr_scheduler.OneCycleLR(
            optimizer,
            max_lr=optimizer.param_groups[0]["lr"],
            total_steps=epochs,
            pct_start=warmup_epochs / epochs,
            **kwargs,
        )
    else:
        raise ValueError(f"Unknown scheduler type: {scheduler_type}")

    logger.info(
        f"Created {scheduler_type.upper()} scheduler with warmup={warmup_epochs} epochs"
    )
    return scheduler
