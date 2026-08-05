from __future__ import annotations


def _torch():
    import torch
    from torch import nn
    return torch, nn


def binary_dice_loss(logits, target, epsilon: float = 1e-6):
    torch, _ = _torch()
    probability = torch.sigmoid(logits)
    dims = tuple(range(1, probability.ndim))
    intersection = (probability * target).sum(dim=dims)
    denominator = probability.sum(dim=dims) + target.sum(dim=dims)
    return 1.0 - ((2.0 * intersection + epsilon) / (denominator + epsilon))


def multiclass_dice_loss(logits, target, epsilon: float = 1e-6, ignore_index: int | None = None):
    torch, _ = _torch()
    probability = torch.softmax(logits, dim=1)
    valid = torch.ones_like(target, dtype=torch.bool) if ignore_index is None else target.ne(int(ignore_index))
    safe_target = torch.where(valid, target.long(), torch.zeros_like(target.long()))
    one_hot = torch.nn.functional.one_hot(safe_target, num_classes=logits.shape[1]).permute(0, 3, 1, 2).float()
    one_hot = one_hot * valid.unsqueeze(1)
    probability = probability * valid.unsqueeze(1)
    intersection = (probability * one_hot).sum(dim=(2, 3))
    denominator = probability.sum(dim=(2, 3)) + one_hot.sum(dim=(2, 3))
    return (1.0 - ((2.0 * intersection + epsilon) / (denominator + epsilon))).mean(dim=1)


def boundary_target(mask, width: int = 2):
    """Morphological gradient for real building masks only."""
    torch, _ = _torch()
    width = max(1, int(width))
    kernel = width * 2 + 1
    dilated = torch.nn.functional.max_pool2d(mask.float(), kernel, stride=1, padding=width)
    eroded = -torch.nn.functional.max_pool2d(-mask.float(), kernel, stride=1, padding=width)
    return (dilated - eroded).clamp_(0.0, 1.0)


def boundary_bce_loss(logits, target, width: int = 2):
    torch, nn = _torch()
    edge = boundary_target(target, width)
    values = nn.functional.binary_cross_entropy_with_logits(logits, target, reduction="none")
    weights = 1.0 + edge
    return (values * weights).flatten(1).mean(1), (values * edge).flatten(1).sum(1) / edge.flatten(1).sum(1).clamp_min(1.0)


class MaskedMultiTaskLossBase:
    pass


def _build_loss_class():
    torch, nn = _torch()

    class MaskedMultiTaskLoss(nn.Module):
        DEFAULT_WEIGHTS = {"binary": 0.5, "new_building": 1.0, "demolition": 1.0, "semantic": 0.5}

        def __init__(self, config: dict | None = None):
            super().__init__()
            config = config or {}
            self.weights = {**self.DEFAULT_WEIGHTS, **config.get("loss_weights", config.get("weights", {}))}
            self.focal_gamma = float(config.get("focal_gamma", 0.0))
            self.semantic_dice = bool(config.get("semantic_dice", True))
            self.boundary_weight = float(config.get("boundary_weight", 0.0))
            self.boundary_width_px = int(config.get("boundary_width_px", 2))
            semantic = config.get("semantic", {})
            self.semantic_ignore_index = int(config.get("semantic_ignore_index", semantic.get("ignore_index", -100)))
            class_weights = config.get("semantic_class_weights")
            self.register_buffer("semantic_class_weights", torch.tensor(class_weights, dtype=torch.float32) if class_weights else None)

        def _binary_per_sample(self, logits, target):
            bce, dice = self._binary_terms(logits, target)
            return bce + dice

        def _binary_terms(self, logits, target):
            bce = nn.functional.binary_cross_entropy_with_logits(logits, target, reduction="none")
            if self.focal_gamma > 0:
                probability = torch.sigmoid(logits)
                pt = probability * target + (1.0 - probability) * (1.0 - target)
                bce = bce * (1.0 - pt).pow(self.focal_gamma)
            return bce.flatten(1).mean(1), binary_dice_loss(logits, target)

        def _semantic_per_sample(self, logits, target):
            # Autocast FP16 is safe for the network but not for the CE/Dice
            # reduction over large semantic masks.  Compute this supervised
            # reduction in FP32; gradients still flow to the original logits.
            logits = logits.float()
            ce = nn.functional.cross_entropy(logits, target.long(), weight=self.semantic_class_weights,
                                              reduction="none", ignore_index=self.semantic_ignore_index)
            valid = target.ne(self.semantic_ignore_index)
            result = (ce * valid).flatten(1).sum(1) / valid.flatten(1).sum(1).clamp_min(1)
            return result + multiclass_dice_loss(logits, target, ignore_index=self.semantic_ignore_index) if self.semantic_dice else result

        @staticmethod
        def _select_targets(values: list, indices, device, *, semantic: bool = False):
            selected = [values[int(index)] for index in indices.tolist()]
            if any(value is None for value in selected):
                raise ValueError("Supervision flag is true but the corresponding target is missing")
            result = torch.stack(selected).to(device, non_blocking=True)
            return result.long() if semantic else result.float()

        def forward(self, outputs: dict, batch: dict):
            zero = outputs["binary_change"].sum() * 0.0
            per_sample_terms: dict[str, tuple[object, object]] = {}
            task_losses: dict[str, object] = {}
            breakdown: dict[str, object] = {}

            binary_indices = torch.where(batch["has_binary"].to(outputs["binary_change"].device))[0]
            if len(binary_indices):
                target = self._select_targets(batch["binary"], binary_indices, outputs["binary_change"].device)
                bce, dice = self._binary_terms(outputs["binary_change"][binary_indices], target); values = bce + dice
                breakdown.update(binary_bce=bce.mean(), binary_dice=dice.mean(), binary_active_batches=len(binary_indices), binary_supervised_pixels=target.numel())
                per_sample_terms["binary"] = (binary_indices, values)
                task_losses["binary"] = values.mean()
            else:
                task_losses["binary"] = zero

            directional_indices = torch.where(batch["has_directional"].to(outputs["binary_change"].device))[0]
            boundary_values = []
            for task, output_name, target_name in (
                ("new_building", "new_building", "new"), ("demolition", "demolition", "demolition")
            ):
                if len(directional_indices):
                    target = self._select_targets(batch[target_name], directional_indices, outputs[output_name].device)
                    bce, dice = self._binary_terms(outputs[output_name][directional_indices], target); values = bce + dice
                    breakdown.update({f"{task}_bce": bce.mean(), f"{task}_dice": dice.mean(), f"{task}_active_batches": len(directional_indices), f"{task}_supervised_pixels": target.numel()})
                    if self.boundary_weight > 0:
                        _, boundary = boundary_bce_loss(
                            outputs[output_name][directional_indices],
                            target,
                            self.boundary_width_px,
                        )
                        values = values + self.boundary_weight * boundary
                        boundary_values.append(boundary)
                        breakdown[f"{task}_boundary"] = boundary.mean()
                    per_sample_terms[task] = (directional_indices, values)
                    task_losses[task] = values.mean()
                else:
                    task_losses[task] = zero

            semantic_indices = torch.where(batch["has_semantic"].to(outputs["binary_change"].device))[0]
            semantic_values = []
            if len(semantic_indices):
                for output_name, target_name in (("semantic_t1", "semantic_t1"), ("semantic_t2", "semantic_t2")):
                    target = self._select_targets(batch[target_name], semantic_indices, outputs[output_name].device, semantic=True)
                    semantic_values.append(self._semantic_per_sample(outputs[output_name][semantic_indices], target))
                    breakdown[f"{output_name}_active_batches"] = len(semantic_indices)
                    breakdown[f"{output_name}_supervised_pixels"] = int(target.ne(self.semantic_ignore_index).sum())
                values = torch.stack(semantic_values).mean(0)
                per_sample_terms["semantic"] = (semantic_indices, values)
                task_losses["semantic"] = values.mean()
                breakdown["semantic_t1"] = semantic_values[0].mean(); breakdown["semantic_t2"] = semantic_values[1].mean()
            else:
                task_losses["semantic"] = zero

            total = sum(task_losses[name] * float(self.weights[name]) for name in self.DEFAULT_WEIGHTS)
            dataset_losses: dict[str, object] = {}
            for dataset in sorted(set(batch["dataset_name"])):
                positions = {index for index, name in enumerate(batch["dataset_name"]) if name == dataset}
                components = []
                for task, (indices, values) in per_sample_terms.items():
                    chosen = [offset for offset, original in enumerate(indices.tolist()) if original in positions]
                    if chosen:
                        components.append(values[chosen].mean() * float(self.weights[task]))
                dataset_losses[dataset] = sum(components, zero) if components else zero
            details = {
                **{f"loss_{name}": value.detach() for name, value in task_losses.items()},
                **{f"loss_dataset_{name}": value.detach() for name, value in dataset_losses.items()},
                "loss_total": total.detach(),
                "interior_loss": (
                    torch.stack([task_losses["new_building"], task_losses["demolition"]]).mean().detach()
                    if len(directional_indices) else zero.detach()
                ),
                "boundary_loss": (
                    torch.cat(boundary_values).mean().detach() if boundary_values else zero.detach()
                ),
                **{f"loss_{name}": value.detach() if hasattr(value, "detach") else value for name, value in breakdown.items()},
            }
            return total, details

    return MaskedMultiTaskLoss


MaskedMultiTaskLoss = _build_loss_class()
