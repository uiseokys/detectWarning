from __future__ import annotations

import torch
from torch import nn

import action_model
from action_training_pipeline import main


class MeanMaxTemporalPoseClassifier(nn.Module):
    """Temporal pose classifier that combines masked mean and max pooling.

    The original classifier uses masked mean pooling over the GRU output. Mean pooling is stable,
    but it can dilute short, decisive motion changes such as collapse onset. This variant keeps
    the same input projection and bidirectional GRU, then concatenates masked mean and masked max
    features before classification.
    """

    def __init__(
        self,
        num_joints: int,
        input_dim: int,
        hidden_dim: int,
        num_layers: int,
        num_classes: int,
        dropout: float,
    ) -> None:
        super().__init__()
        feature_dim = num_joints * input_dim
        self.input_proj = nn.Sequential(
            nn.Linear(feature_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.temporal_encoder = nn.GRU(
            input_size=hidden_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
            bidirectional=True,
        )
        pooled_dim = hidden_dim * 4
        self.classifier = nn.Sequential(
            nn.LayerNorm(pooled_dim),
            nn.Linear(pooled_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_classes),
        )

    def forward(self, pose: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        batch_size, time_steps, num_joints, input_dim = pose.shape
        flattened = pose.view(batch_size, time_steps, num_joints * input_dim)
        encoded = self.input_proj(flattened)
        temporal_out, _hidden = self.temporal_encoder(encoded)

        mask_float = mask.unsqueeze(-1).to(dtype=temporal_out.dtype)
        mean_pooled = (temporal_out * mask_float).sum(dim=1) / mask_float.sum(dim=1).clamp(min=1.0)

        valid_mask = mask.unsqueeze(-1).to(dtype=torch.bool)
        min_value = torch.finfo(temporal_out.dtype).min
        masked_temporal = temporal_out.masked_fill(~valid_mask, min_value)
        max_pooled = masked_temporal.max(dim=1).values
        has_valid_frame = valid_mask.any(dim=1)
        max_pooled = torch.where(has_valid_frame, max_pooled, torch.zeros_like(max_pooled))

        pooled = torch.cat([mean_pooled, max_pooled], dim=-1)
        return self.classifier(pooled)

    def load_state_dict(self, state_dict, strict: bool = True, assign: bool = False):
        """Load compatible tensors and skip old mean-only classifier tensors.

        Mean+max pooling changes the first classifier layer shape from `hidden_dim * 2` to
        `hidden_dim * 4`. Existing mean-only checkpoints should not crash resume; compatible
        layers are warm-started and incompatible classifier tensors are skipped.
        """
        current_state = self.state_dict()
        compatible_state = {}
        skipped_keys = []
        for key, value in dict(state_dict).items():
            if key in current_state and current_state[key].shape == value.shape:
                compatible_state[key] = value
            else:
                skipped_keys.append(key)
        result = super().load_state_dict(compatible_state, strict=False, assign=assign)
        if skipped_keys:
            print(
                "[train] mean+max pooling warm-start skipped incompatible tensors: "
                + ", ".join(skipped_keys[:12])
                + (f" ... (+{len(skipped_keys) - 12} more)" if len(skipped_keys) > 12 else "")
            )
        return result


def enable_mean_max_pooling() -> None:
    action_model.TemporalPoseClassifier = MeanMaxTemporalPoseClassifier
    print("[train] enabled mean+max temporal pooling")


if __name__ == "__main__":
    enable_mean_max_pooling()
    main()
