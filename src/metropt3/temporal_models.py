from __future__ import annotations

from typing import Any


def _torch_modules():
    try:
        import torch
        from torch import nn
    except ImportError as exc:
        raise RuntimeError(
            "Temporal models require requirements-experiment.txt"
        ) from exc
    return torch, nn


def build_temporal_model(
    model_name: str,
    *,
    input_channels: int,
    config: dict[str, Any],
):
    """Build the frozen TCN or Attention-TCN architecture lazily."""
    if model_name not in {"tcn", "attention_tcn"}:
        raise ValueError(f"Unsupported temporal model: {model_name}")
    torch, nn = _torch_modules()
    encoder_config = config["models"]["shared_tcn_encoder"]
    residual_channels = [int(value) for value in encoder_config["residual_channels"]]
    dilations = [int(value) for value in encoder_config["dilations"]]
    kernel_size = int(encoder_config["kernel_size"])
    dropout = float(encoder_config["dropout"])

    if len(residual_channels) != len(dilations):
        raise ValueError("Each residual block requires one dilation")
    if int(encoder_config["convolutions_per_block"]) != 2:
        raise ValueError("The frozen encoder requires two convolutions per block")

    class CausalConvolution(nn.Module):
        def __init__(self, inputs: int, outputs: int, dilation: int):
            super().__init__()
            self.right_padding = (kernel_size - 1) * dilation
            self.convolution = nn.Conv1d(
                inputs,
                outputs,
                kernel_size,
                padding=self.right_padding,
                dilation=dilation,
            )

        def forward(self, values):
            convolved = self.convolution(values)
            if self.right_padding:
                convolved = convolved[:, :, : -self.right_padding]
            return convolved

    class ResidualBlock(nn.Module):
        def __init__(self, inputs: int, outputs: int, dilation: int):
            super().__init__()
            self.network = nn.Sequential(
                CausalConvolution(inputs, outputs, dilation),
                nn.ReLU(),
                nn.Dropout(dropout),
                CausalConvolution(outputs, outputs, dilation),
                nn.ReLU(),
                nn.Dropout(dropout),
            )
            self.residual = (
                nn.Identity() if inputs == outputs else nn.Conv1d(inputs, outputs, 1)
            )

        def forward(self, values):
            return self.network(values) + self.residual(values)

    class TemporalClassifier(nn.Module):
        def __init__(self):
            super().__init__()
            blocks = []
            channels = input_channels
            for outputs, dilation in zip(residual_channels, dilations):
                blocks.append(ResidualBlock(channels, outputs, dilation))
                channels = outputs
            self.encoder = nn.Sequential(*blocks)
            self.uses_attention = model_name == "attention_tcn"
            if self.uses_attention:
                hidden = int(config["models"]["attention_tcn"]["attention_hidden_size"])
                self.attention = nn.Sequential(
                    nn.Conv1d(channels, hidden, 1),
                    nn.Tanh(),
                    nn.Conv1d(hidden, 1, 1),
                )
            self.classifier = nn.Linear(channels, 1)

        def forward(self, values, *, return_attention: bool = False):
            encoded = self.encoder(values.transpose(1, 2))
            weights = None
            if self.uses_attention:
                weights = torch.softmax(self.attention(encoded), dim=-1)
                pooled = (encoded * weights).sum(dim=-1)
            else:
                pooled = encoded[:, :, -1]
            logits = self.classifier(pooled).squeeze(-1)
            if return_attention:
                return logits, None if weights is None else weights.squeeze(1)
            return logits

    return TemporalClassifier()


def temporal_parameter_count(model) -> int:
    return int(sum(parameter.numel() for parameter in model.parameters()))
