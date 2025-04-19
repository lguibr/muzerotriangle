# File: muzerotriangle/nn/model.py
import logging
import math
from typing import cast

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..config import EnvConfig, ModelConfig

logger = logging.getLogger(__name__)


# --- conv_block, PositionalEncoding remain the same ---
def conv_block(
    in_channels, out_channels, kernel_size, stride, padding, use_batch_norm, activation
):
    layers = [
        nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size,
            stride,
            padding,
            bias=not use_batch_norm,
        )
    ]
    if use_batch_norm:
        layers.append(nn.BatchNorm2d(out_channels))
    layers.append(activation())
    return nn.Sequential(*layers)


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model)
        )
        pe = torch.zeros(max_len, d_model)
        pe[:, 0::2] = torch.sin(position * div_term)
        if d_model > 1:
            pe[:, 1::2] = torch.cos(position * div_term[: d_model // 2])
        pe = pe.unsqueeze(1)
        self.register_buffer("pe", pe)

    def forward(self, x):
        pe_buffer = self.pe
        assert isinstance(pe_buffer, torch.Tensor)
        if x.shape[0] > pe_buffer.shape[0]:
            raise ValueError(
                f"Input seq len {x.shape[0]} > max_len {pe_buffer.shape[0]}"
            )
        if x.shape[2] != pe_buffer.shape[2]:
            raise ValueError(f"Input dim {x.shape[2]} != PE dim {pe_buffer.shape[2]}")
        x = x + pe_buffer[: x.size(0)]
        return cast("torch.Tensor", self.dropout(x))


# --- ResidualBlock ---
class ResidualBlock(nn.Module):
    """Standard Residual Block."""

    def __init__(
        self, channels: int, use_batch_norm: bool, activation: type[nn.Module]
    ):
        super().__init__()
        self.conv1 = conv_block(channels, channels, 3, 1, 1, use_batch_norm, activation)
        self.conv2 = nn.Conv2d(channels, channels, 3, 1, 1, bias=not use_batch_norm)
        self.bn2 = nn.BatchNorm2d(channels) if use_batch_norm else nn.Identity()
        self.activation = activation()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual: torch.Tensor = x  # Hint residual type
        out: torch.Tensor = self.conv1(x)
        out = self.conv2(out)
        out = self.bn2(out)
        # --- Explicitly type hint the result of addition ---
        out_sum: torch.Tensor = out + residual
        out_activated: torch.Tensor = self.activation(out_sum)
        # ---
        return out_activated


# --- MuZeroNet Implementation ---
class Permute(nn.Module):
    def __init__(self, *dims):
        super().__init__()
        self.dims = dims

    def forward(self, x):
        return x.permute(*self.dims)


class ConcatFeatures(nn.Module):
    def __init__(self, dim=1):
        super().__init__()
        self.dim = dim

    def forward(self, features_tuple):
        return torch.cat(features_tuple, dim=self.dim)


class RepresentationEncoderWrapper(nn.Module):
    """Wraps CNN/TF layers for representation fn. Returns flattened tensor."""

    def __init__(self, cnn_tf_layers: nn.Module):
        super().__init__()
        self.cnn_tf_layers = cnn_tf_layers
        self.flatten = nn.Flatten(start_dim=1)

    def forward(self, grid_state: torch.Tensor) -> torch.Tensor:
        encoded: torch.Tensor = self.cnn_tf_layers(grid_state)  # Hint type
        encoded_flat: torch.Tensor  # Declare type
        if len(encoded.shape) > 2:
            encoded_flat = self.flatten(encoded)
        else:
            encoded_flat = encoded
        # --- Ensure return type matches annotation ---
        return encoded_flat


class MuZeroNet(nn.Module):
    """MuZero Network Implementation."""

    # ... (init remains the same, including _build_representation_cnn_tf_encoder which now returns RepresentationEncoderWrapper) ...
    def __init__(self, model_config: ModelConfig, env_config: EnvConfig):
        super().__init__()
        self.model_config = model_config
        self.env_config = env_config
        self.action_dim = int(env_config.ACTION_DIM)
        self.hidden_dim = model_config.HIDDEN_STATE_DIM  # type: ignore[call-overload]
        self.activation_cls: type[nn.Module] = getattr(
            nn, model_config.ACTIVATION_FUNCTION
        )
        dummy_input_grid = torch.zeros(
            1, model_config.GRID_INPUT_CHANNELS, env_config.ROWS, env_config.COLS
        )
        dummy_other = torch.zeros(1, model_config.OTHER_NN_INPUT_FEATURES_DIM)
        self.representation_encoder = self._build_representation_cnn_tf_encoder()
        with torch.no_grad():
            encoded_output = self.representation_encoder(dummy_input_grid)
            self.encoded_flat_size = encoded_output.shape[1]
        rep_projector_input_dim = (
            self.encoded_flat_size + model_config.OTHER_NN_INPUT_FEATURES_DIM
        )
        rep_fc_layers: list[nn.Module] = []
        in_features = rep_projector_input_dim
        for hidden_dim_fc in model_config.REP_FC_DIMS_AFTER_ENCODER:
            rep_fc_layers.append(nn.Linear(in_features, hidden_dim_fc))
            in_features = hidden_dim_fc  # Simplified line
        rep_fc_layers.append(nn.Linear(in_features, self.hidden_dim))
        self.representation_projector = nn.Sequential(*rep_fc_layers)
        self.action_encoder = nn.Linear(
            self.action_dim, model_config.ACTION_ENCODING_DIM
        )
        dynamics_input_dim = self.hidden_dim + model_config.ACTION_ENCODING_DIM
        dynamics_layers: list[nn.Module] = [
            nn.Linear(dynamics_input_dim, self.hidden_dim),
            self.activation_cls(),
        ]
        for _ in range(model_config.DYNAMICS_NUM_RESIDUAL_BLOCKS):
            dynamics_layers.extend(
                [nn.Linear(self.hidden_dim, self.hidden_dim), self.activation_cls()]
            )
            self.dynamics_core = nn.Sequential(*dynamics_layers)
        reward_head_layers: list[nn.Module] = []
        reward_in = self.hidden_dim
        for hidden_dim_fc in model_config.REWARD_HEAD_DIMS:
            reward_head_layers.append(nn.Linear(reward_in, hidden_dim_fc))
            reward_in = hidden_dim_fc  # Simplified line
        reward_head_layers.append(
            nn.Linear(reward_in, model_config.REWARD_SUPPORT_SIZE)
        )
        self.reward_head = nn.Sequential(*reward_head_layers)
        prediction_layers: list[nn.Module] = []
        pred_in = self.hidden_dim
        for _ in range(model_config.PREDICTION_NUM_RESIDUAL_BLOCKS):
            prediction_layers.extend(
                [nn.Linear(pred_in, self.hidden_dim), self.activation_cls()]
            )
            pred_in = self.hidden_dim
            self.prediction_core = nn.Sequential(*prediction_layers)
        policy_head_layers: list[nn.Module] = []
        policy_in = self.hidden_dim
        for hidden_dim_fc in model_config.POLICY_HEAD_DIMS:
            policy_head_layers.append(nn.Linear(policy_in, hidden_dim_fc))
            policy_in = hidden_dim_fc  # Simplified line
        policy_head_layers.append(nn.Linear(policy_in, self.action_dim))
        self.policy_head = nn.Sequential(*policy_head_layers)
        value_head_layers: list[nn.Module] = []
        value_in = self.hidden_dim
        for hidden_dim_fc in model_config.VALUE_HEAD_DIMS:
            value_head_layers.append(nn.Linear(value_in, hidden_dim_fc))
            value_in = hidden_dim_fc  # Simplified line
        value_head_layers.append(nn.Linear(value_in, model_config.NUM_VALUE_ATOMS))
        self.value_head = nn.Sequential(*value_head_layers)

    def _build_representation_cnn_tf_encoder(
        self,
    ) -> RepresentationEncoderWrapper:  # Return the wrapper type
        layers: list[nn.Module] = []
        in_channels = self.model_config.GRID_INPUT_CHANNELS
        # CNN Body
        for i, out_channels in enumerate(self.model_config.CONV_FILTERS):
            layers.append(
                conv_block(
                    in_channels,
                    out_channels,
                    self.model_config.CONV_KERNEL_SIZES[i],
                    self.model_config.CONV_STRIDES[i],
                    self.model_config.CONV_PADDING[i],
                    self.model_config.USE_BATCH_NORM,
                    self.activation_cls,
                )
            )
            in_channels = out_channels
        # Residual Body
        if self.model_config.NUM_RESIDUAL_BLOCKS > 0:
            res_channels = self.model_config.RESIDUAL_BLOCK_FILTERS
            if in_channels != res_channels:
                layers.append(
                    conv_block(
                        in_channels,
                        res_channels,
                        1,
                        1,
                        0,
                        self.model_config.USE_BATCH_NORM,
                        self.activation_cls,
                    )
                )
                in_channels = res_channels
            for _ in range(self.model_config.NUM_RESIDUAL_BLOCKS):
                layers.append(
                    ResidualBlock(
                        in_channels,
                        self.model_config.USE_BATCH_NORM,
                        self.activation_cls,
                    )
                )
        # Transformer (Optional)
        if (
            self.model_config.USE_TRANSFORMER_IN_REP
            and self.model_config.REP_TRANSFORMER_LAYERS > 0
        ):
            transformer_input_dim = self.hidden_dim
            if in_channels != transformer_input_dim:
                layers.append(
                    nn.Conv2d(in_channels, transformer_input_dim, kernel_size=1)
                )
                in_channels = transformer_input_dim
            pos_encoder = PositionalEncoding(transformer_input_dim, dropout=0.1)
            encoder_layer = nn.TransformerEncoderLayer(
                d_model=transformer_input_dim,
                nhead=self.model_config.REP_TRANSFORMER_HEADS,
                dim_feedforward=self.model_config.REP_TRANSFORMER_FC_DIM,
                activation=self.model_config.ACTIVATION_FUNCTION.lower(),
                batch_first=False,
                norm_first=True,
            )
            transformer_norm = nn.LayerNorm(transformer_input_dim)
            transformer_encoder = nn.TransformerEncoder(
                encoder_layer,
                num_layers=self.model_config.REP_TRANSFORMER_LAYERS,
                norm=transformer_norm,
            )
            layers.append(nn.Flatten(start_dim=2))
            layers.append(Permute(2, 0, 1))
            layers.append(pos_encoder)
            layers.append(transformer_encoder)
            layers.append(Permute(1, 0, 2))
            layers.append(nn.Flatten(start_dim=1))
        else:
            layers.append(nn.Flatten(start_dim=1))
        return RepresentationEncoderWrapper(nn.Sequential(*layers))

    def represent(
        self, grid_state: torch.Tensor, other_features: torch.Tensor
    ) -> torch.Tensor:
        encoded_grid_flat = self.representation_encoder(grid_state)
        combined_features = torch.cat([encoded_grid_flat, other_features], dim=1)
        hidden_state = self.representation_projector(combined_features)
        return hidden_state

    def dynamics(self, hidden_state, action) -> tuple[torch.Tensor, torch.Tensor]:
        if isinstance(action, int) or (
            isinstance(action, torch.Tensor) and action.numel() == 1
        ):
            action_tensor = torch.tensor([action], device=hidden_state.device)
            action_one_hot = F.one_hot(
                action_tensor, num_classes=self.action_dim
            ).float()
        elif isinstance(action, torch.Tensor) and action.dim() == 1:
            action_one_hot = F.one_hot(action, num_classes=self.action_dim).float()
        elif (
            isinstance(action, torch.Tensor)
            and action.dim() == 2
            and action.shape[1] == self.action_dim
        ):
            action_one_hot = action
        else:
            raise TypeError(
                f"Unsupported action type/shape: {type(action), action.shape if isinstance(action, torch.Tensor) else ''}"
            )
        action_embedding = self.action_encoder(action_one_hot)
        dynamics_input = torch.cat([hidden_state, action_embedding], dim=1)
        next_hidden_state = self.dynamics_core(dynamics_input)
        reward_logits = self.reward_head(next_hidden_state)
        return next_hidden_state, reward_logits

    def predict(self, hidden_state) -> tuple[torch.Tensor, torch.Tensor]:
        prediction_features = self.prediction_core(hidden_state)
        policy_logits = self.policy_head(prediction_features)
        value_logits = self.value_head(prediction_features)
        return policy_logits, value_logits

    def forward(
        self, grid_state: torch.Tensor, other_features: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        initial_hidden_state = self.represent(grid_state, other_features)
        policy_logits, value_logits = self.predict(initial_hidden_state)
        dummy_reward_logits = torch.zeros(
            (initial_hidden_state.shape[0], self.model_config.REWARD_SUPPORT_SIZE),
            device=initial_hidden_state.device,
        )
        return policy_logits, value_logits, initial_hidden_state
