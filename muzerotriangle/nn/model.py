# File: muzerotriangle/nn/model.py
import logging
import math
from typing import cast

import torch
import torch.nn as nn

from ..config import EnvConfig, ModelConfig
from ..utils.types import ActionType

logger = logging.getLogger(__name__)


def conv_block(
    in_channels: int,
    out_channels: int,
    kernel_size: int | tuple[int, int],
    stride: int | tuple[int, int],
    padding: int | tuple[int, int] | str,
    use_batch_norm: bool,
    activation: type[nn.Module],
) -> nn.Sequential:
    """Creates a standard convolutional block."""
    layers: list[nn.Module] = [
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
        residual = x
        out: torch.Tensor = self.conv1(x)
        out = self.conv2(out)
        out = self.bn2(out)
        out += residual
        out = self.activation(out)
        return out


class PositionalEncoding(nn.Module):
    """Injects sinusoidal positional encoding. (Adapted from PyTorch tutorial)"""

    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000):
        super().__init__()
        if d_model <= 0:
            raise ValueError("d_model must be positive for PositionalEncoding")
        self.dropout = nn.Dropout(p=dropout)

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model)
        )
        pe = torch.zeros(max_len, d_model)
        pe[:, 0::2] = torch.sin(position * div_term)
        if d_model > 1:
            pe[:, 1::2] = torch.cos(position * div_term[: d_model // 2])

        pe = pe.unsqueeze(1)  # Shape: [max_len, 1, d_model]
        self.register_buffer("pe", pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Tensor, shape [seq_len, batch_size, embedding_dim]
        """
        pe_buffer = self.pe
        if not isinstance(pe_buffer, torch.Tensor):
            raise TypeError("PositionalEncoding buffer 'pe' is not a Tensor.")
        if x.shape[0] > pe_buffer.shape[0]:
            raise ValueError(
                f"Input sequence length {x.shape[0]} exceeds max_len {pe_buffer.shape[0]} of PositionalEncoding"
            )
        if x.shape[2] != pe_buffer.shape[2]:
            raise ValueError(
                f"Input embedding dimension {x.shape[2]} does not match PositionalEncoding dimension {pe_buffer.shape[2]}"
            )
        x = x + pe_buffer[: x.size(0)]
        return cast("torch.Tensor", self.dropout(x))


class MuZeroNet(nn.Module):
    """
    Neural Network architecture for MuZeroTriangle.
    Implements Representation (h), Dynamics (g), and Prediction (f) functions.
    """

    def __init__(self, model_config: ModelConfig, env_config: EnvConfig):
        super().__init__()
        self.model_config = model_config
        self.env_config = env_config
        self.action_dim = int(env_config.ACTION_DIM)  # type: ignore[call-overload]
        self.hidden_dim = model_config.HIDDEN_STATE_DIM
        self.activation_cls: type[nn.Module] = getattr(
            nn, model_config.ACTIVATION_FUNCTION
        )

        # --- Representation Function (h) ---
        self.representation_encoder = self._build_representation_encoder()
        # Calculate flattened size after encoder
        dummy_input_grid = torch.zeros(
            1, model_config.GRID_INPUT_CHANNELS, env_config.ROWS, env_config.COLS
        )
        dummy_other = torch.zeros(1, model_config.OTHER_NN_INPUT_FEATURES_DIM)
        with torch.no_grad():
            encoded_output, _ = self.representation_encoder(dummy_grid, dummy_other)
            self.encoded_flat_size = encoded_output.numel()

        rep_fc_layers: list[nn.Module] = []
        in_features = self.encoded_flat_size
        for hidden_dim_fc in model_config.REP_FC_DIMS_AFTER_ENCODER:
            rep_fc_layers.append(nn.Linear(in_features, hidden_dim_fc))
            if model_config.USE_BATCH_NORM:
                rep_fc_layers.append(nn.BatchNorm1d(hidden_dim_fc))
            rep_fc_layers.append(self.activation_cls())
            in_features = hidden_dim_fc
        rep_fc_layers.append(nn.Linear(in_features, self.hidden_dim))
        # Optional: Add normalization or activation after final projection? MuZero paper doesn't specify.
        self.representation_projector = nn.Sequential(*rep_fc_layers)

        # --- Dynamics Function (g) ---
        self.action_encoder = nn.Linear(
            self.action_dim, model_config.ACTION_ENCODING_DIM
        )
        dynamics_input_dim = self.hidden_dim + model_config.ACTION_ENCODING_DIM
        dynamics_layers: list[nn.Module] = [
            nn.Linear(dynamics_input_dim, self.hidden_dim),
            self.activation_cls(),
        ]
        for _ in range(model_config.DYNAMICS_NUM_RESIDUAL_BLOCKS):
            # Residual blocks for dynamics need adapting (operate on 1D vector)
            # Placeholder: Use simple Linear layers instead of ResBlocks for now
            dynamics_layers.extend(
                [
                    nn.Linear(self.hidden_dim, self.hidden_dim),
                    self.activation_cls(),
                ]
            )
        self.dynamics_core = nn.Sequential(*dynamics_layers)
        # Reward Head (predicts logits for reward distribution)
        reward_head_layers: list[nn.Module] = []
        reward_in = self.hidden_dim
        for hidden_dim_fc in model_config.REWARD_HEAD_DIMS:
            reward_head_layers.append(nn.Linear(reward_in, hidden_dim_fc))
            if model_config.USE_BATCH_NORM:
                reward_head_layers.append(nn.BatchNorm1d(hidden_dim_fc))
            reward_head_layers.append(self.activation_cls())
            reward_in = hidden_dim_fc
        reward_head_layers.append(nn.Linear(reward_in, model_config.REWARD_SUPPORT_SIZE))
        self.reward_head = nn.Sequential(*reward_head_layers)

        # --- Prediction Function (f) ---
        prediction_layers: list[nn.Module] = []
        pred_in = self.hidden_dim
        # Placeholder: Use simple Linear layers instead of ResBlocks for now
        for _ in range(model_config.PREDICTION_NUM_RESIDUAL_BLOCKS):
            prediction_layers.extend(
                [nn.Linear(pred_in, self.hidden_dim), self.activation_cls()]
            )
            pred_in = self.hidden_dim
        self.prediction_core = nn.Sequential(*prediction_layers)
        # Policy Head
        policy_head_layers: list[nn.Module] = []
        policy_in = self.hidden_dim
        for hidden_dim_fc in model_config.POLICY_HEAD_DIMS:
            policy_head_layers.append(nn.Linear(policy_in, hidden_dim_fc))
            if model_config.USE_BATCH_NORM:
                policy_head_layers.append(nn.BatchNorm1d(hidden_dim_fc))
            policy_head_layers.append(self.activation_cls())
            policy_in = hidden_dim_fc
        policy_head_layers.append(nn.Linear(policy_in, self.action_dim))
        self.policy_head = nn.Sequential(*policy_head_layers)
        # Value Head (predicts logits for value distribution)
        value_head_layers: list[nn.Module] = []
        value_in = self.hidden_dim
        for hidden_dim_fc in model_config.VALUE_HEAD_DIMS:
            value_head_layers.append(nn.Linear(value_in, hidden_dim_fc))
            if model_config.USE_BATCH_NORM:
                value_head_layers.append(nn.BatchNorm1d(hidden_dim_fc))
            value_head_layers.append(self.activation_cls())
            value_in = hidden_dim_fc
        value_head_layers.append(nn.Linear(value_in, model_config.NUM_VALUE_ATOMS))
        self.value_head = nn.Sequential(*value_head_layers)

    def _build_representation_encoder(self) -> nn.Module:
        """Builds the initial encoder part of the representation function."""
        layers: list[nn.Module] = []
        in_channels = self.model_config.GRID_INPUT_CHANNELS
        # --- CNN Body ---
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
        # --- Residual Body ---
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
                        in_channels, self.model_config.USE_BATCH_NORM, self.activation_cls
                    )
                )
        # --- Transformer (Optional) ---
        if (
            self.model_config.USE_TRANSFORMER_IN_REP
            and self.model_config.REP_TRANSFORMER_LAYERS > 0
        ):
            transformer_input_dim = self.hidden_dim  # Use hidden_dim for transformer
            if in_channels != transformer_input_dim:
                layers.append(
                    nn.Conv2d(in_channels, transformer_input_dim, kernel_size=1)
                )
                in_channels = transformer_input_dim
            else:
                layers.append(nn.Identity())

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
            # Add layers to handle transformer input/output shapes
            layers.append(nn.Flatten(start_dim=2))  # Flatten H, W -> (B, C, H*W)
            layers.append(nn.Unflatten(dim=1, unflattened_size=(in_channels, -1)))
            layers.append(
                lambda x: x.permute(2, 0, 1)
            )  # (H*W, B, C) for PositionalEncoding
            layers.append(pos_encoder)
            layers.append(transformer_encoder)
            layers.append(lambda x: x.permute(1, 0, 2))  # (B, H*W, C)
            layers.append(nn.Flatten(start_dim=1))  # Flatten Seq and Dim

        # Final feature combination (Flatten CNN output and concat other features)
        final_layers = [
            nn.Flatten(start_dim=1),
            lambda grid_flat, other: torch.cat([grid_flat, other], dim=1),
        ]

        # Need a wrapper module to handle the two inputs (grid, other)
        class RepresentationEncoderWrapper(nn.Module):
            def __init__(self, cnn_tf_layers: nn.Module, final_layers: list):
                super().__init__()
                self.cnn_tf_layers = cnn_tf_layers
                self.final_layers = nn.ModuleList(final_layers)

            def forward(
                self, grid_state: torch.Tensor, other_features: torch.Tensor
            ) -> tuple[torch.Tensor, torch.Tensor]:
                encoded = self.cnn_tf_layers(grid_state)
                # Check if encoded is already flattened (if Transformer was used)
                if len(encoded.shape) > 2:
                    flattened_encoded = self.final_layers[0](encoded) # Flatten
                else:
                    flattened_encoded = encoded # Already flattened by Transformer part

                combined = self.final_layers[1](flattened_encoded, other_features) # Concat
                return combined, flattened_encoded # Return combined and intermediate flat

        cnn_transformer_part = nn.Sequential(*layers)
        return RepresentationEncoderWrapper(cnn_transformer_part, final_layers)

    def represent(
        self, grid_state: torch.Tensor, other_features: torch.Tensor
    ) -> torch.Tensor:
        """Representation function h(o) -> s_0."""
        encoded_combined, _ = self.representation_encoder(grid_state, other_features)
        hidden_state = self.representation_projector(encoded_combined)
        return hidden_state

    def dynamics(
        self, hidden_state: torch.Tensor, action: ActionType | torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Dynamics function g(s_k, a_{k+1}) -> s_{k+1}, r_{k+1}."""
        # Encode action: Convert scalar action to one-hot, then embed
        if isinstance(action, int) or (
            isinstance(action, torch.Tensor) and action.numel() == 1
        ):
            # Handle single action (create batch of 1)
            action_tensor = torch.tensor([action], device=hidden_state.device)
            action_one_hot = F.one_hot(
                action_tensor, num_classes=self.action_dim
            ).float()
        elif isinstance(action, torch.Tensor) and action.dim() == 1:
            # Handle batch of actions
            action_one_hot = F.one_hot(action, num_classes=self.action_dim).float()
        else:
            raise TypeError(f"Unsupported action type for dynamics: {type(action)}")

        action_embedding = self.action_encoder(action_one_hot)

        # Combine state and action embedding
        dynamics_input = torch.cat([hidden_state, action_embedding], dim=1)

        # Pass through core dynamics layers
        next_hidden_state = self.dynamics_core(dynamics_input)

        # Predict reward logits from the next hidden state
        reward_logits = self.reward_head(next_hidden_state)

        return next_hidden_state, reward_logits

    def predict(
        self, hidden_state: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Prediction function f(s_k) -> p_k, v_k."""
        # Pass through core prediction layers
        prediction_features = self.prediction_core(hidden_state)

        # Predict policy and value logits
        policy_logits = self.policy_head(prediction_features)
        value_logits = self.value_head(prediction_features)

        return policy_logits, value_logits

    def forward(
        self, grid_state: torch.Tensor, other_features: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Initial inference: h(o) -> s_0, then f(s_0) -> p_0, v_0.
        Used for initial MCTS expansion or direct evaluation if needed.
        Returns: (policy_logits, value_logits, initial_hidden_state)
        """
        initial_hidden_state = self.represent(grid_state, other_features)
        policy_logits, value_logits = self.predict(initial_hidden_state)
        return policy_logits, value_logits, initial_hidden_state