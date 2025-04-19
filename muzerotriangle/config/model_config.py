# File: muzerotriangle/config/model_config.py
from typing import Literal

from pydantic import BaseModel, Field, model_validator


class ModelConfig(BaseModel):
    """
    Configuration for the MuZero Neural Network model (Pydantic model).
    Defines parameters for representation, dynamics, and prediction functions.
    """

    # --- Input Representation ---
    GRID_INPUT_CHANNELS: int = Field(default=1, gt=0)
    OTHER_NN_INPUT_FEATURES_DIM: int = Field(
        default=30, ge=0
    )  # Dimension of non-grid features

    # --- Shared Components ---
    HIDDEN_STATE_DIM: int = Field(
        default=128, gt=0, description="Dimension of the MuZero hidden state (s_k)."
    )
    ACTION_ENCODING_DIM: int = Field(
        default=16,
        gt=0,
        description="Dimension for embedding actions before dynamics function.",
    )
    ACTIVATION_FUNCTION: Literal["ReLU", "GELU", "SiLU", "Tanh", "Sigmoid"] = Field(
        default="ReLU"
    )
    USE_BATCH_NORM: bool = Field(default=True)

    # --- Representation Function (h) ---
    # (CNN/ResNet/Transformer part, outputs initial hidden state)
    CONV_FILTERS: list[int] = Field(default=[32, 64, 128])
    CONV_KERNEL_SIZES: list[int | tuple[int, int]] = Field(default=[3, 3, 3])
    CONV_STRIDES: list[int | tuple[int, int]] = Field(default=[1, 1, 1])
    CONV_PADDING: list[int | tuple[int, int] | str] = Field(default=[1, 1, 1])

    NUM_RESIDUAL_BLOCKS: int = Field(default=2, ge=0)
    RESIDUAL_BLOCK_FILTERS: int = Field(default=128, gt=0)  # Match last conv filter

    # Transformer for Representation Encoder (Optional)
    USE_TRANSFORMER_IN_REP: bool = Field(
        default=False, description="Use Transformer in the representation function."
    )
    REP_TRANSFORMER_HEADS: int = Field(default=4, gt=0)
    REP_TRANSFORMER_LAYERS: int = Field(default=2, ge=0)
    REP_TRANSFORMER_FC_DIM: int = Field(default=256, gt=0)

    # Final projection to hidden state dim in representation function
    REP_FC_DIMS_AFTER_ENCODER: list[int] = Field(default=[])
    # If empty, a single Linear layer projects directly to HIDDEN_STATE_DIM

    # --- Dynamics Function (g) ---
    # (Takes s_k, a_{k+1} -> s_{k+1}, r_{k+1})
    DYNAMICS_NUM_RESIDUAL_BLOCKS: int = Field(
        default=2, ge=0, description="Number of ResBlocks in the dynamics function."
    )
    # Dynamics function combines hidden_state + encoded_action

    # Reward Prediction Head (part of Dynamics)
    REWARD_HEAD_DIMS: list[int] = Field(default=[64])
    # Assuming categorical reward prediction (like MuZero paper)
    REWARD_SUPPORT_SIZE: int = Field(
        default=21,
        gt=1,
        description="Number of atoms for categorical reward prediction (e.g., -10 to +10). Must be odd.",
    )

    # --- Prediction Function (f) ---
    # (Takes s_k -> p_k, v_k)
    PREDICTION_NUM_RESIDUAL_BLOCKS: int = Field(
        default=1, ge=0, description="Number of ResBlocks in the prediction function."
    )
    POLICY_HEAD_DIMS: list[int] = Field(default=[64])
    VALUE_HEAD_DIMS: list[int] = Field(default=[64])
    NUM_VALUE_ATOMS: int = Field(
        default=51, gt=1, description="Number of atoms for distributional value head."
    )
    VALUE_MIN: float = Field(default=-10.0)
    VALUE_MAX: float = Field(default=10.0)

    # --- Validation ---
    @model_validator(mode="after")
    def check_conv_layers_consistency(self) -> "ModelConfig":
        n_filters = len(self.CONV_FILTERS)
        if not (
            len(self.CONV_KERNEL_SIZES) == n_filters
            and len(self.CONV_STRIDES) == n_filters
            and len(self.CONV_PADDING) == n_filters
        ):
            raise ValueError(
                "Lengths of CONV_FILTERS, CONV_KERNEL_SIZES, CONV_STRIDES, and CONV_PADDING must match."
            )
        return self

    @model_validator(mode="after")
    def check_residual_filter_match(self) -> "ModelConfig":
        if (
            self.NUM_RESIDUAL_BLOCKS > 0
            and self.CONV_FILTERS
            and self.CONV_FILTERS[-1] != self.RESIDUAL_BLOCK_FILTERS
        ):
            # Representation function will handle projection if needed
            pass
        return self

    @model_validator(mode="after")
    def check_transformer_config(self) -> "ModelConfig":
        if self.USE_TRANSFORMER_IN_REP:
            if self.REP_TRANSFORMER_LAYERS < 0:
                raise ValueError("REP_TRANSFORMER_LAYERS cannot be negative.")
            if self.REP_TRANSFORMER_LAYERS > 0:
                if self.HIDDEN_STATE_DIM <= 0:
                    raise ValueError(
                        "HIDDEN_STATE_DIM must be positive if REP_TRANSFORMER_LAYERS > 0."
                    )
                if self.REP_TRANSFORMER_HEADS <= 0:
                    raise ValueError(
                        "REP_TRANSFORMER_HEADS must be positive if REP_TRANSFORMER_LAYERS > 0."
                    )
                if self.HIDDEN_STATE_DIM % self.REP_TRANSFORMER_HEADS != 0:
                    raise ValueError(
                        f"HIDDEN_STATE_DIM ({self.HIDDEN_STATE_DIM}) must be divisible by REP_TRANSFORMER_HEADS ({self.REP_TRANSFORMER_HEADS})."
                    )
                if self.REP_TRANSFORMER_FC_DIM <= 0:
                    raise ValueError(
                        "REP_TRANSFORMER_FC_DIM must be positive if REP_TRANSFORMER_LAYERS > 0."
                    )
        return self

    @model_validator(mode="after")
    def check_value_distribution_params(self) -> "ModelConfig":
        if self.VALUE_MIN >= self.VALUE_MAX:
            raise ValueError("VALUE_MIN must be strictly less than VALUE_MAX.")
        return self

    @model_validator(mode="after")
    def check_reward_support_size(self) -> "ModelConfig":
        # Often assumed to be odd for symmetry around 0, but not strictly required by algo
        if self.REWARD_SUPPORT_SIZE % 2 == 0:
            # pass # Allow even for now
            raise ValueError("REWARD_SUPPORT_SIZE must be odd.")
        return self


ModelConfig.model_rebuild(force=True)
