# File: muzerotriangle/config/train_config.py
import logging
import time
from typing import Literal

from pydantic import BaseModel, Field, field_validator, model_validator

logger = logging.getLogger(__name__)


class TrainConfig(BaseModel):
    """
    Configuration for the MuZero training process (Pydantic model).
    Defaults tuned for a longer training run (~48h) on capable hardware.
    Worker count is dynamically adjusted based on detected cores during setup.
    """

    RUN_NAME: str = Field(
        default_factory=lambda: f"train_muzero_{time.strftime('%Y%m%d_%H%M%S')}"
    )
    LOAD_CHECKPOINT_PATH: str | None = Field(default=None)
    LOAD_BUFFER_PATH: str | None = Field(default=None)
    AUTO_RESUME_LATEST: bool = Field(default=True)
    DEVICE: Literal["auto", "cuda", "cpu", "mps"] = Field(default="auto")
    RANDOM_SEED: int = Field(default=42)

    # --- Training Loop ---
    MAX_TRAINING_STEPS: int | None = Field(
        default=500_000, ge=1, description="Target number of training steps."
    )

    # --- Workers & Batching ---
    NUM_SELF_PLAY_WORKERS: int = Field(
        default=8,
        ge=1,
        description="Default number of parallel self-play actors (adjusted dynamically).",
    )
    WORKER_DEVICE: Literal["auto", "cuda", "cpu", "mps"] = Field(
        default="cpu", description="Device for self-play workers (usually CPU)."
    )
    BATCH_SIZE: int = Field(
        default=128, ge=1, description="Batch size for training steps."
    )
    BUFFER_CAPACITY: int = Field(
        default=1_000_000, ge=1, description="Total steps capacity across trajectories."
    )
    MIN_BUFFER_SIZE_TO_TRAIN: int = Field(
        default=50_000,
        ge=1,
        description="Minimum total steps in buffer before training starts.",
    )
    WORKER_UPDATE_FREQ_STEPS: int = Field(
        default=1000, ge=1, description="How often to send new weights to workers."
    )

    # --- MuZero Specific ---
    MUZERO_UNROLL_STEPS: int = Field(
        default=5,
        ge=0,
        description="Number of steps to unroll the dynamics model during training.",
    )
    N_STEP_RETURNS: int = Field(
        default=10,
        ge=1,
        description="Number of steps for calculating N-step reward targets.",
    )
    POLICY_LOSS_WEIGHT: float = Field(default=1.0, ge=0)
    VALUE_LOSS_WEIGHT: float = Field(default=0.25, ge=0)  # Often lower than policy
    REWARD_LOSS_WEIGHT: float = Field(default=1.0, ge=0)
    DISCOUNT: float = Field(
        default=0.99,
        gt=0,
        le=1.0,
        description="Discount factor (gamma) used for N-step returns and MCTS.",
    )

    # --- Optimizer ---
    OPTIMIZER_TYPE: Literal["Adam", "AdamW", "SGD"] = Field(default="AdamW")
    LEARNING_RATE: float = Field(default=1e-4, gt=0)  # MuZero often uses smaller LR
    WEIGHT_DECAY: float = Field(default=1e-4, ge=0)
    GRADIENT_CLIP_VALUE: float | None = Field(default=5.0)  # Clip grads

    # --- LR Scheduler ---
    LR_SCHEDULER_TYPE: Literal["StepLR", "CosineAnnealingLR"] | None = Field(
        default="CosineAnnealingLR"
    )
    LR_SCHEDULER_T_MAX: int | None = Field(
        default=None, description="Auto-set from MAX_TRAINING_STEPS for Cosine."
    )
    LR_SCHEDULER_ETA_MIN: float = Field(default=1e-6, ge=0)

    # --- Checkpointing ---
    CHECKPOINT_SAVE_FREQ_STEPS: int = Field(
        default=10000, ge=1, description="Frequency to save model checkpoints."
    )

    # --- Prioritized Experience Replay (PER) ---
    USE_PER: bool = Field(default=True)
    PER_ALPHA: float = Field(default=0.6, ge=0)
    PER_BETA_INITIAL: float = Field(default=0.4, ge=0, le=1.0)
    PER_BETA_FINAL: float = Field(default=1.0, ge=0, le=1.0)
    PER_BETA_ANNEAL_STEPS: int | None = Field(
        default=None, description="Auto-set from MAX_TRAINING_STEPS."
    )
    PER_EPSILON: float = Field(default=1e-5, gt=0)

    # --- Model Compilation ---
    COMPILE_MODEL: bool = Field(
        default=True, description="Use torch.compile() if available."
    )

    @model_validator(mode="after")
    def check_buffer_sizes(self) -> "TrainConfig":
        if self.MIN_BUFFER_SIZE_TO_TRAIN > self.BUFFER_CAPACITY:
            raise ValueError(
                "MIN_BUFFER_SIZE_TO_TRAIN cannot be greater than BUFFER_CAPACITY."
            )
        return self

    @model_validator(mode="after")
    def set_scheduler_t_max(self) -> "TrainConfig":
        if (
            self.LR_SCHEDULER_TYPE == "CosineAnnealingLR"
            and self.LR_SCHEDULER_T_MAX is None
            and self.MAX_TRAINING_STEPS is not None
            and self.MAX_TRAINING_STEPS >= 1
        ):
            self.LR_SCHEDULER_T_MAX = self.MAX_TRAINING_STEPS
            logger.info(
                f"Set LR_SCHEDULER_T_MAX to MAX_TRAINING_STEPS ({self.MAX_TRAINING_STEPS})"
            )
        elif (
            self.LR_SCHEDULER_T_MAX is None
            and self.LR_SCHEDULER_TYPE == "CosineAnnealingLR"
        ):
            # Fallback if MAX_TRAINING_STEPS is None (e.g., running indefinitely)
            self.LR_SCHEDULER_T_MAX = 1_000_000
            logger.warning(
                f"MAX_TRAINING_STEPS is None, using fallback T_max {self.LR_SCHEDULER_T_MAX}"
            )

        if self.LR_SCHEDULER_T_MAX is not None and self.LR_SCHEDULER_T_MAX <= 0:
            raise ValueError("LR_SCHEDULER_T_MAX must be positive if set.")
        return self

    @model_validator(mode="after")
    def set_per_beta_anneal_steps(self) -> "TrainConfig":
        if self.USE_PER and self.PER_BETA_ANNEAL_STEPS is None:
            if self.MAX_TRAINING_STEPS is not None and self.MAX_TRAINING_STEPS >= 1:
                self.PER_BETA_ANNEAL_STEPS = self.MAX_TRAINING_STEPS
                logger.info(
                    f"Set PER_BETA_ANNEAL_STEPS to MAX_TRAINING_STEPS ({self.MAX_TRAINING_STEPS})"
                )
            else:
                # Fallback if MAX_TRAINING_STEPS is None
                self.PER_BETA_ANNEAL_STEPS = 1_000_000
                logger.warning(
                    f"MAX_TRAINING_STEPS invalid or None, using fallback PER_BETA_ANNEAL_STEPS {self.PER_BETA_ANNEAL_STEPS}"
                )

        if (
            self.USE_PER
            and self.PER_BETA_ANNEAL_STEPS is not None
            and self.PER_BETA_ANNEAL_STEPS <= 0
        ):
            raise ValueError("PER_BETA_ANNEAL_STEPS must be positive if set.")
        return self

    @field_validator("GRADIENT_CLIP_VALUE")
    @classmethod
    def check_gradient_clip(cls, v: float | None) -> float | None:
        if v is not None and v <= 0:
            raise ValueError("GRADIENT_CLIP_VALUE must be positive if set.")
        return v

    @field_validator("PER_BETA_FINAL")
    @classmethod
    def check_per_beta_final(cls, v: float, info) -> float:
        data = info.data if info.data else info.values
        initial_beta = data.get("PER_BETA_INITIAL")
        if initial_beta is not None and v < initial_beta:
            raise ValueError("PER_BETA_FINAL cannot be less than PER_BETA_INITIAL")
        return v


TrainConfig.model_rebuild(force=True)
