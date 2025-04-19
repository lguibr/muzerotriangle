# File: muzerotriangle/rl/types.py
import logging

import numpy as np
from pydantic import BaseModel, ConfigDict, Field, model_validator

from ..utils.types import Trajectory

logger = logging.getLogger(__name__)
arbitrary_types_config = ConfigDict(arbitrary_types_allowed=True)


class SelfPlayResult(BaseModel):
    """Pydantic model for structuring results from a self-play worker (MuZero)."""

    model_config = arbitrary_types_config
    trajectory: Trajectory
    final_score: float
    episode_steps: int
    total_simulations: int = Field(..., ge=0)
    avg_root_visits: float = Field(..., ge=0)
    avg_tree_depth: float = Field(..., ge=0)

    @model_validator(mode="after")
    def check_trajectory_structure(self) -> "SelfPlayResult":
        """Basic structural validation for trajectory steps."""
        invalid_count = 0
        valid_steps = []
        for i, step in enumerate(self.trajectory):
            is_valid = False
            try:
                # --- Apply SIM102: Combine nested ifs ---
                if (
                    isinstance(step, dict)
                    and all(
                        k in step
                        for k in [
                            "observation",
                            "action",
                            "reward",
                            "policy_target",
                            "value_target",
                        ]
                    )
                    and isinstance(step["observation"], dict)
                    and "grid" in step["observation"]
                    and "other_features" in step["observation"]
                    and isinstance(step["observation"]["grid"], np.ndarray)
                    and isinstance(step["observation"]["other_features"], np.ndarray)
                    and np.all(np.isfinite(step["observation"]["grid"]))
                    and np.all(np.isfinite(step["observation"]["other_features"]))
                    and isinstance(step["action"], int)
                    and isinstance(step["reward"], float | int)
                    and isinstance(step["policy_target"], dict)
                    and isinstance(step["value_target"], float | int)
                ):
                    is_valid = True
                # --- End Combined If ---
            except Exception as e:
                logger.warning(f"Error validating step {i}: {e}")
            if is_valid:
                valid_steps.append(step)
            else:
                invalid_count += 1
        if invalid_count > 0:
            logger.warning(f"Found {invalid_count} invalid steps. Keeping valid.")
            object.__setattr__(self, "trajectory", valid_steps)
        return self


SelfPlayResult.model_rebuild(force=True)
