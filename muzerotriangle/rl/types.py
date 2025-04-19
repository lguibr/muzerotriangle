# File: muzerotriangle/rl/types.py
import logging

import numpy as np
from pydantic import BaseModel, ConfigDict, Field, model_validator

# Use relative import for Trajectory
from ..utils.types import Trajectory

logger = logging.getLogger(__name__)

arbitrary_types_config = ConfigDict(arbitrary_types_allowed=True)


class SelfPlayResult(BaseModel):
    """Pydantic model for structuring results from a self-play worker (MuZero)."""

    model_config = arbitrary_types_config

    # --- CHANGED: Store the full trajectory ---
    trajectory: Trajectory
    # --- END CHANGED ---

    final_score: float
    episode_steps: int

    # MCTS Stats
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
                # Check if step is a dict (TypedDict) and has required keys
                if isinstance(step, dict) and all(
                    k in step
                    for k in [
                        "observation",
                        "action",
                        "reward",
                        "policy_target",
                        "value_target",
                    ]
                ):
                    obs = step["observation"]
                    # Further checks on observation structure
                    if (
                        isinstance(obs, dict)
                        and "grid" in obs
                        and "other_features" in obs
                        and isinstance(obs["grid"], np.ndarray)
                        and isinstance(obs["other_features"], np.ndarray)
                        and np.all(np.isfinite(obs["grid"]))
                        and np.all(np.isfinite(obs["other_features"]))
                    ):
                        # Check other types
                        if (
                            isinstance(step["action"], int)
                            and isinstance(step["reward"], float | int)
                            and isinstance(step["policy_target"], dict)
                            and isinstance(step["value_target"], float | int)
                        ):
                            is_valid = True
            except Exception as e:
                logger.warning(f"Error validating trajectory step {i}: {e}")

            if is_valid:
                valid_steps.append(step)
            else:
                invalid_count += 1
                # logger.warning(f"SelfPlayResult validation: Invalid trajectory step structure at index {i}: {type(step)}")

        if invalid_count > 0:
            logger.warning(
                f"SelfPlayResult validation: Found {invalid_count} invalid trajectory steps. Keeping only valid ones."
            )
            # Modify trajectory in place (consider alternatives if mutation in validator is problematic)
            object.__setattr__(self, "trajectory", valid_steps)

        return self


SelfPlayResult.model_rebuild(force=True)

# --- REMOVED: PERBatchSample (for now) ---
# class PERBatchSample(TypedDict):
#    batch: SampledBatch # Would need update
#    indices: np.ndarray
#    weights: np.ndarray
