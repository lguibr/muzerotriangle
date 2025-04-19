# File: muzerotriangle/data/serializer.py
# File: muzerotriangle/data/serializer.py
import json
import logging
from collections import deque
from pathlib import Path
from typing import TYPE_CHECKING, Any

import cloudpickle
import numpy as np
import torch
from pydantic import ValidationError

# --- ADDED: Import Trajectory ---
# --- END ADDED ---
from .schemas import BufferData, CheckpointData

if TYPE_CHECKING:
    from torch.optim import Optimizer

    from ..rl.core.buffer import ExperienceBuffer
    from ..utils.types import Trajectory

logger = logging.getLogger(__name__)


class Serializer:
    """Handles serialization and deserialization of training data."""

    def load_checkpoint(self, path: Path) -> CheckpointData | None:
        try:
            with path.open("rb") as f:
                loaded_data = cloudpickle.load(f)
            if isinstance(loaded_data, CheckpointData):
                return loaded_data
            else:
                logger.error(
                    f"Loaded checkpoint {path} type mismatch: {type(loaded_data)}."
                )
                return None
        except ValidationError as e:
            logger.error(
                f"Pydantic validation failed for checkpoint {path}: {e}", exc_info=True
            )
            return None
        except FileNotFoundError:
            logger.warning(f"Checkpoint file not found: {path}")
            return None
        except Exception as e:
            logger.error(f"Error loading checkpoint {path}: {e}", exc_info=True)
            return None

    def save_checkpoint(self, data: CheckpointData, path: Path):
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            with path.open("wb") as f:
                cloudpickle.dump(data, f)
                logger.info(f"Checkpoint data saved to {path}")
        except Exception as e:
            logger.error(f"Failed save checkpoint {path}: {e}", exc_info=True)
            raise

    def load_buffer(self, path: Path) -> BufferData | None:
        """Loads and validates MuZero buffer data (trajectories) from a file."""
        try:
            with path.open("rb") as f:
                loaded_data = cloudpickle.load(f)
            if isinstance(loaded_data, BufferData):
                # --- FIXED: Add type hint ---
                valid_trajectories: list[Trajectory] = []
                # --- END FIXED ---
                total_valid_steps = 0
                invalid_traj_count = 0
                invalid_step_count = 0
                for i, traj in enumerate(loaded_data.trajectories):
                    if not isinstance(traj, list):
                        invalid_traj_count += 1
                        continue
                    valid_steps_in_traj = []
                    for j, step in enumerate(traj):
                        is_valid_step = False
                        try:
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
                                and isinstance(
                                    step["observation"]["other_features"], np.ndarray
                                )
                                and np.all(np.isfinite(step["observation"]["grid"]))
                                and np.all(
                                    np.isfinite(step["observation"]["other_features"])
                                )
                                and isinstance(step["action"], int)
                                and isinstance(step["reward"], float | int)
                                and isinstance(step["policy_target"], dict)
                                and isinstance(step["value_target"], float | int)
                            ):
                                is_valid_step = True
                        except Exception as val_err:
                            logger.warning(
                                f"Validation error step {j} traj {i}: {val_err}"
                            )
                        if is_valid_step:
                            valid_steps_in_traj.append(step)
                        else:
                            invalid_step_count += 1
                    if valid_steps_in_traj:
                        valid_trajectories.append(valid_steps_in_traj)
                        total_valid_steps += len(valid_steps_in_traj)
                    else:
                        invalid_traj_count += 1
                if invalid_traj_count > 0 or invalid_step_count > 0:
                    logger.warning(
                        f"Loaded buffer: Skipped {invalid_traj_count} invalid trajs, {invalid_step_count} invalid steps."
                    )
                loaded_data.trajectories = valid_trajectories
                loaded_data.total_steps = total_valid_steps
                return loaded_data
            else:
                logger.error(
                    f"Loaded buffer {path} type mismatch: {type(loaded_data)}."
                )
                return None
        except ValidationError as e:
            logger.error(
                f"Pydantic validation failed buffer {path}: {e}", exc_info=True
            )
            return None
        except FileNotFoundError:
            logger.warning(f"Buffer file not found: {path}")
            return None
        except Exception as e:
            logger.error(f"Failed load MuZero buffer {path}: {e}", exc_info=True)
            return None

    def save_buffer(self, data: BufferData, path: Path):
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            with path.open("wb") as f:
                cloudpickle.dump(data, f)
                logger.info(
                    f"MuZero Buffer saved to {path} ({len(data.trajectories)} trajs, {data.total_steps} steps)"
                )
        except Exception as e:
            logger.error(f"Error saving MuZero buffer {path}: {e}", exc_info=True)
            raise

    def prepare_optimizer_state(self, optimizer: "Optimizer") -> dict[str, Any]:
        optimizer_state_cpu = {}
        try:
            optimizer_state_dict = optimizer.state_dict()

            def move_to_cpu(item):
                if isinstance(item, torch.Tensor):
                    return item.cpu()
                elif isinstance(item, dict):
                    return {k: move_to_cpu(v) for k, v in item.items()}
                elif isinstance(item, list):
                    return [move_to_cpu(elem) for elem in item]
                else:
                    return item

            optimizer_state_cpu = move_to_cpu(optimizer_state_dict)
        except Exception as e:
            logger.error(f"Could not prepare optimizer state: {e}")
        return optimizer_state_cpu

    def prepare_buffer_data(self, buffer: "ExperienceBuffer") -> BufferData | None:
        try:
            # --- CHANGED: Get trajectories from tuples ---
            if not hasattr(buffer, "buffer") or not isinstance(buffer.buffer, deque):
                logger.error("Buffer missing 'buffer' deque.")
                return None
            trajectories_list: list[Trajectory] = [
                traj for _, traj in buffer.buffer
            ]  # Extract only trajectories
            # --- END CHANGED ---
            total_steps = buffer.total_steps
            valid_trajectories = []
            actual_steps = 0
            for traj in trajectories_list:
                if isinstance(traj, list) and traj:
                    valid_trajectories.append(traj)
                    actual_steps += len(traj)
                else:
                    logger.warning(
                        "Skipping invalid/empty trajectory during save prep."
                    )
            if actual_steps != total_steps:
                logger.warning(
                    f"Buffer total_steps mismatch: Stored={total_steps}, Calc={actual_steps}. Saving calc value."
                )
                total_steps = actual_steps
            return BufferData(trajectories=valid_trajectories, total_steps=total_steps)
        except Exception as e:
            logger.error(f"Error preparing MuZero buffer data: {e}")
            return None

    def save_config_json(self, configs: dict[str, Any], path: Path):
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            with path.open("w") as f:

                def default_serializer(obj):
                    if isinstance(obj, torch.Tensor | np.ndarray):
                        return "<tensor/array>"
                    if isinstance(obj, deque):
                        return list(obj)
                    try:
                        if hasattr(obj, "__dict__"):
                            return obj.__dict__
                        else:
                            return str(obj)
                    except TypeError:
                        return f"<object type {type(obj).__name__}>"

                json.dump(configs, f, indent=4, default=default_serializer)
                logger.info(f"Run config saved to {path}")
        except Exception as e:
            logger.error(f"Failed save run config JSON {path}: {e}", exc_info=True)
            raise
