# File: muzerotriangle/data/schemas.py
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

# Use relative import for Trajectory
from ..utils.types import Trajectory  # Import Trajectory type

arbitrary_types_config = ConfigDict(arbitrary_types_allowed=True)


class CheckpointData(BaseModel):
    """Pydantic model defining the structure of saved checkpoint data."""

    model_config = arbitrary_types_config

    run_name: str
    global_step: int = Field(..., ge=0)
    episodes_played: int = Field(..., ge=0)
    total_simulations_run: int = Field(..., ge=0)
    model_config_dict: dict[str, Any]
    env_config_dict: dict[str, Any]
    model_state_dict: dict[str, Any]
    optimizer_state_dict: dict[str, Any]
    stats_collector_state: dict[str, Any]


class BufferData(BaseModel):
    """Pydantic model defining the structure of saved MuZero buffer data."""

    model_config = arbitrary_types_config

    # --- CHANGED: Store list of Trajectories ---
    trajectories: list[Trajectory]
    total_steps: int = Field(..., ge=0)  # Store total steps for quicker loading
    # --- END CHANGED ---


class LoadedTrainingState(BaseModel):
    """Pydantic model representing the fully loaded state."""

    model_config = arbitrary_types_config

    checkpoint_data: CheckpointData | None = None
    buffer_data: BufferData | None = None


# Rebuild models after changes
CheckpointData.model_rebuild(force=True)
BufferData.model_rebuild(force=True)
LoadedTrainingState.model_rebuild(force=True)
