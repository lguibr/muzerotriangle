# File: muzerotriangle/config/mcts_config.py
from pydantic import BaseModel, Field, field_validator


class MCTSConfig(BaseModel):
    """
    Configuration for Monte Carlo Tree Search (Pydantic model).
    --- TUNED FOR INCREASED EXPLORATION & DEPTH ---
    """

    num_simulations: int = Field(default=128, ge=1)  # Reduced for faster debugging
    puct_coefficient: float = Field(default=1.5, gt=0)  # Adjusted c1/c2 balance
    temperature_initial: float = Field(default=1.0, ge=0)
    temperature_final: float = Field(default=0.1, ge=0)
    temperature_anneal_steps: int = Field(default=100, ge=0)  # Reduced anneal
    dirichlet_alpha: float = Field(default=0.3, gt=0)
    dirichlet_epsilon: float = Field(default=0.25, ge=0, le=1.0)
    max_search_depth: int = Field(default=16, ge=1)  # Reduced depth
    discount: float = Field(
        default=0.99,
        gt=0,
        le=1.0,
        description="Discount factor (gamma) used in MCTS backpropagation and value targets.",
    )  # ADDED

    @field_validator("temperature_final")
    @classmethod
    def check_temp_final_le_initial(cls, v: float, info) -> float:
        data = info.data if info.data else info.values
        initial_temp = data.get("temperature_initial")
        if initial_temp is not None and v > initial_temp:
            raise ValueError(
                "temperature_final cannot be greater than temperature_initial"
            )
        return v


MCTSConfig.model_rebuild(force=True)
