# File: muzerotriangle/rl/core/trainer.py
import logging
from typing import TYPE_CHECKING, cast

import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim

from ...utils.types import (
    SampledBatch,
    SampledBatchPER,  # Import PER batch type
)

if TYPE_CHECKING:
    from torch.optim.lr_scheduler import _LRScheduler

    from ...config import EnvConfig, TrainConfig
    from ...nn import NeuralNetwork

logger = logging.getLogger(__name__)


class Trainer:
    """MuZero Trainer."""

    def __init__(
        self,
        nn_interface: "NeuralNetwork",
        train_config: "TrainConfig",
        env_config: "EnvConfig",
    ):
        self.nn = nn_interface
        self.model = nn_interface.model
        self.train_config = train_config
        self.env_config = env_config
        self.model_config = nn_interface.model_config
        self.device = nn_interface.device
        self.optimizer = self._create_optimizer()
        self.scheduler = self._create_scheduler(self.optimizer)
        self.num_value_atoms = self.nn.num_value_atoms
        self.value_support = self.nn.support.to(self.device)
        self.num_reward_atoms = self.nn.num_reward_atoms
        self.reward_support = self.nn.reward_support.to(self.device)
        self.unroll_steps = self.train_config.MUZERO_UNROLL_STEPS

    def _create_optimizer(self):
        lr = self.train_config.LEARNING_RATE
        wd = self.train_config.WEIGHT_DECAY
        params = self.model.parameters()
        opt_type = self.train_config.OPTIMIZER_TYPE.lower()
        logger.info(f"Creating optimizer: {opt_type}, LR: {lr}, WD: {wd}")
        if opt_type == "adam":
            return optim.Adam(params, lr=lr, weight_decay=wd)
        elif opt_type == "adamw":
            return optim.AdamW(params, lr=lr, weight_decay=wd)
        elif opt_type == "sgd":
            return optim.SGD(params, lr=lr, weight_decay=wd, momentum=0.9)
        else:
            raise ValueError(
                f"Unsupported optimizer type: {self.train_config.OPTIMIZER_TYPE}"
            )

    def _create_scheduler(self, optimizer):
        scheduler_type_config = self.train_config.LR_SCHEDULER_TYPE
        scheduler_type = None
        if scheduler_type_config:
            scheduler_type = scheduler_type_config.lower()
        if not scheduler_type or scheduler_type == "none":
            return None
        logger.info(f"Creating LR scheduler: {scheduler_type}")
        if scheduler_type == "steplr":
            step_size = getattr(self.train_config, "LR_SCHEDULER_STEP_SIZE", 100000)
            gamma = getattr(self.train_config, "LR_SCHEDULER_GAMMA", 0.1)
            return cast(
                "_LRScheduler",
                optim.lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=gamma),
            )
        elif scheduler_type == "cosineannealinglr":
            t_max = self.train_config.LR_SCHEDULER_T_MAX
            eta_min = self.train_config.LR_SCHEDULER_ETA_MIN
            if t_max is None:
                t_max = self.train_config.MAX_TRAINING_STEPS or 1_000_000
            return cast(
                "_LRScheduler",
                optim.lr_scheduler.CosineAnnealingLR(
                    optimizer, T_max=t_max, eta_min=eta_min
                ),
            )
        else:
            raise ValueError(f"Unsupported scheduler type: {scheduler_type_config}")

    def _prepare_batch(self, batch_sequences: SampledBatch) -> dict[str, torch.Tensor]:
        """Prepares batch tensors from sampled sequences."""
        batch_size = len(batch_sequences)
        seq_len = self.unroll_steps + 1
        action_dim = int(self.env_config.ACTION_DIM)
        batch_grids = torch.zeros(
            (
                batch_size,
                seq_len,
                self.model_config.GRID_INPUT_CHANNELS,
                self.env_config.ROWS,
                self.env_config.COLS,
            ),
            dtype=torch.float32,
            device=self.device,
        )
        batch_others = torch.zeros(
            (batch_size, seq_len, self.model_config.OTHER_NN_INPUT_FEATURES_DIM),
            dtype=torch.float32,
            device=self.device,
        )
        batch_actions = torch.zeros(
            (batch_size, seq_len), dtype=torch.long, device=self.device
        )
        batch_n_step_rewards = torch.zeros(
            (batch_size, seq_len), dtype=torch.float32, device=self.device
        )
        batch_policy_targets = torch.zeros(
            (batch_size, seq_len, action_dim), dtype=torch.float32, device=self.device
        )
        batch_value_targets = torch.zeros(
            (batch_size, seq_len), dtype=torch.float32, device=self.device
        )
        for b_idx, sequence in enumerate(batch_sequences):
            if len(sequence) != seq_len:
                raise ValueError(f"Sequence {b_idx} len {len(sequence)} != {seq_len}")
            for s_idx, step_data in enumerate(sequence):
                obs = step_data["observation"]
                batch_grids[b_idx, s_idx] = torch.from_numpy(obs["grid"])
                batch_others[b_idx, s_idx] = torch.from_numpy(obs["other_features"])
                batch_actions[b_idx, s_idx] = step_data["action"]
                batch_n_step_rewards[b_idx, s_idx] = step_data["n_step_reward_target"]
                policy_map = step_data["policy_target"]
                for action, prob in policy_map.items():
                    if 0 <= action < action_dim:
                        batch_policy_targets[b_idx, s_idx, action] = prob
                policy_sum = batch_policy_targets[b_idx, s_idx].sum()
                if abs(policy_sum - 1.0) > 1e-5 and policy_sum > 1e-9:
                    batch_policy_targets[b_idx, s_idx] /= policy_sum
                elif policy_sum <= 1e-9 and action_dim > 0:
                    batch_policy_targets[b_idx, s_idx].fill_(1.0 / action_dim)
                batch_value_targets[b_idx, s_idx] = step_data["value_target"]
        return {
            "grids": batch_grids,
            "others": batch_others,
            "actions": batch_actions,
            "n_step_rewards": batch_n_step_rewards,
            "policy_targets": batch_policy_targets,
            "value_targets": batch_value_targets,
        }

    def _calculate_target_distribution(
        self, target_scalars: torch.Tensor, support: torch.Tensor
    ) -> torch.Tensor:
        """Projects scalar targets onto the fixed support atoms (z or r)."""
        target_shape = target_scalars.shape
        num_atoms = support.size(0)
        v_min = support[0]
        v_max = support[-1]
        delta = (v_max - v_min) / (num_atoms - 1) if num_atoms > 1 else 0.0

        target_scalars_flat = target_scalars.flatten()
        target_scalars_flat = target_scalars_flat.clamp(v_min, v_max)
        b: torch.Tensor = (
            (target_scalars_flat - v_min) / delta
            if delta > 0
            else torch.zeros_like(target_scalars_flat)
        )
        lower_idx: torch.Tensor = b.floor().long()
        upper_idx: torch.Tensor = b.ceil().long()

        lower_idx = torch.max(
            torch.tensor(0, device=self.device, dtype=torch.long), lower_idx
        )
        upper_idx = torch.min(
            torch.tensor(num_atoms - 1, device=self.device, dtype=torch.long), upper_idx
        )
        lower_eq_upper = lower_idx == upper_idx
        lower_idx[lower_eq_upper & (lower_idx > 0)] -= 1
        upper_idx[lower_eq_upper & (upper_idx < num_atoms - 1)] += 1

        m_lower: torch.Tensor = (upper_idx.float() - b).clamp(min=0.0, max=1.0)
        m_upper: torch.Tensor = (b - lower_idx.float()).clamp(min=0.0, max=1.0)

        m = torch.zeros(target_scalars_flat.size(0), num_atoms, device=self.device)
        # Create index tensor explicitly
        batch_indices = torch.arange(
            target_scalars_flat.size(0), device=self.device, dtype=torch.long
        )

        # Use index_put_ for sparse updates (more robust than index_add_)
        m.index_put_((batch_indices, lower_idx), m_lower, accumulate=True)
        m.index_put_((batch_indices, upper_idx), m_upper, accumulate=True)

        m = m.view(*target_shape, num_atoms)
        return m

    def _calculate_loss(
        self,
        policy_logits,
        value_logits,
        reward_logits,
        target_data,
        is_weights,  # Add IS weights
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Calculates MuZero losses, applying IS weights."""
        pi_target = target_data["policy_targets"]
        z_target = target_data["value_targets"]
        r_target_n_step = target_data["n_step_rewards"]
        batch_size, seq_len, action_dim = pi_target.shape

        # --- Expand IS weights ---
        # is_weights has shape [batch_size, 1]
        # Expand to match the shape of per-step losses [batch_size, seq_len]
        is_weights_expanded = is_weights.expand(-1, seq_len).reshape(-1)
        # For reward loss, we only need weights for steps 1 to K (unroll_steps)
        is_weights_reward = is_weights.expand(-1, self.unroll_steps).reshape(-1)
        # --- END Expand IS weights ---

        # --- Policy Loss ---
        policy_logits_flat = policy_logits.view(-1, action_dim)
        pi_target_flat = pi_target.view(-1, action_dim)
        log_pred_p = F.log_softmax(policy_logits_flat, dim=1)
        policy_loss_per_sample = -torch.sum(pi_target_flat * log_pred_p, dim=1)
        policy_loss = (is_weights_expanded * policy_loss_per_sample).mean()

        # --- Value Loss ---
        value_target_dist = self._calculate_target_distribution(
            z_target, self.value_support
        )
        value_logits_flat = value_logits.view(-1, self.num_value_atoms)
        value_target_dist_flat = value_target_dist.view(-1, self.num_value_atoms)
        log_pred_v = F.log_softmax(value_logits_flat, dim=1)
        value_loss_per_sample = -torch.sum(value_target_dist_flat * log_pred_v, dim=1)
        value_loss = (is_weights_expanded * value_loss_per_sample).mean()

        # --- Reward Loss ---
        # Target rewards are for steps k=1 to K (n_step_reward_target[t] is target for r_{t+1})
        r_target_k = r_target_n_step[:, 1 : self.unroll_steps + 1]
        reward_target_dist = self._calculate_target_distribution(
            r_target_k, self.reward_support
        )
        # reward_logits are for steps k=1 to K (output of dynamics for action a_k)
        reward_logits_flat = reward_logits.reshape(-1, self.num_reward_atoms)
        reward_target_dist_flat = reward_target_dist.reshape(-1, self.num_reward_atoms)
        log_pred_r = F.log_softmax(reward_logits_flat, dim=1)
        reward_loss_per_sample = -torch.sum(reward_target_dist_flat * log_pred_r, dim=1)
        reward_loss = (is_weights_reward * reward_loss_per_sample).mean()

        # --- Total Loss ---
        total_loss = (
            self.train_config.POLICY_LOSS_WEIGHT * policy_loss
            + self.train_config.VALUE_LOSS_WEIGHT * value_loss
            + self.train_config.REWARD_LOSS_WEIGHT * reward_loss
        )

        # --- Calculate TD Errors for PER Update ---
        # Use the value prediction for the initial state (k=0) vs its target
        value_pred_scalar_0 = self.nn._logits_to_scalar(
            value_logits[:, 0, :], self.value_support
        )
        value_target_scalar_0 = z_target[:, 0]
        td_errors_tensor = (value_target_scalar_0 - value_pred_scalar_0).detach()

        return total_loss, policy_loss, value_loss, reward_loss, td_errors_tensor

    def train_step(
        self, batch_sample: SampledBatchPER | SampledBatch
    ) -> tuple[dict[str, float], np.ndarray] | None:
        """Performs one training step, handling PER if enabled."""
        if not batch_sample:
            return None

        self.model.train()

        # --- Unpack batch sample ---
        if isinstance(batch_sample, dict) and "sequences" in batch_sample:
            batch_sequences = batch_sample["sequences"]
            is_weights_np = batch_sample["weights"]
            is_weights = torch.from_numpy(is_weights_np).to(self.device).unsqueeze(-1)
        else:
            batch_sequences = batch_sample
            batch_size = len(batch_sequences)
            is_weights = torch.ones((batch_size, 1), device=self.device)

        if not batch_sequences:
            return None

        try:
            target_data = self._prepare_batch(batch_sequences)
        except Exception as e:
            logger.error(f"Error preparing batch: {e}", exc_info=True)
            return None

        self.optimizer.zero_grad()
        obs_grid_0 = target_data["grids"][:, 0].contiguous()
        obs_other_0 = target_data["others"][:, 0].contiguous()

        # Initial inference (h + f)
        policy_logits_0, value_logits_0, initial_hidden_state = self.model(
            obs_grid_0, obs_other_0
        )

        policy_logits_list = [policy_logits_0]
        value_logits_list = [value_logits_0]
        reward_logits_list = []
        hidden_state = initial_hidden_state

        # Unroll dynamics and prediction (g + f)
        for k in range(self.unroll_steps):
            action_k = target_data["actions"][:, k + 1]  # Action a_{k+1}
            hidden_state, reward_logits_k_plus_1 = self.model.dynamics(
                hidden_state, action_k
            )
            policy_logits_k_plus_1, value_logits_k_plus_1 = self.model.predict(
                hidden_state
            )

            policy_logits_list.append(policy_logits_k_plus_1)
            value_logits_list.append(value_logits_k_plus_1)
            reward_logits_list.append(reward_logits_k_plus_1)

        # Stack predictions
        policy_logits_all = torch.stack(policy_logits_list, dim=1)
        value_logits_all = torch.stack(value_logits_list, dim=1)
        reward_logits_k = torch.stack(reward_logits_list, dim=1)

        # Calculate loss
        total_loss, policy_loss, value_loss, reward_loss, td_errors_tensor = (
            self._calculate_loss(
                policy_logits_all,
                value_logits_all,
                reward_logits_k,
                target_data,
                is_weights,
            )
        )

        # Backpropagate
        total_loss.backward()
        if self.train_config.GRADIENT_CLIP_VALUE is not None:
            torch.nn.utils.clip_grad_norm_(
                self.model.parameters(), self.train_config.GRADIENT_CLIP_VALUE
            )
        self.optimizer.step()
        if self.scheduler:
            self.scheduler.step()

        # Calculate entropy (optional logging)
        with torch.no_grad():
            policy_probs = F.softmax(policy_logits_all, dim=-1)
            entropy = (
                -torch.sum(policy_probs * torch.log(policy_probs + 1e-9), dim=-1)
                .mean()
                .item()
            )

        loss_info = {
            "total_loss": float(total_loss.detach().item()),
            "policy_loss": float(policy_loss.detach().item()),
            "value_loss": float(value_loss.detach().item()),
            "reward_loss": float(reward_loss.detach().item()),
            "entropy": float(entropy),
        }

        td_errors_np = td_errors_tensor.cpu().numpy()

        return loss_info, td_errors_np

    def get_current_lr(self) -> float:
        try:
            return float(self.optimizer.param_groups[0]["lr"])
        except Exception:
            logger.warning("Could not retrieve LR.")
            return 0.0

    def load_optimizer_state(self, state_dict: dict):
        try:
            self.optimizer.load_state_dict(state_dict)
            for state in self.optimizer.state.values():
                for k, v in state.items():
                    if isinstance(v, torch.Tensor):
                        state[k] = v.to(self.device)
            logger.info("Optimizer state loaded.")
        except Exception as e:
            logger.error(f"Failed load optimizer state: {e}", exc_info=True)
