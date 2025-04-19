# File: muzerotriangle/rl/core/trainer.py
import logging
from typing import TYPE_CHECKING, cast

import torch
import torch.nn.functional as F
import torch.optim as optim

from ...utils.types import (
    SampledBatch,
)

if TYPE_CHECKING:
    from torch.optim.lr_scheduler import _LRScheduler

logger = logging.getLogger(__name__)


class Trainer:
    """MuZero Trainer."""

    # ... (init, _create_optimizer, _create_scheduler remain the same) ...
    def __init__(self, nn_interface, train_config, env_config):
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
        # (No changes needed here)
        batch_size = len(batch_sequences)
        seq_len = self.unroll_steps + 1
        action_dim = int(self.env_config.ACTION_DIM)  # type: ignore[call-overload]
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
        batch_rewards = torch.zeros(
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
                batch_rewards[b_idx, s_idx] = step_data["reward"]
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
            "rewards": batch_rewards,
            "policy_targets": batch_policy_targets,
            "value_targets": batch_value_targets,
        }

    def _calculate_target_distribution(
        self, target_scalars: torch.Tensor, support: torch.Tensor
    ) -> torch.Tensor:
        """Projects scalar targets onto the fixed support atoms (z or r)."""
        target_shape = target_scalars.shape  # (B, T) or (B, K)
        num_atoms = support.size(0)
        v_min = support[0]
        v_max = support[-1]
        delta = (v_max - v_min) / (num_atoms - 1) if num_atoms > 1 else 0.0

        # Flatten targets for easier processing: (B*T) or (B*K)
        target_scalars_flat = (
            target_scalars.flatten()
        )  # Shape (N,) where N = B*T or B*K

        target_scalars_flat = target_scalars_flat.clamp(v_min, v_max)
        # Add type hints for clarity
        b: torch.Tensor = (
            (target_scalars_flat - v_min) / delta
            if delta > 0
            else torch.zeros_like(target_scalars_flat)
        )
        lower_idx: torch.Tensor = b.floor().long()
        upper_idx: torch.Tensor = b.ceil().long()

        # Handle cases where target hits an atom exactly (l==u)
        # Ensure indices are within bounds
        lower_idx = torch.max(
            torch.tensor(0, device=self.device, dtype=torch.long), lower_idx
        )
        upper_idx = torch.min(
            torch.tensor(num_atoms - 1, device=self.device, dtype=torch.long), upper_idx
        )

        # Calculate weights
        m_lower: torch.Tensor = upper_idx.float() - b
        m_upper: torch.Tensor = b - lower_idx.float()

        # Distribute probability mass
        m = torch.zeros(
            target_scalars_flat.size(0), num_atoms, device=self.device
        )  # Shape (N, num_atoms)
        # Explicitly type batch_indices
        batch_indices: torch.Tensor = torch.arange(
            target_scalars_flat.size(0), device=self.device, dtype=torch.long
        )

        # Use advanced indexing (this should be correct)
        m[batch_indices, lower_idx] += m_lower
        m[batch_indices, upper_idx] += m_upper

        # Reshape back to original batch/sequence shape + atoms dim
        m = m.view(*target_shape, num_atoms)  # (B, T, num_atoms) or (B, K, num_atoms)
        return m

    def _calculate_loss(
        self, policy_logits, value_logits, reward_logits, target_data
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        # (No changes needed here, relies on _calculate_target_distribution)
        pi_target = target_data["policy_targets"]
        z_target = target_data["value_targets"]
        r_target = target_data["rewards"]
        action_dim = pi_target.shape[-1]  # Get action dim from target
        policy_logits_flat = policy_logits.view(-1, action_dim)
        pi_target_flat = pi_target.view(-1, action_dim)
        log_pred_p = F.log_softmax(policy_logits_flat, dim=1)
        policy_loss = -torch.sum(pi_target_flat * log_pred_p, dim=1).mean()
        value_target_dist = self._calculate_target_distribution(
            z_target, self.value_support
        )
        value_logits_flat = value_logits.view(-1, self.num_value_atoms)
        value_target_dist_flat = value_target_dist.view(-1, self.num_value_atoms)
        log_pred_v = F.log_softmax(value_logits_flat, dim=1)
        value_loss = -torch.sum(value_target_dist_flat * log_pred_v, dim=1).mean()
        r_target_k = r_target[:, : self.unroll_steps]  # Rewards r_1 to r_K
        reward_target_dist = self._calculate_target_distribution(
            r_target_k, self.reward_support
        )
        reward_logits_flat = reward_logits.view(-1, self.num_reward_atoms)
        reward_target_dist_flat = reward_target_dist.view(-1, self.num_reward_atoms)
        log_pred_r = F.log_softmax(reward_logits_flat, dim=1)
        reward_loss = -torch.sum(reward_target_dist_flat * log_pred_r, dim=1).mean()
        total_loss = (
            self.train_config.POLICY_LOSS_WEIGHT * policy_loss
            + self.train_config.VALUE_LOSS_WEIGHT * value_loss
            + self.train_config.REWARD_LOSS_WEIGHT * reward_loss
        )
        return total_loss, policy_loss, value_loss, reward_loss

    def train_step(self, batch_sequences: SampledBatch) -> dict[str, float] | None:
        # (No changes needed in main logic flow)
        if not batch_sequences:
            return None
        self.model.train()
        try:
            target_data = self._prepare_batch(batch_sequences)
        except Exception as e:
            logger.error(f"Error preparing batch: {e}", exc_info=True)
            return None

        self.optimizer.zero_grad()
        obs_grid_0 = target_data["grids"][:, 0].contiguous()
        obs_other_0 = target_data["others"][:, 0].contiguous()
        policy_logits_0, value_logits_0, initial_hidden_state = self.model(
            obs_grid_0, obs_other_0
        )
        policy_logits_list = [policy_logits_0]
        value_logits_list = [value_logits_0]
        reward_logits_list = []
        hidden_state = initial_hidden_state

        for k in range(self.unroll_steps):
            action_k_plus_1 = target_data["actions"][:, k]
            policy_logits_k, value_logits_k, reward_logits_k, hidden_state = (
                self.model.recurrent_inference(hidden_state, action_k_plus_1)
            )
            policy_logits_list.append(policy_logits_k)
            value_logits_list.append(value_logits_k)
            reward_logits_list.append(reward_logits_k)

        policy_logits_all = torch.stack(policy_logits_list, dim=1)
        value_logits_all = torch.stack(value_logits_list, dim=1)
        reward_logits_k = torch.stack(reward_logits_list, dim=1)
        total_loss, policy_loss, value_loss, reward_loss = self._calculate_loss(
            policy_logits_all, value_logits_all, reward_logits_k, target_data
        )
        total_loss.backward()
        if self.train_config.GRADIENT_CLIP_VALUE is not None:
            torch.nn.utils.clip_grad_norm_(
                self.model.parameters(), self.train_config.GRADIENT_CLIP_VALUE
            )
        self.optimizer.step()
        if self.scheduler:
            self.scheduler.step()

        with torch.no_grad():
            policy_probs = F.softmax(policy_logits_all, dim=-1)
            entropy = (
                -torch.sum(policy_probs * torch.log(policy_probs + 1e-9), dim=-1)
                .mean()
                .item()
            )
        loss_info = {
            "total_loss": total_loss.item(),
            "policy_loss": policy_loss.item(),
            "value_loss": value_loss.item(),
            "reward_loss": reward_loss.item(),
            "entropy": entropy,
        }
        return loss_info

    def get_current_lr(self) -> float:
        # (No changes needed here)
        try:
            return float(self.optimizer.param_groups[0]["lr"])
        except Exception:
            logger.warning("Could not retrieve LR.")
            return 0.0

    def load_optimizer_state(self, state_dict: dict):
        # (No changes needed here)
        try:
            self.optimizer.load_state_dict(state_dict)
            for state in self.optimizer.state.values():
                for k, v in state.items():
                    if isinstance(v, torch.Tensor):
                        state[k] = v.to(self.device)
            logger.info("Optimizer state loaded.")
        except Exception as e:
            logger.error(f"Failed load optimizer state: {e}", exc_info=True)
