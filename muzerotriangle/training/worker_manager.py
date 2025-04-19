# File: muzerotriangle/training/worker_manager.py
import logging
from typing import TYPE_CHECKING

import ray
from pydantic import ValidationError

from ..rl.self_play.worker import SelfPlayWorker  # Keep worker import

# Import MuZero-specific types
from ..rl.types import SelfPlayResult

if TYPE_CHECKING:
    from .components import TrainingComponents

logger = logging.getLogger(__name__)


class WorkerManager:
    """Manages the pool of SelfPlayWorker actors, task submission, and results."""

    def __init__(self, components: "TrainingComponents"):
        self.components = components
        self.train_config = components.train_config
        self.nn = components.nn
        self.stats_collector_actor = components.stats_collector_actor

        self.workers: list[ray.actor.ActorHandle | None] = []
        self.worker_tasks: dict[ray.ObjectRef, int] = {}  # Map task ref to worker index
        self.active_worker_indices: set[int] = set()

    def initialize_workers(self):
        """Creates the pool of SelfPlayWorker Ray actors."""
        num_workers = self.train_config.NUM_SELF_PLAY_WORKERS
        logger.info(f"Initializing {num_workers} self-play workers...")
        initial_weights = self.nn.get_weights()
        weights_ref = ray.put(initial_weights)
        self.workers = [None] * num_workers

        for i in range(num_workers):
            try:
                # Pass MuZero-compatible configs
                worker = SelfPlayWorker.options(num_cpus=1).remote(
                    actor_id=i,
                    env_config=self.components.env_config,
                    mcts_config=self.components.mcts_config,
                    model_config=self.components.model_config,
                    train_config=self.train_config,  # Contains MuZero params now
                    stats_collector_actor=self.stats_collector_actor,
                    initial_weights=weights_ref,
                    seed=self.train_config.RANDOM_SEED + i,
                    worker_device_str=self.train_config.WORKER_DEVICE,
                )
                self.workers[i] = worker
                self.active_worker_indices.add(i)
            except Exception as e:
                logger.error(f"Failed to initialize worker {i}: {e}", exc_info=True)

        logger.info(f"Initialized {len(self.active_worker_indices)} active workers.")
        del weights_ref  # Clean up reference

    def submit_initial_tasks(self):
        """Submits the first task for each active worker."""
        logger.info("Submitting initial tasks to workers...")
        for worker_idx in self.active_worker_indices:
            self.submit_task(worker_idx)

    def submit_task(self, worker_idx: int):
        """Submits a new run_episode task to a specific worker."""
        if worker_idx not in self.active_worker_indices:
            return
        worker = self.workers[worker_idx]
        if worker:
            try:
                task_ref = worker.run_episode.remote()
                self.worker_tasks[task_ref] = worker_idx
                logger.debug(f"Submitted task to worker {worker_idx}")
            except Exception as e:
                logger.error(
                    f"Failed to submit task to worker {worker_idx}: {e}", exc_info=True
                )
                self.active_worker_indices.discard(worker_idx)
                self.workers[worker_idx] = None
        else:
            logger.error(f"Worker {worker_idx} is None during task submission.")
            self.active_worker_indices.discard(worker_idx)

    def get_completed_tasks(
        self, timeout: float = 0.1
    ) -> list[tuple[int, SelfPlayResult | Exception]]:  # Return type uses MuZero result
        """
        Checks for completed worker tasks using ray.wait.
        Returns a list of tuples: (worker_id, SelfPlayResult | exception).
        """
        completed_results: list[tuple[int, SelfPlayResult | Exception]] = []
        if not self.worker_tasks:
            return completed_results

        ready_refs, _ = ray.wait(
            list(self.worker_tasks.keys()), num_returns=1, timeout=timeout
        )
        if not ready_refs:
            return completed_results

        for ref in ready_refs:
            worker_idx = self.worker_tasks.pop(ref, -1)
            if worker_idx == -1 or worker_idx not in self.active_worker_indices:
                continue

            try:
                result_raw = ray.get(ref)
                # Validate using the MuZero SelfPlayResult model
                result_validated = SelfPlayResult.model_validate(result_raw)
                completed_results.append((worker_idx, result_validated))
            except ValidationError as e_val:
                error_msg = f"Pydantic validation failed for result from worker {worker_idx}: {e_val}"
                logger.error(error_msg, exc_info=False)
                logger.debug(f"Invalid data: {result_raw}")
                completed_results.append((worker_idx, ValueError(error_msg)))
            except ray.exceptions.RayActorError as e_actor:
                logger.error(
                    f"Worker {worker_idx} actor failed: {e_actor}", exc_info=True
                )
                completed_results.append((worker_idx, e_actor))
                self.workers[worker_idx] = None
                self.active_worker_indices.discard(worker_idx)
            except Exception as e_get:
                logger.error(
                    f"Error getting result from worker {worker_idx}: {e_get}",
                    exc_info=True,
                )
                completed_results.append((worker_idx, e_get))

        return completed_results

    def update_worker_networks(self, global_step: int):
        """Sends latest weights and current global_step to active workers."""
        # (No change needed in core logic here, just passes step along)
        active_workers = [
            w
            for i, w in enumerate(self.workers)
            if i in self.active_worker_indices and w is not None
        ]
        if not active_workers:
            return
        logger.debug(f"Updating worker networks for step {global_step}...")
        current_weights = self.nn.get_weights()
        weights_ref = ray.put(current_weights)
        set_weights_tasks = [
            worker.set_weights.remote(weights_ref) for worker in active_workers
        ]
        set_step_tasks = [
            worker.set_current_trainer_step.remote(global_step)
            for worker in active_workers
        ]
        all_tasks = set_weights_tasks + set_step_tasks
        if not all_tasks:
            del weights_ref
            return
        try:
            ray.get(all_tasks, timeout=120.0)
            logger.debug(
                f"Worker networks updated for {len(active_workers)} workers to step {global_step}."
            )
        except ray.exceptions.RayActorError as e:
            logger.error(f"Worker actor failed during update: {e}", exc_info=True)
        except ray.exceptions.GetTimeoutError:
            logger.error("Timeout updating workers.")
        except Exception as e:
            logger.error(f"Error updating workers: {e}", exc_info=True)
        finally:
            del weights_ref

    def get_num_active_workers(self) -> int:
        return len(self.active_worker_indices)

    def get_num_pending_tasks(self) -> int:
        return len(self.worker_tasks)

    def cleanup_actors(self):
        """Kills Ray actors associated with this manager."""
        logger.info("Cleaning up WorkerManager actors...")
        for task_ref in list(self.worker_tasks.keys()):
            try:
                ray.cancel(task_ref, force=True)
            except Exception:
                pass  # Ignore errors during cancellation
        self.worker_tasks = {}
        for i, worker in enumerate(self.workers):
            if worker:
                try:
                    ray.kill(worker, no_restart=True)
                    logger.debug(f"Killed worker {i}.")
                except Exception:
                    pass  # Ignore errors during kill
        self.workers = []
        self.active_worker_indices = set()
        logger.info("WorkerManager actors cleaned up.")
