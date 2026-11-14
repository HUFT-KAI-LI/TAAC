"""PCVRHyFormer pointwise trainer (binary-classification, AUC-monitored).

Despite the historical "Ranking" suffix in the class name, the training loop
uses pointwise BCE / Focal loss and evaluates Binary AUC + binary logloss.
"""

import os
import glob
import shutil
import logging
import csv
import json
from contextlib import nullcontext
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm
from sklearn.metrics import roc_auc_score

from utils import sigmoid_focal_loss, EarlyStopping
from model import ModelInput


class PCVRHyFormerRankingTrainer:
    """PCVRHyFormer trainer for pointwise binary classification.

    Uses PCVR data layout:
    - user_int_feats, user_dense_feats
    - item_int_feats, item_dense_feats
    - seq_a, seq_b, seq_c, seq_d (each with *_len companion)
    - label (binary)

    Loss: BCEWithLogitsLoss or Focal Loss.
    Metrics: BinaryAUROC + binary logloss.
    """

    def __init__(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        valid_loader: DataLoader,
        lr: float,
        num_epochs: int,
        device: str,
        save_dir: str,
        early_stopping: EarlyStopping,
        monday_peak_early_stopping: Optional[EarlyStopping] = None,
        loss_type: str = 'bce',
        focal_alpha: float = 0.1,
        focal_gamma: float = 2.0,
        sparse_lr: float = 0.05,
        sparse_weight_decay: float = 0.0,
        reinit_sparse_after_epoch: int = 1,
        reinit_cardinality_threshold: int = 0,
        ckpt_params: Optional[Dict[str, Any]] = None,
        writer: Optional[Any] = None,
        schema_path: Optional[str] = None,
        ns_groups_path: Optional[str] = None,
        eval_every_n_steps: int = 0,
        train_config: Optional[Dict[str, Any]] = None,
        use_amp: bool = False,
        amp_dtype: str = 'auto',
    ) -> None:
        self.model: nn.Module = model
        self.train_loader: DataLoader = train_loader
        self.valid_loader: DataLoader = valid_loader
        self.writer = writer
        # schema_path is copied alongside every checkpoint so that infer.py can
        # rebuild the exact same feature schema the model was trained with.
        self.schema_path: Optional[str] = schema_path
        # ns_groups_path is optional; copied next to schema.json when provided
        # and points at an existing file. Keeping the JSON inside the ckpt dir
        # makes the checkpoint self-contained for evaluation environments that
        # do not ship ns_groups.json separately.
        self.ns_groups_path: Optional[str] = ns_groups_path
        self.sparse_lr: float = sparse_lr
        self.sparse_weight_decay: float = sparse_weight_decay

        # Dual optimizer: Adagrad for sparse Embeddings, AdamW for dense params.
        self.sparse_optimizer: Optional[torch.optim.Optimizer]
        if hasattr(model, 'get_sparse_params'):
            sparse_params = model.get_sparse_params()
            dense_params = model.get_dense_params()
            sparse_param_count = sum(p.numel() for p in sparse_params)
            dense_param_count = sum(p.numel() for p in dense_params)
            logging.info(f"Sparse params: {len(sparse_params)} tensors, {sparse_param_count:,} parameters (Adagrad lr={sparse_lr})")
            logging.info(f"Dense params: {len(dense_params)} tensors, {dense_param_count:,} parameters (AdamW lr={lr})")
            self.sparse_optimizer = self._make_sparse_optimizer(sparse_params)
            self.dense_optimizer: torch.optim.Optimizer = torch.optim.AdamW(
                dense_params, lr=lr, betas=(0.9, 0.98)
            )
        else:
            self.sparse_optimizer = None
            self.dense_optimizer = torch.optim.AdamW(
                model.parameters(), lr=lr, betas=(0.9, 0.98)
            )

        self.num_epochs: int = num_epochs
        self.device: str = device
        self.save_dir: str = save_dir
        self.early_stopping: EarlyStopping = early_stopping
        self.monday_peak_early_stopping: Optional[EarlyStopping] = monday_peak_early_stopping
        self.loss_type: str = loss_type
        self.focal_alpha: float = focal_alpha
        self.focal_gamma: float = focal_gamma
        self.reinit_sparse_after_epoch: int = reinit_sparse_after_epoch
        self.reinit_cardinality_threshold: int = reinit_cardinality_threshold
        self.ckpt_params: Dict[str, Any] = ckpt_params or {}
        self.eval_every_n_steps: int = eval_every_n_steps
        self.train_config: Optional[Dict[str, Any]] = train_config
        self.use_amp: bool = bool(
            use_amp and str(device).startswith('cuda') and torch.cuda.is_available()
        )
        self.amp_dtype_name: str = amp_dtype
        self.amp_dtype = self._resolve_amp_dtype(amp_dtype)
        self.grad_scaler = self._make_grad_scaler(
            enabled=self.use_amp and self.amp_dtype == torch.float16
        )
        self.train_loss_history: List[Dict[str, float]] = []
        self.eval_history: List[Dict[str, float]] = []
        self.epoch_history: List[Dict[str, float]] = []

        logging.info(f"PCVRHyFormerRankingTrainer loss_type={loss_type}, "
                     f"focal_alpha={focal_alpha}, focal_gamma={focal_gamma}, "
                     f"reinit_sparse_after_epoch={reinit_sparse_after_epoch}, "
                     f"reinit_cardinality_threshold={reinit_cardinality_threshold}")
        if self.use_amp:
            logging.info(f"AMP enabled: dtype={self.amp_dtype}")

    def _make_sparse_optimizer(
        self,
        sparse_params,
    ) -> torch.optim.Optimizer:
        """Create sparse Adagrad with memory-friendly CUDA behavior."""
        try:
            return torch.optim.Adagrad(
                sparse_params,
                lr=self.sparse_lr,
                weight_decay=self.sparse_weight_decay,
                foreach=False,
            )
        except TypeError:
            logging.warning(
                "This PyTorch build does not support Adagrad(foreach=False); "
                "falling back to the default Adagrad implementation."
            )
            return torch.optim.Adagrad(
                sparse_params,
                lr=self.sparse_lr,
                weight_decay=self.sparse_weight_decay,
            )

    def _resolve_amp_dtype(self, amp_dtype: str) -> torch.dtype:
        """Choose the autocast dtype used on CUDA."""
        amp_dtype = (amp_dtype or 'auto').lower()
        if amp_dtype == 'bf16':
            return torch.bfloat16
        if amp_dtype == 'fp16':
            return torch.float16
        bf16_ok = (
            hasattr(torch.cuda, 'is_bf16_supported')
            and torch.cuda.is_available()
            and torch.cuda.is_bf16_supported()
        )
        return torch.bfloat16 if bf16_ok else torch.float16

    @staticmethod
    def _make_grad_scaler(enabled: bool):
        """Create a GradScaler across PyTorch 2.x API variants."""
        if hasattr(torch, 'amp') and hasattr(torch.amp, 'GradScaler'):
            try:
                return torch.amp.GradScaler('cuda', enabled=enabled)
            except TypeError:
                return torch.amp.GradScaler(enabled=enabled)
        return torch.cuda.amp.GradScaler(enabled=enabled)

    def _autocast_context(self):
        if not self.use_amp:
            return nullcontext()
        if hasattr(torch, 'amp') and hasattr(torch.amp, 'autocast'):
            return torch.amp.autocast('cuda', dtype=self.amp_dtype)
        return torch.cuda.amp.autocast(dtype=self.amp_dtype)

    def _build_step_dir_name(
        self,
        global_step: int,
        is_best: bool = False,
        best_tag: str = "best_model",
    ) -> str:
        """Build a checkpoint sub-directory name such as
        ``global_step2500.layer=2.head=4.hidden=64[.best_model]``.
        """
        parts = [f"global_step{global_step}"]
        for key in ("layer", "head", "hidden"):
            if key in self.ckpt_params:
                parts.append(f"{key}={self.ckpt_params[key]}")
        name = ".".join(parts)
        if is_best:
            name += f".{best_tag}"
        return name

    def _write_sidecar_files(self, ckpt_dir: str) -> None:
        """Write sidecar files next to a ``model.pt``.

        Currently persists up to three files, all overwritten on every call:

        - ``schema.json`` (copied from ``self.schema_path``): feature layout
          metadata needed to rebuild the Parquet dataset.
        - ``ns_groups.json`` (copied from ``self.ns_groups_path`` when set
          and the file exists): NS-token grouping used to construct the
          tokenizer. Making a per-ckpt copy lets evaluation environments
          consume the checkpoint without having to ship the original
          project-level ``ns_groups.json``.
        - ``train_config.json`` (serialized from ``self.train_config``):
          full set of training-time hyperparameters. When ``ns_groups.json``
          is copied into ``ckpt_dir``, the ``ns_groups_json`` field is
          rewritten to the bare filename so that ``infer.py`` resolves it
          against ``ckpt_dir`` rather than the original absolute path on
          the training machine.
        """
        os.makedirs(ckpt_dir, exist_ok=True)
        if self.schema_path and os.path.exists(self.schema_path):
            shutil.copy2(self.schema_path, ckpt_dir)

        ns_groups_copied = False
        if self.ns_groups_path and os.path.exists(self.ns_groups_path):
            shutil.copy2(self.ns_groups_path, ckpt_dir)
            ns_groups_copied = True

        if self.train_config:
            import json
            cfg_to_dump = self.train_config
            if ns_groups_copied:
                # Override the stored path to a filename relative to ckpt_dir;
                # infer.py already falls back to `<ckpt_dir>/<basename>` when
                # the recorded path is not absolute, which keeps the ckpt
                # portable across hosts.
                cfg_to_dump = dict(self.train_config)
                cfg_to_dump['ns_groups_json'] = os.path.basename(
                    self.ns_groups_path)
            with open(os.path.join(ckpt_dir, 'train_config.json'), 'w') as f:
                json.dump(cfg_to_dump, f, indent=2)

    def _save_step_checkpoint(
        self,
        global_step: int,
        is_best: bool = False,
        best_tag: str = "best_model",
        skip_model_file: bool = False,
    ) -> str:
        """Save ``model.pt`` plus sidecar files under a ``global_step`` sub-dir.

        Args:
            global_step: current global step used to name the directory.
            is_best: whether this is a new-best checkpoint.
            skip_model_file: if True, skip writing ``model.pt`` (because the
                caller, e.g. EarlyStopping, has already persisted it to the
                same path). Sidecar files are still (re)written.

        Returns:
            The absolute path of the checkpoint directory.
        """
        dir_name = self._build_step_dir_name(
            global_step,
            is_best=is_best,
            best_tag=best_tag,
        )
        ckpt_dir = os.path.join(self.save_dir, dir_name)
        os.makedirs(ckpt_dir, exist_ok=True)
        if not skip_model_file:
            torch.save(self.model.state_dict(), os.path.join(ckpt_dir, "model.pt"))
        self._write_sidecar_files(ckpt_dir)
        logging.info(f"Saved checkpoint to {ckpt_dir}/model.pt")
        return ckpt_dir

    def _remove_old_best_dirs(self, best_tag: str = "best_model") -> None:
        """Delete stale ``*.best-tag`` directories so that only the latest
        best checkpoint is kept on disk.
        """
        pattern = os.path.join(self.save_dir, f"global_step*.{best_tag}")
        for old_dir in glob.glob(pattern):
            shutil.rmtree(old_dir)
            logging.info(f"Removed old {best_tag} dir: {old_dir}")

    def _batch_to_device(self, batch: Dict[str, Any]) -> Dict[str, Any]:
        """Move all tensors in ``batch`` to ``self.device`` (``non_blocking=True``,
        to cooperate with ``pin_memory``). Non-tensor values pass through.
        """
        device_batch: Dict[str, Any] = {}
        for k, v in batch.items():
            if isinstance(v, torch.Tensor):
                device_batch[k] = v.to(self.device, non_blocking=True)
            else:
                device_batch[k] = v
        return device_batch

    def _handle_validation_result(
        self,
        total_step: int,
        val_auc: float,
        val_logloss: float,
    ) -> None:
        self._handle_best_result(
            total_step=total_step,
            score=val_auc,
            score_name="AUC",
            logloss=val_logloss,
            tracker=self.early_stopping,
            best_tag="best_model",
            extra_metrics={
                "best_val_AUC": val_auc,
                "best_val_logloss": val_logloss,
            },
        )

    def _handle_best_result(
        self,
        total_step: int,
        score: float,
        score_name: str,
        logloss: float,
        tracker: Optional[EarlyStopping],
        best_tag: str,
        extra_metrics: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Persist a new-best checkpoint atomically.

        Flow (ordered to avoid leaving empty sidecar-only directories on disk):

        1. Decide whether ``val_auc`` is *likely* to beat the current best
           using the same threshold as ``EarlyStopping._is_not_improved``,
           so our pre-cleanup and EarlyStopping's internal save decision
           stay in sync.
        2. If unlikely, short-circuit: do nothing on disk. We must NOT
           touch ``self.early_stopping.checkpoint_path`` or call
           ``_write_sidecar_files`` because the target directory may not
           exist yet (sidecar-only dirs would otherwise be created here,
           producing checkpoints with missing ``model.pt``).
        3. If likely, point ``EarlyStopping`` at the canonical
           ``global_stepN.best_model/model.pt`` path, remove any stale
           ``*.best_model`` dirs, then run ``EarlyStopping`` (which writes
           ``model.pt`` when it actually confirms a new best).
        4. Only after ``EarlyStopping`` has confirmed a new best
           (``best_score != old_best``) do we write the sidecar files into
           the freshly-created directory; this is guarded so that a
           razor-close score that tripped ``is_likely_new_best`` but not
           ``EarlyStopping``'s own gate does not create a stray dir.
        """
        if tracker is None:
            return

        old_best = tracker.best_score
        is_likely_new_best = (
            old_best is None
            or score > old_best + tracker.delta
        )
        if not is_likely_new_best:
            # No new best anticipated: leave disk untouched. The previous
            # best_model dir (with its model.pt + sidecars) remains valid.
            tracker(score, self.model, extra_metrics)
            return

        # Point EarlyStopping at the canonical best-model location for this
        # step. Only done on the likely-new-best branch so that a skipped
        # save never leaks the unused path into EarlyStopping state.
        best_dir = os.path.join(
            self.save_dir,
            self._build_step_dir_name(
                total_step,
                is_best=True,
                best_tag=best_tag,
            ),
        )
        tracker.checkpoint_path = os.path.join(best_dir, "model.pt")

        # Remove stale best dirs first so EarlyStopping's write is the only
        # I/O needed when a new best is confirmed.
        self._remove_old_best_dirs(best_tag=best_tag)

        tracker(score, self.model, extra_metrics)

        # Write sidecar files only when EarlyStopping actually confirmed a
        # new best and wrote model.pt. If the score tripped our heuristic
        # but EarlyStopping internally declined to save, skip to avoid
        # creating an empty (sidecar-only) checkpoint directory.
        if tracker.best_score != old_best and os.path.exists(
            tracker.checkpoint_path
        ):
            self._save_step_checkpoint(
                total_step,
                is_best=True,
                best_tag=best_tag,
                skip_model_file=True,
            )
            logging.info(
                "Saved %s checkpoint at step %s with %s=%.6f, logloss=%.6f",
                best_tag,
                total_step,
                score_name,
                score,
                logloss,
            )

    @staticmethod
    def _safe_auc(labels_np: np.ndarray, probs_np: np.ndarray) -> float:
        if len(probs_np) == 0 or len(np.unique(labels_np)) < 2:
            return 0.0
        return float(roc_auc_score(labels_np, probs_np))

    @staticmethod
    def _safe_logloss_from_logits(
        logits: torch.Tensor,
        labels: torch.Tensor,
    ) -> float:
        if len(logits) == 0:
            return float('inf')
        return F.binary_cross_entropy_with_logits(logits, labels.float()).item()

    def _record_train_loss(
        self,
        epoch: int,
        step_in_epoch: int,
        total_step: int,
        loss: float,
    ) -> None:
        self.train_loss_history.append({
            "epoch": float(epoch),
            "step_in_epoch": float(step_in_epoch),
            "global_step": float(total_step),
            "batch_loss": float(loss),
        })

    def _record_eval_metrics(
        self,
        scope: str,
        epoch: int,
        total_step: int,
        metrics: Dict[str, float],
        train_epoch_loss: Optional[float] = None,
    ) -> None:
        row = {
            "scope": scope,
            "epoch": float(epoch),
            "global_step": float(total_step),
            "train_epoch_loss": (
                float(train_epoch_loss) if train_epoch_loss is not None else float('nan')
            ),
        }
        row.update({k: float(v) for k, v in metrics.items()})
        self.eval_history.append(row)

    def _record_epoch_metrics(
        self,
        epoch: int,
        total_step: int,
        train_epoch_loss: float,
        metrics: Dict[str, float],
    ) -> None:
        row = {
            "epoch": float(epoch),
            "global_step": float(total_step),
            "train_epoch_loss": float(train_epoch_loss),
        }
        row.update({k: float(v) for k, v in metrics.items()})
        self.epoch_history.append(row)

    def _write_metric_artifacts(self) -> None:
        os.makedirs(self.save_dir, exist_ok=True)
        history_path = os.path.join(self.save_dir, "training_metrics_history.json")
        with open(history_path, "w", encoding="utf-8") as f:
            json.dump(
                {
                    "train_loss": self.train_loss_history,
                    "eval": self.eval_history,
                    "epoch": self.epoch_history,
                },
                f,
                indent=2,
            )

        self._write_csv(
            os.path.join(self.save_dir, "train_batch_loss.csv"),
            self.train_loss_history,
        )
        self._write_csv(
            os.path.join(self.save_dir, "valid_metrics.csv"),
            self.eval_history,
        )
        self._write_csv(
            os.path.join(self.save_dir, "epoch_metrics.csv"),
            self.epoch_history,
        )
        self._plot_metric_curves()

    @staticmethod
    def _write_csv(path: str, rows: List[Dict[str, float]]) -> None:
        if not rows:
            return
        fieldnames = sorted({key for row in rows for key in row.keys()})
        with open(path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)

    def _plot_metric_curves(self) -> None:
        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
        except Exception as exc:
            logging.warning("matplotlib unavailable; skipped metric plot: %s", exc)
            return

        fig, axes = plt.subplots(2, 2, figsize=(14, 9))

        if self.train_loss_history:
            x = [row["global_step"] for row in self.train_loss_history]
            y = [row["batch_loss"] for row in self.train_loss_history]
            axes[0, 0].plot(x, y, linewidth=0.8, alpha=0.65, label="batch loss")
            if len(y) >= 20:
                window = min(100, max(20, len(y) // 20))
                kernel = np.ones(window, dtype=np.float64) / window
                smooth = np.convolve(np.asarray(y, dtype=np.float64), kernel, mode="valid")
                axes[0, 0].plot(x[window - 1:], smooth, linewidth=1.2, label=f"ma{window}")
            axes[0, 0].set_title("Batch Train Loss")
            axes[0, 0].set_xlabel("global step")
            axes[0, 0].set_ylabel("loss")
            axes[0, 0].legend()

        if self.epoch_history:
            x = [row["epoch"] for row in self.epoch_history]
            y = [row["train_epoch_loss"] for row in self.epoch_history]
            axes[0, 1].plot(x, y, marker="o", label="epoch train loss")
            axes[0, 1].set_title("Epoch Train Loss")
            axes[0, 1].set_xlabel("epoch")
            axes[0, 1].set_ylabel("loss")
            axes[0, 1].legend()

        if self.eval_history:
            x = [row["global_step"] for row in self.eval_history]
            axes[1, 0].plot(x, [row["auc"] for row in self.eval_history], marker="o", label="valid AUC")
            axes[1, 0].plot(x, [row["peak_7_9_auc"] for row in self.eval_history], marker="o", label="7-9 AUC")
            axes[1, 0].plot(x, [row["workday_peak_7_9_auc"] for row in self.eval_history], marker="o", label="workday 7-9 AUC")
            axes[1, 0].plot(x, [row["monday_peak_8_10_auc"] for row in self.eval_history], marker="o", label="monday 8-10 AUC")
            axes[1, 0].set_title("Validation AUC")
            axes[1, 0].set_xlabel("global step")
            axes[1, 0].set_ylabel("AUC")
            axes[1, 0].legend()

            axes[1, 1].plot(x, [row["logloss"] for row in self.eval_history], marker="o", label="valid logloss")
            axes[1, 1].plot(x, [row["peak_7_9_logloss"] for row in self.eval_history], marker="o", label="7-9 logloss")
            axes[1, 1].plot(x, [row["positive_rate"] for row in self.eval_history], marker="o", label="positive rate")
            axes[1, 1].set_title("Validation LogLoss / Rate")
            axes[1, 1].set_xlabel("global step")
            axes[1, 1].legend()

        for ax in axes.ravel():
            ax.grid(True, alpha=0.25)

        fig.tight_layout()
        fig.savefig(os.path.join(self.save_dir, "training_curves.png"), dpi=160)
        plt.close(fig)

    def _write_metrics_to_tensorboard(
        self,
        metrics: Dict[str, float],
        total_step: int,
    ) -> None:
        if not self.writer:
            return
        self.writer.add_scalar('AUC/valid', metrics['auc'], total_step)
        self.writer.add_scalar('AUC/valid_peak_7_9', metrics['peak_7_9_auc'], total_step)
        self.writer.add_scalar(
            'AUC/valid_workday_peak_7_9',
            metrics['workday_peak_7_9_auc'],
            total_step,
        )
        self.writer.add_scalar(
            'AUC/valid_monday_peak_8_10',
            metrics['monday_peak_8_10_auc'],
            total_step,
        )
        self.writer.add_scalar('LogLoss/valid', metrics['logloss'], total_step)
        self.writer.add_scalar('LogLoss/valid_peak_7_9', metrics['peak_7_9_logloss'], total_step)
        self.writer.add_scalar(
            'LogLoss/valid_monday_peak_8_10',
            metrics['monday_peak_8_10_logloss'],
            total_step,
        )
        self.writer.add_scalar('Rate/valid_positive', metrics['positive_rate'], total_step)
        self.writer.add_scalar('Count/valid_peak_7_9', metrics['peak_7_9_count'], total_step)
        self.writer.add_scalar(
            'Count/valid_workday_peak_7_9',
            metrics['workday_peak_7_9_count'],
            total_step,
        )
        self.writer.add_scalar(
            'Count/valid_monday_peak_8_10',
            metrics['monday_peak_8_10_count'],
            total_step,
        )

    def train(self) -> None:
        """Main training loop: iterates over epochs, performs step-level and
        epoch-level validation, triggers EarlyStopping and the periodic sparse
        re-initialization strategy.
        """
        print("Start training (PCVRHyFormer)")
        self.model.train()
        total_step = 0

        for epoch in range(1, self.num_epochs + 1):
            train_pbar = tqdm(enumerate(self.train_loader), total=len(self.train_loader),
                              dynamic_ncols=True)
            loss_sum = 0.0

            for step, batch in train_pbar:
                loss = self._train_step(batch)
                total_step += 1
                loss_sum += loss
                self._record_train_loss(epoch, step + 1, total_step, loss)

                if self.writer:
                    self.writer.add_scalar('Loss/train', loss, total_step)

                train_pbar.set_postfix({"loss": f"{loss:.4f}"})

                # Step-level validation (only when eval_every_n_steps > 0).
                if self.eval_every_n_steps > 0 and total_step % self.eval_every_n_steps == 0:
                    logging.info(f"Evaluating at step {total_step}")
                    metrics = self.evaluate(epoch=epoch)
                    val_auc = metrics['auc']
                    val_logloss = metrics['logloss']
                    self.model.train()
                    torch.cuda.empty_cache()

                    logging.info(
                        "Step %s Validation | AUC: %.6f, LogLoss: %.6f, "
                        "AUC_7_9: %.6f(rows=%d), AUC_workday_7_9: %.6f(rows=%d), "
                        "AUC_monday_8_10: %.6f(rows=%d)",
                        total_step,
                        val_auc,
                        val_logloss,
                        metrics['peak_7_9_auc'],
                        int(metrics['peak_7_9_count']),
                        metrics['workday_peak_7_9_auc'],
                        int(metrics['workday_peak_7_9_count']),
                        metrics['monday_peak_8_10_auc'],
                        int(metrics['monday_peak_8_10_count']),
                    )

                    if self.writer:
                        self._write_metrics_to_tensorboard(metrics, total_step)

                    self._record_eval_metrics('step', epoch, total_step, metrics)
                    self._write_metric_artifacts()

                    self._handle_validation_result(total_step, val_auc, val_logloss)
                    self._handle_best_result(
                        total_step=total_step,
                        score=metrics['monday_peak_8_10_auc'],
                        score_name='monday_peak_8_10_AUC',
                        logloss=metrics['monday_peak_8_10_logloss'],
                        tracker=self.monday_peak_early_stopping,
                        best_tag='best_monday_8_10_model',
                        extra_metrics={
                            "best_monday_peak_8_10_auc": metrics['monday_peak_8_10_auc'],
                            "best_monday_peak_8_10_logloss": metrics['monday_peak_8_10_logloss'],
                            "best_monday_peak_8_10_count": metrics['monday_peak_8_10_count'],
                        },
                    )

                    if self.early_stopping.early_stop:
                        logging.info(f"Early stopping at step {total_step}")
                        return

            avg_epoch_loss = loss_sum / max(len(self.train_loader), 1)
            logging.info(f"Epoch {epoch}, Average Loss: {avg_epoch_loss}")

            metrics = self.evaluate(epoch=epoch)
            val_auc = metrics['auc']
            val_logloss = metrics['logloss']
            self.model.train()
            torch.cuda.empty_cache()

            logging.info(
                "Epoch %s Validation | AUC: %.6f, LogLoss: %.6f, "
                "AUC_7_9: %.6f(rows=%d), AUC_workday_7_9: %.6f(rows=%d), "
                "AUC_monday_8_10: %.6f(rows=%d), "
                "TrainLoss: %.6f",
                epoch,
                val_auc,
                val_logloss,
                metrics['peak_7_9_auc'],
                int(metrics['peak_7_9_count']),
                metrics['workday_peak_7_9_auc'],
                int(metrics['workday_peak_7_9_count']),
                metrics['monday_peak_8_10_auc'],
                int(metrics['monday_peak_8_10_count']),
                avg_epoch_loss,
            )

            if self.writer:
                self.writer.add_scalar('Loss/train_epoch', avg_epoch_loss, epoch)
                self._write_metrics_to_tensorboard(metrics, total_step)

            self._record_eval_metrics('epoch', epoch, total_step, metrics, avg_epoch_loss)
            self._record_epoch_metrics(epoch, total_step, avg_epoch_loss, metrics)
            self._write_metric_artifacts()

            self._handle_validation_result(total_step, val_auc, val_logloss)
            self._handle_best_result(
                total_step=total_step,
                score=metrics['monday_peak_8_10_auc'],
                score_name='monday_peak_8_10_AUC',
                logloss=metrics['monday_peak_8_10_logloss'],
                tracker=self.monday_peak_early_stopping,
                best_tag='best_monday_8_10_model',
                extra_metrics={
                    "best_monday_peak_8_10_auc": metrics['monday_peak_8_10_auc'],
                    "best_monday_peak_8_10_logloss": metrics['monday_peak_8_10_logloss'],
                    "best_monday_peak_8_10_count": metrics['monday_peak_8_10_count'],
                },
            )

            if self.early_stopping.early_stop:
                logging.info(f"Early stopping at epoch {epoch}")
                break

            # After the configured epoch, reinitialize high-cardinality sparse
            # params (Embeddings) as a form of cold restart to reduce overfit.
            # Reference: KuaiShou Tech., "MultiEpoch: Reusing Training Data
            # for Click-Through Rate Prediction",
            # https://arxiv.org/pdf/2305.19531
            if epoch >= self.reinit_sparse_after_epoch and self.sparse_optimizer is not None:
                reinit_ptrs = self.model.reinit_high_cardinality_params(
                    self.reinit_cardinality_threshold)

                # Move Adagrad state only for parameters that were not reset.
                # Avoid deepcopying large embedding states: for threshold=0
                # almost every sparse table is reset, and copying all states
                # can double GPU memory at the start of the next epoch.
                old_state: Dict[nn.Parameter, Any] = {}
                old_sparse_optimizer = self.sparse_optimizer
                for group in self.sparse_optimizer.param_groups:
                    for p in group['params']:
                        if p.data_ptr() in reinit_ptrs:
                            continue
                        if p in self.sparse_optimizer.state:
                            old_state[p] = self.sparse_optimizer.state[p]

                sparse_params = self.model.get_sparse_params()
                self.sparse_optimizer = self._make_sparse_optimizer(sparse_params)
                # Restore optimizer state for low-cardinality embeddings only.
                restored = 0
                reset = 0
                for p in sparse_params:
                    if p.data_ptr() in reinit_ptrs:
                        reset += 1
                        continue
                    if p in old_state:
                        self.sparse_optimizer.state[p] = old_state[p]
                        restored += 1
                old_sparse_optimizer.state.clear()
                old_sparse_optimizer.param_groups.clear()
                del old_sparse_optimizer
                del old_state
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                logging.info(f"Rebuilt Adagrad optimizer after epoch {epoch}, "
                             f"reset_embedding={reset}, "
                             f"restored_embedding={restored}")

    def _make_model_input(self, device_batch: Dict[str, Any]) -> ModelInput:
        """Construct a ``ModelInput`` NamedTuple from a device_batch dict."""
        seq_domains = device_batch['_seq_domains']
        seq_data: Dict[str, torch.Tensor] = {}
        seq_lens: Dict[str, torch.Tensor] = {}
        seq_time_buckets: Dict[str, torch.Tensor] = {}
        seq_time_stats: Dict[str, torch.Tensor] = {}
        for domain in seq_domains:
            seq_data[domain] = device_batch[domain]
            seq_lens[domain] = device_batch[f'{domain}_len']
            B = device_batch[domain].shape[0]
            L = device_batch[domain].shape[2]
            seq_time_buckets[domain] = device_batch.get(
                f'{domain}_time_bucket',
                torch.zeros(B, L, dtype=torch.long, device=self.device))
            seq_time_stats[domain] = device_batch.get(
                f'{domain}_time_stats',
                torch.zeros(B, 0, dtype=torch.float32, device=self.device))
        return ModelInput(
            user_int_feats=device_batch['user_int_feats'],
            item_int_feats=device_batch['item_int_feats'],
            user_dense_feats=device_batch['user_dense_feats'],
            item_dense_feats=device_batch['item_dense_feats'],
            seq_data=seq_data,
            seq_lens=seq_lens,
            seq_time_buckets=seq_time_buckets,
            seq_time_stats=seq_time_stats,
            timestamp_weekday=device_batch['timestamp_weekday'],
            timestamp_hour=device_batch['timestamp_hour'],
        )

    def _train_step(self, batch: Dict[str, Any]) -> float:
        """Run a single training step and return the scalar loss value."""
        device_batch = self._batch_to_device(batch)
        label = device_batch['label'].float()

        self.dense_optimizer.zero_grad()
        if self.sparse_optimizer is not None:
            self.sparse_optimizer.zero_grad()

        with self._autocast_context():
            model_input = self._make_model_input(device_batch)
            logits = self.model(model_input)  # (B, 1)
            logits = logits.squeeze(-1)  # (B,)

            if self.loss_type == 'focal':
                loss = sigmoid_focal_loss(
                    logits, label, alpha=self.focal_alpha, gamma=self.focal_gamma)
            else:
                loss = F.binary_cross_entropy_with_logits(logits, label)

        if self.use_amp and self.amp_dtype == torch.float16:
            self.grad_scaler.scale(loss).backward()
            self.grad_scaler.unscale_(self.dense_optimizer)
            if self.sparse_optimizer is not None:
                self.grad_scaler.unscale_(self.sparse_optimizer)
        else:
            loss.backward()

        # foreach=False: avoids a PyTorch _foreach_norm CUDA kernel bug observed
        # with certain tensor shapes in this project.
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0, foreach=False)

        if self.use_amp and self.amp_dtype == torch.float16:
            self.grad_scaler.step(self.dense_optimizer)
            if self.sparse_optimizer is not None:
                self.grad_scaler.step(self.sparse_optimizer)
            self.grad_scaler.update()
        else:
            self.dense_optimizer.step()
            if self.sparse_optimizer is not None:
                self.sparse_optimizer.step()

        return loss.item()

    def evaluate(self, epoch: Optional[int] = None) -> Dict[str, float]:
        """Run validation and return overall plus time-window metrics.

        NaN predictions (which can arise from exploding gradients) are filtered
        out before computing both metrics.
        """
        print("Start Evaluation (PCVRHyFormer) - validation")
        self.model.eval()
        if not epoch:
            epoch = -1

        pbar = tqdm(enumerate(self.valid_loader), total=len(self.valid_loader))

        all_logits_list = []
        all_labels_list = []
        all_hour_list = []
        all_weekday_list = []

        with torch.no_grad():
            for step, batch in pbar:
                logits, labels, hours, weekdays = self._evaluate_step(batch)
                all_logits_list.append(logits.detach().float().cpu())
                all_labels_list.append(labels.detach().cpu())
                all_hour_list.append(hours.detach().cpu())
                all_weekday_list.append(weekdays.detach().cpu())

        all_logits = torch.cat(all_logits_list, dim=0).float()
        all_labels = torch.cat(all_labels_list, dim=0).long()
        all_hours = torch.cat(all_hour_list, dim=0).long()
        all_weekdays = torch.cat(all_weekday_list, dim=0).long()

        # Binary AUC via sklearn.
        probs = torch.sigmoid(all_logits).numpy()
        labels_np = all_labels.numpy()
        hours_np = all_hours.numpy()
        weekdays_np = all_weekdays.numpy()

        # Filter NaN predictions (may appear if gradients explode).
        nan_mask = np.isnan(probs)
        if nan_mask.any():
            n_nan = int(nan_mask.sum())
            logging.warning(f"[Evaluate] {n_nan}/{len(probs)} predictions are NaN, filtering them out")
            valid_mask = ~nan_mask
            probs = probs[valid_mask]
            labels_np = labels_np[valid_mask]
            all_logits = all_logits[valid_mask]
            all_labels = all_labels[valid_mask]
            hours_np = hours_np[valid_mask]
            weekdays_np = weekdays_np[valid_mask]

        auc = self._safe_auc(labels_np, probs)

        # Binary logloss (same NaN filtering).
        valid_logits = all_logits[~torch.isnan(all_logits)]
        valid_labels = all_labels[~torch.isnan(all_logits)]
        logloss = self._safe_logloss_from_logits(valid_logits, valid_labels)

        local_hours = hours_np - 1
        peak_mask = (local_hours >= 7) & (local_hours < 9)
        workday_mask = (weekdays_np >= 1) & (weekdays_np <= 5)
        workday_peak_mask = peak_mask & workday_mask
        monday_peak_8_10_mask = (weekdays_np == 1) & np.isin(local_hours, [8, 9, 10])

        peak_auc = self._safe_auc(labels_np[peak_mask], probs[peak_mask])
        workday_peak_auc = self._safe_auc(
            labels_np[workday_peak_mask],
            probs[workday_peak_mask],
        )
        monday_peak_8_10_auc = self._safe_auc(
            labels_np[monday_peak_8_10_mask],
            probs[monday_peak_8_10_mask],
        )

        peak_logloss = self._safe_logloss_from_logits(
            valid_logits[torch.from_numpy(peak_mask).to(torch.bool)],
            valid_labels[torch.from_numpy(peak_mask).to(torch.bool)],
        )
        workday_peak_logloss = self._safe_logloss_from_logits(
            valid_logits[torch.from_numpy(workday_peak_mask).to(torch.bool)],
            valid_labels[torch.from_numpy(workday_peak_mask).to(torch.bool)],
        )
        monday_peak_8_10_logloss = self._safe_logloss_from_logits(
            valid_logits[torch.from_numpy(monday_peak_8_10_mask).to(torch.bool)],
            valid_labels[torch.from_numpy(monday_peak_8_10_mask).to(torch.bool)],
        )

        metrics = {
            "auc": auc,
            "logloss": logloss,
            "peak_7_9_auc": peak_auc,
            "peak_7_9_logloss": peak_logloss,
            "peak_7_9_count": float(peak_mask.sum()),
            "workday_peak_7_9_auc": workday_peak_auc,
            "workday_peak_7_9_logloss": workday_peak_logloss,
            "workday_peak_7_9_count": float(workday_peak_mask.sum()),
            "monday_peak_8_10_auc": monday_peak_8_10_auc,
            "monday_peak_8_10_logloss": monday_peak_8_10_logloss,
            "monday_peak_8_10_count": float(monday_peak_8_10_mask.sum()),
            "positive_rate": float(labels_np.mean()) if len(labels_np) > 0 else 0.0,
            "pred_mean": float(probs.mean()) if len(probs) > 0 else 0.0,
            "pred_std": float(probs.std()) if len(probs) > 0 else 0.0,
        }

        logging.info(
            "[EvaluateSubset] valid rows=%s AUC=%.6f LogLoss=%.6f "
            "positive_rate=%.6f pred_mean=%.6f pred_std=%.6f",
            len(probs),
            metrics["auc"],
            metrics["logloss"],
            metrics["positive_rate"],
            metrics["pred_mean"],
            metrics["pred_std"],
        )
        logging.info(
            "[EvaluateSubset] 7-9 rows=%d AUC=%.6f LogLoss=%.6f | "
            "workday_7-9 rows=%d AUC=%.6f LogLoss=%.6f | "
            "monday_8_10 rows=%d AUC=%.6f LogLoss=%.6f",
            int(metrics["peak_7_9_count"]),
            metrics["peak_7_9_auc"],
            metrics["peak_7_9_logloss"],
            int(metrics["workday_peak_7_9_count"]),
            metrics["workday_peak_7_9_auc"],
            metrics["workday_peak_7_9_logloss"],
            int(metrics["monday_peak_8_10_count"]),
            metrics["monday_peak_8_10_auc"],
            metrics["monday_peak_8_10_logloss"],
        )

        return metrics

    def _evaluate_step(
        self, batch: Dict[str, Any]
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Run a validation step and return logits, labels, hour, weekday."""
        device_batch = self._batch_to_device(batch)
        label = device_batch['label']

        with self._autocast_context():
            model_input = self._make_model_input(device_batch)
            logits, _ = self.model.predict(model_input)  # (B, 1), (B, D)
            logits = logits.squeeze(-1)  # (B,)

        return (
            logits,
            label,
            device_batch['timestamp_hour'],
            device_batch['timestamp_weekday'],
        )
