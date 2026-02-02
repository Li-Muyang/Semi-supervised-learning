# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import torch
from semilearn.core.hooks import Hook


class PseudoLabelAccuracyHook(Hook):
    """
    Hook for tracking pseudo label accuracy in different confidence ranges.
    Only active when using pre-trained models for semi-supervised learning.
    
    This hook computes and logs the accuracy of pseudo labels binned by 
    their confidence scores, helping to understand how pseudo label quality
    varies with model confidence during SSL fine-tuning of pre-trained models.
    """
    
    def __init__(self, confidence_bins=None):
        """
        Args:
            confidence_bins: List of confidence bin edges. 
                             Default: [0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 1.0]
        """
        super().__init__()
        if confidence_bins is None:
            self.confidence_bins = torch.tensor([0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 1.0])
        else:
            self.confidence_bins = torch.tensor(confidence_bins)
        
        self.num_bins = len(self.confidence_bins) - 1
        
        # Pre-allocate accumulators as tensors for efficiency
        self.correct_per_bin = None
        self.total_per_bin = None
        self._initialized = False
    
    def _init_accumulators(self, device):
        """Initialize accumulators on the correct device"""
        self.correct_per_bin = torch.zeros(self.num_bins, dtype=torch.float32, device=device)
        self.total_per_bin = torch.zeros(self.num_bins, dtype=torch.float32, device=device)
        self.confidence_bins = self.confidence_bins.to(device)
        self._initialized = True
    
    def reset_stats(self):
        """Reset accumulated statistics"""
        if self._initialized:
            self.correct_per_bin.zero_()
            self.total_per_bin.zero_()
    
    @torch.no_grad()
    def compute_pseudo_label_accuracy(self, 
                                       algorithm, 
                                       probs_x_ulb, 
                                       pseudo_labels, 
                                       idx_ulb,
                                       *args, **kwargs):
        """
        Compute and accumulate pseudo label accuracy by confidence bin.
        Optimized for GPU computation with minimal CPU transfers.
        
        Args:
            algorithm: Base algorithm instance
            probs_x_ulb: Probability distribution for unlabeled data (after softmax)
            pseudo_labels: Generated pseudo labels (hard labels)
            idx_ulb: Indices of unlabeled samples in the dataset
        
        Returns:
            None (stats accumulated internally)
        """
        # Only process if using pre-trained model
        if not getattr(algorithm.args, 'use_pretrain', False):
            return
        
        # Initialize accumulators on first call
        if not self._initialized:
            self._init_accumulators(probs_x_ulb.device)
        
        # Get confidence scores (max probability)
        if probs_x_ulb.dim() > 1:
            max_probs, _ = torch.max(probs_x_ulb, dim=-1)
        else:
            max_probs = probs_x_ulb
        
        # Ensure pseudo_labels are hard labels
        if pseudo_labels.dim() > 1:
            pseudo_labels = torch.argmax(pseudo_labels, dim=-1)
        
        # Get true labels from the unlabeled dataset efficiently
        ulb_dataset = algorithm.dataset_dict['train_ulb']
        if not hasattr(ulb_dataset, 'targets') or ulb_dataset.targets is None:
            return
        
        # Convert indices to numpy for efficient indexing
        if isinstance(idx_ulb, torch.Tensor):
            idx_np = idx_ulb.cpu().numpy()
        else:
            idx_np = idx_ulb
        
        # Batch get true labels using numpy indexing (much faster than loop)
        targets = ulb_dataset.targets
        if isinstance(targets, torch.Tensor):
            true_labels = targets[idx_np].to(pseudo_labels.device)
        elif hasattr(targets, '__getitem__'):
            # Works for lists and numpy arrays
            true_labels = torch.tensor(
                [targets[i] for i in idx_np], 
                device=pseudo_labels.device, 
                dtype=pseudo_labels.dtype
            )
        else:
            return
        
        # Compute correctness mask (vectorized)
        correct_mask = (pseudo_labels == true_labels)
        
        # Bin samples by confidence using searchsorted (fully vectorized)
        # This assigns each sample to a bin based on its confidence
        bin_indices = torch.searchsorted(self.confidence_bins, max_probs, right=True) - 1
        bin_indices = bin_indices.clamp(0, self.num_bins - 1)
        
        # Accumulate stats per bin using scatter_add (GPU-efficient)
        ones = torch.ones_like(max_probs)
        correct_float = correct_mask.float()
        
        self.total_per_bin.scatter_add_(0, bin_indices, ones)
        self.correct_per_bin.scatter_add_(0, bin_indices, correct_float)
    
    def _get_current_stats(self):
        """Get current accumulated statistics as a dictionary"""
        if not self._initialized:
            return {}
        
        stats = {}
        total_np = self.total_per_bin.cpu().numpy()
        correct_np = self.correct_per_bin.cpu().numpy()
        bins_np = self.confidence_bins.cpu().numpy()
        
        for i in range(self.num_bins):
            if total_np[i] > 0:
                low, high = bins_np[i], bins_np[i + 1]
                bin_key = f"{low:.2f}-{high:.2f}"
                stats[f"pseudo_acc/{bin_key}"] = correct_np[i] / total_np[i]
                stats[f"pseudo_count/{bin_key}"] = int(total_np[i])
        
        return stats
    
    def get_and_reset_stats(self):
        """Get current stats and reset accumulators"""
        stats = self._get_current_stats()
        self.reset_stats()
        return stats
    
    def get_log_dict(self):
        """Get stats formatted for log_dict integration (without resetting)"""
        if not self._initialized:
            return {}
        
        stats = {}
        total = self.total_per_bin
        correct = self.correct_per_bin
        bins = self.confidence_bins
        
        # Only transfer to CPU once at the end
        mask = total > 0
        if mask.any():
            for i in range(self.num_bins):
                if total[i] > 0:
                    low, high = bins[i].item(), bins[i + 1].item()
                    bin_key = f"{low:.2f}-{high:.2f}"
                    acc = (correct[i] / total[i]).item()
                    stats[f"pl_acc/{bin_key}"] = acc
        
        return stats
    
    def log_pseudo_label_accuracy(self, algorithm, *args, **kwargs):
        """
        Log pseudo label accuracy statistics.
        Called periodically during training.
        
        Args:
            algorithm: Base algorithm instance
        """
        # Only log if using pre-trained model
        if not getattr(algorithm.args, 'use_pretrain', False):
            return {}
        
        stats = self.get_and_reset_stats()
        
        if not stats:
            return {}
        
        # Build compact log message
        log_parts = ["[PL Acc]"]
        
        # Sort bins for consistent output
        acc_keys = sorted([k for k in stats.keys() if k.startswith("pseudo_acc/")])
        
        for key in acc_keys:
            bin_range = key.replace("pseudo_acc/", "")
            acc = stats[key]
            count_key = f"pseudo_count/{bin_range}"
            count = stats.get(count_key, 0)
            log_parts.append(f"{bin_range}:{acc:.3f}({count})")
        
        # Print the statistics (compact single line)
        if len(log_parts) > 1:
            algorithm.print_fn(" ".join(log_parts))
        
        # Also add to tensorboard if available
        if algorithm.tb_log is not None:
            tb_stats = {k: v for k, v in stats.items() if k.startswith("pseudo_acc/")}
            if tb_stats:
                algorithm.tb_log.update(tb_stats, algorithm.it)
        
        return stats
