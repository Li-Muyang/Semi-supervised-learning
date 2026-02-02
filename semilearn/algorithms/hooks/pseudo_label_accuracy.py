# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import torch
import numpy as np
from collections import defaultdict
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
            self.confidence_bins = [0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 1.0]
        else:
            self.confidence_bins = confidence_bins
        
        # Accumulators for computing accuracy per bin
        self.reset_stats()
    
    def reset_stats(self):
        """Reset accumulated statistics"""
        self.correct_per_bin = defaultdict(int)
        self.total_per_bin = defaultdict(int)
    
    @torch.no_grad()
    def compute_pseudo_label_accuracy(self, 
                                       algorithm, 
                                       probs_x_ulb, 
                                       pseudo_labels, 
                                       idx_ulb,
                                       *args, **kwargs):
        """
        Compute and accumulate pseudo label accuracy by confidence bin.
        
        Args:
            algorithm: Base algorithm instance
            probs_x_ulb: Probability distribution for unlabeled data (after softmax)
            pseudo_labels: Generated pseudo labels (hard labels)
            idx_ulb: Indices of unlabeled samples in the dataset
        
        Returns:
            dict: Accuracy statistics per confidence bin
        """
        # Only process if using pre-trained model
        if not getattr(algorithm.args, 'use_pretrain', False):
            return {}
        
        # Get confidence scores (max probability)
        if probs_x_ulb.dim() > 1:
            max_probs, pred_labels = torch.max(probs_x_ulb, dim=-1)
        else:
            max_probs = probs_x_ulb
            pred_labels = pseudo_labels
        
        # Ensure pseudo_labels are hard labels
        if pseudo_labels.dim() > 1:
            pseudo_labels = torch.argmax(pseudo_labels, dim=-1)
        
        # Get true labels from the unlabeled dataset
        ulb_dataset = algorithm.dataset_dict['train_ulb']
        
        # Handle both tensor and list indices
        if isinstance(idx_ulb, torch.Tensor):
            idx_ulb = idx_ulb.cpu().numpy()
        
        # Get true labels
        true_labels = []
        for idx in idx_ulb:
            if hasattr(ulb_dataset, 'targets') and ulb_dataset.targets is not None:
                true_labels.append(ulb_dataset.targets[idx])
            else:
                # If no targets available, cannot compute accuracy
                return {}
        
        true_labels = torch.tensor(true_labels, device=pseudo_labels.device)
        
        # Move to same device
        max_probs = max_probs.detach()
        pseudo_labels = pseudo_labels.detach()
        
        # Compute accuracy per confidence bin
        stats = {}
        for i in range(len(self.confidence_bins) - 1):
            low = self.confidence_bins[i]
            high = self.confidence_bins[i + 1]
            
            # Find samples in this confidence range
            if i == len(self.confidence_bins) - 2:
                # Include upper bound for the last bin
                mask = (max_probs >= low) & (max_probs <= high)
            else:
                mask = (max_probs >= low) & (max_probs < high)
            
            if mask.sum() > 0:
                correct = (pseudo_labels[mask] == true_labels[mask]).sum().item()
                total = mask.sum().item()
                
                # Accumulate stats
                bin_key = f"{low:.2f}-{high:.2f}"
                self.correct_per_bin[bin_key] += correct
                self.total_per_bin[bin_key] += total
        
        return self._get_current_stats()
    
    def _get_current_stats(self):
        """Get current accumulated statistics"""
        stats = {}
        for bin_key in self.correct_per_bin.keys():
            total = self.total_per_bin[bin_key]
            if total > 0:
                acc = self.correct_per_bin[bin_key] / total
                stats[f"pseudo_acc_{bin_key}"] = acc
                stats[f"pseudo_count_{bin_key}"] = total
        return stats
    
    def get_and_reset_stats(self):
        """Get current stats and reset accumulators"""
        stats = self._get_current_stats()
        self.reset_stats()
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
            return
        
        stats = self.get_and_reset_stats()
        
        if not stats:
            return
        
        # Build log message
        log_parts = ["[Pseudo Label Accuracy by Confidence]"]
        
        # Sort bins for consistent output
        bin_keys = sorted([k for k in stats.keys() if k.startswith("pseudo_acc_")])
        
        for key in bin_keys:
            bin_range = key.replace("pseudo_acc_", "")
            acc = stats[key]
            count_key = f"pseudo_count_{bin_range}"
            count = stats.get(count_key, 0)
            log_parts.append(f"  Conf {bin_range}: Acc={acc:.4f} (n={count})")
        
        # Print the statistics
        if log_parts and len(log_parts) > 1:
            algorithm.print_fn("\n".join(log_parts))
        
        # Also add to tensorboard if available
        if algorithm.tb_log is not None:
            tb_stats = {k: v for k, v in stats.items() if k.startswith("pseudo_acc_")}
            if tb_stats:
                algorithm.tb_log.update(tb_stats, algorithm.it)
        
        return stats
