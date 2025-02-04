# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.


import torch
from semilearn.algorithms.hooks import MaskingHook

class AdaMatchThresholdingHook(MaskingHook):
    """
    Relative Confidence Thresholding in AdaMatch
    """

    @torch.no_grad()
    def masking(self, algorithm, logits_x_lb, logits_x_ulb, softmax_x_lb=True, softmax_x_ulb=True,  *args, **kwargs):
        if softmax_x_ulb:
            # probs_x_ulb = torch.softmax(logits_x_ulb.detach(), dim=-1)
            probs_x_ulb = algorithm.compute_prob(logits_x_ulb.detach())
        else:
            # logits is already probs
            probs_x_ulb = logits_x_ulb.detach()

        if softmax_x_lb:
            # probs_x_lb = torch.softmax(logits_x_lb.detach(), dim=-1)
            probs_x_lb = algorithm.compute_prob(logits_x_lb.detach())
        else:
            # logits is already probs
            probs_x_lb = logits_x_lb.detach()

        max_probs, _ = probs_x_lb.max(dim=-1)
        p_cutoff = max_probs.mean() * algorithm.p_cutoff
        max_probs, _ = probs_x_ulb.max(dim=-1)
        mask = max_probs.ge(p_cutoff).to(max_probs.dtype)
        return mask
        
        
class InstantThresholdingHook(MaskingHook):
    """
    Instance-dependnet Thresholding
    """
    def __init__(self, num_classes, momentum=0.999, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.num_classes = num_classes
        self.m = momentum
        
        self.p_model = torch.ones((self.num_classes)) / self.num_classes
        self.label_hist = torch.ones((self.num_classes)) / self.num_classes
        self.time_p = self.p_model.mean()
    def update(self, algorithm, probs_x_ulb):
        if algorithm.distributed and algorithm.world_size > 1:
            probs_x_ulb = concat_all_gather(probs_x_ulb)
        max_probs, max_idx = torch.max(probs_x_ulb, dim=-1,keepdim=True)

        if algorithm.use_quantile:
            self.time_p = self.time_p * self.m + (1 - self.m) * torch.quantile(max_probs,0.8) #* max_probs.mean()
        else:
            self.time_p = self.time_p * self.m + (1 - self.m) * max_probs.mean().detach().cpu()
        
        if algorithm.clip_thresh:
            self.time_p = torch.clip(self.time_p, 0.0, 0.95)

        self.p_model = self.p_model * self.m + (1 - self.m) * probs_x_ulb.mean(dim=0).detach().cpu()
        hist = torch.bincount(max_idx.reshape(-1), minlength=self.p_model.shape[0]).to(self.p_model.dtype) 
        self.label_hist = self.label_hist * self.m + (1 - self.m) * (hist / hist.sum()).detach().cpu()

        algorithm.p_model = self.p_model 
        algorithm.label_hist = self.label_hist 
        algorithm.time_p = self.time_p
    
    @torch.no_grad()
    def masking(self, algorithm, logits_x_lb, logits_x_ulb, x_ulb_w, softmax_x_lb=True, softmax_x_ulb=True,  *args, **kwargs):
        if softmax_x_ulb:
            # probs_x_ulb = torch.softmax(logits_x_ulb.detach(), dim=-1)
            probs_x_ulb = algorithm.compute_prob(logits_x_ulb.detach())
        else:
            # logits is already probs
            probs_x_ulb = logits_x_ulb.detach()

        if softmax_x_lb:
            # probs_x_lb = torch.softmax(logits_x_lb.detach(), dim=-1)
            probs_x_lb = algorithm.compute_prob(logits_x_lb.detach())
        else:
            # logits is already probs
            probs_x_lb = logits_x_lb.detach()
        self.update(algorithm, probs_x_ulb)
        max_probs, _ = probs_x_ulb.max(dim=-1)
        p_cutoff = max_probs.mean() * algorithm.p_cutoff
        max_probs, _ = probs_x_ulb.max(dim=-1)
        probs_x_ulb = torch.softmax(probs_x_ulb / algorithm.T, dim=-1)
        p_label = probs_x_ulb.argmax(dim=-1)
        if algorithm.epoch < 10:
            T = algorithm.class_T.cuda(algorithm.gpu)
            
            # thresholds = torch.zeros(probs_x_ulb.size()[0])
            thresholds = T[p_label,p_label].squeeze() * torch.topk(probs_x_ulb, 2, dim=-1)[0][:,-1]
            # j_vectors = torch.t(T)[p_label] # bs * n_class
            j_vectors = torch.t(T)[p_label]
            # print(j_vectors)
            j_vectors[:,p_label] = 0
        
            thresholds += (j_vectors.squeeze() * probs_x_ulb.squeeze()).sum(dim=-1)
            # thresholds *= 50
            thresholds += p_cutoff
        else:
            batch_T = torch.softmax(algorithm.estimator(x_ulb_w),dim=-1)
            # thresholds = torch.zeros(probs_x_ulb.size()[0])
            indexes = torch.arange(batch_T.size()[0])
            thresholds = batch_T[indexes,p_label,p_label].squeeze() * torch.topk(probs_x_ulb, 2, dim=-1)[0][:,-1]
            j_vectors = batch_T[indexes,:,p_label]
            j_vectors[indexes,p_label] = 0
            thresholds += (j_vectors.squeeze() * probs_x_ulb.squeeze()).sum(dim=-1)
            thresholds += p_cutoff
        thresholds = torch.clip(thresholds,0.0,0.99)
        algorithm.average_thres += thresholds.sum().item()
        mask = max_probs.ge(thresholds.cuda(algorithm.gpu)).to(max_probs.dtype)
        return mask
