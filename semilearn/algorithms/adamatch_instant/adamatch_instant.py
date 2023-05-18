# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import torch
import torch.nn as nn
import torch.nn.functional as F

from .utils import AdaMatchThresholdingHook, InstantThresholdingHook
from semilearn.core import AlgorithmBase
from semilearn.core.utils import ALGORITHMS
from semilearn.algorithms.hooks import PseudoLabelingHook, DistAlignEMAHook
from semilearn.algorithms.utils import SSL_Argument, str2bool
from .T_estimator import ResNet18_F, ResNet34, ResNet50

def replace_inf_to_zero(val):
    val[val == float('inf')] = 0.0
    return val

def entropy_loss(mask, logits_s, prob_model, label_hist):
    mask = mask.bool()

    # select samples
    logits_s = logits_s[mask]

    prob_s = logits_s.softmax(dim=-1)
    _, pred_label_s = torch.max(prob_s, dim=-1)

    hist_s = torch.bincount(pred_label_s, minlength=logits_s.shape[1]).to(logits_s.dtype)
    hist_s = hist_s / hist_s.sum()

    # modulate prob model 
    prob_model = prob_model.reshape(1, -1)
    label_hist = label_hist.reshape(1, -1)
    # prob_model_scaler = torch.nan_to_num(1 / label_hist, nan=0.0, posinf=0.0, neginf=0.0).detach()
    prob_model_scaler = replace_inf_to_zero(1 / label_hist).detach()
    mod_prob_model = prob_model * prob_model_scaler
    mod_prob_model = mod_prob_model / mod_prob_model.sum(dim=-1, keepdim=True)

    # modulate mean prob
    mean_prob_scaler_s = replace_inf_to_zero(1 / hist_s).detach()
    # mean_prob_scaler_s = torch.nan_to_num(1 / hist_s, nan=0.0, posinf=0.0, neginf=0.0).detach()
    mod_mean_prob_s = prob_s.mean(dim=0, keepdim=True) * mean_prob_scaler_s
    mod_mean_prob_s = mod_mean_prob_s / mod_mean_prob_s.sum(dim=-1, keepdim=True)
    loss = mod_prob_model.cuda(mod_mean_prob_s.get_device()) * torch.log(mod_mean_prob_s + 1e-12)
    loss = loss.sum(dim=1)
    return loss.mean(), hist_s.mean()

@ALGORITHMS.register('adamatch_instant')
class AdaMatch_InstanT(AlgorithmBase):
    """
        AdaMatch algorithm (https://arxiv.org/abs/2106.04732).

        Args:
            - args (`argparse`):
                algorithm arguments
            - net_builder (`callable`):
                network loading function
            - tb_log (`TBLog`):
                tensorboard logger
            - logger (`logging.Logger`):
                logger to use
            - T (`float`):
                Temperature for pseudo-label sharpening
            - p_cutoff(`float`):
                Confidence threshold for generating pseudo-labels
            - hard_label (`bool`, *optional*, default to `False`):
                If True, targets have [Batch size] shape with int values. If False, the target is vector
            - ema_p (`float`):
                momentum for average probability
    """
    def __init__(self, args, net_builder, tb_log=None, logger=None):
        super().__init__(args, net_builder, tb_log, logger) 
        self.init(p_cutoff=args.p_cutoff, T=args.T, hard_label=args.hard_label, ema_p=args.ema_p)
        self.instant_start = 0
        if self.dataset == 'cifar10' or self.dataset == 'cifar100':
            self.estimator = ResNet34(self.num_classes*self.num_classes).cuda(self.args.gpu)
        elif self.dataset == 'stl10':
            self.estimator = ResNet34(self.num_classes*self.num_classes).cuda(self.args.gpu)
        self.T_optimizer = torch.optim.SGD(self.estimator.parameters(), lr=0.001, weight_decay=5e-4)
        self.use_quantile=False
        self.clip_thresh=False
    def init(self, p_cutoff, T, hard_label=True, ema_p=0.999):
        self.p_cutoff = p_cutoff
        self.T = T
        self.use_hard_label = hard_label
        self.ema_p = ema_p


    def set_hooks(self):
        self.register_hook(PseudoLabelingHook(), "PseudoLabelingHook")
        self.register_hook(
            DistAlignEMAHook(num_classes=self.num_classes, momentum=self.args.ema_p, p_target_type='model'), 
            "DistAlignHook")
        # self.register_hook(AdaMatchThresholdingHook(), "MaskingHook")
        self.register_hook(InstantThresholdingHook(num_classes=self.num_classes, momentum=0.999), "MaskingHook")
        super().set_hooks()


    def train_step(self, x_lb, y_lb, idx_ulb, x_ulb_w, x_ulb_s):
        num_lb = y_lb.shape[0]

        # inference and calculate sup/unsup losses
        with self.amp_cm():
            if self.use_cat:
                inputs = torch.cat((x_lb, x_ulb_w, x_ulb_s))
                outputs = self.model(inputs)
                logits_x_lb = outputs['logits'][:num_lb]
                logits_x_ulb_w, logits_x_ulb_s = outputs['logits'][num_lb:].chunk(2)
                feats_x_lb = outputs['feat'][:num_lb]
                feats_x_ulb_w, feats_x_ulb_s = outputs['feat'][num_lb:].chunk(2)
            else:
                outs_x_lb = self.model(x_lb) 
                logits_x_lb = outs_x_lb['logits']
                feats_x_lb = outs_x_lb['feat']
                outs_x_ulb_s = self.model(x_ulb_s)
                logits_x_ulb_s = outs_x_ulb_s['logits']
                feats_x_ulb_s = outs_x_ulb_s['feat']
                with torch.no_grad():
                    outs_x_ulb_w = self.model(x_ulb_w)
                    logits_x_ulb_w = outs_x_ulb_w['logits']
                    feats_x_ulb_w = outs_x_ulb_w['feat']
            feat_dict = {'x_lb':feats_x_lb, 'x_ulb_w':feats_x_ulb_w, 'x_ulb_s':feats_x_ulb_s}
                    

            sup_loss = self.ce_loss(logits_x_lb, y_lb, reduction='mean')
            y_ulb = self.dataset_dict['train_ulb'].__sample__(idx_ulb.cpu())[-1]
            # probs_x_lb = torch.softmax(logits_x_lb.detach(), dim=-1)
            # probs_x_ulb_w = torch.softmax(logits_x_ulb_w.detach(), dim=-1)
            probs_x_lb = self.compute_prob(logits_x_lb.detach())
            probs_x_ulb_w = self.compute_prob(logits_x_ulb_w.detach())

            # distribution alignment 
            probs_x_ulb_w = self.call_hook("dist_align", "DistAlignHook", probs_x_ulb=probs_x_ulb_w, probs_x_lb=probs_x_lb)

            # calculate weight
            mask = self.call_hook("masking", "MaskingHook", logits_x_lb=probs_x_lb, logits_x_ulb=probs_x_ulb_w, x_ulb_w=x_ulb_w, softmax_x_lb=False, softmax_x_ulb=False)
            self.total += mask.sum()
            self.total_sum += mask.size()[0]
            # generate unlabeled targets using pseudo label hook
            pseudo_label = self.call_hook("gen_ulb_targets", "PseudoLabelingHook", 
                                          logits=probs_x_ulb_w,
                                          use_hard_label=self.use_hard_label,
                                          T=self.T,
                                          softmax=False)
            bool_mask = torch.gt(mask, 0)
            self.correct += torch.sum(pseudo_label[bool_mask] == torch.LongTensor(y_ulb).cuda(self.gpu)[bool_mask])
            # calculate loss
            unsup_loss = self.consistency_loss(logits_x_ulb_s,
                                               pseudo_label,
                                               'ce',
                                               mask=mask)
            total_loss = sup_loss + self.lambda_u * unsup_loss

        out_dict = self.process_out_dict(loss=total_loss, feat=feat_dict)
        log_dict = self.process_log_dict(sup_loss=sup_loss.item(), 
                                         unsup_loss=unsup_loss.item(), 
                                         total_loss=total_loss.item(), 
                                         util_ratio=mask.float().mean().item())
        return out_dict, log_dict


    def get_save_dict(self):
        save_dict = super().get_save_dict()
        # additional saving arguments
        save_dict['p_model'] = self.hooks_dict['DistAlignHook'].p_model.cpu()
        save_dict['p_target'] = self.hooks_dict['DistAlignHook'].p_target.cpu()
        return save_dict


    def load_model(self, load_path):
        checkpoint = super().load_model(load_path)
        self.hooks_dict['DistAlignHook'].p_model = checkpoint['p_model'].cuda(self.args.gpu)
        self.hooks_dict['DistAlignHook'].p_target = checkpoint['p_target'].cuda(self.args.gpu)
        self.print_fn("additional parameter loaded")
        return checkpoint
        
    def eval_T(self, true, pred):
        return torch.abs(true - pred).mean()

    def estimate_ncp(self, x_lb, y_lb):
        out = self.model(x_lb)
        ncp = F.softmax(out['logits'],dim=-1)
        return ncp, ncp.size()[0]
        
    def train_T(self, ncp, x_lb):
    
        
        loss_function = nn.CrossEntropyLoss()
        for epoch in range(0, 1):
            T_loss = 0.
            T_mae = 0.
            self.estimator.train()
            pointer = 0
            
            for i in range(0,self.num_labels,10):
                data = x_lb[i:i+10].cuda(self.gpu)
                y_lb = torch.LongTensor(self.dataset_dict['train_lb'].targets[i:i+10]).cuda(self.args.gpu)
                batch_matrix = self.estimator(data) # batch_size x nclass x nclass
                noisy_class_post = torch.zeros((10, self.num_classes))
                for j in range(batch_matrix.shape[0]):
                    y = F.one_hot(y_lb[j], self.num_classes).float()
                    y_one_hot = y.unsqueeze(0)
                    noisy_class_post_temp = y_one_hot.float().mm(batch_matrix[j]) # 1*10 noisy
                    noisy_class_post[j, :] = noisy_class_post_temp
                    # noisy_class_post = torch.log(noisy_class_post+1e-12)
                loss = loss_function(noisy_class_post.cuda(self.args.gpu), ncp[pointer:pointer+batch_matrix.shape[0]].cuda(self.args.gpu))
                T_mae += self.eval_T(ncp[pointer:pointer+batch_matrix.shape[0]], noisy_class_post.cuda(self.args.gpu))
                pointer += batch_matrix.shape[0]
                self.T_optimizer.zero_grad()
                loss.backward()
                self.T_optimizer.step()
                T_loss += loss.item()
            print('Bayesian-T Training Epoch [%d], Loss: %.4f, MAE: %.4f'% (epoch + 1, T_loss, T_mae))
        
        '''
        for data in self.loader_dict['train_ulb'].dataset:
            x = data['x_ulb_w'].cuda(self.args.gpu).unsqueeze(0)
            pred = self.estimator(x).detach().cpu()
            T_ulb.append(pred[])
            total += pred.size()[0]
        b = torch.Tensor(total,self.num_classes)
        T_ulb = torch.cat(T_ulb, out=b)
        T_dir = self.save_dir + self.save_name + '_T' + '_' + str(self.it)
        torch.save(T_ulb, T_dir)
        self.T_hat = T_ulb
        '''
        

    @staticmethod
    def get_argument():
        return [
            SSL_Argument('--hard_label', str2bool, True),
            SSL_Argument('--T', float, 0.5),
            SSL_Argument('--ema_p', float, 0.999),
            SSL_Argument('--p_cutoff', float, 0.95),
        ]
