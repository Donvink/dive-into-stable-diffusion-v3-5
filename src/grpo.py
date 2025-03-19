import torch
import torch.nn.functional as F
import numpy as np
from sklearn.neighbors import KernelDensity
from scipy.integrate import simpson
from torch.utils.data import Sampler
from typing import Any, Callable, Optional, Sized, Union


class RepeatRandomSampler(Sampler):
    """
    Sampler that repeats the indices of a dataset N times.

    Args:
        data_source (`Sized`):
            Dataset to sample from.
        repeat_count (`int`):
            Number of times to repeat each index.
        seed (`Optional[int]`):
            Random seed for reproducibility (only affects this sampler).

    Example:
    ```python
    >>> sampler = RepeatRandomSampler(["a", "b", "c", "d"], repeat_count=2)
    >>> list(sampler)
    [2, 2, 0, 0, 3, 3, 1, 1]
    ```
    """

    def __init__(self, data_source: Sized, repeat_count: int, seed: Optional[int] = None):
        self.data_source = data_source
        self.repeat_count = repeat_count
        self.num_samples = len(data_source)
        self.seed = seed
        self.generator = torch.Generator()  # Create a local random generator
        if seed is not None:
            self.generator.manual_seed(seed)

    def __iter__(self):
        indexes = [
            idx
            for idx in torch.randperm(self.num_samples, generator=self.generator).tolist()
            for _ in range(self.repeat_count)
        ]
        return iter(indexes)

    def __len__(self):
        return self.num_samples * self.repeat_count


class GRPOLossCalculator:
    def __init__(self, epsilon=0.2, beta=0.1, scaling_factor=1.5305, log_epsilon=20, kl_mode='gaussian'):
        self.epsilon = epsilon  # 裁剪阈值
        self.beta = beta        # KL惩罚系数
        self.scaling_factor = scaling_factor        # VAE默认缩放因子
        self.log_epsilon = log_epsilon  # 概率裁剪阈值
        self.kl_mode = kl_mode # KL计算模式 ('gaussian'/'kde')

    def compute_loss(self, new_latents, ref_latents, old_log_probs, rewards):
        """
        输入参数：
        - new_latents: 新策略生成的latent (batch_size, C, H, W)
        - ref_latents: 参考模型生成的latent (batch_size, C, H, W)
        - old_log_probs: 旧策略的log概率 (batch_size, seq_len)
        - rewards: 美学评分奖励 (batch_size,)
        """
        # 1. 计算新策略的log概率
        new_log_probs = self._get_latent_log_probs(new_latents)  # (batch_size,)
        
        # 2. 计算重要性采样比率
        log_diff = new_log_probs - old_log_probs
        clip_log_diff = torch.clamp(log_diff, self.log_epsilon, self.log_epsilon)
        ratio = torch.exp(clip_log_diff)  # (batch_size,)
        
        # 3. 计算组内相对优势
        # num_group == 1
        advantages = rewards  # (batch_size,)
        # TODO: add num_group > 1
        # advantages = self._compute_group_advantages(rewards)  # (batch_size,)
        
        # 4. 裁剪比率项
        clipped_ratio = torch.clamp(ratio, 1-self.epsilon, 1+self.epsilon)
        # policy_loss = -torch.min(ratio * advantages, clipped_ratio * advantages).mean()
        policy_loss = -(clipped_ratio * advantages).mean()
        
        # 5. 计算KL散度惩罚项
        kl_penalty = self._compute_kl_divergence(new_latents, ref_latents)
        
        # 6. 组合最终Loss
        total_loss = policy_loss + self.beta * kl_penalty
        return total_loss, policy_loss, kl_penalty, new_log_probs

    def _get_latent_log_probs(self, latents):
        # 添加归一化（根据VAE实际缩放因子调整）
        # latents = latents / self.scaling_factor
        
        # 标准正态分布概率计算
        dist = torch.distributions.Normal(0, 1)
        log_prob = dist.log_prob(latents).sum(dim=(1,2,3))
        return log_prob

    def _compute_group_advantages(self, rewards):
        """组内相对优势计算"""
        # 假设每组包含batch_size个样本
        mean_reward = rewards.mean()
        std_reward = rewards.std()
        return (rewards - mean_reward) / std_reward

    def _compute_kl_divergence(self, p_latents, q_latents):
        """KL散度计算（支持两种模式）"""
        if self.kl_mode == 'gaussian':
            return self._gaussian_kl(p_latents, q_latents)
        elif self.kl_mode == 'kde':
            return self._kde_kl(p_latents, q_latents)
        
    def _gaussian_kl(self, p, q):
        """高斯分布闭式解KL计算"""
        mu_p, sigma_p = p.mean(dim=(0,2,3)), p.std(dim=(0,2,3), unbiased=False)
        mu_q, sigma_q = q.mean(dim=(0,2,3)), q.std(dim=(0,2,3), unbiased=False)
        kl = torch.log(sigma_q/sigma_p) + (sigma_p**2 + (mu_p - mu_q)**2)/(2*sigma_q**2) - 0.5
        return kl.sum()
    
    def _kde_kl(self, latent1, latent2, bandwidth=0.5):
        """
        基于核密度估计的KL散度计算
        输入维度：(batchsize, channels, H, W)
        """
        # 展平为(batch*H*W, channels)
        samples1 = latent1.reshape(-1, latent1.shape[1])
        samples2 = latent2.reshape(-1, latent2.shape[1])
        
        # 训练KDE模型
        kde_p = KernelDensity(bandwidth=bandwidth).fit(samples1)
        kde_q = KernelDensity(bandwidth=bandwidth).fit(samples2)
        
        # 在重叠区域采样计算积分
        min_val = min(samples1.min(), samples2.min())
        max_val = max(samples1.max(), samples2.max())
        x = np.linspace(min_val, max_val, 1000)
        
        # 计算概率密度（取指数转为概率）
        log_p = kde_p.score_samples(x.reshape(-1, 1))
        log_q = kde_q.score_samples(x.reshape(-1, 1))
        p = np.exp(log_p)
        q = np.exp(log_q)
        
        # 计算KL散度积分
        integrand = p * (log_p - log_q)
        return simpson(integrand, x)