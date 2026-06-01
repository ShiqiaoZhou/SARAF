import os

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import copy
import math
from tqdm import tqdm
import pandas as pd
from datetime import datetime, date

from torch.utils.data import Dataset, DataLoader


class RetrievalTool():
    def __init__(
        self,
        seq_len,
        pred_len,
        channels,
        n_period=3,
        temperature=0.1,
        topm=20,
        with_dec=False,
        return_key=False,
        freq='h',  # Data frequency: 'h'=hourly, 't'=minute-level(15min), 'd'=daily, etc.
        time_aware_weight=0,  # weight for time-aware weighting
        # ===== Adaptive retrieval parameters =====
        use_adaptive=True,       # whether to use adaptive retrieval
        sigma_min=0.05,          # sigma for stationary data: sharper weights
        sigma_max=0.3,           # sigma for non-stationary data: smoother weights
        lambda_min=0.3,          # lambda for non-stationary data: more diversity
        lambda_max=0.9,          # lambda for stationary data: closer to TopK
        mmr_candidate_pool=100,  # MMR candidate pool size
    ):
        period_num = [16, 8, 4, 2, 1]
        period_num = period_num[-1 * n_period:]
        
        self.seq_len = seq_len
        self.pred_len = pred_len
        self.channels = channels
        
        self.n_period = n_period
        self.period_num = sorted(period_num, reverse=True)
        
        self.temperature = temperature
        self.topm = topm
        
        self.with_dec = with_dec
        self.return_key = return_key
        
        # Time-aware retrieval parameters
        self.freq = freq
        self.time_aware_weight = time_aware_weight
        print(f"[RetrievalTool.__init__] time_aware_weight = {self.time_aware_weight}")
        
        # ===== Adaptive retrieval parameters =====
        self.use_adaptive = use_adaptive
        self.sigma_min = sigma_min
        self.sigma_max = sigma_max
        self.lambda_min = lambda_min
        self.lambda_max = lambda_max
        self.mmr_candidate_pool = mmr_candidate_pool
        
        # Compute daily and yearly period lengths from the frequency.
        self.steps_per_day, self.steps_per_year = self._compute_time_periods(freq)
    
    def _compute_time_periods(self, freq):
        """
        Compute the number of time steps per day and per year from the data frequency.
        freq: 'h'=hourly, 't'=15-minute, 'd'=daily, 'w'=weekly, 'm'=monthly, etc.
        """
        freq_map = {
            'h': (24, 24 * 365),        # hourly frequency: 24steps/day, 8760steps/year
            't': (24 * 4, 24 * 4 * 365), # 15-minute frequency: 96steps/day, 35040steps/year
            '15min': (24 * 4, 24 * 4 * 365),
            '30min': (24 * 2, 24 * 2 * 365),
            'd': (1, 365),               # daily frequency: 1steps/day, 365steps/year
            'w': (1, 52),                # weekly frequency: 1steps/week, 52steps/year
            'm': (1, 12),                # monthly frequency: 1steps/month, 12steps/year
            's': (24 * 60 * 60, 24 * 60 * 60 * 365),  # second-level frequency
        }
        
        if freq in freq_map:
            return freq_map[freq]
        else:
            # Assume hourly data by default.
            print(f"Warning: Unknown frequency '{freq}', assuming hourly data")
            return (24, 24 * 365)
        
    def prepare_dataset(self, train_data):
        train_data_all = []
        y_data_all = []
        train_data_all_mark = []

        for i in range(len(train_data)):
            td = train_data[i]
            train_data_all.append(td[1])
            train_data_all_mark.append(td[3])
                        
            if self.with_dec:
                y_data_all.append(td[2][-(train_data.pred_len + train_data.label_len):])
            else:
                y_data_all.append(td[2][-train_data.pred_len:])
            
        self.train_data_all = torch.tensor(np.stack(train_data_all, axis=0)).float()
        self.train_data_all_mg, _ = self.decompose_mg(self.train_data_all)
        
        self.y_data_all = torch.tensor(np.stack(y_data_all, axis=0)).float()
        self.y_data_all_mg, _ = self.decompose_mg(self.y_data_all)

        self.n_train = self.train_data_all.shape[0]
        
        # Save timestamp features for time-aware retrieval.
        self.train_data_all_mark = torch.tensor(np.stack(train_data_all_mark, axis=0)).float()
        print(f"Time-aware retrieval enabled. Steps per day: {self.steps_per_day}, Steps per year: {self.steps_per_year}")
        
        # ===== Compute the overall stationarity score for the training set at dataset level. =====
        # if self.use_adaptive:
        self.dataset_stationarity = self.compute_dataset_stationarity(self.train_data_all)
        # Precompute adaptive parameters from dataset stationarity.
        self.adaptive_sigma = self.compute_adaptive_sigma(self.dataset_stationarity)
        self.adaptive_lambda = self.compute_adaptive_lambda(self.dataset_stationarity)
            

    # ===== Stationarity scoring functions =====
    def compute_stationarity_scores(self, data):
        """
        Compute the time-series stationarity score s in [0, 1].
        Larger s means more stationary; smaller s means more non-stationary.
        
        Method: estimate stationarity from rolling-window changes in mean and variance.
        - Stationary series: mean and variance change little across windows.
        - Non-stationary series: mean and variance change substantially.
        
        Args:
            data: [N, seq_len, channels] time-series data
        Returns:
            stationarity: [N] stationarity score for each sample
        """
        N, L, C = data.shape
        
        # Use 6 windows to compute local statistics.
        n_windows = 6
        window_size = L // n_windows
        
        window_means = []
        window_stds = []
        
        for i in range(n_windows):
            start = i * window_size
            end = start + window_size if i < n_windows - 1 else L
            window_data = data[:, start:end, :]  # [N, window_size, C]
            
            # Compute the mean and standard deviation for each window.
            w_mean = window_data.mean(dim=1)  # [N, C]
            w_std = window_data.std(dim=1)    # [N, C]
            
            window_means.append(w_mean)
            window_stds.append(w_std)
        
        window_means = torch.stack(window_means, dim=1)  # [N, n_windows, C]
        window_stds = torch.stack(window_stds, dim=1)    # [N, n_windows, C]
        
        # Compute mean variation as the standard deviation across windows.
        mean_variation = window_means.std(dim=1).mean(dim=1)  # [N]
        
        # Compute variance variation.
        std_variation = window_stds.std(dim=1).mean(dim=1)    # [N]
        
        # Normalize using global statistics.
        global_std = data.std(dim=1).mean(dim=1)  # [N] global standard deviation
        
        mean_score = 1.0 - (mean_variation / (global_std + 1e-8)).clamp(0, 1)
        std_score = 1.0 - (std_variation / (global_std + 1e-8)).clamp(0, 1)
        
        # Combined score with equal weight for mean and variance variation.
        stationarity = 0.5 * mean_score + 0.5 * std_score
        stationarity = stationarity.clamp(0, 1)
        
        return stationarity
    
    def compute_dataset_stationarity(self, data, min_samples=200, sample_ratio=0.05):
        """
        Compute the overall stationarity score for the training set at dataset level.
        Use uniform sampling to avoid clustered windows and cover different time ranges.
        
        Args:
            data: [N, seq_len, channels] training-set data
            min_samples: minimum number of samples
            sample_ratio: sampling ratio
        Returns:
            stationarity: scalar overall dataset stationarity score
        """
        N = data.shape[0]
        
        # Use all samples to compute stationarity; sampling is disabled.
        n_samples = N
        
        # Use all data directly.
        sampled_data = data  # [N, seq_len, channels]
        
        # Compute stationarity scores for each sample.
        sample_stationarity = self.compute_stationarity_scores(sampled_data)  # [N]

        
        # Compute stationarity scores for each sampled sample.
        sample_stationarity = self.compute_stationarity_scores(sampled_data)  # [n_samples]
        
        # Use the mean stationarity as the overall dataset stationarity.
        dataset_stationarity = sample_stationarity.mean().item()
        
        # Print detailed information.
        print(f"\n" + "="*60)
        print(f"Dataset Stationarity Analysis")
        print(f"="*60)
        print(f"  Total training samples: {N}")
        print(f"  Samples analyzed: {n_samples} (100% of dataset)")
        # print(f"  Sampled for analysis: {n_samples} (step={step:.2f})")
        # print(f"  Sample indices range: [{sample_indices.min().item()}, {sample_indices.max().item()}]")
        print(f"  Stationarity scores:")
        print(f"    - Mean:   {dataset_stationarity:.4f}")
        print(f"    - Std:    {sample_stationarity.std().item():.4f}")
        print(f"    - Min:    {sample_stationarity.min().item():.4f}")
        print(f"    - Max:    {sample_stationarity.max().item():.4f}")
        print(f"    - Median: {sample_stationarity.median().item():.4f}")
        
        # Describe dataset characteristics based on stationarity.
        if dataset_stationarity > 0.7:
            characteristic = "Highly Stationary (stable patterns)"
        elif dataset_stationarity > 0.5:
            characteristic = "Moderately Stationary"
        elif dataset_stationarity > 0.3:
            characteristic = "Moderately Non-stationary"
        else:
            characteristic = "Highly Non-stationary (volatile patterns)"
        print(f"  Dataset characteristic: {characteristic}")
        
        # Print the adaptive parameters that will be used.
        sigma = self.sigma_min + (1.0 - dataset_stationarity) * (self.sigma_max - self.sigma_min)
        lambda_val = self.lambda_min + dataset_stationarity * (self.lambda_max - self.lambda_min)

        print(f"  Adaptive parameters:")
        print(f"    - σ (temperature): {sigma:.4f} (range: [{self.sigma_min}, {self.sigma_max}])")
        print(f"    - λ (MMR balance): {lambda_val:.4f} (range: [{self.lambda_min}, {self.lambda_max}])")
        print(f"="*60 + "\n")
        
        return dataset_stationarity
    
    def compute_adaptive_sigma(self, stationarity):
        """
        Compute adaptive sigma from stationarity.
        σ(s) = σ_min + (1-s) * (σ_max - σ_min)
        - Stationary (s -> 1): sigma -> sigma_min, sharper weights and stronger trust in the most similar samples.
        - Non-stationary (s -> 0): sigma -> sigma_max, smoother weights and more risk spreading.
        
        """

        return self.sigma_min + (1.0 - stationarity) * (self.sigma_max - self.sigma_min)
    
    def compute_adaptive_lambda(self, stationarity):
        """
        Compute the MMR lambda from stationarity.
        λ(s) = λ_min + s * (λ_max - λ_min)
        - Stationary (s -> 1): lambda -> lambda_max, closer to standard TopK.
        - Non-stationary (s -> 0): lambda -> lambda_min, more diverse.
        """

        return self.lambda_min + stationarity * (self.lambda_max - self.lambda_min)
    
    def mmr_select(self, sim, k, lambda_val, candidate_pool_size=None, stationarity=None):
        """
        Stochastic MMR (stochastic MMR) selection
        Sample with softmax probabilities at each step instead of deterministic argmax.
        
        MMR score: λ·sim(q,x) - (1-λ)·max_{y∈S} sim(x,y)
        Sampling probability: p(x) ∝ exp(MMR_score(x) / τ)
        
        Args:
            sim: [N_train] similarity between the query and all training samples
            k: number of selected samples
            lambda_val: parameter that balances similarity and diversity
            candidate_pool_size: candidate pool size; TopM is selected first
            stationarity: stationarity score in [0, 1], used to control sampling temperature
        Returns:
            selected_idx: [k] selected indices
        """
        device = sim.device
        n_train = sim.shape[0]
        
        if candidate_pool_size is None:
            candidate_pool_size = self.mmr_candidate_pool
        
        # Compute the temperature parameter from stationarity.
        # Stationary -> lower temperature, concentrating on high-scoring samples.
        # Non-stationary -> higher temperature, more spread and exploration.
        if stationarity is not None:
            # temperature = 0.01 + (1.0 - stationarity) * 0.49  # [0.01, 0.5]
            temperature = 0.1
        else:
            temperature = 0.1  # default temperature
        
        # Step 1: Select TopM as the candidate pool first.
        candidate_pool_size = min(candidate_pool_size, n_train)
        topM = torch.topk(sim, candidate_pool_size, dim=0)
        candidate_idx = topM.indices  # [M]
        candidate_sim = topM.values   # [M] similarity between candidate samples and the query
        
        # Precompute similarity between candidate samples using closeness of similarity values as a proxy.
        candidate_sim_matrix = 1.0 - torch.abs(
            candidate_sim.unsqueeze(0) - candidate_sim.unsqueeze(1)
        )  # [M, M]
        
        # Step 2: Stochastic MMR selection with probabilistic sampling at each step.
        selected = []
        selected_mask = torch.zeros(candidate_pool_size, dtype=torch.bool, device=device)
        
        for step in range(min(k, candidate_pool_size)):
            if step == 0:
                # First step: always choose the most similar sample deterministically.
                best_idx = 0  # The first candidate in the pool has the highest similarity.
            else:
                # Subsequent steps: compute full MMR scores and sample probabilistically.
                relevance = candidate_sim  # [M]
                
                # diversity: maximum similarity to the selected set
                selected_indices = torch.tensor(selected, device=device)
                max_sim_to_selected = candidate_sim_matrix[:, selected_indices].max(dim=1).values  # [M]
                
                # MMR score = λ·similarity - (1-λ)·redundancy
                mmr_scores = lambda_val * relevance - (1 - lambda_val) * max_sim_to_selected
                
                # Set selected entries to -inf so they are not considered again.
                mmr_scores[selected_mask] = float('-inf')
                
                # Key point: use softmax plus sampling instead of argmax.
                # p(x) ∝ exp(MMR_score(x) / τ)
                probs = F.softmax(mmr_scores / temperature, dim=0)
                
                # Sample by probability instead of choosing the maximum.
                best_idx = torch.multinomial(probs, 1).item()
            
            selected.append(best_idx)
            selected_mask[best_idx] = True
        
        # Convert back to original indices.
        selected_idx = candidate_idx[torch.tensor(selected, device=device)]
        
        return selected_idx

    def decompose_mg(self, data_all, remove_offset=True): #decompose time-series data into components with different period lengths
        data_all = copy.deepcopy(data_all) # T, S, C  #T: time steps  S: sequence length  C: number of channels / feature dimension

        mg = []
        for g in self.period_num:
            cur = data_all.unfold(dimension=1, size=g, step=g).mean(dim=-1) # apply a sliding window of size g and step g along dimension 1, the sequence dimension
            cur = cur.repeat_interleave(repeats=g, dim=1) # repeat each average value g times
            mg.append(cur)
#             data_all = data_all - cur
            
        mg = torch.stack(mg, dim=0) # G, T, S, C # stack decomposition results from all granularities

        if remove_offset:
            offset = []
            for i, data_p in enumerate(mg):
                # Original code: subtract the final time-step feature as the offset
                cur_offset = data_p[:,-1:,:]
                mg[i] = data_p - cur_offset # subtract the final time-step feature from every time step to obtain relative offsets
                
                offset.append(cur_offset)
            offset = torch.stack(offset, dim=0)
        else:
            offset = None
            
        return mg, offset
    
    def compute_time_aware_bonus(self, query_mark, train_mark, device):
        """
        Compute the time-aware bonus, prioritizing segments from the same time of day or year.
        
        query_mark: [B, seq_len, mark_features] time markers of the query sequence
        train_mark: [N_train, seq_len, mark_features] time markers of the training set
        
        Time marker format: [year, month, day, weekday, hour] or [year, month, day, weekday, hour, minute]
        
        Return: [B, N_train] time-aware similarity bonus matrix
        """
        B = query_mark.shape[0]
        N_train = train_mark.shape[0]
        
        # Ensure train_mark is on the correct device.
        train_mark = train_mark.to(device)
        
        # Extract query-sequence time features using the middle point for better stability.
        mid_idx = query_mark.shape[1] // 2
        query_time = query_mark[:, mid_idx, :]  # [B, mark_features]
        train_time = train_mark[:, mid_idx, :]  # [N_train, mark_features]
        
        # Initialize the bonus matrix.
        time_bonus = torch.zeros(B, N_train, device=device)
        
        mark_features = query_time.shape[1]
        
        # Choose matching priorities based on temporal granularity.
        # For high-frequency data such as hourly or minute-level data, the same time of day is more important.
        # For low-frequency data such as daily or weekly data, the same time of year, i.e. seasonality, is more important.
        
        if mark_features >= 5:  # at least year, month, day, weekday, and hour are available
            # Extract hour information (index 4)
            query_hour = query_time[:, 4:5]  # [B, 1]
            train_hour = train_time[:, 4]    # [N_train]
            
            # Compute hour-matching bonus for the same time of day.
            # Use circular distance, e.g. 23:00 and 00:00 should be close.
            hour_diff = torch.abs(query_hour - train_hour.unsqueeze(0))  # [B, N_train]
            hour_diff = torch.min(hour_diff, 24 - hour_diff)  # circular distance
            
            # Use seq_len and steps_per_day to determine the importance of daily periodicity.
            # If seq_len < steps_per_day, the same time of day is more important.
            if self.seq_len < self.steps_per_day:
                # For high-frequency data, prioritize the same time of day.
                # Allow a +/-2 hour tolerance.
                hour_bonus = torch.exp(-hour_diff / 2.0)  # exponential decay
                time_bonus += 0.6 * hour_bonus
            else:
                # For lower-frequency data, hour matching is less important.
                hour_bonus = torch.exp(-hour_diff / 4.0)
                time_bonus += 0.3 * hour_bonus
            
            # Extract weekday information (index 3)
            query_weekday = query_time[:, 3:4]  # [B, 1]
            train_weekday = train_time[:, 3]    # [N_train]
            
            # Compute weekday-matching bonus for the same day of the week.
            weekday_match = (query_weekday == train_weekday.unsqueeze(0)).float()  # [B, N_train]
            
            # Weekday/weekend matching bonus.
            query_is_weekend = (query_weekday >= 5).float()
            train_is_weekend = (train_weekday >= 5).float()
            weekend_match = (query_is_weekend == train_is_weekend.unsqueeze(0)).float()
            
            time_bonus += 0.2 * weekday_match + 0.1 * weekend_match
            
            # Extract month information (index 1) - seasonality
            query_month = query_time[:, 1:2]  # [B, 1]
            train_month = train_time[:, 1]    # [N_train]
            
            # Compute month/season matching bonus for the same time of year.
            month_diff = torch.abs(query_month - train_month.unsqueeze(0))
            month_diff = torch.min(month_diff, 12 - month_diff)  # circular distance
            
            # If seq_len >= steps_per_day, seasonality is more important.
            if self.seq_len >= self.steps_per_day:
                # The same time of year, i.e. seasonality, is more important.
                month_bonus = torch.exp(-month_diff / 2.0)  # +/-2 month tolerance
                time_bonus += 0.4 * month_bonus
            else:
                # For high-frequency data, seasonality is less important.
                month_bonus = torch.exp(-month_diff / 3.0)
                time_bonus += 0.2 * month_bonus
        
        # If minute information is available (index 5)
        if mark_features >= 6:
            query_minute = query_time[:, 5:6]  # [B, 1]
            train_minute = train_time[:, 5]    # [N_train]
            
            # Minute-level matching, which is more important for high-frequency data.
            if self.seq_len < self.steps_per_day // 4:  # sequence shorter than 6 hours
                minute_diff = torch.abs(query_minute - train_minute.unsqueeze(0))
                minute_diff = torch.min(minute_diff, 60 - minute_diff)
                minute_bonus = torch.exp(-minute_diff / 15.0)  # +/-15 minute tolerance
                time_bonus += 0.1 * minute_bonus
        
        # Normalize to the [0, 1] range.
        time_bonus = time_bonus / time_bonus.max().clamp(min=1e-8)
        
        return time_bonus
    
    def periodic_batch_corr(self, data_all, key, in_bsz = 512):
        _, bsz, features = key.shape
        _, train_len, _ = data_all.shape
        
        bx = key - torch.mean(key, dim=2, keepdim=True)
        
        iters = math.ceil(train_len / in_bsz) # Batch processing avoids memory overflow.
        
        sim = []
        for i in range(iters):
            start_idx = i * in_bsz
            end_idx = min((i + 1) * in_bsz, train_len)
            
            cur_data = data_all[:, start_idx:end_idx].to(key.device)
            ax = cur_data - torch.mean(cur_data, dim=2, keepdim=True) # center
            
            cur_sim = torch.bmm(F.normalize(bx, dim=2), F.normalize(ax, dim=2).transpose(-1, -2)) #normalize 
            sim.append(cur_sim)
            
        sim = torch.cat(sim, dim=2) # Concatenate similarity results from all batches. # G, B, T
        
        return sim    
            

    def masked_sim(self, sim, index, bsz, x):
        #Prevent leakage during training by avoiding retrieval of time spans overlapping the current sample. #
        sliding_index = torch.arange(2 * (self.seq_len + self.pred_len) - 1).to(x.device)
        sliding_index = sliding_index.unsqueeze(dim=0).repeat(len(index), 1)
        sliding_index = sliding_index + (index - self.seq_len - self.pred_len + 1).unsqueeze(dim=1)
        
        sliding_index = torch.where(sliding_index >= 0, sliding_index, 0)
        sliding_index = torch.where(sliding_index < self.n_train, sliding_index, self.n_train - 1)

        self_mask = torch.zeros((bsz, self.n_train)).to(x.device)
        self_mask = self_mask.scatter_(1, sliding_index, 1.)
        self_mask = self_mask.unsqueeze(dim=0).repeat(self.n_period, 1, 1)
        
        sim = sim.masked_fill_(self_mask.bool(), float('-inf')) # G, B, T

        return sim

    def generate_retrieval_predictions(self, sim, bsz):
        # Top-K selection mechanism #
        sim = sim.reshape(self.n_period * bsz, self.n_train) # G X B, T
                
        topm_index = torch.topk(sim, self.topm, dim=1).indices
        ranking_sim = torch.ones_like(sim) * float('-inf')
        
        rows = torch.arange(sim.size(0)).unsqueeze(-1).to(sim.device)
        ranking_sim[rows, topm_index] = sim[rows, topm_index]
        
        sim = sim.reshape(self.n_period, bsz, self.n_train) # G, B, T
        ranking_sim = ranking_sim.reshape(self.n_period, bsz, self.n_train) # G, B, T

        data_len, seq_len, channels = self.train_data_all.shape
        
        # Weighted prediction generation # 
        ranking_prob = F.softmax(ranking_sim / self.temperature, dim=2)
        ranking_prob = ranking_prob.detach().cpu() # G, B, T
        
        y_data_all = self.y_data_all_mg.flatten(start_dim=2) # G, T, P * C
        
        pred_from_retrieval = torch.bmm(ranking_prob, y_data_all).reshape(self.n_period, bsz, -1, channels) #weighted average of labels from all training samples G B P C
        
        return pred_from_retrieval

    def generate_retrieval_predictions_2(self, sim, bsz, index):
        """
        Adaptive retrieval prediction generation using dataset-level stationarity parameters.
        
        Core idea：
        - Higher stationarity s -> smaller sigma, sharper weights, and stronger trust in the most similar samples.
        - Lower stationarity s -> larger sigma, smoother weights, and more risk spreading.
        - Use MMR selection consistently, with lambda varying continuously with stationarity.
        
        Args:
            sim: [G, B, T] similarity matrix
            bsz: batch size
            index: sample index
        """
        eps = 1e-12
        device = sim.device
        
        # ===== Adaptive retrieval logic with continuous changes and no hard threshold. =====
        if self.use_adaptive and hasattr(self, 'dataset_stationarity'):
            # Use precomputed dataset-level adaptive scalar parameters.
            sigma = self.adaptive_sigma      # stationary -> small sigma; non-stationary -> large sigma
            lambda_val = self.adaptive_lambda  # stationary -> large lambda, prioritizing similarity; non-stationary -> small lambda, increasing diversity

            # sigma = 0.1
            # lambda_val = 0.5
            
            sim_flat = sim.reshape(self.n_period * bsz, self.n_train)  # [G*B, T]
            
            # Use MMR selection consistently and control diversity continuously through lambda.
            # When lambda is close to 1, MMR degenerates to TopK and considers only similarity.
            # When lambda is close to 0, MMR emphasizes diversity.
            all_masks = []
            for i in range(sim_flat.size(0)):
                sample_sim = sim_flat[i]  # [T]
                # Pass stationarity to control selection of the first sample.
                topk_idx = self.mmr_select(sample_sim, self.topm, lambda_val, 
                                          stationarity=self.dataset_stationarity)
                
                mask_i = torch.zeros(self.n_train, dtype=torch.bool, device=device)
                mask_i[topk_idx] = True
                all_masks.append(mask_i)
            
            mask = torch.stack(all_masks, dim=0)  # [G*B, T]
            mask = mask.reshape(self.n_period, bsz, self.n_train)
            
            # Compute distance.
            dist = 1.0 - sim  # [G, B, T]
            dist_masked = dist.masked_fill(~mask, float('inf'))
            
            # Use adaptive sigma to compute Gaussian-kernel weights.
            # Small sigma -> weights concentrate on the most similar samples.
            # Large sigma -> weights spread over more candidates.
            gauss = torch.exp(-(dist_masked ** 2) / (2.0 * (sigma ** 2)))
            
        else:
            # ===== Original fixed-parameter logic =====
            # sigma = getattr(self, "kernel_sigma", None) or float(self.temperature)
            sigma = self.adaptive_sigma
            
            sim_flat = sim.reshape(self.n_period * bsz, self.n_train)  # [G*B, T]
            topk = torch.topk(sim_flat, self.topm, dim=1)
            topk_idx = topk.indices
            
            mask = torch.zeros_like(sim_flat, dtype=torch.bool)
            rows = torch.arange(sim_flat.size(0), device=device).unsqueeze(-1)
            mask[rows, topk_idx] = True
            mask = mask.reshape(self.n_period, bsz, self.n_train)
            
            dist = 1.0 - sim
            dist_masked = dist.masked_fill(~mask, float('inf'))
            
            gauss = torch.exp(-(dist_masked ** 2) / (2.0 * (sigma ** 2)))
        
        # ===== Normalization and aggregation shared logic =====
        den = gauss.sum(dim=2, keepdim=True).clamp_min(eps)
        ranking_prob = (gauss / den).detach().cpu()  # [G, B, T]
        
        # Aggregate future segments.
        data_len, seq_len, channels = self.train_data_all.shape
        y_data_all = self.y_data_all_mg.flatten(start_dim=2)  # [G, T, P*C]
        
        pred_from_retrieval = torch.bmm(ranking_prob, y_data_all)  # [G, B, P*C]
        pred_from_retrieval = pred_from_retrieval.reshape(self.n_period, bsz, -1, channels)
        
        return pred_from_retrieval

    def retrieve(self, x, index, train=True, x_mark=None):
        index = index.to(x.device)
        
        bsz, seq_len, channels = x.shape
        assert seq_len == self.seq_len and channels == self.channels
        
        # Note: per-query stationarity is no longer computed; dataset-level stationarity is used.
        
        x_mg, mg_offset = self.decompose_mg(x) # G, B, S, C

        # Compute temporal similarity.
        temporal_sim = self.periodic_batch_corr(
            self.train_data_all_mg.flatten(start_dim=2), # G, T, S * C
            x_mg.flatten(start_dim=2), # G, B, S * C
        ) # G, B, T
        
        # Add the time-aware bonus.
        if x_mark is not None and hasattr(self, 'train_data_all_mark'):
            print(f"\n[Time-Aware] weight={self.time_aware_weight:.4f}")
            if self.time_aware_weight > 0:
                print(f"[Time-Aware] Computing time bonus...")
                time_bonus = self.compute_time_aware_bonus(x_mark, self.train_data_all_mark, x.device)
                print(f"[Time-Aware] time_bonus stats: min={time_bonus.min():.4f}, max={time_bonus.max():.4f}, mean={time_bonus.mean():.4f}")
                time_bonus = time_bonus.unsqueeze(0).repeat(self.n_period, 1, 1)
                
                sim_before = temporal_sim.clone()
                temporal_sim = (1 - self.time_aware_weight) * temporal_sim + self.time_aware_weight * time_bonus
                
                # Check whether the similarity actually changed.
                sim_diff = (temporal_sim - sim_before).abs().mean()
                print(f"[Time-Aware] Applied! sim_before: mean={sim_before.mean():.4f}, sim_after: mean={temporal_sim.mean():.4f}, diff={sim_diff:.6f}")
            else:
                print(f"[Time-Aware] SKIPPED (weight=0)")
            
        if train:
            temporal_sim = self.masked_sim(temporal_sim, index, bsz, x)
        
        # ===== Use adaptive retrieval based on dataset-level stationarity. =====
        pred_from_temporal_sim = self.generate_retrieval_predictions_2(
            temporal_sim, bsz, index
        )
        pred_from_temporal_sim = pred_from_temporal_sim.to(x.device)
        
        return pred_from_temporal_sim
    
    def retrieve_all(self, data, train=False, device=torch.device('cpu')):
        assert(self.train_data_all_mg != None)

        # Windows + CUDA can hit WinError 1455 when worker processes import torch DLLs.
        retrieval_workers = 0 if os.name == 'nt' else 8
        
        rt_loader = DataLoader(
            data,
            batch_size=1024,
            shuffle=False,
            num_workers=retrieval_workers,
            drop_last=False
        )
        
        preds_from_temporal_sims = []
        with torch.no_grad():
            for index, batch_x, batch_y, batch_x_mark, batch_y_mark in tqdm(rt_loader):
                pred_from_temporal_sim = self.retrieve(batch_x.float().to(device), index, train=train, x_mark=batch_x_mark.float().to(device))
                pred_from_temporal_sim = pred_from_temporal_sim.cpu()
                preds_from_temporal_sims.append(pred_from_temporal_sim)
                
        preds_from_temporal_sims = torch.cat(preds_from_temporal_sims, dim=1)
        
        return preds_from_temporal_sims