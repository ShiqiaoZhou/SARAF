import torch
import torch.nn as nn
import torch.nn.functional as F

from layers.Retrieval import RetrievalTool

def print_gpu_memory_by_module(model, device_id=0):
    """
    Print detailed GPU memory allocation by module.
    """
    print("\n" + "="*80)
    print(f"GPU Memory Allocation by Module (Device: cuda:{device_id})")
    print("="*80)
    
    total_memory = 0
    module_memory = {}
    
    # Count parameter memory for each module.
    for name, module in model.named_modules():
        module_params = sum(p.numel() * p.element_size() for p in module.parameters())
        if module_params > 0:
            module_memory[name] = module_params
            total_memory += module_params
    
    # Sort by memory size.
    sorted_modules = sorted(module_memory.items(), key=lambda x: x[1], reverse=True)
    
    # Print the top 20 largest modules.
    print(f"\n{'Module Name':<60} {'Memory (MB)':<15} {'Percentage':<10}")
    print("-" * 85)
    for name, size_bytes in sorted_modules[:20]:
        size_mb = size_bytes / (1024 ** 2)
        percentage = (size_bytes / total_memory * 100) if total_memory > 0 else 0
        print(f"{name:<60} {size_mb:>13.2f} {percentage:>8.2f}%")
    
    total_memory_mb = total_memory / (1024 ** 2)
    print("-" * 85)
    print(f"{'Total Parameters Memory':<60} {total_memory_mb:>13.2f} {100.0:>8.2f}%")
    
    # Print actual GPU memory usage.
    if torch.cuda.is_available():
        torch.cuda.synchronize(device_id)
        allocated = torch.cuda.memory_allocated(device_id) / (1024 ** 3)  # GB
        reserved = torch.cuda.memory_reserved(device_id) / (1024 ** 3)    # GB
        total = torch.cuda.get_device_properties(device_id).total_memory / (1024 ** 3)  # GB
        
        print(f"\nActual GPU Memory Status:")
        print(f"  Allocated: {allocated:.2f} GB")
        print(f"  Reserved:  {reserved:.2f} GB")
        print(f"  Total:     {total:.2f} GB")
        print(f"  Free:      {total - allocated:.2f} GB")
    
    print("="*80 + "\n")

def print_gpu_memory_summary(device_id=0):
    """
    Print a GPU memory usage summary.
    """
    if not torch.cuda.is_available():
        print("CUDA not available")
        return
    
    torch.cuda.synchronize(device_id)
    
    allocated = torch.cuda.memory_allocated(device_id) / (1024 ** 3)
    reserved = torch.cuda.memory_reserved(device_id) / (1024 ** 3)
    total = torch.cuda.get_device_properties(device_id).total_memory / (1024 ** 3)
    
    print(f"\n{'GPU Memory Summary (Device: cuda:{device_id})':-^80}")
    print(f"Allocated: {allocated:>8.2f} GB | Reserved: {reserved:>8.2f} GB | Total: {total:>8.2f} GB | Free: {total - allocated:>8.2f} GB")
    print("-" * 80)

class Model(nn.Module):
    """
    Paper link: https://arxiv.org/pdf/2205.13504.pdf
    """

    def __init__(self, configs, individual=False):
        """
        individual: Bool, whether shared model among different variates.
        """
        super(Model, self).__init__()
        use_cuda = bool(getattr(configs, 'use_gpu', False)) and torch.cuda.is_available()
        self.device = torch.device(f'cuda:{configs.gpu}' if use_cuda else 'cpu')
        self.task_name = configs.task_name
        self.seq_len = configs.seq_len
        if self.task_name == 'classification' or self.task_name == 'anomaly_detection' or self.task_name == 'imputation':
            self.pred_len = configs.seq_len
        else:
            self.pred_len = configs.pred_len

        self.channels = 1 if configs.features == 'S' else configs.enc_in

        self.linear_x = nn.Linear(self.seq_len, self.pred_len)
        
        self.n_period = configs.n_period
        self.topm = configs.topm
        
        # Get the frequency parameter for time-aware retrieval.
        self.freq = getattr(configs, 'freq', 'h')
        self.time_aware_weight = getattr(configs, 'time_aware_weight', 0.3)
        
        self.rt = RetrievalTool(
            seq_len=self.seq_len,
            pred_len=self.pred_len,
            channels=self.channels,
            n_period=self.n_period,
            topm=self.topm,
            freq=self.freq,
            time_aware_weight=self.time_aware_weight,
        )
        
        self.period_num = self.rt.period_num[-1 * self.n_period:]
        
        # Fusion method selection.
        self.fusion_method = getattr(configs, 'fusion_method', 'weighted')  # Use gated attention by default.
        
        module_list = [
            nn.Linear(self.pred_len // g, self.pred_len)
            for g in self.period_num
        ]
        self.retrieval_pred = nn.ModuleList(module_list)

        self.norm = nn.LayerNorm(self.channels)
        # self.linear_pred = nn.Linear(2 * self.pred_len, self.pred_len) # Keep the original method as a fallback.
        self.linear_pred_2 = nn.Linear(self.pred_len, self.pred_len) 
        
        # Weight for retrieval prediction fusion.
        self.retrieval_fusion_weight = nn.Parameter(torch.tensor(0.5))  # Learnable weight initialized to 0.5.
        self.retrieval_fusion_gate = nn.Sequential(
            nn.Linear(2 * self.channels, self.channels),
            nn.Sigmoid()
        ) 

    def prepare_dataset(self, train_data, valid_data, test_data):
        self.rt.prepare_dataset(train_data)
        
        self.retrieval_temporal_dict = {}
        
        print('Doing Train Retrieval')
        train_temporal_rt = self.rt.retrieve_all(train_data, train=True, device=self.device)

        print('Doing Valid Retrieval')
        valid_temporal_rt = self.rt.retrieve_all(valid_data, train=False, device=self.device)

        print('Doing Test Retrieval')
        test_temporal_rt = self.rt.retrieve_all(test_data, train=False, device=self.device)

        del self.rt
        torch.cuda.empty_cache()

        self.retrieval_temporal_dict['train'] = train_temporal_rt.detach().to(self.device, non_blocking=True)
        self.retrieval_temporal_dict['valid'] = valid_temporal_rt.detach().to(self.device, non_blocking=True)
        self.retrieval_temporal_dict['test']  = test_temporal_rt.detach().to(self.device, non_blocking=True)


    def encoder(self, x, batch_x_mark, index, mode):
        index = index.to(self.device)
        
        bsz, seq_len, channels = x.shape
        assert(seq_len == self.seq_len, channels == self.channels)
        
        # use the last value as the offset.
        x_offset = x[:, -1:, :].detach()
        x_norm = x - x_offset

        x_pred_from_x = self.linear_x(x_norm.permute(0, 2, 1)).permute(0, 2, 1) # B, P, C #torch.Size([32, 96, 7])
        
        # ⚠️ Optimization: data is already on the GPU, so index directly without another to(device).
        # Use detach() so these precomputed retrieval results are not included in the computation graph.
        # Use only temporal-similarity retrieval results; semantic retrieval has been removed.
        preds_from_temporal_sims = self.retrieval_temporal_dict[mode][:, index].detach() # G, B, P, C

        # Single-period optimization: process directly to avoid loops and stack operations.
        pr = preds_from_temporal_sims[0]  # Take the first and only element directly.
        assert((bsz, self.pred_len, channels) == pr.shape)
        preds_from_temporal_sims = self.retrieval_pred[0](pr.permute(0, 2, 1)).permute(0, 2, 1)

        
        # Use the simplified fusion method with only the base prediction and temporal retrieval prediction.
        pred = self.simple_fusion(x_pred_from_x, preds_from_temporal_sims)
        pred = pred.reshape(bsz, self.pred_len, self.channels)
        
        pred = pred + x_offset
        
        return pred
    
    def simple_fusion(self, x_pred_from_x, preds_from_temporal_sims):
        """
        Simplified fusion method: fuse only the base prediction and temporal retrieval prediction.
        """
        return self.weighted_fusion_simple(x_pred_from_x, preds_from_temporal_sims)
    
    
    def weighted_fusion_simple(self, x_pred_from_x, preds_from_temporal_sims):
        """Weighted fusion: simplified version with only two inputs."""
        alpha = 0.5
        pred = alpha * x_pred_from_x + (1 - alpha) * preds_from_temporal_sims
        pred = self.linear_pred_2(pred.permute(0, 2, 1)).permute(0, 2, 1)

        return pred
    
    def forecast(self, x_enc, batch_x_mark, index, mode):
        # Encoder
        return self.encoder(x_enc, batch_x_mark, index, mode)

    def imputation(self, x_enc, index, mode):
        # Encoder
        return self.encoder(x_enc, index, mode)

    def anomaly_detection(self, x_enc, index, mode):
        # Encoder
        return self.encoder(x_enc, index, mode)

    def classification(self, x_enc, index, mode):
        # Encoder
        enc_out = self.encoder(x_enc, index, mode)
        # Output
        # (batch_size, seq_length * d_model)
        output = enc_out.reshape(enc_out.shape[0], -1)
        # (batch_size, num_classes)
        output = self.projection(output)
        return output

    def forward(self, x_enc, batch_x_mark, index, mode='train'):
        if self.task_name == 'long_term_forecast' or self.task_name == 'short_term_forecast':
            dec_out = self.forecast(x_enc, batch_x_mark, index, mode)
            return dec_out[:, -self.pred_len:, :]  # [B, L, D]
        if self.task_name == 'imputation':
            dec_out = self.imputation(x_enc, index, mode)
            return dec_out  # [B, L, D]
        if self.task_name == 'anomaly_detection':
            dec_out = self.anomaly_detection(x_enc, index, mode)
            return dec_out  # [B, L, D]
        if self.task_name == 'classification':
            dec_out = self.classification(x_enc, index, mode)
            return dec_out  # [B, N]
        return None
