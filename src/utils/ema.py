from lightning.pytorch.callbacks import StochasticWeightAveraging 
from torch.optim.swa_utils import get_ema_avg_fn

# Enable Exponential Moving Average after 100 steps
class EMAWeightAveraging(StochasticWeightAveraging):
    def __init__(self, swa_lrs):
        super().__init__(swa_lrs=swa_lrs, avg_fn=get_ema_avg_fn())
        
    def should_update(self, step_idx=None, epoch_idx=None):
        return (step_idx is not None) and (step_idx >= 100)
    
