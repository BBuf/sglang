"""Use PyTorch's deterministic cuDNN convolution selection in every server process."""
import torch

torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True
