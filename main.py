import torch

from utils.utils import fix_random_seed

fix_random_seed(1234)
print(torch.cuda.is_available())