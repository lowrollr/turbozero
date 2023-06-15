import torch
GLOB_FLOAT_TYPE = torch.float16 if torch.cuda.is_available() else torch.float32