
from core.env import Env
from core.resnet import TurboZeroResnet
import torch


def load_checkpoint(path: str):
    return torch.load(path, map_location=torch.device('cpu'))

def load_model_and_optimizer_from_checkpoint(checkpoint: dict, env: Env, device: torch.device):
    model = TurboZeroResnet(checkpoint['model_arch_params'], env.state_shape, env.policy_shape)
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    model.eval()
    optimizer = torch.optim.SGD(model.parameters(), 
                    lr = checkpoint['raw_train_config']['learning_rate'], 
                    momentum = checkpoint['raw_train_config']['momentum'], 
                    weight_decay = checkpoint['raw_train_config']['c_reg'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    return model, optimizer
