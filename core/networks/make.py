




from typing import Any


def make_model(model_type: str, config: dict) -> Any:
   
    if model_type == 'az_resnet':
        from core.networks.azresnet import AZResnet, AZResnetConfig
        config = AZResnetConfig(model_type=model_type, **config)
        return AZResnet(config)
    else:
        raise NotImplementedError(f'Unknown model type {model_type}')