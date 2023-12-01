




from typing import Any


def make_model(model_config: dict) -> Any:
    model_type = model_config.get('model_type')
    if model_type == 'az_resnet':
        from core.networks.azresnet import AZResnet, AZResnetConfig
        config = AZResnetConfig(**model_config)
        return AZResnet(config)
    else:
        raise NotImplementedError(f'Unknown model type {model_type}')