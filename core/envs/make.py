
from core.envs.env import Env, EnvConfig

def make_env(env_pkg: str, env_name: str, config: dict) -> Env:
    config = EnvConfig(env_pkg=env_pkg, env_name=env_name, base_config=config)
    if config.env_pkg == 'jumanji':
        from core.envs.jumanji import make_jumanji_env
        return make_jumanji_env(config.env_name, **config.base_config)
    elif config.env_pkg == 'pgx':
        from core.envs.pgx import make_pgx_env
        return make_pgx_env(config.env_name, **config.base_config)
    elif config.env_pkg == 'custom':
        # TODO: handle custom envs
        pass
    else:
        raise NotImplementedError(f'Unknown env pkg {config.env_pkg}')