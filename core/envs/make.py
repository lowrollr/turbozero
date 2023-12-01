


from core.envs.env import Env, EnvConfig


def make_env(env_config: dict) -> Env:
    config = EnvConfig(**env_config)
    if config.env_type == 'jumanji':
        from core.envs.jumanji import make_jumanji_env
        return make_jumanji_env(config.env_name, **config.base_config)
    elif config.env_type == 'pgx':
        from core.envs.pgx import make_pgx_env
        return make_pgx_env(config.env_name, **config.base_config)
    else:
        raise NotImplementedError(f'Unknown env type {config.env_type}')