



from core.memory.replay_memory import EndRewardReplayBuffer


def make_replay_buffer(
    buff_config: dict
) -> EndRewardReplayBuffer:
    buff_type = buff_config.get('buff_type')
    if buff_type == 'end_reward':
        from core.memory.replay_memory import EndRewardReplayBuffer, EndRewardReplayBufferConfig
        config = EndRewardReplayBufferConfig(**buff_config)
        return EndRewardReplayBuffer(config)
    elif buff_type == 'ranked_reward':
        from core.memory.ranked_reward_replay_memory import RankedRewardReplayBuffer, RankedRewardReplayBufferConfig
        config = RankedRewardReplayBufferConfig(**buff_config)
        return RankedRewardReplayBuffer(config)
    else:
        raise NotImplementedError(f'Unknown buff type {buff_type}')