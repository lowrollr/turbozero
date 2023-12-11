



from core.memory.replay_memory import EndRewardReplayBuffer


def make_replay_buffer(
    buff_type: str,
    batch_size: int,
    reward_size: int,
    config: dict
) -> EndRewardReplayBuffer:
    if buff_type == 'end_reward':
        from core.memory.replay_memory import EndRewardReplayBuffer, EndRewardReplayBufferConfig
        config = EndRewardReplayBufferConfig(buff_type=buff_type, batch_size=batch_size, reward_size=reward_size, **config)
        return EndRewardReplayBuffer(config)
    elif buff_type == 'ranked_reward':
        from core.memory.ranked_reward_replay_memory import RankedRewardReplayBuffer, RankedRewardReplayBufferConfig
        config = RankedRewardReplayBufferConfig(buff_type=buff_type, batch_size=batch_size, reward_size=reward_size, **config)
        return RankedRewardReplayBuffer(config)
    else:
        raise NotImplementedError(f'Unknown buff type {buff_type}')