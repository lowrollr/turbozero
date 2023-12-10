from typing import Optional, Tuple

import jax
import optax
import yaml
from core.envs.make import make_env
from core.evaluators.make import make_evaluator
from core.memory.make import make_replay_buffer
from core.networks.make import make_model
from flax.training import orbax_utils
import orbax.checkpoint

from core.training.train import BNTrainState, Trainer, TrainerConfig, TrainerState
from core.training.train_2p import TwoPlayerTrainer


def init_from_config(
    config: dict,
    debug=False
) -> Tuple[Trainer, TrainerState]:
    env_config = config['env_config']
    env = make_env(
        env_pkg=env_config['pkg'],
        env_name=env_config['name'],
        config=env_config['config']
    )

    evaluator_config = config['evaluator_config']
    evaluator_type = evaluator_config['type']
    evaluator_config_train = evaluator_config['train']
    evaluator_config_test = evaluator_config['test']
    evaluator_model_config = evaluator_config['model']
    model_type = evaluator_model_config['type']
    model_config = evaluator_model_config['config']

    model_config.update(
        policy_head_out_size=env.num_actions,
        value_head_out_size=1
    )

    model = make_model(
        model_type=model_type,
        config=model_config
    )

    evaluator_train = make_evaluator(
        evaluator_type=evaluator_type,
        config=evaluator_config_train, 
        env=env,
        model=model
    )

    evaluator_test = make_evaluator(
        evaluator_type=evaluator_type,
        config=evaluator_config_test,
        env=env,
        model=model
    )

    train_config = config['training_config']
    batch_size = train_config['selfplay_batch_size']

    buff_config = train_config['replay_buffer_config']
    buff = make_replay_buffer(
        buff_type=buff_config['type'],
        batch_size=batch_size,
        config=buff_config['config']
    )

    trainer_config = TrainerConfig(
        warmup_steps=train_config['warmup_steps'],
        collection_steps_per_epoch=train_config['collection_steps_per_epoch'],
        train_steps_per_epoch=train_config['train_steps_per_epoch'],
        test_every_n_epochs=train_config['test_every_n_epochs'],
        test_episodes=train_config['test_episodes'],
        checkpoint_every_n_epochs=train_config['checkpoint_every_n_epochs'],
        checkpoint_dir=train_config['checkpoint_dir'],
        retain_n_checkpoints=train_config['retain_n_checkpoints'],
        learning_rate=train_config['learning_rate'],
        momentum=train_config['momentum'],
        l2_lambda=train_config['l2_lambda'],
        policy_factor=train_config['policy_factor'],
        disk_store_location=train_config['disk_store_location'],
        selfplay_batch_size=batch_size,
        max_episode_steps=train_config['max_episode_steps']
    )

    if env.num_players() == 1:
        trainer = Trainer(
            trainer_config,
            env,
            evaluator_train,
            evaluator_test,
            buff,
            model,
            debug=debug
        )
    else:
        trainer = TwoPlayerTrainer(
            trainer_config,
            env,
            evaluator_train,
            evaluator_test,
            buff,
            model,
            debug=debug
        )

    return trainer, trainer.init(jax.random.PRNGKey(train_config['seed']))


def init_from_config_file(
    config_file: str,
    debug=False
) -> Tuple[Trainer, TrainerState]:
    with open(config_file, 'r') as f:
        config = yaml.safe_load(f)
    return init_from_config(config, debug)


def init_from_checkpoint(
    checkpoint_dir: str,
    checkpoint_num: Optional[int] = None,
    debug=False
) -> Tuple[Trainer, TrainerState]:
    checkpointer = orbax.checkpoint.PyTreeCheckpointer()
    checkpoint_mngr = orbax.checkpoint.CheckpointManager(
        checkpoint_dir,
        checkpointer
    )
    if checkpoint_num is None:
        checkpoint_num = checkpoint_mngr.latest_step()

    # load once to get configs
    ckpt = checkpoint_mngr.restore(checkpoint_num)

    env = make_env(ckpt['env_config'])
    model = make_model(ckpt['model_config'])
    evaluator_train = make_evaluator(ckpt['eval_config_train'], env, model=model)
    evaluator_test = make_evaluator(ckpt['eval_config_test'], env, model=model)
    buff = make_replay_buffer(ckpt['buff_config'])

    if env.num_players() == 1:
        trainer = Trainer(
            TrainerConfig(**ckpt['train_config']),
            env,
            evaluator_train,
            evaluator_test,
            buff,
            model,
            debug=debug
        )
    else:
        trainer = TwoPlayerTrainer(
            TrainerConfig(**ckpt['train_config']),
            env,
            evaluator_train,
            evaluator_test,
            buff,
            model,
            debug=debug
        )

    # build target
    target = {
        'state': BNTrainState.create(
                apply_fn=model.apply,
                params=ckpt['state']['params'],
                batch_stats=ckpt['state'].get('batch_stats'),
                tx=optax.sgd(learning_rate=trainer.config.learning_rate, momentum=trainer.config.momentum)
        ),
        'train_config': None,
        'env_config': None,
        'eval_config_train': None,
        'eval_config_test': None,
        'buff_config': None,
        'model_config': None
    }

    # load again now that we have proper TrainState target (why why why why)
    ckpt = checkpoint_mngr.restore(checkpoint_num, items=target)

    trainer_state = trainer.init(jax.random.PRNGKey(trainer.config.seed))
    
    trainer_state = trainer_state.replace(
        train_state=ckpt['state']
    )

    return trainer, trainer_state
