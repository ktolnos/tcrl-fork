# %%
import warnings

warnings.filterwarnings("ignore")
import os

os.environ['MKL_SERVICE_FORCE_INTEL'] = '1'
import random
import time
from datetime import timedelta

import hydra
import numpy as np
import torch
import wandb
from pathlib import Path
from dm_env import specs

from tcrl import TCRL
import utils
import utils.helper as helper
from utils.video import VideoRecorder
from utils.logger import Logger
import statistics
import matplotlib.pyplot as plt

torch.backends.cudnn.benchmark = True
__CONFIG__, __LOGS__ = "cfgs", "logs"


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)


last_expected_ret = 1

def log_q_bias(time_steps, agent, cfg):
    global last_expected_ret
    ret = 0
    min_biases = []
    mean_biases = []
    max_biases = []
    q_diff = 0
    expected_ret = 0
    for ts in reversed(time_steps):
        ret += ts.reward
        obs_t = torch.tensor(ts.observation, dtype=torch.float32, device=agent.device).unsqueeze(0)
        act_t = torch.tensor(ts.action, dtype=torch.float32, device=agent.device).unsqueeze(0)
        z = agent.enc(obs_t)
        q1, q2 = agent.critic(z, act_t)
        q1 = float(q1.item())
        q2 = float(q2.item())
        q_diff += abs(q1 - q2)
        min_biases.append((min(q1, q2) - ret) / last_expected_ret)
        max_biases.append((max(q1, q2) - ret) / last_expected_ret)
        mean_biases.append(((q1 + q2) / 2 - ret) / last_expected_ret)
        expected_ret += ret

        ret *= cfg.gamma
    if last_expected_ret == 1:
        last_expected_ret = expected_ret / len(time_steps)
    else:
        last_expected_ret = last_expected_ret * 0.9 + (expected_ret / len(time_steps)) * 0.1
    plt.plot(list(reversed(mean_biases)))
    return {
        'min_bias_avg': statistics.mean(min_biases),
        'min_bias_std': statistics.stdev(min_biases),
        'max_bias_avg': statistics.mean(max_biases),
        'max_bias_std': statistics.stdev(max_biases),
        'mean_bias_avg': statistics.mean(mean_biases),
        'mean_bias_std': statistics.stdev(mean_biases),
        'q_diff': q_diff / len(time_steps),
        'mean_biases_plot': plt
    }


@hydra.main(config_path='cfgs', config_name='default')
def main(cfg):
    ###### set random seed ######
    if cfg.seed == -1: cfg.seed = random.randint(0, 10000)  # generate random seed
    set_seed(cfg.seed)

    ###### change some config values ######
    cfg.device = 'cuda' if torch.cuda.is_available() else 'cpu'
    cfg.episode_length = cfg.episode_length // cfg.action_repeat
    cfg.train_step = cfg.train_episode * cfg.episode_length

    ###### set up workspace ######
    work_dir = Path().cwd() / __LOGS__ / cfg.algo_name / cfg.exp_name / cfg.env_name / str(cfg.seed)
    if cfg.save_model:
        model_dir = work_dir / 'models'
        helper.make_dir(model_dir)
    if cfg.save_buffer:
        buffer_dir = work_dir / 'buffer'
        helper.make_dir(buffer_dir)
    if cfg.save_logging:
        logs_dir = work_dir / 'logging'
        helper.make_dir(work_dir / "logging")
        logger = Logger(logs_dir)

    video_recorder = VideoRecorder(work_dir) if cfg.save_video else None

    ###### initialize environments ######
    env = utils.make_env(cfg.env_name, cfg.seed, cfg)
    eval_env = utils.make_env(cfg.env_name, cfg.seed + 100, cfg)

    cfg.obs_shape = tuple(int(x) for x in env.observation_spec().shape)
    cfg.action_shape = tuple(int(x) for x in env.action_spec().shape)

    # initialize wandb logging if needed
    if cfg.use_wandb:
        import omegaconf
        wandb.init(project="tcrl",
                   name=f'{cfg.env_name}-{cfg.algo_name}-{cfg.exp_name}-{str(cfg.seed)}-{int(time.time())}',
                   group=f'{cfg.env_name}-{cfg.algo_name}',
                   tags=[cfg.algo_name, cfg.env_name, cfg.exp_name, str(cfg.seed)],
                   config=omegaconf.OmegaConf.to_container(cfg, resolve=True, throw_on_missing=True))
    print("CONFIG", cfg)

    ###### initialize the TCRL agent ######
    tcrl_kwargs = {
        "obs_shape": cfg.obs_shape,
        "action_shape": cfg.action_shape,
        "mlp_dims": cfg.mlp_dims,
        "latent_dim": cfg.latent_dim,
        "lr": cfg.lr,
        "weight_decay": cfg.weight_decay,
        'tau': cfg.tau,
        'nstep': cfg.nstep,
        "horizon": cfg.horizon,
        "rho": cfg.rho,
        "gamma": cfg.gamma,
        "state_coef": cfg.state_coef,
        "reward_coef": cfg.reward_coef,
        "grad_clip_norm": cfg.grad_clip_norm,
        "std_schedule": cfg.std_schedule,
        "std_clip": cfg.std_clip,
        "device": cfg.device,
        "value_expansion": cfg.value_expansion,
        "value_aggregation": cfg.value_aggregation,
        "normalize_z": cfg.normalize_z
    }

    agent = TCRL(**tcrl_kwargs)

    ###### prepare replay buffer ######
    data_specs = (env.observation_spec(),
                  env.action_spec(),
                  specs.Array((1,), np.float32, 'reward'),
                  specs.Array((1,), np.float32, 'discount'))
    replay_storage = utils.ReplayBufferStorage(data_specs, work_dir / 'buffer')

    replay_loader = utils.make_replay_loader(replay_dir=work_dir / 'buffer', max_size=int(cfg.train_step),
                                             batch_size=int(cfg.batch_size),
                                             num_workers=cfg.buffer_num_workers, save_snapshot=cfg.save_buffer,
                                             nstep=int(cfg.horizon), discount=1.0)
    replay_iter = None

    ###################################
    ########## Training Loop ##########    
    ###################################
    timer = helper.Timer()
    global_step, start_time = 0, time.time()
    total_training_time, total_eval_time, data_collection_time = 0, 0, 0
    for ep in range(cfg.train_episode + 1):
        ###### collect an episodic trajectory ######
        time_step = env.reset()
        replay_storage.add(time_step)
        ep_step, ep_reward = 0, 0
        timer.reset()
        time_steps = [time_step]
        while not time_step.last():
            if ep < cfg.random_episode:
                action = np.random.uniform(-1, 1, env.action_spec().shape).astype(dtype=env.action_spec().dtype)
            else:
                action = agent.select_action(time_step.observation, eval_mode=False, t0=time_step.first)
                action = action.cpu().numpy()

            time_step = env.step(action)
            time_steps.append(time_step)
            replay_storage.add(time_step)

            global_step += 1
            ep_reward += time_step.reward
            ep_step += 1


        data_collection_time += timer.reset()

        ###### update model ######
        if ep >= cfg.random_episode:
            train_info = dict()
            train_info.update(log_q_bias(time_steps, agent, cfg))
            timer.reset()
            for _ in range(cfg.episode_length // cfg.update_every_steps):
                if replay_iter is None:
                    replay_iter = iter(replay_loader)
                train_info.update(agent.update(ep, replay_iter, cfg.batch_size))  # log training every episode
            elapsed_time = timer.reset()
            total_training_time += elapsed_time
            # logging
            if cfg.save_logging or cfg.use_wandb:
                episode_len = ep_step * cfg.action_repeat
                train_info.update({'episode_reward': ep_reward,
                                   'fps': episode_len / elapsed_time,
                                   'total_time': timer.total_time(),
                                   'total_training_time': total_training_time,
                                   'data_collection_time': data_collection_time,
                                   'episode_length': episode_len,
                                   'episode': ep,
                                   'step': global_step,
                                   'env_step': global_step * cfg.action_repeat})
                # save to logger and wandb
                if cfg.save_logging:
                    with logger.log_and_dump_ctx(global_step * cfg.action_repeat, ty='train') as log:
                        log.log_metrics(train_info)
                if cfg.use_wandb: wandb.log({'train/': train_info})

        ###### evaluation ######
        if ep % cfg.eval_interval == 0:
            timer.reset()
            Gs = utils.evaluate(eval_env, agent, ep=ep, num_episode=cfg.eval_episode, video=video_recorder)
            elapsed_time = timer.reset()
            total_eval_time += elapsed_time
            if cfg.save_logging or cfg.use_wandb:
                eval_metrics = {
                    'episode': ep,
                    'step': global_step,
                    'env_step': global_step * cfg.action_repeat,
                    'time': time.time() - start_time,
                    'episode_reward': np.mean(Gs),
                    'eval_total_time': total_eval_time}
                if cfg.save_logging:
                    with logger.log_and_dump_ctx(global_step * cfg.action_repeat, ty='eval') as log:
                        log.log_metrics(eval_metrics)
                if cfg.use_wandb: wandb.log({"eval/": eval_metrics})

        ###### save model and buffers ######
        if cfg.save_model and ep != 0 and ep % cfg.save_interval == 0:
            agent.save(work_dir / "models" / f"model_{ep}.pt")

    print(f"------ Training is done, training duration is: {timedelta(seconds=(time.time() - start_time))} ------")


if __name__ == "__main__":
    main()
