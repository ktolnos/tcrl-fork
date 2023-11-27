import os
import sys

from utils.helper import Timer, FreezeParameters

sys.path.insert(0, os.path.abspath(".."))
import numpy as np
import torch
import torch.nn as nn
from utils import helper as h
from utils import net


def to_torch(xs, device, dtype=torch.float32):
    return tuple(torch.as_tensor(x, device=device, dtype=dtype) for x in xs)


class Actor(nn.Module):
    def __init__(self, latent_dim, mlp_dims, action_shape):
        super().__init__()
        self.trunk = nn.Sequential(nn.Linear(latent_dim, mlp_dims[0]),
                                   nn.LayerNorm(mlp_dims[0]), nn.Tanh())
        self._actor = net.mlp(mlp_dims[0], mlp_dims[1:], action_shape[0])
        self.apply(net.orthogonal_init)

    def forward(self, obs, std):
        feature = self.trunk(obs)
        mu = self._actor(feature)
        mu = torch.tanh(mu)
        std = torch.ones_like(mu) * std
        return h.TruncatedNormal(mu, std)


class Critic(nn.Module):
    def __init__(self, latent_dim, mlp_dims, action_shape):
        super().__init__()
        self.trunk = nn.Sequential(nn.Linear(latent_dim + action_shape[0], mlp_dims[0]),
                                   nn.LayerNorm(mlp_dims[0]), nn.Tanh())
        self._critic1 = net.mlp(mlp_dims[0], mlp_dims[1:], 1)
        self._critic2 = net.mlp(mlp_dims[0], mlp_dims[1:], 1)
        self.apply(net.orthogonal_init)

    def forward(self, z, a):
        feature = torch.cat([z, a], dim=-1)
        feature = self.trunk(feature)
        return self._critic1(feature), self._critic2(feature)


class ValueCritic(nn.Module):
    def __init__(self, latent_dim, mlp_dims, transition1, transition2, critic_model_grad):
        super().__init__()
        self.transition2 = transition2
        self.transition1 = transition1

        if critic_model_grad == 'none':
            self.freeze_params_list = [transition1, transition2]
        elif critic_model_grad == 'first':
            self.freeze_params_list = [transition1.reward, transition2]
        elif critic_model_grad == 'both':
            self.freeze_params_list = [transition1.reward, transition2.reward]
        else:
            raise ValueError(f'Unsupported critic_model_grad {critic_model_grad}')

        self.trunk = nn.Sequential(nn.Linear(latent_dim, mlp_dims[0]),
                                   nn.LayerNorm(mlp_dims[0]), nn.Tanh())
        self._critic1 = net.mlp(mlp_dims[0], mlp_dims[1:], 1)
        self._critic2 = net.mlp(mlp_dims[0], mlp_dims[1:], 1)
        self.apply(net.orthogonal_init)

    def value(self, z1, z2):
        feature1 = self.trunk(z1)
        feature2 = self.trunk(z2)
        return self._critic1(feature1), self._critic2(feature2)

    def forward(self, z, a):
        with FreezeParameters(self.freeze_params_list):
            z1, reward1 = self.transition1(z, a)
            z2, reward2 = self.transition2(z, a)

        return self.value(z1, z2)


class Encoder(nn.Module):
    def __init__(self, obs_shape, mlp_dims, latent_dim, normalize_z):
        super().__init__()
        self.normalize_z = normalize_z
        self._encoder = net.mlp(obs_shape[0], mlp_dims, latent_dim, )
        self.apply(net.orthogonal_init)

    def forward(self, obs):
        out = self._encoder(obs)
        if self.normalize_z:
            return torch.nn.functional.normalize(out) * out.shape[-1]
        return out



class LatentModel(nn.Module):
    def __init__(self, latent_dim, action_shape, mlp_dims, normalize_z):
        super().__init__()
        self.normalize_z = normalize_z
        self.dynamics = net.mlp(latent_dim + action_shape[0], mlp_dims, latent_dim)
        self.reward = net.mlp(latent_dim + action_shape[0], mlp_dims, 1)
        self.apply(net.orthogonal_init)

    def forward(self, z, action):
        """Perform one step forward rollout to predict the next latent state and reward."""
        assert z.ndim == action.ndim == 2  # [batch_dim, a/s_dim]

        x = torch.cat([z, action], dim=-1)  # shape[B, xdim]
        next_z = self.dynamics(x)
        if self.normalize_z:
            next_z = torch.nn.functional.normalize(next_z) * z.shape[-1]
        reward = self.reward(x)
        return next_z, reward


class TCRL(object):
    def __init__(self, obs_shape, action_shape, mlp_dims, latent_dim,
                 lr, weight_decay=1e-6, tau=0.005, rho=0.9, gamma=0.99,
                 nstep=3, horizon=5, state_coef=1.0, reward_coef=1.0, grad_clip_norm=10.,
                 std_schedule="", std_clip=0.3,
                 device='cuda', value_expansion='td-k', value_aggregation='min', normalize_z=False,
                 lambda_=0.95, policy_update='ddpg', critic_mode='q', critic_model_grad='none', model_loss='cosine'):
        self.model_loss = model_loss
        self.critic_model_grad = critic_model_grad
        self.policy_update = policy_update
        if value_expansion == 'double-mve' and value_aggregation == 'mean':
            raise ValueError(f'Cant use {value_expansion} with {value_aggregation}')
        self.value_expansion = value_expansion
        self.value_aggregation = value_aggregation
        self.lambda_ = lambda_
        self.device = torch.device(device)

        # models
        self.encoder = Encoder(obs_shape, mlp_dims, latent_dim, normalize_z).to(self.device)
        self.encoder_tar = Encoder(obs_shape, mlp_dims, latent_dim, normalize_z).to(self.device)

        self.trans1 = LatentModel(latent_dim, action_shape, mlp_dims, normalize_z).to(self.device)
        self.trans2 = LatentModel(latent_dim, action_shape, mlp_dims, normalize_z).to(self.device)

        self.actor = Actor(latent_dim, mlp_dims, action_shape).to(self.device)

        if critic_mode == 'q':
            self.critic = Critic(latent_dim, mlp_dims, action_shape).to(self.device)
            self.critic_tar = Critic(latent_dim, mlp_dims, action_shape).to(self.device)
        elif critic_mode == 'value':
            self.critic = ValueCritic(latent_dim, mlp_dims,
                                      transition1=self.trans1,
                                      transition2=self.trans1,
                                      critic_model_grad=critic_model_grad
                                      ).to(self.device)
            self.critic_tar = ValueCritic(latent_dim, mlp_dims,
                                          transition1=self.trans1,
                                          transition2=self.trans1,
                                          critic_model_grad=critic_model_grad
                                          ).to(self.device)
        elif critic_mode == 'double_model_value':
            self.critic = ValueCritic(latent_dim, mlp_dims,
                                      transition1=self.trans1,
                                      transition2=self.trans2,
                                      critic_model_grad=critic_model_grad
                                      ).to(self.device)
            self.critic_tar = ValueCritic(latent_dim, mlp_dims,
                                          transition1=self.trans1,
                                          transition2=self.trans2,
                                          critic_model_grad=critic_model_grad
                                          ).to(self.device)
        else:
            raise ValueError(f'Unsupported critic_mode={critic_mode}')

        # init optimizer
        self.enc_trans_optim = torch.optim.Adam(list(self.encoder.parameters()) + \
                                                list(self.trans1.parameters()) + \
                                                list(self.trans2.parameters()), lr=lr, weight_decay=weight_decay)
        self.actor_optim = torch.optim.Adam(self.actor.parameters(), lr=lr)
        self.critic_optim = torch.optim.Adam(self.critic.parameters(), lr=lr)

        # assign variables
        # common
        self.counter = 0
        self.action_shape = action_shape
        self.tau = tau  # EMA coef

        self.std_schedule = std_schedule
        self.std = h.linear_schedule(self.std_schedule, 0)
        self.std_clip = std_clip

        # transition related 
        self.state_coef, self.reward_coef, self.horizon, self.rho, self.grad_clip_norm = state_coef, reward_coef, horizon, rho, grad_clip_norm

        # policy related
        self.gamma, self.nstep = gamma, nstep

        self.data_loading_time = 0
        self.torching_time = 0
        self.enc_latent_time = 0
        self.z_prep_time = 0
        self.q_time = 0
        self.pi_time = 0
        self.soft_time = 0

    def save(self, fp):
        torch.save({
            'encoder': self.encoder.state_dict(),
            'trans': self.trans.state_dict(),
            'actor': self.actor.state_dict(),
            'critic': self.critic.state_dict(),
        }, fp)

    def load(self, fp):
        d = torch.load(fp)
        self.encoder.load_state_dict(d['encoder'])
        self.trans.load_state_dict(d['trans'])
        self.actor.load_state_dict(d['actor'])
        self.critic.load_state_dict(d['critic'])

    @torch.no_grad()
    def enc(self, obs):
        """ Only replace part of the states from the original observations to check which one have the highest impacts."""
        if isinstance(obs, np.ndarray):
            obs = torch.tensor(obs, dtype=torch.float32, device=self.device).unsqueeze(0)
        return self.encoder(obs)

    def aggregate_values(self, tensor1, tensor2):
        if self.value_aggregation == 'min':
            return torch.min(tensor1, tensor2)
        if self.value_aggregation == 'max':
            return torch.min(tensor1, tensor2)
        if self.value_aggregation == 'mean':
            return (tensor1 + tensor2) / 2
        raise ValueError(f'Incorrect value aggregation {self.value_aggregation}')

    def _update_enc_trans(self, obs, action, next_obses, reward):

        self.enc_trans_optim.zero_grad(set_to_none=True)
        self.trans1.train()
        self.trans2.train()

        state_loss, reward_loss = 0, 0

        z = self.encoder(obs)
        mse = 0
        critic_loss = torch.zeros((z.shape[0]), device=self.device)
        if self.model_loss == 'mse':
            consistency_loss_fn = h.mse
        elif self.model_loss == 'cosine':
            consistency_loss_fn = h.cosine
        else:
            raise ValueError(f'Unrecognized model loss {self.model_loss}')

        prev_v1, prev_v2 = None, None
        prev_r1, prev_r2 = None, None
        for t in range(self.horizon):
            next_z_pred1, r_pred1 = self.trans1(z, action[t])
            next_z_pred2, r_pred2 = self.trans2(z, action[t])

            with torch.no_grad():
                next_obs = next_obses[t]
                next_z_tar = self.encoder_tar(next_obs)
                r_tar = reward[t]
                assert next_obs.ndim == r_tar.ndim == 2

            # Losses
            rho = (self.rho ** t)
            state_loss += rho * torch.mean(consistency_loss_fn(next_z_pred1, next_z_tar) + consistency_loss_fn(next_z_pred2, next_z_tar), dim=-1)
            reward_loss += rho * torch.mean(h.mse(r_pred1, r_tar) + h.mse(r_pred2, r_tar), dim=-1)
            mse += rho * h.mse(next_z_pred1, next_z_tar).mean()

            # don't forget this
            batch_size = z.shape[0]
            z2_idx = np.random.choice(batch_size, batch_size // 2, replace=False)
            z = next_z_pred1
            z = z.clone()
            z[z2_idx] = next_z_pred2[z2_idx]

            if self.value_expansion == 'joint':
                if self.critic_model_grad == 'first':
                    z2 = next_z_pred1
                elif self.critic_model_grad == 'both':
                    z2 = next_z_pred2
                else:
                    raise f'critic_model_grad={self.critic_model_grad} cannot be used with joint optimization'

                if prev_v1 is not None:
                    with torch.no_grad():
                        v1_tar, v2_tar = self.critic_tar.value(next_z_pred1, z2)
                        td_target = torch.min(prev_r1 + self.gamma * v1_tar, prev_r2 + self.gamma * v2_tar)
                    critic_loss += rho * (torch.mean(h.mse(prev_v1, td_target)) + torch.mean(h.mse(prev_v2, td_target)))

                prev_v1, prev_v2 = self.critic.value(next_z_pred1, z2)
                prev_r1 = r_pred1
                if self.critic_model_grad == 'first':
                    prev_r2 = r_pred1
                elif self.critic_model_grad == 'both':
                    prev_r2 = r_pred2

        total_loss = (self.state_coef * state_loss.clamp(max=1e4) + \
                      self.reward_coef * reward_loss.clamp(max=1e4) +\
                      critic_loss.clamp(max=1e4)).mean()

        total_loss.register_hook(lambda grad: grad * (1 / self.horizon))
        total_loss.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(list(self.encoder.parameters()) + list(self.trans1.parameters()) + list(self.trans2.parameters()),
                                                   self.grad_clip_norm, error_if_nonfinite=True)

        self.enc_trans_optim.step()

        self.trans1.eval()
        self.trans2.eval()
        return {
            'trans_loss': float(total_loss.mean().item()),
            'consistency_loss': float(state_loss.mean().item()),
            'mse_consistency_loss': float(mse.item()),
            'reward_loss': float(reward_loss.mean().item()),
            'trans_grad_norm': float(grad_norm),
            'z_mean': z.mean().item(), 'z_max': z.max().item(), 'z_min': z.min().item()
        }

    def _update_q(self, z, act, rew, discount, next_z):
        with torch.no_grad():
            action = self.actor(next_z, std=self.std).sample(clip=self.std_clip)

            td_target = rew + discount * self.aggregate_values(*self.critic_tar(next_z, action))

        q1, q2 = self.critic(z, act)
        q_loss = torch.mean(h.mse(q1, td_target) + h.mse(q2, td_target))

        self.critic_optim.zero_grad(set_to_none=True)
        q_loss.backward()
        self.critic_optim.step()

        return {'q': q1.mean().item(), 'q_loss': q_loss.item()}

    def _update_q_mve(self, z):
        zs, acts, rs, qs = [z], [], [], []

        with torch.no_grad():
            for t in range(self.nstep):
                act = self.actor(z, self.std).sample(self.std_clip)
                acts.append(act)
                z, r = self.trans1(z, act)
                zs.append(z)
                rs.append(r)

            # calculate td_target
            next_q = self.aggregate_values(*self.critic_tar(z, self.actor(z, self.std).sample(self.std_clip)))

            td_targets = []
            for t in reversed(range(len(rs))):
                next_q = rs[t] + self.gamma * next_q
                td_targets.append(next_q)
            td_targets = list(reversed(td_targets))

        # calculate the td error
        q_loss = 0
        for t, td_target in enumerate(td_targets):
            q1, q2 = self.critic(zs[t], acts[t])
            q_loss += h.mse(q1, td_target) + h.mse(q2, td_target)
        q_loss = torch.mean(q_loss)
        # H-step td
        # q1, q2 = self.critic(zs[0], acts[0])
        # q_loss = torch.mean(h.mse(q1, td_targets[0]) + h.mse(q2, td_targets[0]))

        self.critic_optim.zero_grad(set_to_none=True)
        q_loss.register_hook(lambda grad: grad / self.nstep)
        q_loss.backward()
        self.critic_optim.step()

        return {'q': q1.mean().item(), 'q_loss': q_loss.item()}

    def _update_q_gae(self, z):
        zs, acts, rs, qs = [z], [], [], []

        with torch.no_grad():
            for t in range(self.nstep):
                act = self.actor(zs[t], self.std).sample(self.std_clip)
                acts.append(act)
                qs.append(self.aggregate_values(*self.critic_tar(zs[t], acts[t])))
                z, r = self.trans1(zs[t], act)
                zs.append(z)
                rs.append(r)

            # calculate td_target
            next_q = self.aggregate_values(*self.critic_tar(z, self.actor(z, self.std).sample(self.std_clip)))

            td_targets = []
            for t in reversed(range(len(rs))):
                if t == len(rs) - 1:  # the last timestep
                    next_values = next_q
                else:
                    next_values = (1-self.lambda_) * qs[t+1] + self.lambda_*td_targets[-1]

                td_targets.append(rs[t] + self.gamma * next_values)

            td_targets = list(reversed(td_targets))

        # calculate the td error
        q_loss = 0
        for t, td_target in enumerate(td_targets):
            q1, q2 = self.critic(zs[t], acts[t])
            q_loss += h.mse(q1, td_target) + h.mse(q2, td_target)
        q_loss = torch.mean(q_loss)

        self.critic_optim.zero_grad(set_to_none=True)
        q_loss.register_hook(lambda grad: grad * (1 / self.nstep))
        q_loss.backward()
        self.critic_optim.step()

        return {'q': q1.mean().item(), 'q_loss': q_loss.item()}


    def _update_q_double_mve(self, z):
        zs, acts, rs = [z], [], []
        act = self.actor(z, self.std).sample(self.std_clip)
        acts.append(act)

        with torch.no_grad():
            for t in range(self.nstep):
                z1, r1 = self.trans1(z, act)
                z2, r2 = self.trans2(z, act)
                act1 = self.actor(z1, self.std).sample(self.std_clip)
                act2 = self.actor(z2, self.std).sample(self.std_clip)
                q1 = torch.max(*self.critic_tar(z1, act1))
                q2 = torch.max(*self.critic_tar(z2, act2))
                if self.value_aggregation == 'min':
                    z1_mask = q1 < q2
                else:
                    z1_mask = q1 > q2
                z2_mask = ~z1_mask
                z = z1 * z1_mask + z2 * z2_mask
                r = r1 * z1_mask + r2 * z2_mask
                act = act1 * z1_mask + act2 * z2_mask

                zs.append(z)
                rs.append(r)
                acts.append(act)

            # calculate td_target
            next_q = self.aggregate_values(*self.critic_tar(z, self.actor(z, self.std).sample(self.std_clip)))

            td_targets = []
            for t in reversed(range(len(rs))):
                next_q = rs[t] + self.gamma * next_q
                td_targets.append(next_q)
            td_targets = list(reversed(td_targets))

        # calculate the td error
        q_loss = 0
        for t, td_target in enumerate(td_targets):
            q1, q2 = self.critic(zs[t], acts[t])
            q_loss += h.mse(q1, td_target) + h.mse(q2, td_target)
        q_loss = torch.mean(q_loss)
        # H-step td
        # q1, q2 = self.critic(zs[0], acts[0])
        # q_loss = torch.mean(h.mse(q1, td_targets[0]) + h.mse(q2, td_targets[0]))

        self.critic_optim.zero_grad(set_to_none=True)
        q_loss.register_hook(lambda grad: grad / self.nstep)
        q_loss.backward()
        self.critic_optim.step()

        return {'q': q1.mean().item(), 'q_loss': q_loss.item()}


    def _update_pi(self, z):
        a = self.actor(z, std=self.std).sample(clip=self.std_clip)
        Q = torch.min(*self.critic(z, a))
        pi_loss = -Q.mean()

        self.actor_optim.zero_grad(set_to_none=True)
        pi_loss.backward()
        self.actor_optim.step()

        return {'pi_loss': pi_loss.item()}


    def _update_pi_bp(self, z):
        Q = 0
        with FreezeParameters([self.trans1, self.trans2]):
            for t in range(self.nstep):
                act = self.actor(z, self.std).sample(self.std_clip)
                z, r = self.trans1(z, act)
                Q += (self.gamma ** t) * r

        # calculate td_target
        with FreezeParameters([self.critic]):
            next_q = self.aggregate_values(*self.critic(z, self.actor(z, self.std).sample(self.std_clip)))
        Q += (self.gamma ** self.nstep) * next_q

        pi_loss = -Q.mean()

        self.actor_optim.zero_grad(set_to_none=True)
        pi_loss.backward()
        self.actor_optim.step()
        return {'pi_loss': pi_loss.item()}

    def _update_pi_lambda_bp(self, z):
        zs, acts, rs, qs = [z], [], [], []

        with FreezeParameters([self.critic, self.trans2, self.trans1]):
            for t in range(self.nstep):
                act = self.actor(zs[t], self.std).sample(self.std_clip)
                acts.append(act)
                qs.append(self.aggregate_values(*self.critic(zs[t], acts[t])))
                z, r = self.trans1(zs[t], act)
                zs.append(z)
                rs.append(r)

            # calculate td_target
            next_q = self.aggregate_values(*self.critic(z, self.actor(z, self.std).sample(self.std_clip)))

            td_targets = []
            for t in reversed(range(len(rs))):
                if t == len(rs) - 1:  # the last timestep
                    next_values = next_q
                else:
                    next_values = (1 - self.lambda_) * qs[t + 1] + self.lambda_ * td_targets[-1]

                td_targets.append(rs[t] + self.gamma * next_values)

            lambda_q_sum = 0
            for q in reversed(td_targets):
                lambda_q_sum += q

        pi_loss = -lambda_q_sum.mean()

        self.actor_optim.zero_grad(set_to_none=True)
        pi_loss.register_hook(lambda grad: grad / self.nstep)
        pi_loss.backward()
        self.actor_optim.step()
        return {'pi_loss': pi_loss.item()}

    def update(self, step, replay_iter, batch_size):
        info = dict()
        timer = Timer()
        self.std = h.linear_schedule(self.std_schedule, step)  # linearly udpate std

        # obs, action, next_obses, reward = replay_buffer.sample(batch_size) 
        batch = next(replay_iter)
        self.data_loading_time += timer.reset()
        obs, action, reward, discount, next_obses = to_torch(batch, self.device, dtype=torch.float32)
        # swap the batch and horizon dimension -> [H, B, _shape]
        action, reward, discount, next_obses = torch.swapaxes(action, 0, 1), torch.swapaxes(reward, 0, 1), \
            torch.swapaxes(discount, 0, 1), torch.swapaxes(next_obses, 0, 1)
        self.torching_time += timer.reset()

        # update encoder and latent dynamics
        info.update(self._update_enc_trans(obs, action, next_obses, reward))
        self.enc_latent_time += timer.reset()

        # form n-step samples
        z0 = self.enc(obs)
        if self.value_expansion == 'mve':
            info.update(self._update_q_mve(z0))
        elif self.value_expansion == 'double-mve':
            info.update(self._update_q_double_mve(z0))
        elif self.value_expansion == 'lambda-mve':
            info.update(self._update_q_gae(z0))
        elif self.value_expansion == 'td-k':
            zt = self.enc(next_obses[self.nstep - 1])
            _rew, _discount = 0, 1
            for t in range(self.nstep):
                _rew += _discount * reward[t]
                _discount *= self.gamma
            self.z_prep_time += timer.reset()
            # udpate policy and value functions
            info.update(self._update_q(z0, action[0], _rew, _discount, zt))
        elif self.value_expansion == 'joint':
            ...  # handled by the model update
        else:
            raise ValueError(f'Unsupported self.value_expansion = {self.value_expansion}')

        self.q_time += timer.reset()
        if self.policy_update == 'ddpg':
            info.update(self._update_pi(z0))
        elif self.policy_update == 'backprop':
            info.update(self._update_pi_bp(z0))
        elif self.policy_update == 'lambda_bp':
            info.update(self._update_pi_lambda_bp(z0))
        else:
            raise ValueError(f'Unsupported self.policy_update = {self.policy_update}')
        self.pi_time += timer.reset()

        # update target networks
        h.soft_update_params(self.encoder, self.encoder_tar, self.tau)
        h.soft_update_params(self.critic, self.critic_tar, self.tau)
        self.soft_time += timer.reset()

        self.counter += 1
        info.update({
            'std': self.std,
            'data_loading_time': self.data_loading_time,
            'torching_time': self.torching_time,
            'enc_latent_time': self.enc_latent_time,
            'z_prep_time': self.z_prep_time,
            'q_time': self.q_time,
            'pi_time': self.pi_time,
            'soft_time': self.soft_time,
        })

        return info

    @torch.no_grad()
    def select_action(self, obs, eval_mode=False, t0=True):
        if isinstance(obs, np.ndarray):
            obs = torch.tensor(obs, dtype=torch.float32, device=self.device).unsqueeze(0)

        dist = self.actor(self.enc(obs), std=self.std)
        if eval_mode:
            action = dist.mean
        else:
            action = dist.sample(clip=None)
        return action[0]
