import numpy as np
import scipy.signal
import torch
import torch.nn as nn
from gym.spaces import Box, Discrete
from torch.distributions.categorical import Categorical
from torch.distributions.normal import Normal


def combined_shape(length, shape=None):
    if shape is None:
        return (length,)
    return (length, shape) if np.isscalar(shape) else (length, *shape)


def mlp(sizes, activation, output_activation=nn.Identity):
    layers = []
    for j in range(len(sizes) - 1):
        act = activation if j < len(sizes) - 2 else output_activation
        layers += [nn.Linear(sizes[j], sizes[j + 1]), act()]
    return nn.Sequential(*layers)


def count_vars(module):
    return sum([np.prod(p.shape) for p in module.parameters()])


def discount_cumsum(x, discount):
    """
    magic from rllab for computing discounted cumulative sums of vectors.

    input:
        vector x,
        [x0,
         x1,
         x2]

    output:
        [x0 + discount * x1 + discount^2 * x2,
         x1 + discount * x2,
         x2]
    """
    return scipy.signal.lfilter([1], [1, float(-discount)], x[::-1], axis=0)[::-1]


class Actor(nn.Module):
    def _distribution(self, obs):
        raise NotImplementedError

    def _log_prob_from_distribution(self, pi, act):
        raise NotImplementedError

    def forward(self, obs, act=None):
        # Produce action distributions for given observations, and
        # optionally compute the log likelihood of given actions under
        # those distributions.
        pi = self._distribution(obs)
        logp_a = None
        if act is not None:
            logp_a = self._log_prob_from_distribution(pi, act)
        return pi, logp_a


class MLPCategoricalActor(Actor):
    def __init__(self, obs_dim, act_dim, hidden_sizes, activation):
        super().__init__()
        self.logits_net = mlp(
            [obs_dim] + list(hidden_sizes) + [act_dim], activation)

    def _distribution(self, obs):
        logits = self.logits_net(obs)
        return Categorical(logits=logits)

    def _log_prob_from_distribution(self, pi, act):
        return pi.log_prob(act)


class MLPGaussianActor(Actor):
    def __init__(self, obs_dim, act_dim, hidden_sizes, activation):
        super().__init__()
        log_std = -0.5 * np.ones(act_dim, dtype=np.float32)
        self.log_std = torch.nn.Parameter(torch.as_tensor(log_std))
        self.mu_net = mlp([obs_dim] + list(hidden_sizes) +
                          [act_dim], activation)

    def _distribution(self, obs):
        mu = self.mu_net(obs)
        std = torch.exp(self.log_std)
        return Normal(mu, std)

    def _log_prob_from_distribution(self, pi, act):
        # Last axis sum needed for Torch Normal distribution
        return pi.log_prob(act).sum(axis=-1)


class MLPCritic(nn.Module):
    def __init__(self, obs_dim, hidden_sizes, activation):
        super().__init__()
        self.v_net = mlp([obs_dim] + list(hidden_sizes) + [1], activation)

    def forward(self, obs):
        # Critical to ensure v has right shape.
        return torch.squeeze(self.v_net(obs), -1)


class MLPActorCritic(nn.Module):
    def __init__(self, observation_space, action_space,
                 hidden_sizes=(64, 64), activation=nn.Tanh):
        super().__init__()

        obs_dim = observation_space.shape[0]

        # policy builder depends on action space
        if isinstance(action_space, Box):
            self.pi = MLPGaussianActor(
                obs_dim, action_space.shape[0], hidden_sizes, activation)
        elif isinstance(action_space, Discrete):
            self.pi = MLPCategoricalActor(
                obs_dim, action_space.n, hidden_sizes, activation)

        # build value function
        self.v = MLPCritic(obs_dim, hidden_sizes, activation)

    def step(self, obs):
        with torch.no_grad():
            pi = self.pi._distribution(obs)
            a = pi.sample()
            logp_a = self.pi._log_prob_from_distribution(pi, a)
            v = self.v(obs)
        return a.numpy(), v.numpy(), logp_a.numpy()

    def act(self, obs):
        return self.step(obs)[0]


class MLPActorCritic2Heads(nn.Module):

    def __init__(self, observation_space, action_space,
                 hidden_sizes=(64, 64), activation=nn.Tanh):
        super().__init__()

        obs_dim = observation_space.shape[0]

        # policy builder depends on action space
        if isinstance(action_space, Box):
            self.pi = MLPGaussianActor(obs_dim, action_space.shape[0], hidden_sizes, activation)
        elif isinstance(action_space, Discrete):
            self.pi = MLPCategoricalActor(obs_dim, action_space.n, hidden_sizes, activation)

        # build value functions for extristic and intristic rewards
        self.v_extr = MLPCritic(obs_dim, hidden_sizes, activation)
        self.v_intr = MLPCritic(obs_dim, hidden_sizes, activation)

    def step(self, obs):
        with torch.no_grad():
            pi = self.pi._distribution(obs)
            a = pi.sample()
            logp_a = self.pi._log_prob_from_distribution(pi, a)
            v_extr = self.v_extr(obs)
            v_intr = self.v_intr(obs)
        return a.numpy(), v_extr.numpy(), v_intr.numpy(), logp_a.numpy()

    def act(self, obs):
        return self.step(obs)[0]
    
class MLPActorCritic2V(nn.Module):

    def __init__(self, observation_space, action_space, 
                 hidden_sizes=(64,64), activation=nn.Tanh):
        super().__init__()

        obs_dim = observation_space.shape[0]
        
        # policy builder depends on action space
        if isinstance(action_space, Box):
            self.pi = MLPGaussianActor(obs_dim, action_space.shape[0], hidden_sizes, activation)
        elif isinstance(action_space, Discrete):
            self.pi = MLPCategoricalActor(obs_dim, action_space.n, hidden_sizes, activation)

        # build value functions
        self.v_extr  = MLPCritic(obs_dim, hidden_sizes, activation)
        self.v_intr  = MLPCritic(obs_dim, hidden_sizes, activation)
        
    def step(self, obs):
        with torch.no_grad():
            pi = self.pi._distribution(obs)
            a = pi.sample()
            logp_a = self.pi._log_prob_from_distribution(pi, a)
            v_extr = self.v_extr(obs)
            v_intr = self.v_intr(obs)
        return a.numpy(), v_extr.numpy(), v_intr.numpy(), logp_a.numpy()

    def act(self, obs):
        return self.step(obs)[0]



class IntrMotivation(nn.Module):
    def __init__(self):
        super().__init__()

    def loss(self, o, next_o, a):
        pass

    def reward(self, o, next_o, a):
        pass


class ForwardDynamics(IntrMotivation):

    def __init__(self, observation_space, action_space,
                 hidden_sizes=(64, 64), activation=nn.Tanh,
                 scaling_factor=20000):
        super().__init__()
        self.scaling_factor = scaling_factor
        obs_dim = observation_space.shape[0]
        """
        currently only dicsrete
        if isinstance(action_space, Box):
            act_dim = action_space.shape[0]
        else:
        """
        self.act_dim = action_space.n
        self.net = mlp([obs_dim + self.act_dim] + list(hidden_sizes) +
                       [obs_dim], activation=activation)
    def loss(self, o, next_o, a):
        a_t = torch.as_tensor(a)
        o_t = torch.as_tensor(o, dtype=torch.float32)
        next_o_t = torch.as_tensor(next_o, dtype=torch.float32)
        x = torch.cat([o_t, nn.functional.one_hot(
            a_t.to(torch.int64), self.act_dim).float()], dim=-1)
        pred_next_o = self.net(x)

        return (pred_next_o - next_o_t).pow(2).mean(dim=-1)

    def reward(self, o, next_o, a):
        return self.scaling_factor / 2 * self.loss(o, next_o, a).detach().numpy()


class RND(nn.Module):
    def __init__(self, obs_dim, hidden_sizes, activation):
        super().__init__()
        self.target_network = mlp([obs_dim] + list(hidden_sizes), activation)
        for p in self.target_network.parameters():
            p.requires_grad = False
        self.predictor_network = mlp([obs_dim] + list(hidden_sizes), activation)
        for p in self.modules():
            if isinstance(p, nn.Linear):
                nn.init.orthogonal_(p.weight, np.sqrt(2))
                p.bias.data.zero_()

    def loss(self, o):
        return ((self.target_network(o) - self.predictor_network(o)) ** 2).mean()

    def reward(self, o):
        with torch.no_grad():
            return self.loss(o).detach().item()


    
    def loss(self, o, next_o, a):
        raise NotImplementedError
        
    def reward(self, o, next_o, a):
        raise NotImplementedError


class InverseDynamic(IntrMotivation):
    
    def __init__(self, observation_space, action_space,
                 encoder_output_size,
                 hidden_sizes_encoder=(64, 64),
                 hidden_sizes_DM=(64, 64), 
                 activation_encoder=nn.ELU,
                 activation_DM=nn.ELU, 
                 scaling_factor=10):
        super().__init__()
        
        self.action_space = action_space
        self.scaling_factor = scaling_factor
        obs_dim = observation_space.shape[0]
        
        self.encoder = mlp([obs_dim] + list(hidden_sizes_encoder) + [encoder_output_size], activation_encoder)
        
        if isinstance(action_space, Box):
            self.IDM = mlp(2 * [obs_dim] + list(hidden_sizes_DM) + [action_space.shape[0]], activation_DM)
            self.loss_func = nn.MSELoss()
        elif isinstance(action_space, Discrete):
            self.IDM = mlp([2 * encoder_output_size] + list(hidden_sizes_DM) + [action_space.n], activation_DM)
            self.loss_func = nn.CrossEntropyLoss()
            
    def loss(self, o, next_o, a):
        if isinstance(self.action_space, Discrete):
            a = a.long()
        phi = torch.cat((self.encoder(o), self.encoder(next_o)), dim=1)
        a_pred = self.IDM(phi)
        return self.loss_func(a_pred, a)
    
    def reward(self, o, next_o, a):
        with torch.no_grad():
            if isinstance(self.action_space, Discrete):
                a = a.long()
            phi = torch.cat((self.encoder(o), self.encoder(next_o)), dim=1)
            a_pred = self.IDM(phi)
            intr_rew = self.scaling_factor * self.loss_func(a_pred, a)
        return intr_rew


class InverseDynamicNoEnc(IntrMotivation):
    
    def __init__(self, observation_space, action_space,
                 encoder_output_size,
                 hidden_sizes_encoder=(64, 64),
                 hidden_sizes_DM=(64, 64), 
                 activation_encoder=nn.ELU,
                 activation_DM=nn.ELU, 
                 scaling_factor=10):
        super().__init__()
        
        self.action_space = action_space
        self.scaling_factor = scaling_factor
        obs_dim = observation_space.shape[0]
        
        if isinstance(action_space, Box):
            self.IDM = mlp(2 * [obs_dim] + list(hidden_sizes_DM) + [action_space.shape[0]], activation_DM)
            self.loss_func = nn.MSELoss()
        elif isinstance(action_space, Discrete):
            self.IDM = mlp([2 * obs_dim] + list(hidden_sizes_DM) + [action_space.n], activation_DM)
            self.loss_func = nn.CrossEntropyLoss()
            
    def loss(self, o, next_o, a):
        if isinstance(self.action_space, Discrete):
            a = a.long()
        a_pred = self.IDM(torch.cat((o, next_o), dim=1))
        return self.loss_func(a_pred, a)
    
    def reward(self, o, next_o, a):
        with torch.no_grad():
            if isinstance(self.action_space, Discrete):
                a = a.long()
            a_pred = self.IDM(torch.cat((o, next_o), dim=1))
            intr_rew = self.scaling_factor * self.loss_func(a_pred, a)
        return intr_rew


class ForwardDynamic(IntrMotivation):
    
    def __init__(self, observation_space, action_space,
                 encoder_output_size,
                 hidden_sizes_encoder=(64, 64),
                 hidden_sizes_DM=(64, 64), 
                 activation_encoder=nn.ELU,
                 activation_DM=nn.ELU, 
                 scaling_factor=10):
        super().__init__()
        
        self.action_space = action_space
        self.scaling_factor = scaling_factor
        obs_dim = observation_space.shape[0]
        
        self.encoder = mlp([obs_dim] + list(hidden_sizes_encoder) + [encoder_output_size], activation_encoder)
        
        if isinstance(action_space, Box):
            self.FDM = mlp([encoder_output_size +  action_space.shape[0]] + list(hidden_sizes_DM) + [encoder_output_size], activation_DM)
        elif isinstance(action_space, Discrete):
            self.FDM = mlp([encoder_output_size +  action_space.n] + list(hidden_sizes_DM) + [encoder_output_size], activation_DM)

        self.loss_func = nn.MSELoss()
            
    def loss(self, o, next_o, a):
        if isinstance(self.action_space, Discrete):
            a = nn.functional.one_hot(a.long(), self.action_space.n).float()
        phi = torch.cat((self.encoder(o), a), dim=1)
        o_pred = self.FDM(phi)
        return self.loss_func(o_pred, self.encoder(next_o))
    
    def reward(self, o, next_o, a):
        with torch.no_grad():
            if isinstance(self.action_space, Discrete):
                a = nn.functional.one_hot(a.long(), self.action_space.n).float()
            phi = torch.cat((self.encoder(o), a), dim=1)
            o_pred = self.FDM(phi)
            intr_rew = self.scaling_factor * 0.5 * self.loss_func(o_pred, self.encoder(next_o))
        return intr_rew


class ForwardDynamicNoEnc(IntrMotivation):
    
    def __init__(self, observation_space, action_space,
                 encoder_output_size,
                 hidden_sizes_encoder=(64, 64),
                 hidden_sizes_DM=(64, 64), 
                 activation_encoder=nn.ELU,
                 activation_DM=nn.ELU, 
                 scaling_factor=10):
        super().__init__()
        
        self.action_space = action_space
        self.scaling_factor = scaling_factor
        obs_dim = observation_space.shape[0]
        
        if isinstance(action_space, Box):
            self.FDM = mlp([encoder_output_size +  action_space.shape[0]] + list(hidden_sizes_DM) + [encoder_output_size], activation_DM)
        elif isinstance(action_space, Discrete):
            self.FDM = mlp([obs_dim  + action_space.n] + list(hidden_sizes_DM) + [obs_dim], activation_DM)

        self.loss_func = nn.MSELoss()
            
    def loss(self, o, next_o, a):
        if isinstance(self.action_space, Discrete):
            a = nn.functional.one_hot(a.long(), self.action_space.n).float()
        o_pred = self.FDM(torch.cat((o, a), dim=1))
        return self.loss_func(o_pred, next_o)
    
    def reward(self, o, next_o, a):
        with torch.no_grad():
            if isinstance(self.action_space, Discrete):
                a = nn.functional.one_hot(a.long(), self.action_space.n).float()
            o_pred = self.FDM(torch.cat((o, a), dim=1))
            intr_rew = self.scaling_factor * 0.5 * self.loss_func(o_pred, next_o)
        return intr_rew


class ICM(IntrMotivation):
    
    def __init__(self, observation_space, action_space,
                 encoder_output_size,
                 hidden_sizes_encoder=(64, 64),
                 hidden_sizes_DM=(64, 64), 
                 activation_encoder=nn.ELU,
                 activation_DM=nn.ELU, 
                 scaling_factor=10,
                 beta=0.2):
        super().__init__()
        
        self.action_space = action_space
        self.scaling_factor = scaling_factor
        self.beta = beta
        obs_dim = observation_space.shape[0]
        
        self.encoder = mlp([obs_dim] + list(hidden_sizes_encoder) + [encoder_output_size], activation_encoder)
        
        if isinstance(action_space, Box):
            self.FDM = mlp([encoder_output_size +  action_space.shape[0]] + list(hidden_sizes_DM) + [encoder_output_size], activation_DM)
            self.IDM = mlp(2 * [obs_dim] + list(hidden_sizes_DM) + [action_space.shape[0]], activation_DM)
            self.IDM_loss = nn.MSELoss()
        elif isinstance(action_space, Discrete):
            self.FDM = mlp([encoder_output_size +  action_space.n] + list(hidden_sizes_DM) + [encoder_output_size], activation_DM)
            self.IDM = mlp([2 * encoder_output_size] + list(hidden_sizes_DM) + [action_space.n], activation_DM)
            self.IDM_loss = nn.CrossEntropyLoss()

        self.FDM_loss = nn.MSELoss()
            
    def loss(self, o, next_o, a):
        if isinstance(self.action_space, Discrete):
            a = a.long()
        phi = torch.cat((self.encoder(o), self.encoder(next_o)), dim=1)
        a_pred = self.IDM(phi)
        inverse_loss = self.IDM_loss(a_pred, a)
        
        if isinstance(self.action_space, Discrete):
            a = nn.functional.one_hot(a, self.action_space.n).float()
        phi = torch.cat((self.encoder(o), a), dim=1)
        o_pred = self.FDM(phi)
        forward_loss = 0.5 * self.FDM_loss(o_pred, self.encoder(next_o))
        
        return self.beta * forward_loss + (1 - self.beta) * inverse_loss
    
    def reward(self, o, next_o, a):
      with torch.no_grad():
            if isinstance(self.action_space, Discrete):
                a = nn.functional.one_hot(a.long(), self.action_space.n).float()
            phi = torch.cat((self.encoder(o), a), dim=1)
            o_pred = self.FDM(phi)
            intr_rew = self.scaling_factor * 0.5 * self.FDM_loss(o_pred, self.encoder(next_o))
        return intr_rew


class running_estimator:
    def __init__(self):
        """
        Welford's online algorithm
        """
        self.iter = 0
        self.mean = 0
        self.M = 0  # sum of squares of differences from the current mean

    def update(self, x: float):
        self.iter += 1
        d = x - self.mean
        self.mean += d / self.iter
        self.M += d * (x - self.mean)

    def get_std(self):
        return 1 if self.iter < 2  else (self.M / self.iter) ** 0.5


class running_exp_estimator:
    def __init__(self, alpha=0.05):
        self.alpha = alpha
        self.iter = 0
        self.mean = 0
        self.var = 0

    def update(self, x: float):
        if self.iter == 0:
            self.mean = x
        else:
            self.var = (1 - self.alpha) * (self.var + self.alpha * (x - self.mean) ** 2)
            self.mean = (1 - self.alpha) * self.mean + self.alpha * x
        self.iter += 1

    def get_std(self):
        return self.var ** 0.5