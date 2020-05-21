import time

import gym
import numpy as np
import torch
import torch.nn as nn
from torch.optim import Adam

import spinup.algos.pytorch.ppo.core as core
from spinup.algos.pytorch.ppo.core import RND
from spinup.utils.logx import EpochLogger
from spinup.utils.mpi_pytorch import (mpi_avg_grads, setup_pytorch_for_mpi,
                                      sync_params)
from spinup.utils.mpi_tools import (mpi_avg, mpi_fork, mpi_statistics_scalar,
                                    num_procs, proc_id)


class PPOBuffer:
    """
    A buffer for storing trajectories experienced by a PPO agent interacting
    with the environment, and using Generalized Advantage Estimation (GAE-Lambda)
    for calculating the advantages of state-action pairs.
    """

    def __init__(self, obs_dim, act_dim, size, gamma=0.99, lam=0.95):
        self.obs_buf = np.zeros(core.combined_shape(size, obs_dim), dtype=np.float32)
        self.act_buf = np.zeros(core.combined_shape(size, act_dim), dtype=np.float32)
        self.adv_buf = np.zeros(size, dtype=np.float32)
        self.rew_extr_buf = np.zeros(size, dtype=np.float32)
        self.rew_intr_buf = np.zeros(size, dtype=np.float32)
        self.ret_extr_buf = np.zeros(size, dtype=np.float32)
        self.ret_intr_buf = np.zeros(size, dtype=np.float32)
        self.val_extr_buf = np.zeros(size, dtype=np.float32)
        self.val_intr_buf = np.zeros(size, dtype=np.float32)
        self.logp_buf = np.zeros(size, dtype=np.float32)
        self.gamma, self.lam = gamma, lam
        self.ptr, self.path_start_idx, self.max_size = 0, 0, size

    def store(self, obs, act, rew_extr, rew_intr, val_extr, val_intr, logp):
        """
        Append one timestep of agent-environment interaction to the buffer.
        """
        assert self.ptr < self.max_size  # buffer has to have room so you can store
        self.obs_buf[self.ptr] = obs
        self.act_buf[self.ptr] = act
        self.rew_extr_buf[self.ptr] = rew_extr
        self.rew_intr_buf[self.ptr] = rew_intr
        self.val_extr_buf[self.ptr] = val_extr
        self.val_intr_buf[self.ptr] = val_intr
        self.logp_buf[self.ptr] = logp
        self.ptr += 1

    def finish_path(self, last_val_extr, last_val_intr):
        """
        Call this at the end of a trajectory, or when one gets cut off
        by an epoch ending. This looks back in the buffer to where the
        trajectory started, and uses rewards and value estimates from
        the whole trajectory to compute advantage estimates with GAE-Lambda,
        as well as compute the rewards-to-go for each state, to use as
        the targets for the value function.

        The "last_val" argument should be 0 if the trajectory ended
        because the agent reached a terminal state (died), and otherwise
        should be V(s_T), the value function estimated for the last state.
        This allows us to bootstrap the reward-to-go calculation to account
        for timesteps beyond the arbitrary episode horizon (or epoch cutoff).
        """

        path_slice = slice(self.path_start_idx, self.ptr)
        rews_extr = np.append(self.rew_extr_buf[path_slice], last_val_extr)
        rews_intr = np.append(self.rew_intr_buf[path_slice], last_val_intr)
        rews = rews_extr + rews_intr
        vals = np.append(self.val_extr_buf[path_slice], last_val_extr) + np.append(
            self.val_intr_buf[path_slice], last_val_intr
        )

        # the next two lines implement GAE-Lambda advantage calculation
        deltas = rews[:-1] + self.gamma * vals[1:] - vals[:-1]
        self.adv_buf[path_slice] = core.discount_cumsum(deltas, self.gamma * self.lam)

        # the next line computes rewards-to-go, to be targets for the value function
        self.ret_extr_buf[path_slice] = core.discount_cumsum(rews_extr, self.gamma)[:-1]
        self.ret_intr_buf[path_slice] = core.discount_cumsum(rews_intr, self.gamma)[:-1]

        self.path_start_idx = self.ptr

    def get(self):
        """
        Call this at the end of an epoch to get all of the data from
        the buffer, with advantages appropriately normalized (shifted to have
        mean zero and std one). Also, resets some pointers in the buffer.
        """
        assert self.ptr == self.max_size  # buffer has to be full before you can get
        self.ptr, self.path_start_idx = 0, 0
        # the next two lines implement the advantage normalization trick
        adv_mean, adv_std = mpi_statistics_scalar(self.adv_buf)
        self.adv_buf = (self.adv_buf - adv_mean) / adv_std
        data = dict(
            obs=self.obs_buf,
            act=self.act_buf,
            ret_extr=self.ret_extr_buf,
            ret_intr=self.ret_intr_buf,
            adv=self.adv_buf,
            logp=self.logp_buf,
        )
        return {k: torch.as_tensor(v, dtype=torch.float32) for k, v in data.items()}


def ppo(
    env_fn,
    actor_critic=core.MLPActorCritic2Heads,
    ac_kwargs=dict(),
    seed=0,
    steps_per_epoch=4000,
    epochs=100,
    epochs_rnd_warmup=1,
    gamma=0.99,
    clip_ratio=0.2,
    pi_lr=3e-4,
    vf_lr=1e-3,
    rnd_lr=1e-3,
    train_pi_iters=80,
    train_v_iters=80,
    train_rnd_iters=80,
    lam=0.97,
    max_ep_len=200,
    target_kl=0.01,
    logger_kwargs=dict(),
    save_freq=10,
    scale_reward=100,
    only_intr=False,
    norm_intr=False,
    alpha_std_est=0.05,
    single_head=False,
):
    """
    Proximal Policy Optimization (by clipping),

    with early stopping based on approximate KL

    Args:
        env_fn : A function which creates a copy of the environment.
            The environment must satisfy the OpenAI Gym API.

        actor_critic: The constructor method for a PyTorch Module with a
            ``step`` method, an ``act`` method, a ``pi`` module, and a ``v``
            module. The ``step`` method should accept a batch of observations
            and return:

            ===========  ================  ======================================
            Symbol       Shape             Description
            ===========  ================  ======================================
            ``a``        (batch, act_dim)  | Numpy array of actions for each
                                           | observation.
            ``v``        (batch,)          | Numpy array of value estimates
                                           | for the provided observations.
            ``logp_a``   (batch,)          | Numpy array of log probs for the
                                           | actions in ``a``.
            ===========  ================  ======================================

            The ``act`` method behaves the same as ``step`` but only returns ``a``.

            The ``pi`` module's forward call should accept a batch of
            observations and optionally a batch of actions, and return:

            ===========  ================  ======================================
            Symbol       Shape             Description
            ===========  ================  ======================================
            ``pi``       N/A               | Torch Distribution object, containing
                                           | a batch of distributions describing
                                           | the policy for the provided observations.
            ``logp_a``   (batch,)          | Optional (only returned if batch of
                                           | actions is given). Tensor containing
                                           | the log probability, according to
                                           | the policy, of the provided actions.
                                           | If actions not given, will contain
                                           | ``None``.
            ===========  ================  ======================================

            The ``v`` module's forward call should accept a batch of observations
            and return:

            ===========  ================  ======================================
            Symbol       Shape             Description
            ===========  ================  ======================================
            ``v``        (batch,)          | Tensor containing the value estimates
                                           | for the provided observations. (Critical:
                                           | make sure to flatten this!)
            ===========  ================  ======================================


        ac_kwargs (dict): Any kwargs appropriate for the ActorCritic object
            you provided to PPO.

        seed (int): Seed for random number generators.

        steps_per_epoch (int): Number of steps of interaction (state-action pairs)
            for the agent and the environment in each epoch.

        epochs (int): Number of epochs of interaction (equivalent to
            number of policy updates) to perform.

        epochs_rnd_warmup (int): Number of epochs of training RND before starting training agent.

        gamma (float): Discount factor. (Always between 0 and 1.)

        clip_ratio (float): Hyperparameter for clipping in the policy objective.
            Roughly: how far can the new policy go from the old policy while
            still profiting (improving the objective function)? The new policy
            can still go farther than the clip_ratio says, but it doesn't help
            on the objective anymore. (Usually small, 0.1 to 0.3.) Typically
            denoted by :math:`\\epsilon`.

        pi_lr (float): Learning rate for policy optimizer.

        vf_lr (float): Learning rate for value function optimizer.

        rnd_lr (float): Learning rate for RND optimizer.

        train_pi_iters (int): Maximum number of gradient descent steps to take
            on policy loss per epoch. (Early stopping may cause optimizer
            to take fewer than this.)

        train_v_iters (int): Number of gradient descent steps to take on
            value function per epoch.

        train_rnd_iters (int): Number of gradient descent steps to take on
            RND per epoch.

        lam (float): Lambda for GAE-Lambda. (Always between 0 and 1,
            close to 1.)

        max_ep_len (int): Maximum length of trajectory / episode / rollout.

        target_kl (float): Roughly what KL divergence we think is appropriate
            between new and old policies after an update. This will get used
            for early stopping. (Usually small, 0.01 or 0.05.)

        logger_kwargs (dict): Keyword args for EpochLogger.

        save_freq (int): How often (in terms of gap between epochs) to save
            the current policy and value function.

        scale_reward (float): total_reward = extr_reward + scale_reward*intr_reward

    """

    # Special function to avoid certain slowdowns from PyTorch + MPI combo.
    setup_pytorch_for_mpi()

    # Set up logger and save configuration
    logger = EpochLogger(**logger_kwargs)
    logger.save_config(locals())

    # Random seed
    seed += 10000 * proc_id()
    torch.manual_seed(seed)
    np.random.seed(seed)

    # Instantiate environment
    env = env_fn()
    obs_dim = env.observation_space.shape
    act_dim = env.action_space.shape

    # Create actor-critic module
    ac = actor_critic(env.observation_space, env.action_space, **ac_kwargs)

    # Sync params across processes
    sync_params(ac)

    # Create RND module and optimizer
    rnd = RND(obs_dim[0], (32, 32), nn.Sigmoid)
    sync_params(rnd)
    rnd_optimizer = Adam(rnd.predictor_network.parameters(), lr=rnd_lr)
    # Create running estimator for reward normalization
    reward_std_estimator = core.running_exp_estimator(alpha_std_est)

    # Count variables
    var_counts = tuple(core.count_vars(module) for module in [ac.pi, ac.v_extr, ac.v_intr, rnd.predictor_network])
    logger.log("\nNumber of parameters: \t pi: %d, \t v_extr: %d, \t v_intr: %d, \t rnd: %d\n" % var_counts)

    local_steps_per_epoch = int(steps_per_epoch / num_procs())
    o = env.reset()
    # Train RND on random agent for 'epochs_rnd_warmup' epochs
    for epoch in range(epochs_rnd_warmup):
        for t in range(local_steps_per_epoch):
            a, _, _, _ = ac.step(torch.as_tensor(o, dtype=torch.float32))

            next_o, r, d, _ = env.step(a)
            rnd_loss = rnd.loss(torch.as_tensor(next_o, dtype=torch.float32))
            reward_std_estimator.update(rnd_loss.item())
            rnd_optimizer.zero_grad()

            rnd_loss.backward()
            mpi_avg_grads(rnd.predictor_network)  # average grads across MPI processes
            rnd_optimizer.step()

    # Set up experience buffer
    buf = PPOBuffer(obs_dim, act_dim, local_steps_per_epoch, gamma, lam)

    # Set up function for computing PPO policy loss
    def compute_loss_pi(data):
        obs, act, adv, logp_old = data["obs"], data["act"], data["adv"], data["logp"]

        # Policy loss
        pi, logp = ac.pi(obs, act)
        ratio = torch.exp(logp - logp_old)
        clip_adv = torch.clamp(ratio, 1 - clip_ratio, 1 + clip_ratio) * adv
        loss_pi = -(torch.min(ratio * adv, clip_adv)).mean()

        # Useful extra info
        approx_kl = (logp_old - logp).mean().item()
        ent = pi.entropy().mean().item()
        clipped = ratio.gt(1 + clip_ratio) | ratio.lt(1 - clip_ratio)
        clipfrac = torch.as_tensor(clipped, dtype=torch.float32).mean().item()
        pi_info = dict(kl=approx_kl, ent=ent, cf=clipfrac)

        return loss_pi, pi_info

    # Set up functions for computing value loss
    def compute_loss_v_extr(data):
        obs, ret = data["obs"], data["ret_extr"]
        return ((ac.v_extr(obs) - ret) ** 2).mean()

    def compute_loss_v_intr(data):
        obs, ret = data["obs"], data["ret_intr"]
        return ((ac.v_intr(obs) - ret) ** 2).mean()

    # Set up optimizers for policy and value function
    pi_optimizer = Adam(ac.pi.parameters(), lr=pi_lr)
    vf_extr_optimizer = Adam(ac.v_extr.parameters(), lr=vf_lr)
    if not single_head:
        vf_intr_optimizer = Adam(ac.v_intr.parameters(), lr=vf_lr)

    # Set up model saving
    logger.setup_pytorch_saver(ac)

    def update(epoch):
        data = buf.get()

        pi_l_old, pi_info_old = compute_loss_pi(data)
        pi_l_old = pi_l_old.item()
        v_extr_l_old = compute_loss_v_extr(data).item()
        if not single_head:
            v_intr_l_old = compute_loss_v_intr(data).item()
        loss_rnd_old = rnd.loss(data["obs"]).item()

        # Train policy with multiple steps of gradient descent
        for i in range(train_pi_iters):
            pi_optimizer.zero_grad()
            loss_pi, pi_info = compute_loss_pi(data)
            kl = mpi_avg(pi_info["kl"])
            if kl > 1.5 * target_kl:
                logger.log("Early stopping at step %d due to reaching max kl." % i)
                break
            loss_pi.backward()
            mpi_avg_grads(ac.pi)  # average grads across MPI processes
            pi_optimizer.step()
        logger.store(StopIter=i)

        # Value function learning
        for i in range(train_v_iters):
            vf_extr_optimizer.zero_grad()
            loss_v_extr = compute_loss_v_extr(data)
            loss_v_extr.backward()
            mpi_avg_grads(ac.v_extr)  # average grads across MPI processes
            vf_extr_optimizer.step()

        if not single_head:
            for i in range(train_v_iters):
                vf_intr_optimizer.zero_grad()
                loss_v_intr = compute_loss_v_intr(data)
                loss_v_intr.backward()
                mpi_avg_grads(ac.v_intr)  # average grads across MPI processes
                vf_intr_optimizer.step()

        for i in range(train_rnd_iters):
            rnd_optimizer.zero_grad()
            loss_rnd = rnd.loss(data["obs"])
            loss_rnd.backward()
            mpi_avg_grads(rnd.predictor_network)  # average grads across MPI processes
            rnd_optimizer.step()

        # Log changes from update
        kl, ent, cf = pi_info["kl"], pi_info_old["ent"], pi_info["cf"]
        logger.store(
            LossPi=pi_l_old,
            LossV_extr=v_extr_l_old,
            LossRND=loss_rnd_old,
            KL=kl,
            Entropy=ent,
            ClipFrac=cf,
            DeltaLossPi=(loss_pi.item() - pi_l_old),
            DeltaLossV_extr=(loss_v_extr.item() - v_extr_l_old),
            DeltaLossRND=(loss_rnd.item() - loss_rnd_old),
        )
        if not single_head:
            logger.store(LossV_intr=v_intr_l_old, DeltaLossV_intr=(loss_v_intr.item() - v_intr_l_old))

    # Prepare for interaction with environment
    start_time = time.time()
    o, ep_ret_extr, ep_ret_intr, ep_len = env.reset(), 0, 0, 0

    # Main loop: collect experience in env and update/log each epoch
    for epoch in range(epochs):
        for t in range(local_steps_per_epoch):
            a, v_extr, v_intr, logp = ac.step(torch.as_tensor(o, dtype=torch.float32))

            next_o, r_extr, d, _ = env.step(a)
            rnd_reward = rnd.reward(torch.as_tensor(next_o, dtype=torch.float32))
            if norm_intr:
                reward_std_estimator.update(rnd_reward)
                r_intr = rnd_reward / reward_std_estimator.get_std()
                logger.store(EpRet_exp_std=reward_std_estimator.get_std())
            else:
                r_intr = rnd_reward

            # save and log
            ep_ret_extr += r_extr
            ep_ret_intr += r_intr
            ep_len += 1

            if only_intr:
                r_extr = 0
            if single_head:
                buf.store(o, a, r_extr + scale_reward * r_intr, 0, v_extr, 0, logp)
            else:
                buf.store(o, a, r_extr, scale_reward * r_intr, v_extr, v_intr, logp)

            logger.store(VVals_extr=v_extr, VVals_intr=v_intr)

            # Update obs (critical!)
            o = next_o

            timeout = ep_len == max_ep_len
            terminal = d or timeout
            epoch_ended = t == local_steps_per_epoch - 1

            if terminal or epoch_ended:
                # if epoch_ended and not(terminal):
                #     print('Warning: trajectory cut off by epoch at %d steps.' % ep_len, flush=True)
                #     logger.log('Warning: trajectory cut off by epoch at %d steps.' % ep_len)

                _, v_extr, v_intr, _ = ac.step(torch.as_tensor(o, dtype=torch.float32))
                # if trajectory reached terminal state, value_extr target is zero, else bootstrap value target
                if not (timeout or epoch_ended):
                    v_extr = 0
                if single_head:
                    buf.finish_path(v_extr + v_intr, 0)
                else:
                    buf.finish_path(v_extr, v_intr)
                if terminal:
                    # only save EpRet / EpLen if trajectory finished
                    logger.store(EpRet_extr=ep_ret_extr, EpLen=ep_len, EpRet_intr=ep_ret_intr)
                o, ep_ret_extr, ep_ret_intr, ep_len = env.reset(), 0, 0, 0

        # Save model
        if (epoch % save_freq == 0) or (epoch == epochs - 1):
            logger.save_state({"env": env}, None)

        # Perform PPO update!
        update(epoch)

        # Log info about epoch
        logger.log_tabular("Epoch", epoch)
        logger.log_tabular("EpRet_extr", with_min_and_max=True)
        logger.log_tabular("EpRet_intr", average_only=True)
        if norm_intr:
            logger.log_tabular("EpRet_exp_std", average_only=True)
        logger.log_tabular("EpLen", average_only=True)
        logger.log_tabular("VVals_extr", average_only=True)
        if not single_head:
            logger.log_tabular("VVals_intr", average_only=True)
        logger.log_tabular("LossPi", average_only=True)
        logger.log_tabular("LossV_extr", average_only=True)
        if not single_head:
            logger.log_tabular("LossV_intr", average_only=True)
        logger.log_tabular("LossRND", average_only=True)
        logger.log_tabular("DeltaLossPi", average_only=True)
        logger.log_tabular("DeltaLossV_extr", average_only=True)
        if not single_head:
            logger.log_tabular("DeltaLossV_intr", average_only=True)
        logger.log_tabular("TotalEnvInteracts", (epoch + 1) * steps_per_epoch)
        logger.log_tabular("Entropy", average_only=True)
        logger.log_tabular("KL", average_only=True)
        logger.log_tabular("ClipFrac", average_only=True)
        logger.log_tabular("StopIter", average_only=True)
        logger.log_tabular("Time", time.time() - start_time)

        logger.dump_tabular()


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--env", type=str, default="MountainCar-v0")
    parser.add_argument("--hid", type=int, default=64)
    parser.add_argument("--l", type=int, default=2)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--seed", "-s", type=int, default=0)
    parser.add_argument("--cpu", type=int, default=12)
    parser.add_argument("--steps", type=int, default=4000)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--exp_name", type=str, default="ppo_rnd")

    parser.add_argument("--train_pi_iters", type=int, default=80)
    parser.add_argument("--train_v_iters", type=int, default=80)
    parser.add_argument("--train_rnd_iters", type=int, default=80)

    parser.add_argument("--scale_reward", type=int, default=100)
    parser.add_argument("--only_intr", type=bool, default=False)
    parser.add_argument("--norm_intr", type=bool, default=True)

    args = parser.parse_args()

    mpi_fork(args.cpu)  # run parallel code with mpi

    from spinup.utils.run_utils import setup_logger_kwargs

    logger_kwargs = setup_logger_kwargs(args.exp_name, args.seed)

    ppo(
        lambda: gym.make(args.env),
        actor_critic=core.MLPActorCritic2Heads,
        ac_kwargs=dict(hidden_sizes=[args.hid] * args.l),
        gamma=args.gamma,
        seed=args.seed,
        steps_per_epoch=args.steps,
        epochs=args.epochs,
        logger_kwargs=logger_kwargs,
        train_pi_iters=args.train_pi_iters,
        train_v_iters=args.train_v_iters,
        train_rnd_iters=args.train_rnd_iters,
        scale_reward=args.scale_reward,
        only_intr=args.only_intr,
        norm_intr=args.norm_intr,
    )
