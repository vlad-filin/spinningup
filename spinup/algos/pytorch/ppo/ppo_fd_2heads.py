import numpy as np
import torch
from torch.optim import Adam
import gym
import time
import spinup.algos.pytorch.ppo.core as core
from spinup.utils.logx import EpochLogger
from spinup.utils.mpi_pytorch import setup_pytorch_for_mpi, sync_params, mpi_avg_grads
from spinup.utils.mpi_tools import mpi_fork, mpi_avg, proc_id, mpi_statistics_scalar, num_procs


class PPOBuffer:
    """
    A buffer for storing trajectories experienced by a PPO agent interacting
    with the environment, and using Generalized Advantage Estimation (GAE-Lambda)
    for calculating the advantages of state-action pairs.
    """

    def __init__(self, obs_dim, act_dim, size, gamma=0.99, lam=0.95):
        self.obs_buf = np.zeros(core.combined_shape(
            size, obs_dim), dtype=np.float32)
        self.act_buf = np.zeros(core.combined_shape(
            size, act_dim), dtype=np.float32)
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
        assert self.ptr < self.max_size     # buffer has to have room so you can store
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
        vals = np.append(self.val_extr_buf[path_slice], last_val_extr) + \
            np.append(self.val_intr_buf[path_slice], last_val_intr)
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
        assert self.ptr == self.max_size    # buffer has to be full before you can get
        self.ptr, self.path_start_idx = 0, 0
        # the next two lines implement the advantage normalization trick
        adv_mean, adv_std = mpi_statistics_scalar(self.adv_buf)
        self.adv_buf = (self.adv_buf - adv_mean) / adv_std
        data = dict(obs=self.obs_buf, act=self.act_buf, ret_extr=self.ret_extr_buf,
                    ret_intr=self.ret_intr_buf, adv=self.adv_buf, logp=self.logp_buf)
        return {k: torch.as_tensor(v, dtype=torch.float32) for k, v in data.items()}


# ToDo: add forward_dynamics parameter
def ppo(env_fn, actor_critic=core.MLPActorCritic2Heads, ac_kwargs=dict(), seed=0,
        steps_per_epoch=4000, epochs=50, gamma=0.99, clip_ratio=0.2, pi_lr=3e-4,
        vf_lr=1e-3, train_pi_iters=80, train_v_iters=80, lam=0.97, max_ep_len=1000,
        target_kl=0.01, logger_kwargs=dict(), save_freq=10,
        fd_kwargs=dict(), fd_scaling_factor=20000, fd_lr=1e-3, train_fd_iters=80):
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

        gamma (float): Discount factor. (Always between 0 and 1.)

        clip_ratio (float): Hyperparameter for clipping in the policy objective.
            Roughly: how far can the new policy go from the old policy while
            still profiting (improving the objective function)? The new policy
            can still go farther than the clip_ratio says, but it doesn't help
            on the objective anymore. (Usually small, 0.1 to 0.3.) Typically
            denoted by :math:`\epsilon`.

        pi_lr (float): Learning rate for policy optimizer.

        vf_lr (float): Learning rate for value function optimizer.

        train_pi_iters (int): Maximum number of gradient descent steps to take
            on policy loss per epoch. (Early stopping may cause optimizer
            to take fewer than this.)

        train_v_iters (int): Number of gradient descent steps to take on
            value function per epoch.

        lam (float): Lambda for GAE-Lambda. (Always between 0 and 1,
            close to 1.)

        max_ep_len (int): Maximum length of trajectory / episode / rollout.

        target_kl (float): Roughly what KL divergence we think is appropriate
            between new and old policies after an update. This will get used
            for early stopping. (Usually small, 0.01 or 0.05.)

        logger_kwargs (dict): Keyword args for EpochLogger.

        save_freq (int): How often (in terms of gap between epochs) to save
            the current policy and value function.

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
    # ToDo create forward_dynamics module
    fd = core.ForwardDynamics(env.observation_space, env.action_space, scaling_factor=fd_scaling_factor,
                              **fd_kwargs)

    # Sync params across processes
    sync_params(ac)
    sync_params(fd)

    # Count variables
    var_counts = tuple(core.count_vars(module)
                       for module in [ac.pi, ac.v_extr, ac.v_intr, fd.net])
    logger.log(
        '\nNumber of parameters: \t pi: %d, \t v_extr: %d,\t v_intr: %d \t fd: %d\n' % var_counts)

    # Set up experience buffer
    local_steps_per_epoch = int(steps_per_epoch / num_procs())
    buf = PPOBuffer(obs_dim, act_dim, local_steps_per_epoch, gamma, lam)

    # Set up function for computing PPO policy loss
    def compute_loss_pi(data):
        obs, act, adv, logp_old = data['obs'], data['act'], data['adv'], data['logp']

        # Policy loss
        pi, logp = ac.pi(obs, act)
        ratio = torch.exp(logp - logp_old)
        clip_adv = torch.clamp(ratio, 1-clip_ratio, 1+clip_ratio) * adv
        loss_pi = -(torch.min(ratio * adv, clip_adv)).mean()

        # Useful extra info
        approx_kl = (logp_old - logp).mean().item()
        ent = pi.entropy().mean().item()
        clipped = ratio.gt(1+clip_ratio) | ratio.lt(1-clip_ratio)
        clipfrac = torch.as_tensor(clipped, dtype=torch.float32).mean().item()
        pi_info = dict(kl=approx_kl, ent=ent, cf=clipfrac)

        return loss_pi, pi_info

    # Set up function for computing value loss
    def compute_loss_v_extr(data):
        obs, ret = data['obs'], data['ret_extr']
        return ((ac.v_extr(obs) - ret)**2).mean()

    def compute_loss_v_intr(data):
        obs, ret = data['obs'], data['ret_intr']
        return ((ac.v_intr(obs) - ret)**2).mean()

    def compute_intr_loss(data):
        obs, act = data['obs'][:-2], data['act'][:-2]
        next_obs = data['obs'][1:-1]
        losses = fd.loss(obs, next_obs, act)
        return losses.sum(dim=0)
    # Set up optimizers for policy and value function
    pi_optimizer = Adam(ac.pi.parameters(), lr=pi_lr)
    vf_extr_optimizer = Adam(ac.v_extr.parameters(), lr=vf_lr)
    vf_intr_optimizer = Adam(ac.v_intr.parameters(), lr=vf_lr)
    fd_optimizer = Adam(fd.net.parameters(), lr=fd_lr)

    # Set up model saving
    logger.setup_pytorch_saver(ac)

    # ToDo add forwardDynamics loss

    def update():
        data = buf.get()

        pi_l_old, pi_info_old = compute_loss_pi(data)
        pi_l_old = pi_l_old.item()
        v_extr_l_old = compute_loss_v_extr(data).item()
        v_intr_l_old = compute_loss_v_intr(data).item()
        intr_loss_old = compute_intr_loss(data).item()
        for i in range(train_fd_iters):
            fd_optimizer.zero_grad()
            loss_fd = compute_intr_loss(data)
            loss_fd.backward()
            mpi_avg_grads(fd.net)
            fd_optimizer.step()
        # Train policy with multiple steps of gradient descent
        for i in range(train_pi_iters):
            pi_optimizer.zero_grad()
            loss_pi, pi_info = compute_loss_pi(data)
            kl = mpi_avg(pi_info['kl'])
            if kl > 1.5 * target_kl:
                logger.log(
                    'Early stopping at step %d due to reaching max kl.' % i)
                break
            loss_pi.backward()
            mpi_avg_grads(ac.pi)    # average grads across MPI processes
            pi_optimizer.step()
        logger.store(StopIter=i)

        # Value function learning
        for i in range(train_v_iters):
            vf_extr_optimizer.zero_grad()
            loss_v_extr = compute_loss_v_extr(data)
            loss_v_extr.backward()
            mpi_avg_grads(ac.v_extr)    # average grads across MPI processes
            vf_extr_optimizer.step()

        for i in range(train_v_iters):
            vf_intr_optimizer.zero_grad()
            loss_v_intr = compute_loss_v_intr(data)
            loss_v_intr.backward()
            mpi_avg_grads(ac.v_intr)    # average grads across MPI processes
            vf_intr_optimizer.step()

        # Log changes from update
        kl, ent, cf = pi_info['kl'], pi_info_old['ent'], pi_info['cf']
        logger.store(LossPi=pi_l_old, LossV_extr=v_extr_l_old, LossV_intr=v_intr_l_old,
                     KL=kl, Entropy=ent, ClipFrac=cf,
                     DeltaLossPi=(loss_pi.item() - pi_l_old),
                     DeltaLossV_extr=(loss_v_extr.item() - v_extr_l_old),
                     DeltaLossV_intr=(loss_v_intr.item() - v_intr_l_old),
                     IntrLoss=intr_loss_old,
                     DeltaLossIntr=(loss_fd.item() - intr_loss_old))

    # Prepare for interaction with environment
    start_time = time.time()
    o, ep_ret, ep_len = env.reset(), 0, 0
    ep_intr_ret, ep_extr_ret = 0, 0

    # Main loop: collect experience in env and update/log each epoch
    for epoch in range(epochs):
        for t in range(local_steps_per_epoch):
            a, v_extr, v_intr, logp = ac.step(torch.as_tensor(o, dtype=torch.float32))

            next_o, r, d, _ = env.step(a)
            # ToDo calc intr motivation -> buf
            intr_r = fd.reward(o, next_o, a)
            ep_intr_ret += intr_r
            ep_extr_ret += r
            ep_len += 1

            # save and log
            buf.store(o, a, r, intr_r, v_extr, v_intr, logp)
            logger.store(VVals_extr=v_extr, VVals_intr=v_intr)

            # Update obs (critical!)
            o = next_o

            timeout = ep_len == max_ep_len
            terminal = d or timeout
            epoch_ended = t == local_steps_per_epoch-1

            if terminal or epoch_ended:
                if epoch_ended and not(terminal):
                    print('Warning: trajectory cut off by epoch at %d steps.' %
                          ep_len, flush=True)
                _, v_extr, v_intr, _ = ac.step(torch.as_tensor(o, dtype=torch.float32))
                # if trajectory reached terminal state, value_extr target is zero, else bootstrap value target
                if not (timeout or epoch_ended):
                    v_extr = 0
                buf.finish_path(v_extr, v_intr)
                if terminal:
                    # only save EpRet / EpLen if trajectory finished
                    logger.store(EpRet_extr=ep_extr_ret, EpLen=ep_len, EpRet_intr=ep_intr_ret)
                o, ep_ret, ep_len = env.reset(), 0, 0
                ep_intr_ret, ep_extr_ret = 0, 0
        # Save model
        if (epoch % save_freq == 0) or (epoch == epochs-1):
            if (epoch == epochs-1):
                print(f'before save {epoch} == {epochs-1}', flush=True)
            logger.save_state({'env': env}, None)
            if (epoch == epochs-1):
                print(f'after save {epoch} == {epochs-1}', flush=True)

        # Perform PPO update!
        update()

        # Log info about epoch
        logger.log_tabular('Epoch', epoch)
        logger.log_tabular('EpRet_extr', with_min_and_max=True)
        logger.log_tabular('EpRet_intr', with_min_and_max=True)
        logger.log_tabular('EpLen', average_only=True)
        logger.log_tabular('VVals_extr', average_only=True)
        logger.log_tabular('VVals_intr', average_only=True)
        logger.log_tabular('TotalEnvInteracts', (epoch+1)*steps_per_epoch)
        logger.log_tabular('LossPi', average_only=True)
        logger.log_tabular('LossV_extr', average_only=True)
        logger.log_tabular('LossV_intr', average_only=True)
        logger.log_tabular('DeltaLossPi', average_only=True)
        logger.log_tabular('DeltaLossV_extr', average_only=True)
        logger.log_tabular('DeltaLossV_intr', average_only=True)
        logger.log_tabular('Entropy', average_only=True)
        logger.log_tabular('KL', average_only=True)
        logger.log_tabular('ClipFrac', average_only=True)
        logger.log_tabular('StopIter', average_only=True)
        logger.log_tabular('Time', time.time()-start_time)
        logger.log_tabular("IntrLoss", average_only=True)
        logger.dump_tabular()

        # ToDo add logger update
print("last", flush=True)

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--env', type=str, default='HalfCheetah-v2')
    parser.add_argument('--hid', type=int, default=64)
    parser.add_argument('--l', type=int, default=2)
    parser.add_argument('--gamma', type=float, default=0.99)
    parser.add_argument('--seed', '-s', type=int, default=0)
    parser.add_argument('--cpu', type=int, default=4)
    parser.add_argument('--steps', type=int, default=4000)
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--exp_name', type=str, default='ppo')
    # Forward Dynamics params
    parser.add_argument('--fd_hid', type=int, default=64)
    parser.add_argument('--fd_l', type=int, default=2)
    parser.add_argument('--fd_scaling_factor', type=int, default=20000)

    args = parser.parse_args()

    mpi_fork(args.cpu)  # run parallel code with mpi

    from spinup.utils.run_utils import setup_logger_kwargs
    logger_kwargs = setup_logger_kwargs(args.exp_name, args.seed)

    ppo(lambda: gym.make(args.env), actor_critic=core.MLPActorCritic,
        ac_kwargs=dict(hidden_sizes=[args.hid]*args.l), gamma=args.gamma,
        seed=args.seed, steps_per_epoch=args.steps, epochs=args.epochs,
        logger_kwargs=logger_kwargs, fd_scaling_factor=args.fd_scaling_factor,
        fd_kwargs=dict(hidden_sizes=[args.fd_hid]*args.fd_l))
