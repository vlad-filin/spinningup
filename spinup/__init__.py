import tensorflow as tf

from spinup.algos.pytorch.ddpg.ddpg import ddpg as ddpg_pytorch
from spinup.algos.pytorch.ppo.ppo import ppo as ppo_pytorch
from spinup.algos.pytorch.ppo.ppo_fd_1head import ppo as ppo_fd_1head_pytorch
from spinup.algos.pytorch.ppo.ppo_fd_2heads import ppo as ppo_fd_2heads_pytorch
from spinup.algos.pytorch.ppo.ppo_rnd import ppo as ppo_rnd_pytorch
from spinup.algos.pytorch.sac.sac import sac as sac_pytorch
from spinup.algos.pytorch.td3.td3 import td3 as td3_pytorch
from spinup.algos.pytorch.trpo.trpo import trpo as trpo_pytorch
from spinup.algos.pytorch.vpg.vpg import vpg as vpg_pytorch

# Algorithms
from spinup.algos.tf1.ddpg.ddpg import ddpg as ddpg_tf1
from spinup.algos.tf1.ppo.ppo import ppo as ppo_tf1
from spinup.algos.tf1.sac.sac import sac as sac_tf1
from spinup.algos.tf1.td3.td3 import td3 as td3_tf1
from spinup.algos.tf1.trpo.trpo import trpo as trpo_tf1
from spinup.algos.tf1.vpg.vpg import vpg as vpg_tf1

# Loggers
from spinup.utils.logx import EpochLogger, Logger

# Version
from spinup.version import __version__

tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
