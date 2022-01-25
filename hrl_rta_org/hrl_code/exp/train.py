import torch
import random
import numpy as np
from gym_microrts.envs.vec_env import MicroRTSGridModeVecEnv
from gym_microrts import microrts_ai
from hrl_code.agents.cnn_agent import RTSAgent, RTSNet
from stable_baselines3.common.vec_env import VecEnvWrapper, VecMonitor, VecVideoRecorder
from numpy.random import choice
from hrl_code.tools.dimension_reduction import get_nodes_vectors
import sys


def sample(l):
    # sample 1 or 2 from logits [0, 1 ,1, 0] but not 0 or 3
    if sum(l) == 0:
        return 0
    return choice(range(len(l)), p=l / sum(l))


def get_units_number(unit_type, bef_obs, ind_obs):
    return int(bef_obs.permute((0, 3, 1, 2))[ind_obs][unit_type].sum())


def init_seeds(torch_seed=0, seed=0):
    torch.manual_seed(torch_seed)  # sets the seed for generating random numbers.
    torch.cuda.manual_seed(torch_seed)  # Sets the seed for generating random numbers for the current GPU.
    torch.cuda.manual_seed_all(torch_seed)  # Sets the seed for generating random numbers on all GPUs.
    if seed == 0:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


if __name__ == '__main__':
    seed = 0

    num_envs = 1

    device = torch.device('cuda:0' if torch.cuda.is_available() and True else 'cpu')
    path_pt = './model/gnn53/2021122901.pt'
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    ais = []
    for i in range(num_envs):
        ais.append(microrts_ai.naiveMCTSAI)

    size = 10
    map_path = "maps/24x24/basesWorkers24x24.xml"
    if size == 10:
        map_path = "maps/10x10/basesWorkers10x10.xml"
    elif size == 16:
        map_path = "maps/16x16/basesWorkers16x16.xml"

    init_seeds()
    envs = MicroRTSGridModeVecEnv(
        num_selfplay_envs=0,
        num_bot_envs=num_envs,
        max_steps=5000,
        render_theme=2,
        ai2s=[microrts_ai.tiamat for _ in range(num_envs)],
        map_paths=[map_path for _ in range(num_envs)],
        reward_weight=np.array([10.0, 1.0, 1.0, 0.2, 1.0, 4.0])
    )
    envs = VecMonitor(envs)
    next_obs = envs.reset()

    net = RTSNet().to(device)
    agent = RTSAgent(net, action_space=[100, 6, 4, 4, 4, 4, 7, 49])

    # net.load_state_dict(torch.load(path_pt, map_location=device))

    tittle = 'GNN against worker rush in ' + str(size) + 'x' + str(size)

    outcomes = []
    for games in range(100):
        for step in range(5000):
            action = np.zeros((num_envs, size*size, 7))
            envs.render()
            obs = next_obs
            with torch.no_grad():
                invalid_action_masks = torch.tensor(np.array(envs.get_action_mask())).float()
                action_plane_space = envs.action_plane_space.nvec
                unit_mask = torch.where(invalid_action_masks.sum(dim=2) > 0, 1.0, 0.0)
                unit_action, _, _, _ = agent.get_action(torch.Tensor(obs).to(device), unit_masks=unit_mask, action_masks=invalid_action_masks)
            for i in range(num_envs):
                action[i][unit_action[i][0]] = unit_action[i][1:]
            next_obs, rs, ds, infos = envs.step(action)
            if ds[0]:
                if get_units_number(11, torch.Tensor(obs), 0) > get_units_number(12, torch.Tensor(obs), 0):
                    outcomes.append(1)
                else:
                    outcomes.append(0)
                break
