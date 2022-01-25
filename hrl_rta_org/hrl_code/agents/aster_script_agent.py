import numpy as np
from hrl_code.tools.lower_level_script import astar_script, simple_astar_path_finding


class ScriptAgent:
    def __init__(self, size, base, op_base, barracks, op_barracks, resource, action_size):
        self.size = size
        self.base = base
        self.op_base = op_base
        self.barracks = barracks
        self.op_barracks = op_barracks
        self.resource = resource
        self.action_size = action_size

    def get_action(self, states, masks):
        """

        :param states:
        :param masks:
        :return:
        """
        actions = np.zeros(self.action_size)
        for i in range(len(states)):
            units = masks.sum(dim=1)
            for pos in range(len(units)):
                target = pos
                if units(pos) > 0:
                    pos_x = pos % self.size
                    pos_y = pos//self.size
                    if pos_x < 3 and pos_y < 3:
                        if states[i][pos][1] == 1:
                            target = self.base
                        else:
                            target = self.resource[0]
                    elif self.op_barracks > 0:
                        target = self.op_barracks
                    else:
                        target = self.op_base
                    action = astar_script(states[i], pos, target, self.size, masks[i][pos])



