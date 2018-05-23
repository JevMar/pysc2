import random
import os.path

import numpy as np
import pandas as pd

from pysc2.agents import base_agent
from pysc2.lib import actions
from pysc2.lib import features

_NO_OP = actions.FUNCTIONS.no_op.id
_SELECT_POINT = actions.FUNCTIONS.select_point.id
_TRAIN_MARINE = actions.FUNCTIONS.Train_Marine_quick.id
_TRAIN_REAPER = actions.FUNCTIONS.Train_Reaper_quick.id
_TRAIN_MARAUDER = actions.FUNCTIONS.Train_Marauder_quick.id
_TRAIN_GHOST = actions.FUNCTIONS.Train_Ghost_quick.id
_SELECT_ARMY = actions.FUNCTIONS.select_army.id
_ATTACK_MINIMAP = actions.FUNCTIONS.Attack_minimap.id

_PLAYER_RELATIVE = features.SCREEN_FEATURES.player_relative.index
_UNIT_TYPE = features.SCREEN_FEATURES.unit_type.index
_PLAYER_ID = features.SCREEN_FEATURES.player_id.index

_PLAYER_SELF = 1
_PLAYER_HOSTILE = 4

_ARMY_CAP_USED = 3
_ARMY_CAP = 4
_ARMY_SUPPLY = 5
_ARMY_COUNT = 8
_SPENT_MINERALS = 11
_SPENT_VESPENE = 12
_KILLED_UNITS = 5

_TERRAN_COMMANDCENTER = 18
_TERRAN_SUPPLY_DEPOT = 19
_TERRAN_BARRACKS = 21
_NEUTRAL_MINERAL_FIELD = 341

_NOT_QUEUED = [0]
_QUEUED = [1]
_SELECT_ALL = [2]

DATA_FILE = 'sparse_agent_data'

ACTION_DO_NOTHING = 'donothing'
ACTION_BUILD_MARINE = 'buildmarine'
ACTION_BUILD_REAPER = 'buildreaper'
ACTION_BUILD_MARAUDER = 'buildmarauder'
ACTION_BUILD_GHOST = 'buildghost'
ACTION_ATTACK = 'attack'

smart_actions = [
    ACTION_DO_NOTHING,
    ACTION_BUILD_MARINE,
    ACTION_BUILD_REAPER,
    ACTION_BUILD_MARAUDER,
    ACTION_BUILD_GHOST,
    ACTION_ATTACK
]

# from https://github.com/MorvanZhou/Reinforcement-learning-with-tensorflow
class QLearningTable:
    def __init__(self, actions, learning_rate=0.05, reward_decay=0.9, e_greedy=0.9):
        self.actions = actions  # a list
        self.lr = learning_rate
        self.gamma = reward_decay
        self.epsilon = e_greedy
        self.q_table = pd.DataFrame(columns=self.actions, dtype=np.float64)

    def choose_action(self, observation):
        self.check_state_exist(observation)

        if np.random.uniform() < self.epsilon:
            # choose best action
            state_action = self.q_table.ix[observation, :]

            # some actions have the same value
            state_action = state_action.reindex(np.random.permutation(state_action.index))

            action = state_action.idxmax()
        else:
            # choose random action
            action = np.random.choice(self.actions)

        return action

    def learn(self, s, a, r, s_):
        if s == s_:
            return

        self.check_state_exist(s_)
        self.check_state_exist(s)

        q_predict = self.q_table.ix[s, a]

        if s_ != 'terminal':
            q_target = r + self.gamma * self.q_table.ix[s_, :].max()
        else:
            q_target = r  # next state is terminal

        # update
        self.q_table.ix[s, a] += self.lr * (q_target - q_predict)

    def check_state_exist(self, state):
        if state not in self.q_table.index:
            # append new state to q table
            self.q_table = self.q_table.append(
                pd.Series([0] * len(self.actions), index=self.q_table.columns, name=state))


class SparseAgent(base_agent.BaseAgent):
    def __init__(self):
        super(SparseAgent, self).__init__()

        self.qlearn = QLearningTable(actions=list(range(len(smart_actions))))

        self.previous_action = None
        self.previous_state = None

        self.previous_killed = 0
        self.killed_score = 0

        self.cc_y = None
        self.cc_x = None

        self.move_number = 0

        if os.path.isfile(DATA_FILE + '.gz'):
            self.qlearn.q_table = pd.read_pickle(DATA_FILE + '.gz', compression='gzip')

    def splitAction(self, action_id):
        smart_action = smart_actions[action_id]

        x = 0
        y = 0
        if '_' in smart_action:
            smart_action, x, y = smart_action.split('_')

        return (smart_action, x, y)

    def step(self, obs):
        super(SparseAgent, self).step(obs)

        if obs.last():
            reward = obs.reward

            killed_units = obs.observation['score_cumulative'][_KILLED_UNITS]

            if killed_units > self.previous_killed:
                self.killed_score += 0.25

            print('reward,killed,army,score;{0};{1};{2};{3}'.format(reward,
                                                                    self.killed_score,
                                                                    obs.observation['player'][_ARMY_CAP_USED] /
                                                                    obs.observation['player'][_ARMY_CAP],
                                                                    obs.observation["score_cumulative"][0] / 35))

            self.qlearn.learn(str(self.previous_state), self.previous_action, reward, 'terminal')

            self.qlearn.q_table.to_pickle(DATA_FILE + '.gz', 'gzip')

            self.previous_action = None
            self.previous_state = None

            self.killed_score = 0

            self.move_number = 0

            return actions.FunctionCall(_NO_OP, [])

        unit_type = obs.observation['screen'][_UNIT_TYPE]

        if obs.first():
            self.cc_y, self.cc_x = (unit_type == _TERRAN_COMMANDCENTER).nonzero()

        cc_y, cc_x = (unit_type == _TERRAN_COMMANDCENTER).nonzero()

        barracks_y, barracks_x = (unit_type == _TERRAN_BARRACKS).nonzero()

        if self.move_number == 0:
            self.move_number += 1

            killed_units = obs.observation['score_cumulative'][_KILLED_UNITS]

            if killed_units > self.previous_killed:
                self.killed_score += 0.25

            current_state = np.zeros(4)
            current_state[0] = obs.observation['player'][_ARMY_CAP_USED] / obs.observation['player'][_ARMY_CAP]
            current_state[1] = obs.observation["score_cumulative"][0] / 35
            current_state[2] = self.killed_score

            if self.previous_action is not None:
                self.qlearn.learn(str(self.previous_state), self.previous_action, 0, str(current_state))

            rl_action = self.qlearn.choose_action(str(current_state))

            self.previous_state = current_state
            self.previous_action = rl_action
            self.previous_killed = killed_units

            smart_action, x, y = self.splitAction(self.previous_action)

            if smart_action == ACTION_DO_NOTHING:
                return actions.FunctionCall(_NO_OP, [])

            elif smart_action == ACTION_BUILD_MARINE or smart_action == ACTION_BUILD_GHOST or smart_action == ACTION_BUILD_MARAUDER or smart_action == ACTION_BUILD_REAPER:
                if barracks_y.any():
                    i = random.randint(0, len(barracks_y) - 1)
                    target = [barracks_x[i], barracks_y[i]]

                    return actions.FunctionCall(_SELECT_POINT, [_SELECT_ALL, target])

            elif smart_action == ACTION_ATTACK:
                if _SELECT_ARMY in obs.observation['available_actions']:
                    return actions.FunctionCall(_SELECT_ARMY, [_NOT_QUEUED])

        elif self.move_number == 1:
            self.move_number = 0

            smart_action, x, y = self.splitAction(self.previous_action)

            if smart_action == ACTION_DO_NOTHING:
                return actions.FunctionCall(_NO_OP, [])

            elif smart_action == ACTION_BUILD_MARINE:
                if _TRAIN_MARINE in obs.observation['available_actions']:
                    return actions.FunctionCall(_TRAIN_MARINE, [_QUEUED])

            elif smart_action == ACTION_BUILD_GHOST:
                if _TRAIN_GHOST in obs.observation['available_actions']:
                    return actions.FunctionCall(_TRAIN_GHOST, [_QUEUED])

            elif smart_action == ACTION_BUILD_MARAUDER:
                if _TRAIN_MARAUDER in obs.observation['available_actions']:
                    return actions.FunctionCall(_TRAIN_MARAUDER, [_QUEUED])

            elif smart_action == ACTION_BUILD_REAPER:
                if _TRAIN_REAPER in obs.observation['available_actions']:
                    return actions.FunctionCall(_TRAIN_REAPER, [_QUEUED])

            elif smart_action == ACTION_ATTACK:
                if _ATTACK_MINIMAP in obs.observation["available_actions"]:
                    x_offset = random.randint(-1, 1)
                    y_offset = random.randint(-1, 1)

                    return actions.FunctionCall(_ATTACK_MINIMAP, [_NOT_QUEUED, [48, 36]])

        return actions.FunctionCall(_NO_OP, [])
