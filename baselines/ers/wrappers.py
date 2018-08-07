from collections import deque
from functools import reduce
from operator import mul

import gym
import numpy as np
from gym import error

from baselines import logger


class CartPoleWrapper(gym.Wrapper):
    def __init__(self, env):
        super().__init__(env)
        self.action_space = gym.spaces.Box(-1., 1., shape=[1])

    def step(self, action):
        if action[0] < 0:
            a = 0
        else:
            a = 1
        return super().step(a)


class ActionSpaceNormalizeWrapper(gym.Wrapper):
    def __init__(self, env):
        super().__init__(env)
        self._ac_low = self.action_space.low
        self._ac_high = self.action_space.high
        self.action_space = gym.spaces.Box(
            -1, 1, shape=self.env.action_space.shape, dtype=np.float32)

    def step(self, action):
        action = np.clip(action, -1, 1)
        action_correct = self._ac_low + \
            (self._ac_high - self._ac_low) * (action + 1) / 2
        return super().step(action_correct)


class LinearFrameStackWrapper(gym.Wrapper):
    k = 3

    def __init__(self, env, k=3):
        super().__init__(env)
        k = LinearFrameStackWrapper.k
        self.k = k
        self.frames = deque([], maxlen=k)
        space = env.observation_space  # type: gym.spaces.Box
        assert len(space.shape) == 1  # can only stack 1-D frames
        self.observation_space = gym.spaces.Box(
            low=np.array(list(space.low) * k), high=np.array(list(space.high) * k))

    def reset(self):
        """Clear buffer and re-fill by duplicating the first observation."""
        ob = self.env.reset()
        for _ in range(self.k):
            self.frames.append(ob)
        return self._observation()

    def step(self, action):
        ob, reward, done, info = self.env.step(action)
        self.frames.append(ob)
        return self._observation(), reward, done, info

    def _observation(self):
        assert len(self.frames) == self.k
        obs = np.concatenate(self.frames, axis=0)
        assert list(np.shape(obs)) == list(self.observation_space.shape)
        return obs


class BipedalWrapper(gym.Wrapper):
    max_episode_length = 400

    def __init__(self, env):
        super().__init__(env)
        self.frame_count = 0

    def reset(self):
        self.frame_count = 0
        return self.env.reset()

    def step(self, action):
        obs, reward, done, info = self.env.step(action)
        self.frame_count += 1
        if self.frame_count >= self.max_episode_length:
            # reward -= 100
            done = True
        return obs, reward, done, info


class DiscreteToContinousWrapper(gym.Wrapper):
    def __init__(self, env):
        super().__init__(env)
        self.action_space = gym.spaces.Box(0, 1, shape=[env.action_space.n])

    def step(self, action):
        a = np.argmax(action)
        return super().step(a)


class ERSEnvWrapper(gym.Wrapper):
    k = 3

    def __init__(self, env: gym.Env):
        super().__init__(env)
        self.k = ERSEnvWrapper.k
        self.request_heat_maps = deque([], maxlen=self.k)
        self.n_ambs = self.metadata['nambs']
        self.n_bases = env.action_space.shape[0]
        self.action_space = gym.spaces.Box(0, 1, shape=[self.n_bases])
        self.observation_space = gym.spaces.Box(
            0, 1, shape=[self.k * self.n_bases + self.n_bases + 1])

    def compute_alloc(self, action):
        action = np.clip(action, 0, 1)
        remaining = 1
        alloc = np.zeros([self.n_bases])
        for i in range(len(action)):
            alloc[i] = action[i] * remaining
            remaining -= alloc[i]
        alloc[-1] = remaining
        assert all(alloc >= 0) and all(
            alloc <= 1), "alloc is {0}".format(alloc)
        # assert sum(alloc) == 1, "sum is {0}".format(sum(alloc))
        return alloc

    def reset(self):
        """Clear buffer and re-fill by duplicating the first observation."""
        self.obs = self.env.reset()
        for _ in range(self.k):
            self.request_heat_maps.append(self.obs[0:self.n_bases])
        return self._observation()

    def step(self, action):
        # action = self.compute_alloc(action)
        logger.log('alloc: {0}'.format(
            np.round(action * self.n_ambs, 2)), level=logger.DEBUG)
        self.obs, r, d, _ = super().step(action)
        self.request_heat_maps.append(self.obs[0:self.n_bases])
        return self._observation(), r, d, _

    def _observation(self):
        assert len(self.request_heat_maps) == self.k
        obs = np.concatenate((np.concatenate(
            self.request_heat_maps, axis=0), self.obs[self.n_bases:]), axis=0)
        if logger.Logger.CURRENT.level <= logger.DEBUG:
            logger.log('req_heat_map: {0}'.format(
                np.round(self.obs[0:self.n_bases], 2)), level=logger.DEBUG)
        return obs


class ERSEnvImWrapper(gym.Wrapper):
    k = 3

    def __init__(self, env: gym.Env):
        super().__init__(env)
        self.k = ERSEnvImWrapper.k
        self.request_heat_maps = deque([], maxlen=self.k)
        self.n_ambs = self.metadata['nambs']
        self.n_bases = env.action_space.shape[0]
        self.action_space = gym.spaces.Box(0, 1, shape=[self.n_bases])
        shp = env.observation_space.shape
        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=(shp[0], shp[1], self.k + shp[2] - 1))

    def compute_alloc(self, action):
        action = np.clip(action, 0, 1)
        remaining = 1
        alloc = np.zeros([self.n_bases])
        for i in range(len(action)):
            alloc[i] = action[i] * remaining
            remaining -= alloc[i]
        alloc[-1] = remaining
        assert all(alloc >= 0) and all(
            alloc <= 1), "alloc is {0}".format(alloc)
        # assert sum(alloc) == 1, "sum is {0}".format(sum(alloc))
        return alloc

    def reset(self):
        """Clear buffer and re-fill by duplicating the first observation."""
        self.obs = self.env.reset()
        for _ in range(self.k):
            self.request_heat_maps.append(self.obs[:, :, 0:1])
        return self._observation()

    def step(self, action):
        # action = self.compute_alloc(action)
        logger.log('alloc: {0}'.format(
            np.round(action * self.n_ambs, 2)), level=logger.DEBUG)
        self.obs, r, d, _ = super().step(action)
        self.request_heat_maps.append(self.obs[:, :, 0:1])
        return self._observation(), r, d, _

    def _observation(self):
        assert len(self.request_heat_maps) == self.k
        obs = np.concatenate((np.concatenate(
            self.request_heat_maps, axis=2), self.obs[:, :, 1:]), axis=2)
        if logger.Logger.CURRENT.level <= logger.DEBUG:
            logger.log('req_heat_map: {0}'.format(
                np.round(self.obs[:, :, 0], 2)), level=logger.DEBUG)
        assert list(obs.shape) == [21, 21, 5]
        return obs


class ERStoMMDPWrapper(gym.Wrapper):
    def __init__(self, env: gym.Env):
        super().__init__(env)
        self.metadata['nzones'] = self.metadata['nbases']
        self.metadata['nresources'] = self.metadata['nambs']

    def reset(self):
        return self.env.reset()

    def step(self, action):
        return self.env.step(action)


class BSStoMMDPWrapper(gym.Wrapper):
    def __init__(self, env):
        super().__init__(env)
        self.metadata['nzones'] = self.metadata['nzones']
        self.metadata['nresources'] = self.metadata['nbikes']

    def reset(self):
        return self.env.reset()

    def step(self, action):
        return self.env.step(action)


class MMDPObsStackWrapper(gym.Wrapper):
    k = 3

    def __init__(self, env):
        super().__init__(env)
        self.last_k_demands = deque([], maxlen=self.k)
        self.nzones = self.metadata['nzones']
        low = list(self.env.observation_space.low)
        low = low[0:self.nzones] * self.k + low[self.nzones:]
        high = list(self.env.observation_space.high)
        high = high[0:self.nzones] * self.k + high[self.nzones:]
        self.observation_space = gym.spaces.Box(
            low=np.array(low), high=np.array(high), dtype=self.env.observation_space.dtype)

    def _observation(self):
        assert len(self.last_k_demands) == self.k
        obs = np.concatenate((np.concatenate(
            self.last_k_demands, axis=0), self.obs[self.nzones:]), axis=0)
        return obs

    def reset(self):
        """Clear buffer and re-fill by duplicating the first observation."""
        self.obs = self.env.reset()
        for _ in range(self.k):
            self.last_k_demands.append(self.obs[0:self.nzones])
        return self._observation()

    def step(self, action):
        self.obs, r, d, info = self.env.step(action)
        self.last_k_demands.append(self.obs[0:self.nzones])
        return self._observation(), r, d, info


class MMDPInfeasibleActionHandlerWrapper(gym.Wrapper):
    """
    This should be used when constraints are not handled by the actor network.
    Transforms an infeasible action to a feasible action.
    It should wrap action rounder wrapper/action normalizer wrapper.
    """
    def __init__(self, env: gym.Env):
        super().__init__(env)
        self.ac_space = self.action_space  # type: gym.spaces.Box
        self.ac_shape = list(self.ac_space.shape)
        self.calculate_epsilons(self.ac_space.high)

    def calculate_epsilons(self, constraints: np.ndarray):
        '''
        for some epsilons vector,
        our output z needs to be (exp + epsilons)/(sigma + sum(epsilons))
        to satisfy the constraints, we get the following set of linear equations:
        for all i:
            (constraints[i] - 1) * epsilons[i] + constraints[i] * sum(epsilons[except i]) = 1 - constraints[i]
        '''
        if np.any(constraints < 0) or np.any(constraints > 1):
            raise ValueError(
                "constraints needs to be in range [0, 1]")
        if np.sum(constraints) <= 1:
            raise ValueError("sum of constrains need to be greater than 1")

        dimensions = reduce(mul, self.ac_shape, 1)
        constraints = np.asarray(constraints)
        constraints_flat = constraints.flatten()
        # to solve the epsilons linear equation:
        # coefficient matrix:
        coeffs = np.array([[(constraints_flat[row] - 1 if col == row else constraints_flat[row])
                            for col in range(dimensions)] for row in range(dimensions)])
        constants = np.array([1 - constraints_flat[row] for row in range(dimensions)])
        epsilons_flat = np.linalg.solve(coeffs, constants)
        self.epsilons = np.reshape(epsilons_flat, self.ac_shape)
        logger.log("wrapper: episilons are {0}".format(self.epsilons), level=logger.INFO)
        self.epsilons_sigma = np.sum(self.epsilons)

    def softmax_with_non_uniform_individual_constraints(self, inputs: np.ndarray):
        """assumes that inputs lie between range 0 and 1"""
        # y = inputs - tf.reduce_max(inputs, axis=1, keepdims=True)
        # y = tf.minimum(inputs, 0)
        # exp = tf.exp(y)
        sigma = np.sum(inputs, axis=-1, keepdims=True)
        return (inputs + self.epsilons) / (sigma + self.epsilons_sigma)

    def reset(self):
        return self.env.reset()

    def step(self, action):
        action = self.softmax_with_non_uniform_individual_constraints(action)
        return self.env.step(action)


class MMDPActionRounder:
    def __init__(self, env: gym.Env):
        self.env = env
        self.nresources = self.env.metadata['nresources']

    def round_action(self, action):
        # print('inside rounder')
        action = self.get_allocation(action) / self.nresources
        return action

    def get_allocation(self, action):
        if not isinstance(action, np.ndarray):
            action = np.array(action)
        if abs(sum(action) - 1) > 1e-6:
            raise error.InvalidAction(
                "Invalid action. The action must sum to 1. Provided action was {0}".format(action))
        try:
            if np.any(action < -0.0):
                raise ValueError(
                    "Each dimension of action must be >=0. Provided action was {0}".format(action))
        except Exception as e:
            print(e)
            print(action)
            raise e
        allocation_fraction = action * self.nresources
        allocation = np.round(allocation_fraction)
        # print(allocation)
        allocated = np.sum(allocation)
        deficit_per_zone = allocation_fraction - allocation
        deficit = self.nresources - allocated
        # print('deficit: {0}'.format(deficit))
        while deficit != 0:
            increase = int(deficit > 0) - int(deficit < 0)
            # print('increase: {0}'.format(increase))
            target_zone = np.argmax(increase * deficit_per_zone)
            # print('target zone: {0}'.format(target_zone))
            allocation[target_zone] += increase
            # print('alloction: {0}'.format(allocation))
            allocated += increase
            deficit_per_zone[target_zone] -= increase
            deficit -= increase
            # print('deficit: {0}'.format(deficit))
        return allocation


class MMDPActionRounderWrapper(gym.Wrapper):
    """Must wrap MMDPActionSpaceNormalizerWrapper. i.e. assumes action space is already normalized"""

    def __init__(self, env: gym.Env):
        super().__init__(env)
        self.action_rounder = MMDPActionRounder(env)

    def reset(self):
        return self.env.reset()

    def step(self, action):
        # print("inside action round wrapper")
        rounded_action = self.action_rounder.round_action(action)
        return self.env.step(rounded_action)


class MMDPActionSpaceNormalizerWrapper(gym.Wrapper):

    def __init__(self, env: gym.Env):
        super().__init__(env)
        self.nzones = self.metadata['nzones']
        self.nresources = self.metadata['nresources']
        ac_space_low = np.zeros(env.action_space.shape)
        ac_space_high = env.action_space.high / self.nresources
        assert np.all(
            ac_space_high <= 1), 'Invalid action space. Individual capacities must be less than or equal the global capacity'
        assert np.sum(
            ac_space_high) > 1, 'Invalid action space. Sum of individual capacities must be greater than global capacity'
        self.action_space = gym.spaces.Box(
            low=ac_space_low, high=ac_space_high, dtype=np.float32)
        logger.log('ac space high: {0}'.format(self.action_space.high))

    def reset(self):
        self.obs = self.env.reset()
        return self.obs

    def step(self, action):
        # action = [1 / self.nzones] * self.nzones
        allocation_fraction = action * self.nresources
        allocation = np.round(allocation_fraction)
        if np.sum(allocation) != self.nresources:
            raise error.InvalidAction(
                "Invalid action. The action, when rounded, should sum to nresources. Provided action was {0}".format(allocation))
        logger.log("action: {0}".format(allocation), level=logger.DEBUG)
        self.obs, r, d, info = self.env.step(allocation)
        return self.obs, r, d, info


class MMDPObsNormalizeWrapper(gym.Wrapper):
    """Must be used before MMDPObsStackWrapper"""
    demand_log_transform_t = 0.005
    alloc_log_transform_t = 5

    def __init__(self, env):
        super().__init__(env)
        self.nzones = self.env.metadata['nzones']
        self.max_demand = self.env.observation_space.high[0:self.nzones]
        self.max_alloc = self.env.observation_space.high[self.nzones:2 * self.nzones]
        self.max_time = self.env.observation_space.high[-1]
        self.observation_space = gym.spaces.Box(
            low=0, high=1, shape=self.env.observation_space.shape, dtype=np.float32)
        self.metadata['alloc_log_transform_t'] = self.alloc_log_transform_t
        self.metadata['demand_log_transform_t'] = self.demand_log_transform_t
        self.metadata['max_demand'] = self.max_demand
        self.metadata['max_alloc'] = self.max_alloc
        logger.log('obs space high: {0}'.format(self.observation_space.high))

    def _transform(self, x, max_x, t):
        # return self._log_transform(x, max_x, t)
        return x / max_x

    def _log_transform(self, x, max_x, t):
        return np.log(1 + x / t) / np.log(1 + max_x / t)

    def _transform_obs(self, obs):
        obs = np.copy(obs)
        logger.log('demand: {0}'.format(
            np.round(obs[0:self.nzones], 2)), level=logger.DEBUG)
        obs[0:self.nzones] = self._transform(
            obs[0:self.nzones], self.max_demand, self.demand_log_transform_t)
        logger.log('cur_alloc: {0}'.format(
            np.round(obs[self.nzones: 2 * self.nzones], 2)), level=logger.DEBUG)
        obs[self.nzones: 2 * self.nzones] = self._transform(
            obs[self.nzones: 2 * self.nzones], self.max_alloc, self.alloc_log_transform_t)
        logger.log('cur_time: {0}'.format(
            np.round(obs[-1], 2)), level=logger.DEBUG)
        obs[-1] = obs[-1] / self.max_time
        return obs

    def reset(self):
        return self._transform_obs(self.env.reset())

    def step(self, action):
        obs, r, d, info = self.env.step(action)
        return self._transform_obs(obs), r, d, info
