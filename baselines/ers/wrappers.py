from collections import deque

import gym
import numpy as np

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

    def _reset(self):
        """Clear buffer and re-fill by duplicating the first observation."""
        ob = self.env.reset()
        for _ in range(self.k):
            self.frames.append(ob)
        return self._observation()

    def _step(self, action):
        ob, reward, done, info = self.env.step(action)
        self.frames.append(ob)
        return self._observation(), reward, done, info

    def _observation(self):
        assert len(self.frames) == self.k
        obs = np.concatenate(self.frames, axis=0)
        assert list(np.shape(obs)) == list(self.observation_space.shape)
        return obs


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

    def _reset(self):
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

    def _reset(self):
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
