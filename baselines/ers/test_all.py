import unittest

import numpy as np
from baselines.ers.utils import my_video_schedule
from gym.wrappers.monitor import capped_cubic_video_schedule


class TestVideoSchedules(unittest.TestCase):
    '''Tests my_video_schedule function'''

    total_episodes = 10000

    def test_false_for_negative_video_interval(self):
        '''my_video_schedule should always return false for non-positive video intervals'''
        video_interval = [0, -1, -10]
        for vi in video_interval:
            for ep_no in range(0, self.total_episodes):
                self.assertFalse(my_video_schedule(
                    ep_no, self.total_episodes, vi))
                    
    def test_capped_cubic_for_none_video_interval(self):
        '''my_video_schedule should return same value as capped_cubic_shedule when video_interval is None (except for last episode)'''
        video_interval = None
        for ep_no in range(0, self.total_episodes - 1):
            self.assertEqual(my_video_schedule(
                ep_no, self.total_episodes, video_interval), capped_cubic_video_schedule(ep_no))

    def test_extreme_episodes_always_true(self):
        '''should be always true for first and last episode'''
        vi = [None, 1, 10]
        te = [1, 2, 3, 8, 9, 1000, 1001, 1000000, 1000001]
        for v in vi:
            for t in te:
                self.assertTrue(my_video_schedule(0, t, v))
                self.assertTrue(my_video_schedule(t-1, t, v))

    def test_dummy(self):
        'always passes'
        pass


class TestGreat(unittest.TestCase):

    def test_dumbo(self):
        'always passes'
        self.assertTrue(True)

    def test_don(self):
        pass
