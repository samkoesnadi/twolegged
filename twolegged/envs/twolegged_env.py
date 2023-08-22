"""
"""
import math
import time

import gymnasium as gym
import pybullet
import matplotlib.pyplot as plt
import numpy as np

from twolegged.resources.robot import Robot
from twolegged.resources.plane import Plane


class TwoLeggedEnv(gym.Env):
    metadata = {"render_modes": ["human", "rgb_array"]}

    CAMERA_REFRESH_PERIOD = 1/15
    BOUNDARY = 15
    GOAL_Z_POS = 1.1
    MAX_REWARD = 1

    def __init__(self, render_mode=None, render_fps=30):
        assert render_mode is None or render_mode in self.metadata["render_modes"]
        self.render_mode = render_mode

        self.simulation_period = 1/render_fps
        self.action_space = gym.spaces.Box(
            low=-math.pi,
            high=math.pi,
            shape=(3,),
            dtype=np.float32
        )
        self.observation_space = gym.spaces.Dict({
            "joints": gym.spaces.Box(
                low=np.array([-math.pi * 2, -self.BOUNDARY] * 3, dtype=np.float32),
                high=np.array([math.pi * 2, self.BOUNDARY] * 3, dtype=np.float32),
            ),
            "body_z": gym.spaces.Box(
                low=np.array([-self.BOUNDARY] * 1, dtype=np.float32),
                high=np.array([self.BOUNDARY] * 1, dtype=np.float32),
            )
        })
        # self.observation_space = gym.spaces.Box(
        #     low=np.array([-math.pi * 2, -self.BOUNDARY] * 3, dtype=np.float32),
        #     high=np.array([math.pi * 2, self.BOUNDARY] * 3, dtype=np.float32),
        # )
        self.client_id = pybullet.connect(pybullet.DIRECT if render_mode != "human" else pybullet.GUI)
        if render_mode == "human":
            pybullet.configureDebugVisualizer(pybullet.COV_ENABLE_SEGMENTATION_MARK_PREVIEW, 1, physicsClientId=self.client_id)
            pybullet.configureDebugVisualizer(pybullet.COV_ENABLE_DEPTH_BUFFER_PREVIEW, 1, physicsClientId=self.client_id)
            pybullet.configureDebugVisualizer(pybullet.COV_ENABLE_RGB_BUFFER_PREVIEW, 1, physicsClientId=self.client_id)

        
        # Reduce length of episodes for RL algorithms
        pybullet.setTimeStep(self.simulation_period, self.client_id)

        self.robot : Robot = None
        self.plane : Plane = None
        self.truncated = False
        self.terminated = False
        self.reset()

    def _get_obs(self):
        _joints = np.array(self.robot.get_observation(), dtype=np.float32)
        _body = np.array(self.robot.get_whole_body_observation(), dtype=np.float32)
        return {"joints": _joints, "body": _body}

    def _get_info(self):
        return {}

    def step(self, action):
        if self.render_mode == "human":
            start_time = time.time()

        pybullet.stepSimulation(self.client_id)

        # Feed action to the robot and get observation of robot's state
        self.robot.apply_action(action)

        observation = self._get_obs()
        info = self._get_info()
        robot_pos = observation["body"]

        # Compute reward as L2 change in distance to goal
        reward = self.MAX_REWARD * math.sin(
            math.pi / 2 * min(1, robot_pos[4] / self.GOAL_Z_POS)
        )
        # reward = 0
        # if robot_pos[4] >= 0.1:  # TODO
        #     reward = self.MAX_REWARD

        # Done by running off boundaries
        if (robot_pos[0] >= self.BOUNDARY or robot_pos[0] <= -self.BOUNDARY or
                robot_pos[1] >= self.BOUNDARY or robot_pos[1] <= -self.BOUNDARY):
            self.truncated = True

        # # Done by falling down or contact
        # if robot_pos[4] < 0.1:  # TODO
        #     self.terminated = True

        ### ENDING ###
        if self.render_mode == "human":
            gap_time = time.time() - start_time
            time.sleep(self.simulation_period - gap_time)

        obs = {
            "joints": np.array(observation["joints"]),
            "body_z": np.array([observation["body"][4]])
        }
        return obs, reward, self.terminated, self.truncated, info

    def reset(self, seed=None, options=None):
        # We need the following line to seed self.np_random
        super().reset(seed=seed)

        self.truncated = False
        self.terminated = False

        pybullet.resetSimulation(self.client_id)
        pybullet.setGravity(0, 0, -9.807)

        self.plane = Plane(self.client_id)
        self.robot = Robot(self.client_id)

        # Set the goal to a random target
        self.truncated = False

        observation = self._get_obs()
        info = self._get_info()
        obs = {
            "joints": np.array(observation["joints"]),
            "body_z": np.array([observation["body"][4]])
        }

        # Get observation to return
        return obs, info

    def render(self):
        pass

    def close(self):
        pybullet.disconnect(self.client_id)
