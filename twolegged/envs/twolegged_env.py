"""
"""
import math

import gymnasium as gym
import pybullet
import matplotlib.pyplot as plt
import numpy as np

from twolegged.resources.robot import Robot
from twolegged.resources.plane import Plane


class TwoLeggedEnv(gym.Env):
    metadata = {'render.modes': ['human']}

    SIMULATION_PERIOD = 1/30
    CAMERA_REFRESH_PERIOD = 1/15

    def __init__(self):
        self.action_space = gym.spaces.box.Box(
            low=-math.pi/2,
            high=math.pi/2,
            shape=3,
            dtype=np.float32
        )
        self.observation_space = gym.spaces.box.Box(
            low=-math.pi/2,
            high=math.pi/2,
            shape=3,
            dtype=np.float32
        )
        self.client_id = pybullet.connect(pybullet.DIRECT)
        # Reduce length of episodes for RL algorithms
        pybullet.setTimeStep(self.SIMULATION_PERIOD, self.client_id)

        self.robot : Robot = None
        self.prev_robot_pos : Robot = None
        self.done = False
        self.rendered_img = None
        self.render_rot_matrix = None
        self.reset()

    def step(self, action):
        # Feed action to the robot and get observation of robot's state
        self.robot.apply_action(action)
        pybullet.stepSimulation(self.client_id)
        robot_pos = self.robot.get_whole_body_observation()

        if self.prev_robot_pos is None:
            self.prev_robot_pos = robot_pos

        # Compute reward as L2 change in distance to goal
        dist_to_last_pos = np.linalg.norm(
            np.array(self.prev_robot_pos[:3], dtype=np.float32))
            - np.array(robot_pos[:3], dtype=np.float32))
        )  # jumping is also considered a reward here : z-axis
        self.prev_robot_pos = robot_pos
        reward = max(dist_to_last_pos, 0)

        # Done by running off boundaries
        if (robot_pos[0] >= 10 or robot_pos[0] <= -10 or
                robot_pos[1] >= 10 or robot_pos[1] <= -10):
            self.done = True

        ob = np.array(self.robot.get_observation(), dtype=np.float32)
        return ob, reward, self.done, dict()

    def seed(self, seed=None):
        self.np_random, seed = gym.utils.seeding.np_random(seed)
        return [seed]

    def reset(self):
        pybullet.resetSimulation(self.client_id)
        pybullet.setGravity(0, 0, -9.807)

        Plane(self.client_id)
        self.robot = Robot(self.client_id)

        # Set the goal to a random target
        self.done = False

        # Get observation to return
        return np.array(self.robot.get_observation(), dtype=np.float32)

    def render(self, mode='human'):
        if self.rendered_img is None:
            self.rendered_img = plt.imshow(np.zeros((100, 100, 4)))

        # Base information
        robot_id, client_id = self.robot.get_ids()
        proj_matrix = pybullet.computeProjectionMatrixFOV(fov=80, aspect=1,
                                                   nearVal=0.01, farVal=100)
        pos, ori = [list(l) for l in
                    pybullet.getBasePositionAndOrientation(robot_id, client_id)]
        pos[2] = 0.2

        # Rotate camera direction
        rot_mat = np.array(pybullet.getMatrixFromQuaternion(ori)).reshape(3, 3)
        camera_vec = np.matmul(rot_mat, [1, 0, 0])
        up_vec = np.matmul(rot_mat, np.array([0, 0, 1]))
        view_matrix = pybullet.computeViewMatrix(pos, pos + camera_vec, up_vec)

        # Display image
        frame = pybullet.getCameraImage(100, 100, view_matrix, proj_matrix)[2]
        frame = np.reshape(frame, (100, 100, 4))
        self.rendered_img.set_data(frame)
        plt.draw()
        plt.pause(.00001)

    def close(self):
        pybullet.disconnect(self.client_id)
