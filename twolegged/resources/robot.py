"""
"""
from collections import OrderedDict
import math

import numpy as np
import pybullet

from twolegged.utils import Joint


class Robot:
  KNOWN_JOINTS = ["hip_pitch", "hip_roll", "knee_roll"]

  def __init__(self, client_id):
    self.client_id = client_id
    startPos = [0,0,1]
    startOrientation = pybullet.getQuaternionFromEuler(
      [math.pi / 2, 0, 0], physicsClientId=self.client_id
    )

    f_name = "twolegged/resources/urdf/robot_to_export.xacro"
    self.robot = pybullet.loadURDF(
      f_name,
      startPos,
      startOrientation,
      flags=pybullet.URDF_USE_SELF_COLLISION
          # | pybullet.URDF_USE_SELF_COLLISION_INCLUDE_PARENT
          | pybullet.URDF_USE_SELF_COLLISION_EXCLUDE_PARENT  # TODO: remove
      , physicsClientId=self.client_id
    )

    self.joints = OrderedDict()
    for i in range(pybullet.getNumJoints(self.robot, physicsClientId=self.client_id)):
      joint = Joint(*pybullet.getJointInfo(self.robot, i, physicsClientId=self.client_id))
      self.joints[joint.name] = joint

  def get_ids(self):
    return self.car, self.client

  def apply_action(self, action):
    for i_joint, joint_name in enumerate(self.KNOWN_JOINTS):
      pybullet.setJointMotorControl2(
        self.robot,
        self.joint[joint_name].index,
        pybullet.POSITION_CONTROL,
        action[i_joint],
        physicsClientId=self.client_id
      )

  def get_observation(self):
    # get joint position
    obs = []
    for i_joint, joint_name in enumerate(self.KNOWN_JOINTS):
      position, velocity, _, _ = pybullet.getJointState(
        self.robot, self.joints[joint_name].index, physicsClientId=self.client_id)
      obs += [position, velocity]
    return obs

  def get_whole_body_observation(self):
    position, orientation = pybullet.getBasePositionAndOrientation(self.robot)
    roll, pitch, yaw = pybullet.getEulerFromQuaternion(orientation)
    return position + [roll, pitch, yaw]

  def check_contact(self):
    contact_points = pybullet.getClosestPoints(
      self.robot, self.robot, 1, physicsClientId=self.client_id)
    is_contact = False
    for contact_point in contact_points:
      linkA = contact_point[3]
      linkB = contact_point[4]
      if (
        linkA != linkB
        and contact_point[8] <= 0
        and abs(linkB - linkA) > 1  # TODO: remove
      ):
        is_contact = True
    return is_contact

  def get_camera(self, camera_width=100, camera_height=100):  # TODO
    proj_matrix = pybullet.computeProjectionMatrixFOV(
        fov=87, aspect=1, nearVal=0.01, farVal=10, physicsClientId=self.client_id)
    # link_state = pybullet.getLinkState(robot, 0, physicsClientId=self.client_id)
    link_state = pybullet.getBasePositionAndOrientation(self.robot, physicsClientId=self.client_id)
    pos, ori = link_state[0], link_state[1]

    # Rotate camera direction
    rot_mat = np.array(pybullet.getMatrixFromQuaternion(ori, physicsClientId=self.client_id)).reshape(3, 3)
    camera_vec = np.matmul(rot_mat, [1, 0, 0])
    up_vec = np.matmul(rot_mat, np.array([0, 0, 1]))
    view_matrix = pybullet.computeViewMatrix(pos, pos + camera_vec, up_vec, physicsClientId=self.client_id)

    # Display image
    width, height, rgba, depth, mask = pybullet.getCameraImage(camera_width, camera_height, view_matrix, proj_matrix)
