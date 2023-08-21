import pybullet as p
import os


class Plane:
  def __init__(self, client_id):
    p.loadURDF(fileName="plane.urdf",
                basePosition=[0, 0, 0],
                physicsClientId=client_id)
