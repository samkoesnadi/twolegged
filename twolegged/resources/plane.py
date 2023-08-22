import pybullet
import pybullet_data


class Plane:
  def __init__(self, client_id):
    pybullet.setAdditionalSearchPath(pybullet_data.getDataPath())
    self.id = pybullet.loadURDF(
      "plane.urdf",
      basePosition=[0, 0, 0],
      physicsClientId=client_id
    )
