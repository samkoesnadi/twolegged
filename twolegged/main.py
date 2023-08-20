"""
"""
import math
import time

import numpy as np
import pybullet
import pybullet_data

from twolegged.utils import Joint


CONNECTION_MODE = pybullet.GUI
SIMULATION_PERIOD = 1/30


pybullet.connect(CONNECTION_MODE)
pybullet.setAdditionalSearchPath(pybullet_data.getDataPath())

# pybullet.resetSimulation()
pybullet.setTimeStep(SIMULATION_PERIOD)  # default: 1/240 which is 240Hz
pybullet.setGravity(0, 0, -9.807)


if CONNECTION_MODE == pybullet.GUI:
  pybullet.configureDebugVisualizer(pybullet.COV_ENABLE_SEGMENTATION_MARK_PREVIEW, 1)
  pybullet.configureDebugVisualizer(pybullet.COV_ENABLE_DEPTH_BUFFER_PREVIEW, 1)
  pybullet.configureDebugVisualizer(pybullet.COV_ENABLE_RGB_BUFFER_PREVIEW, 1)


plane = pybullet.loadURDF("plane.urdf")


# TODO
startPos = [0,0,1]
startOrientation = pybullet.getQuaternionFromEuler([0,0,math.pi / 2])

robot = pybullet.loadURDF(
  'urdf/robot_to_export.xacro',
  startPos,
  startOrientation,
  flags=pybullet.URDF_USE_SELF_COLLISION
    # | pybullet.URDF_USE_SELF_COLLISION_INCLUDE_PARENT
    | pybullet.URDF_USE_SELF_COLLISION_EXCLUDE_PARENT  # TODO: remove
)

joints = {}
for i in range(pybullet.getNumJoints(robot)):
  joint = Joint(*pybullet.getJointInfo(robot, i))
  joints[joint.name] = joint


# Run the simulation for a fixed amount of steps.
start_time = time.time()
while True:
    pybullet.stepSimulation()

    # action
    for joint_name, joint_obj in joints.items():
      pybullet.setJointMotorControl2(robot, joint_obj.index, pybullet.POSITION_CONTROL, math.pi)

    # get joint position
    for joint_name, joint_obj in joints.items():
      position, velocity, _, _ = pybullet.getJointState(robot, joint_obj.index)
      # print(position, velocity)

    # get whole body position
    # position, velocity = pybullet.getBasePositionAndOrientation(robot)
    # x, y, z = position
    # roll, pitch, yaw = pybullet.getEulerFromQuaternion(orientation)
    # print(f"{i:3}: x={x:0.10f}, y={y:0.10f}, z={z:0.10f}), roll={roll:0.10f}, pitch={pitch:0.10f}, yaw={yaw:0.10f}")

    contact_points = pybullet.getClosestPoints(robot, robot, 1)
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

    # get the camera sensor position
    camera_width = 100
    camera_height = 100

    proj_matrix = pybullet.computeProjectionMatrixFOV(
      fov=87, aspect=1, nearVal=0.01, farVal=10)
    link_state = pybullet.getLinkState(robot, 0)
    pos, ori = link_state[0], link_state[1]

    # Rotate camera direction
    rot_mat = np.array(pybullet.getMatrixFromQuaternion(ori)).reshape(3, 3)
    camera_vec = np.matmul(rot_mat, [1, 0, 0])
    up_vec = np.matmul(rot_mat, np.array([0, 0, 1]))
    view_matrix = pybullet.computeViewMatrix(pos, pos + camera_vec, up_vec)

    # Display image
    # width, height, rgba, depth, mask = pybullet.getCameraImage(camera_width, camera_height, view_matrix, proj_matrix)

    # print(f"rgba shape={rgba.shape}, dtype={rgba.dtype}")
    # Image.fromarray(rgba, 'RGBA').show()
    # print(f"depth shape={depth.shape}, dtype={depth.dtype}, as values from 0.0 (near) to 1.0 (far)")
    # Image.fromarray((depth*255).astype('uint8')).show()
    # print(f"mask shape={mask.shape}, dtype={mask.dtype}, as unique values from 0 to N-1 entities, and -1 as None")
    # Image.fromarray(np.interp(mask, (-1, mask.max()), (0, 255)).astype('uint8')).show()


    ### ENDING ###
    gap_time = time.time() - start_time

    time.sleep(SIMULATION_PERIOD - gap_time)
    start_time = time.time()



from threading import Thread

def runA():
    while True:
        print 'A\n'

def runB():
    while True:
        print 'B\n'

if __name__ == "__main__":
    t1 = Thread(target = runA)
    t2 = Thread(target = runB)
    t1.setDaemon(True)
    t2.setDaemon(True)
    t1.start()
    t2.start()
    while True:
        pass
