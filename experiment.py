import pybullet
from utils import Joint
import pybullet_data


pybullet.connect(pybullet.GUI)
pybullet.resetSimulation()
pybullet.setTimeStep(1/30)  # default: 1/240 which is 240Hz

pybullet.setAdditionalSearchPath(pybullet_data.getDataPath())

pybullet.configureDebugVisualizer(pybullet.COV_ENABLE_SEGMENTATION_MARK_PREVIEW, 1)
pybullet.configureDebugVisualizer(pybullet.COV_ENABLE_DEPTH_BUFFER_PREVIEW, 1)
pybullet.configureDebugVisualizer(pybullet.COV_ENABLE_RGB_BUFFER_PREVIEW, 1)

plane = pybullet.loadURDF("plane.urdf")
# Load an R2D2 droid at the position at 0.5 meters height in the z-axis.
r2d2 = pybullet.loadURDF('/home/kos4st/Documents/robot/urdfs/robot_to_export_description/urdf/robot_to_export.xacro', [0, 0, 0.5])

# Let's analyze the R2D2 droid!
print(f"r2d2 unique ID: {r2d2}")
for i in range(pybullet.getNumJoints(r2d2)):
  joint = Joint(*pybullet.getJointInfo(r2d2, i))
  print(joint)

# Set the gravity to Earth's gravity.
pybullet.setGravity(0, 0, -9.807)

# Run the simulation for a fixed amount of steps.
while True:
    position, orientation = pybullet.getBasePositionAndOrientation(r2d2)

    # width = 320
    # height = 200
    # width, height, rgba, depth, mask = pybullet.getCameraImage(
    #     width,
    #     height,
    #     viewMatrix=pybullet.computeViewMatrixFromYawPitchRoll(
    #         cameraTargetPosition=[0, 0, 0],
    #         distance=4,
    #         yaw=60,
    #         pitch=-10,
    #         roll=0,
    #         upAxisIndex=2,
    #     ),
    #     projectionMatrix=pybullet.computeProjectionMatrixFOV(
    #         fov=60,
    #         aspect=width/height,
    #         nearVal=0.01,
    #         farVal=100,
    #     ),
    #     shadow=True,
    #     lightDirection=[1, 1, 1],
    # )
    # print(f"rgba shape={rgba.shape}, dtype={rgba.dtype}")
    # Image.fromarray(rgba, 'RGBA').show()
    # print(f"depth shape={depth.shape}, dtype={depth.dtype}, as values from 0.0 (near) to 1.0 (far)")
    # Image.fromarray((depth*255).astype('uint8')).show()
    # print(f"mask shape={mask.shape}, dtype={mask.dtype}, as unique values from 0 to N-1 entities, and -1 as None")
    # Image.fromarray(np.interp(mask, (-1, mask.max()), (0, 255)).astype('uint8')).show()

    x, y, z = position
    roll, pitch, yaw = pybullet.getEulerFromQuaternion(orientation)
    print(f"{i:3}: x={x:0.10f}, y={y:0.10f}, z={z:0.10f}), roll={roll:0.10f}, pitch={pitch:0.10f}, yaw={yaw:0.10f}")
    pybullet.stepSimulation()

pybullet.disconnect()
