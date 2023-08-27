import math
import pygame
from pygame.locals import *
from OpenGL.GL import *
from OpenGL.GLU import *
from figure import *
from tkinter import *
import time
import threading
import imufusion
import pyrealsense2 as rs
import numpy as np

class IMU:
    Roll = 0
    Pitch = 0
    Yaw = 0

myimu  = IMU()


def InitPygame():
    global display
    pygame.init()
    display = (640,480)
    pygame.display.set_mode(display, DOUBLEBUF|OPENGL)
    pygame.display.set_caption('IMU visualizer   (Press Esc to exit)')


def InitGL():
    glClearColor((1.0/255*46),(1.0/255*45),(1.0/255*64),1)
    glEnable(GL_DEPTH_TEST)
    glDepthFunc(GL_LEQUAL)
    glHint(GL_PERSPECTIVE_CORRECTION_HINT, GL_NICEST)

    gluPerspective(100, (display[0]/display[1]), 0.1, 50.0)
    glTranslatef(0.0,0.0, -5)


def DrawText(textString):     
    font = pygame.font.SysFont ("Courier New",25, True)
    textSurface = font.render(textString, True, (255,255,0), (46,45,64,255))     
    textData = pygame.image.tostring(textSurface, "RGBA", True)         
    glDrawPixels(textSurface.get_width(), textSurface.get_height(), GL_RGBA, GL_UNSIGNED_BYTE, textData)    


def DrawBoard():
    glBegin(GL_QUADS)
    x = 0

    for surface in surfaces:
        for vertex in surface:  
            glColor3fv((colors[x]))          
            glVertex3fv(vertices[vertex])
        x += 1
    glEnd()


def DrawGL():
    glClear(GL_COLOR_BUFFER_BIT|GL_DEPTH_BUFFER_BIT)
    glLoadIdentity() 
    gluPerspective(90, (display[0]/display[1]), 0.1, 50.0)
    glTranslatef(0.0, 0.0, -5)   

    glRotatef(myimu.Roll, -1, 0, 0)
    glRotatef(myimu.Pitch, 0, 0, 1)
    glRotatef(myimu.Yaw, 0, 1, 0)

    DrawText("Roll: {}°  Pitch: {}°  Yaw: {}°".format(round(myimu.Roll),round(myimu.Pitch),round(myimu.Yaw)))
    DrawBoard()
    pygame.display.flip()


def initialize_camera():
    # start the frames pipe
    p = rs.pipeline()
    conf = rs.config()
    conf.enable_stream(rs.stream.accel)
    conf.enable_stream(rs.stream.gyro)
    p.start(conf)
    return p


def gyro_data(gyro):  # convert to NWU convension
    return np.asarray([gyro.x / math.pi * 180, gyro.z / math.pi * 180, -gyro.y / math.pi * 180])


def accel_data(accel):
    # if abs(accel.y) < 1 and abs(accel.z) < 1:
    #     accel.z = 0
    return np.asarray([accel.x / 9.807, accel.z / 9.807, -accel.y / 9.807])

p = initialize_camera()
sample_rate = 30
offset = imufusion.Offset(sample_rate)
ahrs = imufusion.Ahrs()

ahrs.settings = imufusion.Settings(imufusion.CONVENTION_NWU,  # convention
                                   0.5,  # gain
                                   90,  # acceleration rejection
                                   0,  # magnetic rejection
                                   0)  # rejection timeout

last_ts_gyro = None
def get_sensor_data():
    global last_ts_gyro
    f = p.wait_for_frames()
    accel = accel_data(f[0].as_motion_frame().get_motion_data())
    gyro = gyro_data(f[1].as_motion_frame().get_motion_data())
    gyro = offset.update(gyro)
    ts = f.get_timestamp()
    if last_ts_gyro is None:
        last_ts_gyro = ts
        return [False]
    ahrs.update_no_magnetometer(gyro, accel, (ts - last_ts_gyro) / 1000)
    last_ts_gyro = ts
    euler = ahrs.quaternion.to_euler()
    # euler[0] += 90  # add 90 degree to roll axis because the default is -90 for camera facing forward
    np.matmul(ahrs.quaternion.to_matrix(), np.array([0,1,0]))
    print(np.argmax(np.abs(ahrs.earth_acceleration)))
    return euler, ahrs.linear_acceleration

def ReadData():
    while True:
        euler = get_sensor_data()[0]
        
        if euler is False: continue
        
        # print(starting_point)
        # euler = [x - starting_point[i] for i, x in enumerate(euler)]
        myimu.Roll = euler[0]
        myimu.Pitch = euler[1]
        myimu.Yaw = euler[2]


def main():
    InitPygame()
    InitGL()

    try:
        myThread1 = threading.Thread(target = ReadData)
        myThread1.daemon = True
        myThread1.start() 
        while True:
            event = pygame.event.poll()
            if event.type == QUIT or (event.type == KEYDOWN and event.key == K_ESCAPE):
                pygame.quit()
                break 

            DrawGL()
            pygame.time.wait(10)

    except:
        glClear(GL_COLOR_BUFFER_BIT|GL_DEPTH_BUFFER_BIT)
        DrawText("Sorry, something is wrong :c")
        pygame.display.flip()
        time.sleep(5)


if __name__ == '__main__': main()
