"""
Evaluation Agent.
"""

import time
import numpy as np
from device.robot.flexiv import FlexivRobot
from utils.transformation import xyz_rot_transform
from device.gripper.dahuan import DahuanModbusGripper
from device.camera.realsense import RealSenseRGBDCamera
from easyrobot.arm.flexiv import FlexivArm

class Agent:
    """
    Evaluation agent with Flexiv arm, Dahuan gripper and Intel RealSense RGB-D camera.

    Follow the implementation here to create your own real-world evaluation agent.
    """
    def __init__(
        self,
        robot_ip,
        pc_ip,
        gripper_port,
        camera_serial,
        **kwargs
    ): 
        self.camera_serial = camera_serial

        print("Init robot, gripper, and camera.")
        #self.robot = FlexivRobot(robot_ip_address = robot_ip, pc_ip_address = pc_ip)
        self.robot = FlexivArm("Rizon4-062027")
        self.robot.send_tcp_pose(self.ready_pose)
        time.sleep(1.5)
        
        self.gripper = DahuanModbusGripper(port = gripper_port)
        self.gripper.set_force(30)
        self.gripper.set_width(0)
        time.sleep(0.5)


        self.camera = RealSenseRGBDCamera(serial = camera_serial)
        for _ in range(30): 
            self.camera.get_rgbd_image()
        print("Initialization Finished.")
    
    @property
    def intrinsics(self):
        return np.array([
            [914.81945801,   0.        , 630.63891602,   0.        ],
       [  0.        , 913.88464355, 352.51571655,   0.        ],
       [  0.        ,   0.        ,   1.        ,   0.        ]
        ])
    
    @property
    def ready_pose(self):
        #return np.array([0.5, 0.0, 0.17, 0.0, 0.0, 1.0, 0.0])
        return np.array([0.4, 0.0, 0.25, 0.0, 0.0, 1.0, 0.0])

    @property
    def ready_rot_6d(self):
        return np.array([-1, 0, 0, 0, 1, 0])

    def get_observation(self):
        colors, depths = self.camera.get_rgbd_image()
        return colors, depths
    
    def set_tcp_pose(self, pose, rotation_rep, rotation_rep_convention = None, blocking = False):
        tcp_pose = xyz_rot_transform(
            pose,
            from_rep = rotation_rep, 
            to_rep = "quaternion",
            from_convention = rotation_rep_convention
        )
        self.robot.send_tcp_pose(tcp_pose)
        if blocking:
            time.sleep(0.1)
    
    def set_gripper_width(self, width, blocking = False):
        width = int(np.clip(width / 0.095 * 1000., 0, 1000))
        self.gripper.set_width(width)
        if blocking:
            time.sleep(0.5)
    
    def stop(self):
        self.robot.stop()
    
    #新增获取当前tcp gripper的接口函数
    
    def get_tcp_pose(self):
        """get current robot's tool pose in world frame.

        Returns:
            7-dim list consisting of (x,y,z,rw,rx,ry,rz)"""
        return self.robot.get_tcp_pose()
    
    def get_gripper_width(self):
        return self.gripper.width
    
    
class Agent_slave:
    """副相机类，绑定腕部相机，用于返回腕部相机的观察"""
    def __init__(
            self,
            camera_serial,
            **kwargs
        ): 
            self.camera_serial = camera_serial
    
            self.camera = RealSenseRGBDCamera(serial = camera_serial)
            for _ in range(30): 
                self.camera.get_rgbd_image()
            print("Agent_slave Initialization Finished.")
            
    def get_observation(self):
            colors, depths = self.camera.get_rgbd_image()
            return colors, depths