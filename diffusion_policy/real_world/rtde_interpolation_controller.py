import os
import time
import threading
import enum
from typing import Any
import multiprocessing as mp
from multiprocessing.managers import SharedMemoryManager
import scipy.interpolate as si
import scipy.spatial.transform as st
import numpy as np
import numbers
import rclpy
from rclpy.node import Node
from geometry_msgs.msg import PoseStamped, TwistStamped
from sensor_msgs.msg import JointState
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint
from diffusion_policy.shared_memory.shared_memory_queue import (
    SharedMemoryQueue, Empty)
from diffusion_policy.shared_memory.shared_memory_ring_buffer import SharedMemoryRingBuffer
from diffusion_policy.common.pose_trajectory_interpolator import PoseTrajectoryInterpolator

class Command(enum.Enum):
    STOP = 0
    SERVOL = 1
    SCHEDULE_WAYPOINT = 2

class FrankaDataCollector(Node):
    def __init__(self,shm_manager,get_max_k,frequency):
        super().__init__('franka_data_collector')

        self.shm_manager = shm_manager
        self.frequency = frequency
        self.timer_period = 1.0 / frequency  # seconds
        self.get_max_k = get_max_k

        self.create_subscription(
            TwistStamped,
            '/franka_robot_state_broadcaster/desired_end_effector_twist',
            self.eef_twist_callback,
            10
        )
        self.create_subscription(
            JointState,
            '/franka_robot_state_broadcaster/measured_joint_states',
            self.joint_state_callback,
            10
        )
        self.create_subscription(
            PoseStamped,
            '/franka_robot_state_broadcaster/current_pose',
            self.pose_callback,
            10
        )

        # Data container
        self.example = dict()

        # Start timer to periodically push data
        self.timer = self.create_timer(1/125, self.timer_callback)

    def eef_twist_callback(self, twist_msg: TwistStamped):
        # Get eff velocity (q).
        self.example['ActualTCPSpeed'] = np.array([
            twist_msg.twist.linear.x,
            twist_msg.twist.linear.y,
            twist_msg.twist.linear.z,
            twist_msg.twist.angular.x,
            twist_msg.twist.angular.y,
            twist_msg.twist.angular.z,],dtype=np.float32)

        # Optional: External forces, torques

    def joint_state_callback(self, msg: JointState):
        # Just in case you want additional joint info
        self.example['ActualQ'] = np.array(msg.position, dtype=np.float32)
        self.example["ActualQd"] = np.array(msg.velocity,dtype=np.float32)

    def pose_callback(self, msg: PoseStamped):
        # End effector position and orientation
        pos = msg.pose.position
        ori = msg.pose.orientation
        self.example['ActualTCPPose'] = np.array([pos.x, pos.y, pos.z, ori.x, ori.y, ori.z, ori.w])
        #return self.example['ActualTCPPose']
    
    def timer_callback(self):
        if not self.example:
            # Skip if no data yet
            return

        # Add timestamp
        self.example['robot_receive_timestamp'] = time.time()

        # Create ring buffer if not already created
        self.ring_buffer = SharedMemoryRingBuffer.create_from_examples(
                shm_manager=self.shm_manager,
                examples=self.example,
                get_max_k=self.get_max_k,
                get_time_budget=0.2,
                put_desired_frequency=self.frequency
            )

        # Push the latest data into the ring buffer
        self.ring_buffer.put(self.example)
        self.get_logger().info('Data pushed to ring buffer.')

    def get_ring_buffer(self):
        return self.ring_buffer
    def getActualTCPPose(self):
        return self.example['ActualTCPPose']
    
class FrankaDataPublisher(Node):
    def __init__(self,frequency):
        super().__init__('franka_pose_controller')
        self.frequency = frequency
        self.time = 1/frequency

        self.publisher_ = self.create_publisher(JointTrajectory,'/fr3_arm_controller/joint_trajectory',10)


        #self.timer = self.create_timer(self.time)
        self.start_time = time.time()
        self.joint_names = [
            "fr3_joint1",
            "fr3_joint2",
            "fr3_joint3",
            "fr3_joint4",
            "fr3_joint5",
            "fr3_joint6",
            "fr3_joint7",
        ]

    def joint_init(self,joint_pose,max_vel,max_accl):
        traj_msg = JointTrajectory()
        traj_msg.header.stamp = self.get_clock().now().to_msg()
        traj_msg.joint_names = self.joint_names
        traj_msg.points = []
        self.joint_pose = joint_pose
        self.max_vel = max_vel
        self.max_accl = max_accl

        point = JointTrajectoryPoint()

        point.positions = self.joint_pose

        # Optional: leave velocities/accelerations empty or zero
        point.velocities = [self.max_vel] * len(self.joint_names)
        point.accelerations = [self.max_accl] * len(self.joint_names)
        point.effort = []

        # Time from start specifies how long to reach this pose
        point.time_from_start.sec = 2
        point.time_from_start.nanosec = 0

        # Add point to trajectory
        traj_msg.points.append(point)

        # Publish
        self.publisher_.publish(traj_msg)
        self.get_logger().info(f'Published JointTrajectory Point: {point.positions}')
    
    def joint_trajectory(self, pose, vel, acc, dt, lookahead_time, gain):
        self.eef_pose = pose
        self.max_vel = vel
        self.max_accl = acc
        self.dt = dt
        self.lookahead_time = lookahead_time
        self.gain = gain
        
        self.joint_init(None,self.max_vel, self.max_accl)
        

        

class FrankaInterpolationController(mp.Process):
    """
    To ensure sending command to the robot with predictable latency
    this controller need its separate process (due to python GIL)
    """


    def __init__(self,
            shm_manager: SharedMemoryManager, 
            robot_ip, 
            frequency=125, 
            lookahead_time=0.1, 
            gain=300,
            max_pos_speed=0.25, # 5% of max speed
            max_rot_speed=0.16, # 5% of max speed
            launch_timeout=3,
            tcp_offset_pose=None,
            payload_mass=None,
            payload_cog=None,
            joints_init=None,
            joints_init_speed=1.05,
            soft_real_time=False,
            verbose=False,
            receive_keys=None,
            get_max_k=128,
            ):
        """
        frequency: CB2=125, UR3e=500
        lookahead_time: [0.03, 0.2]s smoothens the trajectory with this lookahead time
        gain: [100, 2000] proportional gain for following target position
        max_pos_speed: m/s
        max_rot_speed: rad/s
        tcp_offset_pose: 6d pose
        payload_mass: float
        payload_cog: 3d position, center of gravity
        soft_real_time: enables round-robin scheduling and real-time priority
            requires running scripts/rtprio_setup.sh before hand.

        """
        # verify
        assert 0 < frequency <= 500
        assert 0.03 <= lookahead_time <= 0.2
        assert 100 <= gain <= 2000
        assert 0 < max_pos_speed
        assert 0 < max_rot_speed
        if tcp_offset_pose is not None:
            tcp_offset_pose = np.array(tcp_offset_pose)
            assert tcp_offset_pose.shape == (6,)
        if payload_mass is not None:
            assert 0 <= payload_mass <= 5
        if payload_cog is not None:
            payload_cog = np.array(payload_cog)
            assert payload_cog.shape == (3,)
            assert payload_mass is not None
        if joints_init is not None:
            joints_init = np.array(joints_init)
            assert joints_init.shape == (6,)

        super().__init__(name="FrankaPositionalController")
        self.robot_ip = robot_ip
        self.frequency = frequency
        self.lookahead_time = lookahead_time
        self.gain = gain
        self.max_pos_speed = max_pos_speed
        self.max_rot_speed = max_rot_speed
        self.launch_timeout = launch_timeout
        self.tcp_offset_pose = tcp_offset_pose
        self.payload_mass = payload_mass
        self.payload_cog = payload_cog
        self.joints_init = joints_init
        self.joints_init_speed = joints_init_speed
        self.soft_real_time = soft_real_time
        self.verbose = verbose
        self.get_max_k = get_max_k
        self.shm_manager = shm_manager

        # build input queue
        example = {
            'cmd': Command.SERVOL.value,
            'target_pose': np.zeros((6,), dtype=np.float64),
            'duration': 0.0,
            'target_time': 0.0
        }
        input_queue = SharedMemoryQueue.create_from_examples(
            shm_manager=shm_manager,
            examples=example,
            buffer_size=256
        )

        # build ring buffer
        if receive_keys is None:
            receive_keys = [
                'ActualTCPPose',
                'ActualTCPSpeed',
                'ActualQ',
                'ActualQd',

                'TargetTCPPose',
                'TargetTCPSpeed',
                'TargetQ',
                'TargetQd'
            ]
        ring_buffer = FrankaDataCollector(shm_manager,self.frequency,self.get_max_k).get_ring_buffer()

        self.ready_event = mp.Event()
        self.input_queue = input_queue
        self.ring_buffer = ring_buffer
        self.receive_keys = receive_keys
    
    # ========= launch method ===========
    def start(self, wait=True):
        super().start()
        if wait:
            self.start_wait()
        if self.verbose:
            print(f"[FrankaPositionalController] Controller process spawned at {self.pid}")

    def stop(self, wait=True):
        message = {'cmd': np.array([int(Command.STOP.value)], dtype=np.int32)}
        self.input_queue.put(message)
        if wait:
            self.stop_wait()

    def start_wait(self):
        self.ready_event.wait(self.launch_timeout)
        assert self.is_alive()
    
    def stop_wait(self):
        self.join()
    
    @property
    def is_ready(self):
        return self.ready_event.is_set()

    # ========= context manager ===========
    def __enter__(self):
        self.start()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.stop()
        
    # ========= command methods ============
    def servoL(self, pose, duration=0.1):
        """
        duration: desired time to reach pose
        """
        assert self.is_alive()
        assert(duration >= (1/self.frequency))
        pose = np.array(pose)
        assert pose.shape == (6,)

        message = {
            'cmd': Command.SERVOL.value,
            'target_pose': pose,
            'duration': duration
        }
        self.input_queue.put(message)
    
    def schedule_waypoint(self, pose, target_time):
        assert target_time > time.time()
        pose = np.array(pose)
        assert pose.shape == (6,) or (7,)

        message = {
            'cmd': Command.SCHEDULE_WAYPOINT.value,
            'target_pose': pose,
            'target_time': target_time
        }
        self.input_queue.put(message)

    # ========= receive APIs =============
    def get_state(self, k=None, out=None):
        if k is None:
            return self.ring_buffer.get(out=out)
        else:
            return self.ring_buffer.get_last_k(k=k,out=out)
    
    def get_all_state(self):
        return self.ring_buffer.get_all()
    
    # ========= main loop in process ============
    def run(self):
        rclpy.init()

        # Create ROS2 nodes
        rtde_c = FrankaDataPublisher(self.frequency)
        rtde_r = FrankaDataCollector(self.shm_manager, self.get_max_k, self.frequency)

        # Spin both nodes in background threads
        executor = rclpy.executors.MultiThreadedExecutor()
        executor.add_node(rtde_c)
        executor.add_node(rtde_r)
        spin_thread = threading.Thread(target=executor.spin, daemon=True)
        spin_thread.start()

        try:
            if self.verbose:
                print(f"[FrankaPositionalController] Running at {self.frequency}Hz")

            # Initial joint pose if needed
            if self.joints_init is not None:
                rtde_c.joint_init(self.joints_init, self.joints_init_speed, 1.4)

            dt = 1. / self.frequency
            curr_pose = rtde_r.getActualTCPPose()
            curr_t = time.monotonic()
            last_waypoint_time = curr_t
            pose_interp = PoseTrajectoryInterpolator(
            times=np.array(curr_t),
            poses=np.array(curr_pose))

            iter_idx = 0
            keep_running = True
            while keep_running:
                t_start = time.perf_counter()
                t_now = time.monotonic()

                pose_command = pose_interp(t_now)
                vel = 0.5
                acc = 0.5

                rtde_c.joint_trajectory(pose_command, vel, acc, dt, self.lookahead_time, self.gain)

                state = {}
                for key in self.receive_keys:
                    state[key] = np.array(getattr(rtde_r, 'get' + key)())
                state['robot_receive_timestamp'] = time.time()
                self.ring_buffer.put(state)

                try:
                    commands = self.input_queue.get_all()
                    n_cmd = len(commands['cmd'])
                except Empty:
                    n_cmd = 0

                for i in range(n_cmd):
                    command = {key: value[i] for key, value in commands.items()}
                    cmd = command['cmd']

                    if cmd == Command.STOP.value:
                        keep_running = False
                        break
                    elif cmd == Command.SERVOL.value:
                        target_pose = command['target_pose']
                        duration = float(command['duration'])
                        curr_time = t_now + dt
                        t_insert = curr_time + duration
                        pose_interp = pose_interp.drive_to_waypoint(
                        pose=target_pose,
                        time=t_insert,
                        curr_time=curr_time,
                        max_pos_speed=self.max_pos_speed,
                        max_rot_speed=self.max_rot_speed
                    )
                    last_waypoint_time = t_insert
                    if self.verbose:
                        print(f"[FrankaPositionalController] New pose target: {target_pose}, duration: {duration}")
                    elif cmd == Command.SCHEDULE_WAYPOINT.value:
                        target_pose = command['target_pose']
                        target_time = float(command['target_time'])
                        target_time = time.monotonic() - time.time() + target_time
                        curr_time = t_now + dt
                        pose_interp = pose_interp.schedule_waypoint(
                        pose=target_pose,
                        time=target_time,
                        max_pos_speed=self.max_pos_speed,
                        max_rot_speed=self.max_rot_speed,
                        curr_time=curr_time,
                        last_waypoint_time=last_waypoint_time
                    )
                    last_waypoint_time = target_time
                else:
                    keep_running = False
                    break

            # regulate frequency
            time_elapsed = time.perf_counter() - t_start
            sleep_duration = max(0.0, dt - time_elapsed)
            time.sleep(sleep_duration)

            if iter_idx == 0:
                self.ready_event.set()
            iter_idx += 1

            if self.verbose:
                print(f"[FrankaPositionalController] Actual frequency {1 / (time.perf_counter() - t_start)}")

        finally:
        # Optional stop motion publishing

        # Clean shutdown
            executor.shutdown()
            rtde_c.destroy_node()
            rtde_r.destroy_node()
            rclpy.shutdown()
            self.ready_event.set()

            if self.verbose:
                print("[FrankaPositionalController] ROS 2 nodes shut down.")
