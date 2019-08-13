from __future__ import division
import random
import os,sys
curr_dir = os.path.dirname(os.path.abspath(__file__))
airsim_path = os.path.join(curr_dir, '..', 'airsim')
sys.path.insert(0, airsim_path)
import setup_path
import airsim
import airsim.types
import airsim.utils
import math
import copy
import time
import numpy as np
import threading
import pdb

random.seed(911)

class GatePoseGenerator(object):
    def __init__(self):
        self.default_gate_facing_vector = type("", (), dict(x=0, y=1, z=0))()
        self.GATE_HEIGHT = 16

    def generate_cubic_curve(self, num_gates, race_course_radius):
        coeffs = [random.uniform(-1,1), random.uniform(-1,1), random.uniform(-1,1)]
        # coeffs = [random.uniform(-1,1), random.uniform(-1,1), 0.0]
        ts = [t / (num_gates - 1) for t in range(0, num_gates)]
        samples = [race_course_radius * (coeffs[0] * t * t * t + coeffs[1] * t * t + coeffs[2] * t) for t in ts]
        derivatives = [race_course_radius * (3 * coeffs[0] * t * t + 2 * coeffs[1] * t + coeffs[2]) for t in ts]
        return list(zip(samples, derivatives))

    def generate_circle(self, i, num_gates, race_course_radius):
        # ts = [t / (num_gates - 1) for t in range(0, num_gates)]
        ts = [t / (num_gates) for t in range(0, num_gates)]
        samples = [0 for t in ts]
        derivatives = [0 for t in ts]

        prev_radius = None

        # min_radius = race_course_radius - 15.0
        min_radius = race_course_radius + 4.0
        max_radius = race_course_radius - 4.0
        max_radius_delta = 5.0

        radius_list = [random.uniform(min_radius, max_radius) for t in ts]


        # not a circle, but hey it's random-ish. and the wrong derivative actually make the track challenging
        # come back again later.  
        if i == 0:
            for (idx, t) in enumerate(ts):
                radius = radius_list[idx]
                if idx > 0:
                    radius = np.clip(radius, radius_list[idx-1] - max_radius_delta, radius_list[idx-1] + max_radius_delta)
                    radius = np.clip(radius, 0.0, radius)
                samples[idx] = radius * math.cos(2.*math.pi * t)
                derivatives[idx] = radius * -math.sin(2.*math.pi * t)
        elif i == 1:
            for (idx, t) in enumerate(ts):
                radius = radius_list[idx]
                if idx > 0:
                    radius = np.clip(radius, radius_list[idx-1] - max_radius_delta, radius_list[idx-1] + max_radius_delta)
                    radius = np.clip(radius, 0.0, radius)
                samples[idx] = radius * math.sin(2.*math.pi * t)
                derivatives[idx] = radius * math.cos(2.*math.pi * t)
        else:
            for (idx, t) in enumerate(ts):
                samples[idx] = 0.
                derivatives[idx] = 0.

        # if random.random() > 0.5:
        #     samples = [race_course_radius * (math.cos(2.*math.pi * t) if i == 0 else math.sin(2.*math.pi * t) if i == 1 else 0) for t in ts]
        #     derivatives = [race_course_radius * (-math.sin(2.*math.pi * t) if i == 0 else math.cos(2.*math.pi * t) if i == 1 else 0) for t in ts]
        # else:
        #     samples = [race_course_radius * (math.sin(2.*math.pi * t) if i == 0 else math.cos(2.*math.pi * t) if i == 1 else 0) for t in ts]
        #     derivatives = [race_course_radius * (math.cos(2.*math.pi * t) if i == 0 else -math.sin(2.*math.pi * t) if i == 1 else 0) for t in ts]
        return list(zip(samples, derivatives))
  
    def quaternionFromUnitGradient(self, dx_dt, dy_dt, dz_dt):
        r0 = self.default_gate_facing_vector
        q = airsim.Quaternionr(
                r0.y * dz_dt - r0.z * dy_dt,
                r0.z * dx_dt - r0.x * dz_dt,
                r0.x * dy_dt - r0.y * dx_dt,
                math.sqrt((r0.x**2 + r0.y**2 + r0.z**2) * (dx_dt**2 + dy_dt**2 + dz_dt**2)) + (r0.x * dx_dt + r0.y * dy_dt + r0.z * dz_dt)
            )

        #Normalize
        length = q.get_length()
        if (length == 0.0):
            q.w_val = 1.0
        else:
            q.w_val /= length
            q.x_val /= length
            q.y_val /= length
            q.z_val /= length

        return q

    # type_of_segment supported is only "circle" for now
    def generate_gate_poses(self, num_gates, race_course_radius, type_of_segment = "circle"):
        if type_of_segment == "circle":
            (x_t, y_t, z_t) = tuple([self.generate_circle(i, num_gates, race_course_radius) for i in range(3)])
            # todo unreadable code
            # todo un-hardcode
            gate_poses = [\
                            airsim.Pose(\
                            airsim.Vector3r((x_t[t_i][0] - x_t[0][0] - 4.0), (y_t[t_i][0] - y_t[0][0] - 4.0), random.uniform(-5.0, -9.0)),\
                            self.quaternionFromUnitGradient(x_t[t_i][1], y_t[t_i][1], z_t[t_i][1])\
                          )\
                        for t_i in range(0, num_gates)]

        # elif type_of_segment == "cubic":
        return gate_poses

class DroneRacingDataGenerator(object):
    def __init__(self, 
                drone_name = "drone_0",
                gate_passed_thresh = 2.0,
                race_course_radius = 30.0,
                odom_loop_rate_sec = 0.015):

        self.curr_track_gate_poses = None
        self.next_track_gate_poses = None
        self.gate_object_names_sorted = None
        self.num_training_laps = None
        self.track_generator = GatePoseGenerator()

        # gate idx trackers
        self.gate_passed_thresh = gate_passed_thresh
        self.last_gate_passed_idx = -1
        self.last_gate_idx_moveOnSpline_was_called_on = -1
        self.next_gate_idx = 0
        self.train_lap_idx = 0

        # should be same as settings.json
        self.drone_name = drone_name
        # training params
        self.race_course_radius = race_course_radius

        # todo encapsulate in function
        self.client = airsim.MultirotorClient()
        self.client.confirmConnection()
        self.client.enableApiControl(True, vehicle_name=self.drone_name)
        time.sleep(0.01)    

        # threading stuff
        self.got_odom = False
        self.is_expert_planner_controller_thread_active = False
        self.expert_planner_controller_thread = threading.Thread(target=self.repeat_timer_expert, args=(self.expert_planner_controller_callback, odom_loop_rate_sec))
        # self.image_loop = threading.Thread(target=self.repeat_timer, args=(self.image_callback, 0.05))

    # def image_callback(self):
    #     self.client.()

    def repeat_timer_expert(self, task, period):
        while self.is_expert_planner_controller_thread_active:
            task()
            time.sleep(period)

    # def repeat_timer_image_cb(self, task, period):
    #     while self.is_expert_planner_controller_thread_active:
    #         task()
            # time.sleep(period)

    def load_level(self, level_name='Soccer_Field_Easy'):
        self.client.simLoadLevel(level_name)
        time.sleep(2)

        self.set_current_track_gate_poses_from_default_track_in_binary() # oh-so-specific function for the hackathon
        self.next_track_gate_poses = self.get_next_generated_track()

        for gate_idx in range(len(self.gate_object_names_sorted)):
            print(self.next_track_gate_poses[gate_idx].position.x_val, self.next_track_gate_poses[gate_idx].position.y_val, self.next_track_gate_poses[gate_idx].position.z_val)
            self.client.simSetObjectPose(self.gate_object_names_sorted[gate_idx], self.next_track_gate_poses[gate_idx])
            time.sleep(0.05)

        self.set_current_track_gate_poses_from_default_track_in_binary() # oh-so-specific function for the hackathon
        self.next_track_gate_poses = self.get_next_generated_track()

    def set_current_track_gate_poses_from_default_track_in_binary(self):
        gate_names_sorted_bad = sorted(self.client.simListSceneObjects("Gate.*"))
        # gate_names_sorted_bad is ['Gate0', 'Gate10_21', 'Gate11_23', 'Gate1_3', 'Gate2_5', 'Gate3_7', 'Gate4_9', 'Gate5_11', 'Gate6_13', 'Gate7_15', 'Gate8_17', 'Gate9_19']
        # number after underscore is unreal garbage. also leading zeros are not there. 
        gate_indices_bad = [int(gate_name.split('_')[0][4:]) for gate_name in gate_names_sorted_bad]
        gate_indices_correct = sorted(range(len(gate_indices_bad)), key=lambda k:gate_indices_bad[k])
        self.gate_object_names_sorted = [gate_names_sorted_bad[gate_idx] for gate_idx in gate_indices_correct]
        self.curr_track_gate_poses = [self.client.simGetObjectPose(gate_name) for gate_name in self.gate_object_names_sorted]
        # for gate_pose in self.curr_track_gate_poses:
        #     print(gate_pose.position.x_val, gate_pose.position.y_val,gate_pose.position.z_val)

    def takeoff_with_moveOnSpline(self, takeoff_height=1.2, vel_max=15.0, acc_max=5.0):
        self.client.moveOnSplineAsync([airsim.Vector3r(0, 0, -takeoff_height)], vel_max=vel_max, acc_max=acc_max, vehicle_name=self.drone_name).join()

    def expert_planner_controller_callback(self):
        self.curr_multirotor_state = self.client.getMultirotorState()
        airsim_xyz = self.curr_multirotor_state.kinematics_estimated.position
        self.curr_xyz = [airsim_xyz.x_val, airsim_xyz.y_val, airsim_xyz.z_val]
        self.got_odom = True

        if ((self.train_lap_idx == 0) and (self.last_gate_passed_idx == -1)):
            if (self.last_gate_idx_moveOnSpline_was_called_on == -1):
                self.fly_to_next_gate_with_moveOnSpline()
                self.last_gate_idx_moveOnSpline_was_called_on = 0
                return
            
        # todo transcribe hackathon shitshow of lists to np arrays
        # todo this NOT foolproof. future self: check for passing inside or outside of gate.
        if (self.curr_track_gate_poses is not None):
            dist_from_next_gate = math.sqrt( (self.curr_xyz[0] - self.curr_track_gate_poses[self.next_gate_idx].position.x_val)**2
                                            + (self.curr_xyz[1] - self.curr_track_gate_poses[self.next_gate_idx].position.y_val)**2
                                            + (self.curr_xyz[2] - self.curr_track_gate_poses[self.next_gate_idx].position.z_val)**2)

            # print(self.last_gate_passed_idx, self.next_gate_idx, dist_from_next_gate)

            if dist_from_next_gate < self.gate_passed_thresh:
                self.last_gate_passed_idx += 1
                self.next_gate_idx += 1
                # self.set_pose_of_gate_just_passed()
                self.set_pose_of_gate_passed_before_the_last_one()

                # if current lap is complete, generate next track
                if (self.last_gate_passed_idx == len(self.curr_track_gate_poses)-1):
                    print("Generating next track")
                    self.last_gate_passed_idx = -1
                    self.next_gate_idx = 0
                    self.curr_track_gate_poses = self.next_track_gate_poses 
                    self.next_track_gate_poses = self.get_next_generated_track()
                    self.train_lap_idx += 1

                    # if last gate of last training lap was just passed, chill out and stop the expert thread!
                    # todo stopping thread from callback seems pretty stupid. watchdog?
                    if (self.train_lap_idx == self.num_training_laps-1):
                        self.stop_expert_planner_controller_thread()

                # todo this is pretty ugly
                if (not(self.last_gate_idx_moveOnSpline_was_called_on == self.next_gate_idx)):
                    self.fly_to_next_gate_with_moveOnSpline()
                    self.last_gate_idx_moveOnSpline_was_called_on = self.next_gate_idx
                # self.fly_to_next_gate_with_learner()
                # self.fly_to_next_gate_with_moveToPostion()

    def set_moveOnSpline_limits(self, vel_max=20.0, acc_max=10.0):
        self.vel_max = vel_max
        self.acc_max = acc_max

    def fly_to_next_gate_with_moveOnSpline(self):
        self.last_future = self.client.moveOnSplineAsync([self.curr_track_gate_poses[self.next_gate_idx].position], vel_max=self.vel_max, acc_max=self.acc_max, vehicle_name=self.drone_name, add_curr_odom_position_constraint=True, add_curr_odom_velocity_constraint= True)

    # maybe maintain a list of futures, or else unreal binary will crash if join() is not called at the end of script
    def join_all_pending_futures(self):
        self.last_future.join()

    def get_next_generated_track(self):
        # todo enable gate spawning in neurips environments for variable number of gates in training laps
        # self.next_track_gate_poses = self.track_generator.generate_gate_poses(num_gates=random.randint(6,10), race_course_radius=30.0, type_of_segment = "circle")
        return self.track_generator.generate_gate_poses(num_gates=len(self.curr_track_gate_poses), \
                                                            race_course_radius=self.race_course_radius, \
                                                            type_of_segment = "circle")

    def set_pose_of_gate_just_passed(self):
        if (self.last_gate_passed_idx == -1):
            return

        self.client.simSetObjectPose(self.gate_object_names_sorted[self.last_gate_passed_idx], self.next_track_gate_poses[self.last_gate_passed_idx])
        # todo unhardcode 100+, ensure unique object ids or just set all non-gate objects to 0, and gates to range(self.next_track_gate_poses)... not needed for hackathon
        # self.client.simSetSegmentationObjectID(self.gate_object_names_sorted[self.last_gate_passed_idx], 100+self.last_gate_passed_idx);
        # todo do we really need this sleep
        time.sleep(0.05)  


    def set_pose_of_gate_passed_before_the_last_one(self):
        gate_idx_to_move = self.last_gate_passed_idx - 1

        # if last_gate passed was -1 or 0, it means the "next" track is already the "current" track. 

        if (self.train_lap_idx > 0):
            if (self.last_gate_passed_idx in [-1,0]):
                print("last_gate_passed_idx", self.last_gate_passed_idx, "moving gate idx from CURRENT track", gate_idx_to_move)
                self.client.simSetObjectPose(self.gate_object_names_sorted[gate_idx_to_move], self.curr_track_gate_poses[gate_idx_to_move])
                return
            else:
                print("last_gate_passed_idx", self.last_gate_passed_idx, "moving gate idx from NEXT track", gate_idx_to_move)
                self.client.simSetObjectPose(self.gate_object_names_sorted[gate_idx_to_move], self.next_track_gate_poses[gate_idx_to_move])
                return

        if (self.train_lap_idx == 0):
            if (self.last_gate_passed_idx in [-1,0]):
                return
            else:
                print("last_gate_passed_idx", self.last_gate_passed_idx, "moving gate idx from NEXT track", gate_idx_to_move)
                self.client.simSetObjectPose(self.gate_object_names_sorted[gate_idx_to_move], self.next_track_gate_poses[gate_idx_to_move])

        # todo unhardcode 100+, ensure unique object ids or just set all non-gate objects to 0, and gates to range(self.next_track_gate_poses)... not needed for hackathon
        # self.client.simSetSegmentationObjectID(self.gate_object_names_sorted[self.last_gate_passed_idx], 100+self.last_gate_passed_idx);
        # todo do we really need this sleep
        time.sleep(0.05)  

    def start_expert_planner_controller_thread(self):
        if not self.is_expert_planner_controller_thread_active:
            self.is_expert_planner_controller_thread_active = True
            self.expert_planner_controller_thread.start()
            print("Started expert_planner_controller thread")

    def stop_expert_planner_controller_thread(self):
        if self.is_expert_planner_controller_thread_active:
            self.is_expert_planner_controller_thread_active = False
            self.expert_planner_controller_thread.join()
            print("Stopped expert_planner_controller thread")

    def set_num_training_laps(self, num_training_laps):
        self.num_training_laps = num_training_laps

    def start_training_data_generator(self, num_training_laps=100, level_name='Soccer_Field_Easy'):
        self.load_level(level_name)
        # todo encapsulate in functions
        self.set_moveOnSpline_limits(vel_max=20.0, acc_max=10.0)
        self.client.enableApiControl(True, vehicle_name=self.drone_name)
        time.sleep(0.01)
        self.client.armDisarm(True, vehicle_name=self.drone_name)
        time.sleep(0.01)
        self.client.setTrajectoryTrackerGains(airsim.TrajectoryTrackerGains().to_list(), vehicle_name=self.drone_name)
        time.sleep(0.01)
        self.takeoff_with_moveOnSpline()
        self.set_num_training_laps(num_training_laps)
        self.start_expert_planner_controller_thread()

if __name__ == "__main__":
    drone_racing_datagenerator = DroneRacingDataGenerator()
    drone_racing_datagenerator.start_training_data_generator()