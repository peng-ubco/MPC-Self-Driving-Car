import numpy as np
from sim_2d import sim_run

# Simulator options.
sim_opt = {}
sim_opt['OBSTACLES'] = False
sim_opt['FIG_SIZE'] = [7.5, 7.5]


class MPC:
    def __init__(self):
        self.horizon = 15
        self.dt = 0.1

        # Reference or set point
        self.reference1 = [10, 10, 0]
        self.reference2 = [10, 2, 3*3.14/2]

    def motion_model(self, state, dt, pedal, steering):
        # Assume we know the state of the car,
        # and state_vector=[pos_x, pos_y, car_angle, velocity]
        # Two control inputs: pedal: pedal position, and steering: steering angle

        # Get the state of the car
        pos_x = state[0]
        pos_y = state[1]
        psi = state[2]
        v = state[3]  # m/s

        alpha = pedal  # control input 1: pedal_position
        delta = steering  # control input 2: steering_angle

        pos_x += v * np.cos(psi) * dt
        pos_y += v * np.sin(psi) * dt
        # Assume acceleration = pedal_position * 5;
        # Assume a natural resistive force "-v/25" from air friction
        v += alpha * 5 * dt - v / 25
        psi += v * np.tan(delta) / 2.5 * dt  # Length of the car is 2.5 m
        # Return the predicted new state
        return [pos_x, pos_y, psi, v]

    def cost_func(self, u, state, ref):
        cost = 0.0
        for i in range(0, self.horizon):
            v_start = state[3]
            state = self.motion_model(state, self.dt, u[2*i], u[2*i+1])
            cost += abs(state[0] - ref[0])
            cost += abs(state[1] - ref[1])
            cost += abs(state[2] - ref[2])**2

            # cost += abs(state[0] - ref[0])**2 + abs(state[1] - ref[1])**2 + 2*abs(state[2] - ref[2])**3

            ## If smooth driving is required:
            # speed = state[3]
            # if speed > 1:
            #     cost += (speed - v_start)**2 * 2
            # cost += u[2 * i] ** 2 * self.dt
            # cost += u[2*i+1]**2 * self.dt * 5

        return cost


sim_run(sim_opt, MPC)
