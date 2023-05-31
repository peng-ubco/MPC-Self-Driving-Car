import numpy as np
from sim_2d import sim_run

# Simulator options.
sim_opt = {}
sim_opt['OBSTACLES'] = True
sim_opt['FIG_SIZE'] = [7.5, 7.5]


class MPC:
    def __init__(self):
        self.horizon = 15
        self.dt = 0.1

        # Reference or set point
        self.reference1 = [10, 0, 0]
        self.reference2 = None

        self.x_obs = 5
        self.y_obs = 0.1

    def motion_model(self, state, dt, pedal, steering):
        # Assume we know the state of the car,
        # and state_vector=[pos_x, pos_y, car_angle, velocity]
        # Two control inputs: pedal (pedal_position), and steering (steering_angle)
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
    # def cost_func(self, u, *args):
    #     state = args[0]
    #     ref = args[1]
        cost = 0.0
        for i in range(0, self.horizon):
            state = self.motion_model(state, self.dt, u[i * 2], u[i * 2 + 1])
            cost += abs(state[0] - ref[0])**2
            cost += abs(state[1] - ref[1])**2
            cost += abs(state[2]-ref[2])**2
            distance = np.sqrt((state[0]-self.x_obs)**2 + (state[1] - self.y_obs)**2)
            if distance > 2:
                cost_obs = 15
            else:
                cost_obs = 1/max(distance,0.1) * 30  # cost_obs = 30 * 1 / distance ** 2
            cost += cost_obs
        return cost

sim_run(sim_opt, MPC)
