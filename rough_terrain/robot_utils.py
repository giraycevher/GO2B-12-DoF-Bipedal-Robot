import numpy as np


class DynamicStepPlanner:
    def __init__(self, footstep_matrix, step_time, initial_delay):
        self.footsteps = footstep_matrix
        self.step_time = step_time
        self.initial_delay = initial_delay
        self.total_steps = len(footstep_matrix)

        MAX = 20000
        self.p_ref_x = np.zeros(MAX)
        self.p_ref_y = np.zeros(MAX)

        self._create_zmp_references()

    def _create_zmp_references(self):
        for i in range(self.total_steps):
            start_idx = self.initial_delay + i * self.step_time
            end_idx = start_idx + self.step_time

            px = self.footsteps[i, 0]
            py = self.footsteps[i, 1]

            if start_idx < len(self.p_ref_x):
                self.p_ref_x[start_idx:min(end_idx, len(self.p_ref_x))] = px
                self.p_ref_y[start_idx:min(end_idx, len(self.p_ref_y))] = py

        last_idx = self.initial_delay + self.total_steps * self.step_time

        if last_idx < len(self.p_ref_x):
            self.p_ref_x[last_idx:] = self.footsteps[-1, 0]
            self.p_ref_y[last_idx:] = self.footsteps[-1, 1]

    def swing_target(self, step_index, state_x, state_y):
        if step_index < self.initial_delay:
            return 0.0, 0.0, False

        current_step = step_index - self.initial_delay
        step_number = current_step // self.step_time
        target_step = step_number + 1

        if target_step < self.total_steps:
            nominal_x = self.footsteps[target_step, 0]
            nominal_y = self.footsteps[target_step, 1]
        else:
            nominal_x = self.footsteps[-1, 0]
            nominal_y = self.footsteps[-1, 1]

        is_left = (target_step % 2 == 0)

        # Push recovery gains
        k_p_x, k_v_x = 0.15, 0.12
        k_p_y, k_v_y = 0.15, 0.12

        com_x, com_vx = state_x[0], state_x[1]
        com_y, com_vy = state_y[0], state_y[1]

        ref_zmp_x = self.p_ref_x[min(step_index, len(self.p_ref_x) - 1)]
        ref_zmp_y = self.p_ref_y[min(step_index, len(self.p_ref_y) - 1)]

        error_x = com_x - ref_zmp_x
        error_y = com_y - ref_zmp_y

        delta_x = k_p_x * error_x + k_v_x * com_vx
        delta_y = k_p_y * error_y + k_v_y * com_vy

        delta_x = np.clip(delta_x, -0.06, 0.06)
        delta_y = np.clip(delta_y, -0.05, 0.05)

        return nominal_x + delta_x, nominal_y + delta_y, is_left


class JoystickStepPlanner:
    def __init__(self, step_time, inital_delay):
        self.step_time = step_time
        self.inital_delay = inital_delay

        # Bacak açıklıkları
        self.D_left = 0.05
        self.D_right = -0.05

        # MPC için ZMP referans tamponu
        self.p_ref_x = [0.0] * inital_delay
        self.p_ref_y = [0.0] * inital_delay

        # Gelecekteki ayak basma noktalarının listesi (x, y, left_swing)
        self.footsteps = []

        self.current_x = 0.0
        self.current_y = 0.0
        self.is_left_stance = True

        # Hareketsiz duruş için ilk birkaç adımı sisteme yükle
        for _ in range(3):
            self._generate_next_step(0.0, 0.0)

    def _generate_next_step(self, dx, dy):
        self.current_x += dx

        if self.is_left_stance:
            self.current_y = dy + self.D_right
            left_swing = False
        else:
            self.current_y = dy + self.D_left
            left_swing = True

        self.footsteps.append((self.current_x, self.current_y, left_swing))

        for _ in range(self.step_time):
            self.p_ref_x.append(self.current_x)
            self.p_ref_y.append(self.current_y)

        self.is_left_stance = not self.is_left_stance

    def update_future_steps(self, target_dx, target_dy, step_index, N_preview):
        while len(self.p_ref_x) < step_index + N_preview + self.step_time:
            self._generate_next_step(target_dx, target_dy)

    def swing_target(self, step_index, state_x, state_y):
        if step_index < self.inital_delay:
            return 0.0, self.D_right, False

        real_time = step_index - self.inital_delay
        step_count = real_time // self.step_time

        step_count = min(step_count, len(self.footsteps) - 1)

        target_x, target_y, left_swing = self.footsteps[step_count]
        return target_x, target_y, left_swing