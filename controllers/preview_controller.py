import numpy as np
import scipy.linalg as linalg

GRAVITY = 9.81


class ZMPPreviewController:

    def __init__(self, dt: float, z_c: float, n_preview: int,
                 q_zmp: float = 1e6, r_jerk: float = 1.0):
        self.dt = dt
        self.z_c = z_c
        self.n_preview = n_preview

        A = np.array([[1.0, dt, (dt ** 2) / 2.0],
                      [0.0, 1.0, dt],
                      [0.0, 0.0, 1.0]])
        B = np.array([[(dt ** 3) / 6.0], [(dt ** 2) / 2.0], [dt]])
        C = np.array([[1.0, 0.0, -z_c / GRAVITY]])

        A_t = np.block([[1.0, C @ A], [np.zeros((3, 1)), A]])
        B_t = np.block([[C @ B], [B]])

        Q = np.zeros((4, 4))
        Q[0, 0] = q_zmp
        R = np.array([[r_jerk]])

        P = linalg.solve_discrete_are(A_t, B_t, Q, R)
        K = np.linalg.inv(R + B_t.T @ P @ B_t) @ (B_t.T @ P @ A_t)

        self.A, self.B, self.C = A, B.flatten(), C
        self.Ks = float(K[0, 0])
        self.Kx = K[0, 1:4]

        Ac_t = A_t - B_t @ K
        self.F = np.zeros(n_preview)
        X_t = -Ac_t.T @ P @ np.block([[np.array([[1.0]])], [np.zeros((3, 1))]])
        inv = np.linalg.inv(R + B_t.T @ P @ B_t)
        for i in range(n_preview):
            self.F[i] = float((inv @ B_t.T @ X_t)[0, 0])
            X_t = Ac_t.T @ X_t

        self.reset()

    def reset(self, pos: float = 0.0):
        self.state = np.array([pos, 0.0, 0.0])
        self.error_sum = 0.0

    @property
    def omega(self) -> float:
        return np.sqrt(GRAVITY / self.z_c)

    @property
    def zmp(self) -> float:
        return float((self.C @ self.state).item())

    @property
    def position(self) -> float:
        return float(self.state[0])

    @property
    def velocity(self) -> float:
        return float(self.state[1])

    def dcm(self, omega: float = None) -> float:
        w = self.omega if omega is None else omega
        return self.state[0] + self.state[1] / w

    def update(self, zmp_ref_now: float, zmp_ref_preview: np.ndarray) -> np.ndarray:
        self.error_sum += self.zmp - zmp_ref_now
        u = (-self.Ks * self.error_sum
             - float(np.dot(self.Kx, self.state))
             - float(np.dot(self.F, zmp_ref_preview)))
        self.state = self.A @ self.state + self.B * u
        return self.state
