import numpy as np

def quat_to_rot_mtx(quats: np.ndarray) -> np.ndarray:
    w = -quats[:, 0]
    x = -quats[:, 1]
    y = -quats[:, 2]
    z = -quats[:, 3]

    theta = np.zeros((quats.shape[0], 3, 3))

    norm = np.einsum("fq,fq->f", quats, quats)

    sel = norm != 0

    w = w[sel]
    x = x[sel]
    y = y[sel]
    z = z[sel]

    s = 1.0 / norm[sel]

    # front direction
    theta[sel, 0, 0] = 1.0 - 2.0 * s * (y * y + z * z)
    theta[sel, 1, 0] = 2.0 * s * (x * y + z * w)
    theta[sel, 2, 0] = 2.0 * s * (x * z - y * w)

    # left direction
    theta[sel, 0, 1] = 2.0 * s * (x * y - z * w)
    theta[sel, 1, 1] = 1.0 - 2.0 * s * (x * x + z * z)
    theta[sel, 2, 1] = 2.0 * s * (y * z + x * w)

    # up direction
    theta[sel, 0, 2] = 2.0 * s * (x * z + y * w)
    theta[sel, 1, 2] = 2.0 * s * (y * z - x * w)
    theta[sel, 2, 2] = 1.0 - 2.0 * s * (x * x + y * y)

    return np.concatenate((theta[:, :, 0], theta[:, :, 1], theta[:, :, 2]), 1)

def make_lookup_table():
        actions = []
        # Ground
        for throttle in (-1, 0, 1):
            for steer in (-1, 0, 1):
                for boost in (0, 1):
                    for handbrake in (0, 1):
                        if boost == 1 and throttle != 1:
                            continue
                        actions.append([throttle or boost, steer, 0, steer, 0, 0, boost, handbrake])
        # Aerial
        for pitch in (-1, 0, 1):
            for yaw in (-1, 0, 1):
                for roll in (-1, 0, 1):
                    for jump in (0, 1):
                        for boost in (0, 1):
                            if jump == 1 and yaw != 0:  # Only need roll for sideflip
                                continue
                            if pitch == roll == jump == 0:  # Duplicate with ground
                                continue
                            # Enable handbrake for potential wavedashes
                            handbrake = jump == 1 and (pitch != 0 or yaw != 0 or roll != 0)
                            actions.append([boost, yaw, pitch, yaw, roll, jump, boost, handbrake])
        actions = np.array(actions)
        return actions