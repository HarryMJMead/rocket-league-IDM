import numpy as np
import numba

def quats_to_rot_mtx(quats: np.ndarray) -> np.ndarray:
    # From rlgym.utils.math.quat_to_rot_mtx
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

    return theta

# from Rolv-Arid's replay-pretraining
@numba.njit
def corrupted_indices(k, n):
    indices = np.zeros((k, n))
    for j in range(k):
        i = 0
        while i < n:
            r = np.random.random()
            if r < 0.075:
                repeats = 1
            elif r < 0.7:
                repeats = 2
            else:
                repeats = 3
            indices[j, i:i + repeats] = i
            i += repeats
    return indices