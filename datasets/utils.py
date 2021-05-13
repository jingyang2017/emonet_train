import numpy as np

def get_scale_center(bb, scale_=220.0):
    center = np.array([bb[2] - (bb[2] - bb[0]) / 2, bb[3] - (bb[3] - bb[1]) / 2])
    scale = (bb[2] - bb[0] + bb[3] - bb[1]) / scale_

    return scale, center


def inv_mat(mat):
    ans = np.linalg.pinv(np.array(mat).tolist() + [[0, 0, 1]])
    return ans[:2]


def get_transform(center, scale, res, rot=0):
    # Generate transformation matrix

    h = 200 * scale
    t = np.zeros((3, 3))
    t[0, 0] = float(res[1]) / h
    t[1, 1] = float(res[0]) / h
    t[0, 2] = res[1] * (-float(center[0]) / h + .5)
    t[1, 2] = res[0] * (-float(center[1]) / h + .5)
    t[2, 2] = 1

    if not rot == 0:
        rot = -rot  # To match direction of rotation from cropping
        rot_mat = np.zeros((3, 3))
        rot_rad = rot * np.pi / 200
        sn, cs = np.sin(rot_rad), np.cos(rot_rad)
        rot_mat[0, :2] = [cs, -sn]
        rot_mat[1, :2] = [sn, cs]
        rot_mat[2, 2] = 1
        # Need to rotate around center
        t_mat = np.eye(3)
        t_mat[0, 2] = -res[1] / 2
        t_mat[1, 2] = -res[0] / 2
        t_inv = t_mat.copy()
        t_inv[:2, 2] *= -1
        t = np.dot(t_inv, np.dot(rot_mat, np.dot(t_mat, t)))

    return t