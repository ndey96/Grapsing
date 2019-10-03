import numpy as np


def perturb_xyz(pose, perturbations):
    R = axis_angle_to_rotmat(*(pose[3:6] * pose[6]))
    H = homogenous_transform(R, pose[0:3])
    return H @ perturbations


def axis_angle_to_rotmat(Rx, Ry, Rz):

    R = np.zeros((3, 3), dtype=float)
    a1 = [Rx, Ry, Rz]
    angle = np.linalg.norm(a1)
    a1 = a1 / angle

    c = np.cos(angle)
    s = np.sin(angle)
    t = 1.0 - c

    R[0, 0] = c + a1[0] * a1[0] * t
    R[1, 1] = c + a1[1] * a1[1] * t
    R[2, 2] = c + a1[2] * a1[2] * t

    tmp1 = a1[0] * a1[1] * t
    tmp2 = a1[2] * s
    R[1, 0] = tmp1 + tmp2
    R[0, 1] = tmp1 - tmp2

    tmp1 = a1[0] * a1[2] * t
    tmp2 = a1[1] * s
    R[2, 0] = tmp1 - tmp2
    R[0, 2] = tmp1 + tmp2

    tmp1 = a1[1] * a1[2] * t
    tmp2 = a1[0] * s
    R[2, 1] = tmp1 + tmp2
    R[1, 2] = tmp1 - tmp2

    return R


def homogenous_transform(R, vect):
    '''
    :param R: 3x3 matrix
    :param vect: list x,y,z
    :return:Homogenous transformation 4x4 matrix using R and vect
    '''

    H = np.zeros((4, 4))
    H[0:3, 0:3] = R
    frame_displacement = vect + [1]
    D = np.array(frame_displacement)
    D.shape = (1, 4)
    H[:, 3] = D
    return H


def inverse_homogenous_transform(H):
    '''
    :param H: Homogenous Transform Matrix
    :return: Inverse Homegenous Transform Matrix
    '''

    R = H[0:3, 0:3]
    origin = H[:-1, 3]
    origin.shape = (3, 1)

    R = R.T
    origin = -R.dot(origin)
    return homogenous_transform(R, list(origin.flatten()))


def rotmat_to_axis_angle(R):

    r00 = R[0, 0]
    r01 = R[0, 1]
    r02 = R[0, 2]
    r10 = R[1, 0]
    r11 = R[1, 1]
    r12 = R[1, 2]
    r20 = R[2, 0]
    r21 = R[2, 1]
    r22 = R[2, 2]
    # catch the error
    angle = (r00 + r11 + r22 - 1) / 2
    if angle > 1:
        angle = 0.99999
    elif angle < -1:
        angle = -0.99999
    theta = math.acos(angle)
    sinetheta = math.sin(theta)
    v = (2 * sinetheta) * theta

    cz = ((r10 - r01) / (2 * sinetheta)) * theta
    by = ((r02 - r20) / (2 * sinetheta)) * theta
    ax = ((r21 - r12) / (2 * sinetheta)) * theta

    return ax, by, cz
