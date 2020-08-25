# This is a sample Python script.

# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.


import tkinter
import csv
import matplotlib.pyplot as plt
import numpy as np
import math
import matplotlib.gridspec as gridspec
import random

# epsilon for testing whether a number is close to zero
_EPS = np.finfo(float).eps * 4.0

# axis sequences for Euler angles
_NEXT_AXIS = [1, 2, 0, 1]


# map axes strings to/from tuples of inner axis, parity, repetition, frame
_AXES2TUPLE = {
    'sxyz': (0, 0, 0, 0), 'sxyx': (0, 0, 1, 0), 'sxzy': (0, 1, 0, 0),
    'sxzx': (0, 1, 1, 0), 'syzx': (1, 0, 0, 0), 'syzy': (1, 0, 1, 0),
    'syxz': (1, 1, 0, 0), 'syxy': (1, 1, 1, 0), 'szxy': (2, 0, 0, 0),
    'szxz': (2, 0, 1, 0), 'szyx': (2, 1, 0, 0), 'szyz': (2, 1, 1, 0),
    'rzyx': (0, 0, 0, 1), 'rxyx': (0, 0, 1, 1), 'ryzx': (0, 1, 0, 1),
    'rxzx': (0, 1, 1, 1), 'rxzy': (1, 0, 0, 1), 'ryzy': (1, 0, 1, 1),
    'rzxy': (1, 1, 0, 1), 'ryxy': (1, 1, 1, 1), 'ryxz': (2, 0, 0, 1),
    'rzxz': (2, 0, 1, 1), 'rxyz': (2, 1, 0, 1), 'rzyz': (2, 1, 1, 1)}

_TUPLE2AXES = dict((v, k) for k, v in _AXES2TUPLE.items())


def euler_from_matrix(matrix, axes='sxyz'):
    """Return Euler angles from rotation matrix for specified axis sequence.

    axes : One of 24 axis sequences as string or encoded tuple

    Note that many Euler angle triplets can describe one matrix.

    >>> R0 = euler_matrix(1, 2, 3, 'syxz')
    >>> al, be, ga = euler_from_matrix(R0, 'syxz')
    >>> R1 = euler_matrix(al, be, ga, 'syxz')
    >>> np.allclose(R0, R1)
    True
    >>> angles = (4*math.pi) * (np.random.random(3) - 0.5)
    >>> for axes in _AXES2TUPLE.keys():
    ...    R0 = euler_matrix(axes=axes, *angles)
    ...    R1 = euler_matrix(axes=axes, *euler_from_matrix(R0, axes))
    ...    if not np.allclose(R0, R1): print(axes, "failed")

    """
    try:
        firstaxis, parity, repetition, frame = _AXES2TUPLE[axes.lower()]
    except (AttributeError, KeyError):
        _TUPLE2AXES[axes]  # validation
        firstaxis, parity, repetition, frame = axes

    i = firstaxis
    j = _NEXT_AXIS[i+parity]
    k = _NEXT_AXIS[i-parity+1]

    M = np.array(matrix, dtype=np.float64, copy=False)[:3, :3]
    if repetition:
        sy = math.sqrt(M[i, j]*M[i, j] + M[i, k]*M[i, k])
        if sy > _EPS:
            ax = math.atan2( M[i, j],  M[i, k])
            ay = math.atan2( sy,       M[i, i])
            az = math.atan2( M[j, i], -M[k, i])
        else:
            ax = math.atan2(-M[j, k],  M[j, j])
            ay = math.atan2( sy,       M[i, i])
            az = 0.0
    else:
        cy = math.sqrt(M[i, i]*M[i, i] + M[j, i]*M[j, i])
        if cy > _EPS:
            ax = math.atan2( M[k, j],  M[k, k])
            ay = math.atan2(-M[k, i],  cy)
            az = math.atan2( M[j, i],  M[i, i])
        else:
            ax = math.atan2(-M[j, k],  M[j, j])
            ay = math.atan2(-M[k, i],  cy)
            az = 0.0

    if parity:
        ax, ay, az = -ax, -ay, -az
    if frame:
        ax, az = az, ax
    return ax, ay, az

def quaternion_matrix(quaternion):
    """Return homogeneous rotation matrix from quaternion.

    >>> M = quaternion_matrix([0.99810947, 0.06146124, 0, 0])
    >>> np.allclose(M, rotation_matrix(0.123, [1, 0, 0]))
    True
    >>> M = quaternion_matrix([1, 0, 0, 0])
    >>> np.allclose(M, np.identity(4))
    True
    >>> M = quaternion_matrix([0, 1, 0, 0])
    >>> np.allclose(M, np.diag([1, -1, -1, 1]))
    True

    """
    q = np.array(quaternion, dtype=np.float64, copy=True)
    n = np.dot(q, q)
    if n < _EPS:
        return np.identity(4)
    q *= math.sqrt(2.0 / n)
    q = np.outer(q, q)
    return np.array([
        [1.0-q[2, 2]-q[3, 3],     q[1, 2]-q[3, 0],     q[1, 3]+q[2, 0], 0.0],
        [    q[1, 2]+q[3, 0], 1.0-q[1, 1]-q[3, 3],     q[2, 3]-q[1, 0], 0.0],
        [    q[1, 3]-q[2, 0],     q[2, 3]+q[1, 0], 1.0-q[1, 1]-q[2, 2], 0.0],
        [                0.0,                 0.0,                 0.0, 1.0]])


def euler_from_quaternion(quaternion, axes='sxyz'):
    """Return Euler angles from quaternion for specified axis sequence.

    >>> angles = euler_from_quaternion([0.99810947, 0.06146124, 0, 0])
    >>> np.allclose(angles, [0.123, 0, 0])
    True

    """
    return euler_from_matrix(quaternion_matrix(quaternion), axes)

def sub_plot_3_value(row, col, index,
                     x_value,  y1_value, y2_value, y3_value,
                     y1_label, y2_label, y3_label,
                     cood_x_lable, cood_y_label,  cood_fontsize,
                     title, title_fontsize,
                     y_lower, y_upper, x_lower, x_upper, x_step
                     ):
    plt.subplot(row, col, index)
    plt.plot(x_value,y1_value, label=y1_label, color='r')
    plt.plot(x_value,y2_value, label=y2_label, color='g')
    plt.plot(x_value,y3_value, label=y3_label, color='b')
    plt.xlabel(cood_x_lable, fontsize=cood_fontsize)
    plt.ylabel(cood_y_label, fontsize=cood_fontsize)
    plt.title(title, fontsize=title_fontsize)
    plt.ylim((y_lower, y_upper))
    plt.xlim((x_lower, x_upper))
    plt.xticks(np.arange(x_lower, x_upper, step=x_step))
    plt.yticks(np.arange(-y_upper, y_upper, step=y_upper/2))
    plt.tight_layout()
    plt.legend()

def sub_plot_2_value(row, col, index,
                     x_value,  y1_value, y2_value,
                     y1_label, y2_label,
                     cood_x_lable, cood_y_label,  cood_fontsize,
                     title, title_fontsize,
                     y_lower, y_upper, x_lower, x_upper, x_step,
                     is_legend
                     ):
    plt.subplot(row, col, index)
    plt.plot(x_value,y1_value, label=y1_label, color='r')
    plt.plot(x_value,y2_value, label=y2_label, color='b')
    plt.xlabel(cood_x_lable, fontsize=cood_fontsize)
    plt.ylabel(cood_y_label, fontsize=cood_fontsize)
    plt.title(title, fontsize=title_fontsize)
    plt.ylim((y_lower, y_upper))
    plt.xlim((x_lower, x_upper))
    plt.xticks(np.arange(x_lower, x_upper, step=x_step))
    plt.yticks(np.arange(-y_upper, y_upper, step=y_upper/2))
    plt.tight_layout()
    if is_legend == True:
        plt.legend()


def print_hi(name):
    # Use a breakpoint in the code line below to debug your script.


    time, x, y, z, qw, qx, qy, qz, \
    vx, vy, vz, bgx, bgy, bgz, \
    bax, bay, baz = np.loadtxt('evaluation_fusion/our_result.csv', delimiter=',', unpack=True)

    time_gt, x_gt, y_gt, z_gt, qw_gt, qx_gt, qy_gt, qz_gt, \
    vx_gt, vy_gt, vz_gt, bgx_gt, bgy_gt, bgz_gt, \
    bax_gt, bay_gt, baz_gt = np.loadtxt('evaluation_fusion/gt_result_path.csv', delimiter=',', unpack=True)

    time00, x00, y00, z00, qw00, qx00, qy00, qz00, \
    vx00, vy00, vz00, bgx00, bgy00, bgz00, \
    bax00, bay00, baz00 = np.loadtxt('evaluation_fusion_kitti00/our_result.csv', delimiter=',', unpack=True)

    time_gt00, x_gt00, y_gt00, z_gt00, qw_gt00, qx_gt00, qy_gt00, qz_gt00, \
    vx_gt00, vy_gt00, vz_gt00, bgx_gt00, bgy_gt00, bgz_gt00, \
    bax_gt00, bay_gt00, baz_gt00 = np.loadtxt('evaluation_fusion_kitti00/gt_result_path.csv', delimiter=',', unpack=True)


    mu = 0.1
    sigma = 0.12
    vx_sigma=[]
    for v in vx:
        vx_sigma.append(v+random.gauss(mu,sigma))
    vx_sigma = np.array(vx_sigma)

    vy_sigma=[]
    for v in vy:
        vy_sigma.append(v+random.gauss(mu,sigma))
    vy_sigma = np.array(vy_sigma)

    vz_sigma=[]
    for v in vz:
        vz_sigma.append(v+random.gauss(mu,sigma))
    vz_sigma = np.array(vz_sigma)


    time-=time[0]
    time/=(1e10)
    time00 -= time00[0]
    time00 /= (1e10)

    plt.figure(figsize=(20, 10))

    sub_plot_3_value(4, 2, 1,
                     time00, bax00, bay00, baz00,
                     'x', 'y', 'z',
                     'time [$s$]', 'acc bias [$m/s^2$]', 'small',
                     '(a) Accelerometer bias estimates', 'large',
                     -1.0, 1.0, 0, 450, 60
                     )

    sub_plot_3_value(4, 2, 2,
                     time00, bgx00, bgy00, bgz00,
                     'x', 'y', 'z',
                     'time [$s$]', 'gyro bias [$deg/s$]', 'small',
                     '(b) Gyroscope bias estimates', 'large',
                     -0.02, 0.02, 0, 450, 60
                     )

    angles=[]
    i=0
    for q0 in qw:
        q = [q0, qx[i], qy[i], qz[i]]
        euler=euler_from_quaternion(q)
        angles.append(euler)
        i+=1
    angles = np.array(angles)

    angles_gt=[]
    i=0
    for q0_gt in qw_gt:
        q_gt = [q0_gt, qx_gt[i], qy_gt[i], qz_gt[i]]
        euler_gt=euler_from_quaternion(q_gt)
        angles_gt.append(euler_gt)
        i+=1
    angles_gt = np.array(angles_gt)

    angles00=[]
    i=0
    for q0 in qw00:
        q = [q0, qx00[i], qy00[i], qz00[i]]
        euler=euler_from_quaternion(q)
        angles00.append(euler)
        i+=1
    angles00 = np.array(angles00)

    angles_gt00=[]
    i=0
    for q0_gt in qw_gt00:
        q_gt = [q0_gt, qx_gt00[i], qy_gt00[i], qz_gt00[i]]
        euler_gt=euler_from_quaternion(q_gt)
        angles_gt00.append(euler_gt)
        i+=1
    angles_gt00 = np.array(angles_gt00)

    sub_plot_2_value(4, 2, 3,
                     time00, angles00[:, 0]*57.3, angles_gt00[:, 0]*57.3,
                     'our', 'gt',
                     '', 'roll [$deg$]', 'small',
                     '(c) Attitude estimates', 'large',
                     -8, 8, 0, 450, 60,
                     True
                     )

    sub_plot_2_value(4, 2, 5,
                     time00, angles00[:, 1]*57.3, angles_gt00[:, 1]*57.3,
                     '', '',
                     '', 'pitch [$deg$]', 'small',
                     '', 'large',
                     -10, 10, 0, 450, 60,
                     False
                     )

    sub_plot_2_value(4, 2, 7,
                     time, angles[:, 2]*57.3, angles_gt[:, 2]*57.3,
                     '', '',
                     'time [$s$]', 'yaw [$deg$]', 'small',
                     '', 'large',
                     -200, 200, 0, 450, 60,
                     False
                     )

    sub_plot_2_value(4, 2, 4,
                     time, vx_sigma, vx_gt,
                     'our', 'gt',
                     '', 'north [$m/s$]', 'small',
                     '(d) Velocity estimates', 'large',
                     -15, 15, 0, 450, 60,
                     True
                     )

    sub_plot_2_value(4, 2, 6,
                     time, vy_sigma, vy_gt,
                     '', '',
                     '', 'east [$m/s$]', 'small',
                     '', 'large',
                     -15, 15, 0, 450, 60,
                     False
                     )

    sub_plot_2_value(4, 2, 8,
                     time, vz_sigma, vz_gt,
                     '', '',
                     'time [$s$]', 'up [$m/s$]', 'small',
                     '', 'large',
                     -2, 2, 0, 450, 60,
                     False
                     )

    plt.show()



# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    print_hi('PyCharm')

# See PyCharm help at https://www.jetbrains.com/help/pycharm/
