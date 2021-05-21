from __future__ import print_function
import numpy as np
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
from scipy.stats import norm
from sympy import Symbol, symbols, Matrix, sin, cos, sqrt, atan2
from sympy import init_printing

init_printing(use_latex=True)
import numdifftools as nd
import math

dataset = []

# read the measurement data, use 0.0 to stand LIDAR data and 1.0 stand RADAR data
# 在数据文件中，各列代表的意义如下：第 1 列使用 L/R 来表示测量的数据是来自 LIDAR 还是 RADAR，第 2,3 列表示测量的目标 (x,y)，
# 第 4 列表示测量的时间点，第 5,6,7 列表示的是真实的 (x, y, vx, vy)，如果是 RADAR 的话，那么测量的前 3 列表示的是 (p, theta, vp)
with open('data_synthetic.txt', 'rb') as f:
    lines = f.readlines()
    for line in lines:
        line = line.decode().strip('\n')
        line = line.strip()
        numbers = line.split()
        # result 列表用来存储一行的数据
        result = []
        for i, item in enumerate(numbers):
            item.strip()

            # 第 1 列（这里下标为 0）表示的是 LIDAR 还是 RADAR
            if i == 0:
                # 如果是 LIDAR，则用 0 表示；如果是 RADAR，则用 1 表示
                if item == 'L':
                    result.append(0.0)
                else:
                    result.append(1.0)
            else:
                result.append(float(item))

        dataset.append(result)
    f.close()

# 初始化 P 矩阵
P = np.diag([1.0, 1.0, 1.0, 1.0, 1.0])
print(' P 矩阵: ', P, ' P shape: ', P.shape)

# 初始化激光雷达的测量矩阵（线性）HL
H_lidar = np.array([[1., 0., 0., 0., 0.],
                    [0., 1., 0., 0., 0.]])
print(H_lidar, H_lidar.shape)

# 初始化测量噪声 R
R_lidar = np.array([[0.0225, 0.], [0., 0.0225]])
R_radar = np.array([[0.09, 0., 0.], [0., 0.0009, 0.], [0., 0., 0.09]])
print('lidar 噪声:', R_lidar, R_lidar.shape)
print('radar 噪声:', R_radar, R_radar.shape)

# 分别处理噪声中的直线加速度项的标准差以及角速度项的标准差（即下面的 yaw_dd）
# process noise standard deviation for a
std_noise_a = 2.0
# process noise standard deviation for yaw acceleration
std_noise_yaw_dd = 0.3


def control_psi(psi):
    while psi > np.pi or psi < -np.pi:
        if psi > np.pi:
            psi = psi - 2 * np.pi
        if psi < -np.pi:
            psi = psi + 2 * np.pi
    return psi


# state 为物体的运动状态，有 5 个分量，即 [x，y，v，theta，w]，其中 w 表示物体的角速度，而 theta 表示物体的速度和 x 轴形成的夹角
state = np.zeros(5)
# 使用第一个雷达的测量数据初始化我们的状态
init_measurement = dataset[0]
current_time = 0.0

# init_measurement 数据中的第 1 列表示是 LIDAR 还是 RADAR 测量的数据
# LIDAR，对于激光雷达数据，可以直接将测量到的目标的 (x,y) 坐标作为初始 (x,y)，其余状态项初始化为 0
if init_measurement[0] == 0.0:
    print('Initialize with LIDAR measurement!')
    # init_measurement 中的第 4 列表示的是测量的时间
    current_time = init_measurement[3]
    # init_measurement 中的第 2,3 列分别表示测量得到的 x,y 坐标
    state[0] = init_measurement[1]
    state[1] = init_measurement[2]
# RADAR
else:
    print('Initialize with RADAR measurement!')
    # init_measurement 中的第 5 列表示测量的时间
    current_time = init_measurement[4]
    # rho 表示的是测量到的位置与当前车辆之间的距离
    init_rho = init_measurement[1]
    # psi 表示目标车辆与 x 轴的夹角
    init_psi = init_measurement[2]
    init_psi = control_psi(init_psi)
    state[0] = init_rho * np.cos(init_psi)
    state[1] = init_rho * np.sin(init_psi)

print(state, state.shape)

# pre-allocation for saving
px = []
py = []
vx = []
vy = []

gpx = []
gpy = []
gvx = []
gvy = []

mx = []
my = []


def save_states(ss, gx, gy, gv1, gv2, m1, m2):
    px.append(ss[0])
    py.append(ss[1])
    vx.append(np.cos(ss[3]) * ss[2])
    vy.append(np.sin(ss[3]) * ss[2])

    gpx.append(gx)
    gpy.append(gy)
    gvx.append(gv1)
    gvy.append(gv2)
    mx.append(m1)
    my.append(m2)

measurement_step = len(dataset)
# state 原本为一个长度为 5 的行向量，在这里转变为一个列向量
state = state.reshape([5, 1])
# 这里我们先设 t=0.05，只是为了占一个位置，当实际运行 EKF 时会计算出前后两次测量的时间差，一次来替换这里的  Δt
dt = 0.05

I = np.eye(5)

# 表示一个状态转移函数，这里的 y 代表上面的 state 列向量
# 在线性的 KF 算法中，叫做状态转移矩阵，但是 EKF 这里是非线性的，所以无法表示成矩阵的形式，因此称为状态转移函数
# y[0] = x, y[1] = y, y[2] = v, y[3] = theta, y[4] = w（表示恒定角速度）
transition_function = lambda y: np.vstack((
    # x + v/w * [sin(theta + w * Δt) - sin(theta)]
    y[0] + (y[2] / y[4]) * (np.sin(y[3] + y[4] * dt) - np.sin(y[3])),
    # y + v/w * [cos(theta) - cos(theta + w * Δt)]
    y[1] + (y[2] / y[4]) * (-np.cos(y[3] + y[4] * dt) + np.cos(y[3])),
    # v
    y[2],
    # theta + w * Δt
    y[3] + y[4] * dt,
    # w
    y[4]))

# when omega is 0
# 这个状态转移函数是当 w 小于一定的数值（默认为 e-4），就认为 w 为 0，物体做匀速直线运动，CTRV 模型就变为 CV 模型
transition_function_1 = lambda m: np.vstack((
    # x + v * cos(theta) * Δt
    m[0] + m[2] * np.cos(m[3]) * dt,
    # y + v * sin(theta) * Δt
    m[1] + m[2] * np.sin(m[3]) * dt,
    # v
    m[2],
    # theta + w * Δt
    m[3] + m[4] * dt,
    # w
    m[4]))

# 使用 numdifftools 库来计算上面两个状态转移函数对应的雅克比矩阵
J_A = nd.Jacobian(transition_function)
J_A_1 = nd.Jacobian(transition_function_1)

# 定义了一个测量函数
measurement_function = lambda k: np.vstack((
    # (x * x + y * y) ^ (0.5)
    np.sqrt(k[0] * k[0] + k[1] * k[1]),
    # arctan(y / x)
    math.atan2(k[1], k[0]),
    # v (x * cos + y * sin) / ((x * x + y * y) ^ (0.5))
    (k[0] * k[2] * np.cos(k[3]) + k[1] * k[2] * np.sin(k[3])) / np.sqrt(k[0] * k[0] + k[1] * k[1])))

# 使用 numdifftools 库来计算测量函数对应的雅克比矩阵
J_H = nd.Jacobian(measurement_function)

for step in range(1, measurement_step):

    # Prediction
    t_measurement = dataset[step]
    # 如果是 LIDAR 的话
    if t_measurement[0] == 0.0:
        m_x = t_measurement[1]
        m_y = t_measurement[2]
        z = np.array([[m_x], [m_y]])

        # 计算出当前测量的时间点和上一次测量时间之差
        dt = (t_measurement[3] - current_time) / 1000000.0
        current_time = t_measurement[3]

        # 如果是 LIDAR 的话，第 5,6,7,8 列表示真实的 [x, y, vx, vy]
        g_x = t_measurement[4]
        g_y = t_measurement[5]
        g_v_x = t_measurement[6]
        g_v_y = t_measurement[7]

    # 如果是 RADAR 的话
    else:
        # 如果是 RADAR 的话，那么前三列是 (rho, psi, rho_dot)
        m_rho = t_measurement[1]
        m_psi = t_measurement[2]
        m_dot_rho = t_measurement[3]
        z = np.array([[m_rho], [m_psi], [m_dot_rho]])

        dt = (t_measurement[4] - current_time) / 1000000.0
        current_time = t_measurement[4]

        # 如果是 RADAR 的话，第 6,7,8,9 列表示真实的 [x, y, vx, vy]
        g_x = t_measurement[5]
        g_y = t_measurement[6]
        g_v_x = t_measurement[7]
        g_v_y = t_measurement[8]

    if np.abs(state[4, 0]) < 0.0001:  # omega is 0, Driving straight
        state = transition_function_1(state.ravel().tolist())
        state[3, 0] = control_psi(state[3, 0])
        JA = J_A_1(state.ravel().tolist())
    else:  # otherwise
        # ravel 方法将数组维度拉成一维数组
        # 通过状态转移函数，将通过上一时刻 t- 1 物体的后验状态 state 转变为当前时刻 t 物体的前验状态
        # x_t_prior = f(x_t-1_post)
        state = transition_function(state.ravel().tolist())
        state[3, 0] = control_psi(state[3, 0])
        # 由于状态转移函数是非线性函数，求出这个状态转移函数的
        JA = J_A(state.ravel().tolist())

    G = np.zeros([5, 2])
    G[0, 0] = 0.5 * dt * dt * np.cos(state[3, 0])
    G[1, 0] = 0.5 * dt * dt * np.sin(state[3, 0])
    G[2, 0] = dt
    G[3, 1] = 0.5 * dt * dt
    G[4, 1] = dt

    # 这两步就是求出噪声的协方差矩阵
    Q_v = np.diag([std_noise_a * std_noise_a, std_noise_yaw_dd * std_noise_yaw_dd])
    Q = np.dot(np.dot(G, Q_v), G.T)

    # Project the error covariance ahead
    # P 为误差协方差矩阵
    P = np.dot(np.dot(JA, P), JA.T) + Q

    # Measurement Update (Correction)
    # 如果是 LIDAR 的话，就使用 KF 的算法进行更新
    if t_measurement[0] == 0.0:
        # Lidar
        S = np.dot(np.dot(H_lidar, P), H_lidar.T) + R_lidar
        K = np.dot(np.dot(P, H_lidar.T), np.linalg.inv(S))
        y = z - np.dot(H_lidar, state)

        y[1, 0] = control_psi(y[1, 0])
        state = state + np.dot(K, y)
        state[3, 0] = control_psi(state[3, 0])
        # Update the error covariance
        P = np.dot((I - np.dot(K, H_lidar)), P)

        # Save states for Plotting
        save_states(state.ravel().tolist(), g_x, g_y, g_v_x, g_v_y, m_x, m_y)

    # 如果是 RADAR 的话，就使用 EKF 的算法进行更新
    else:
        # Radar
        # 如果是 Radar 测量的话，由于测量函数 h(x) 为非线性的，即为了把 state 中的 [x, y, v, theta1, w] 变为测量的 [p, theta2, vp]
        # 所以 h(state) 是非线性的函数，需要求出其雅各比矩阵
        JH = J_H(state.ravel().tolist())

        S = np.dot(np.dot(JH, P), JH.T) + R_radar
        # 求出卡尔曼增益 K
        K = np.dot(np.dot(P, JH.T), np.linalg.inv(S))
        # 求出 h(state) 的值，h(state) 会把 state 转变为 [p, theta2, vp]
        # 这里的 state 其实也就是根据 t - 1 时刻的后验状态算出的 t 时刻的先验状态
        map_pred = measurement_function(state.ravel().tolist())
        if np.abs(map_pred[0, 0]) < 0.0001:
            # if rho is 0
            map_pred[2, 0] = 0

        # z 是 t 时刻的测量数据
        y = z - map_pred
        y[1, 0] = control_psi(y[1, 0])

        # 求出 t 时刻的后验状态 state
        state = state + np.dot(K, y)
        state[3, 0] = control_psi(state[3, 0])
        # Update the error covariance
        # 更新误差协方差矩阵
        P = np.dot((I - np.dot(K, JH)), P)

        save_states(state.ravel().tolist(), g_x, g_y, g_v_x, g_v_y, m_rho * np.cos(m_psi), m_rho * np.sin(m_psi))


def rmse(estimates, actual):
    result = np.sqrt(np.mean((estimates - actual) ** 2))
    return result


print(rmse(np.array(px), np.array(gpx)),
      rmse(np.array(py), np.array(gpy)),
      rmse(np.array(vx), np.array(gvx)),
      rmse(np.array(vy), np.array(gvy)))

# write to the output file
stack = [px, py, vx, vy, mx, my, gpx, gpy, gvx, gvy]
stack = np.array(stack)
stack = stack.T
np.savetxt('output.txt', stack, '%.6f')
