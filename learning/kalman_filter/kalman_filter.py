import numpy as np
import matplotlib.pyplot as plt

'''
这里参考了视频 https://www.bilibili.com/video/BV1dV411B7ME/?spm_id_from=333.788.videocard.3 上的例子，使用 python 进行了实现
使用两个状态变量（x1_k, x2_k）建立模型来描述一个匀速运动的物体，x1_k 中的 1 表示第一个分量，表示物体的位置，x2_k 中的 2 表示第二个分量，
表示物体的速度，其中 k 是采样的时间。

同样使用两个状态变量（z1_k, z2_k）表示一个匀速运动的物体的测量值，z1_k 中的 1 表示第一个分量，表示测量得到的物体的位置，x2_k 中的 2 表示第二个
分量，表示物体速度的测量值，其中 k 表示采样的时间
'''

# 初始化矩阵
# Q 是使用模型估计的噪声或者说不准确度
Q = np.array([[0.1, 0], [0, 0.1]])
# R 是测量的噪声
R = np.array([[1, 0], [0, 1]])
# A 是模型估计的状态转移矩阵
A = np.array([[1, 1], [0, 1]])
# H 是实际测量的状态转移矩阵
H = np.array([[1, 0], [0, 1]])
I = np.array([[1, 0], [0, 1]])
# pk_post 是后验估计误差的协方差矩阵，这里是初始值，后续会不断进行更新
pk_post = np.array([[1,0], [0,1]])
# 这里是一个二维分布的模型，x1 表示的是物体的位置，x2 表示的是物体的速度，后续会不断进行更新
# x_post_prev 是 x1, x2 的后验估计的初始值，是一个 1*2 的矩阵
x_post_prev = np.array([0, 1])
# x_prev 是 x1, x2 的真实状态的初始值，是一个 1*2 的矩阵，后续会不断进行更新
x_prev = np.array([0, 1])

iterations = 50

real = []
measurement = []
prior = []
post = []

for k in range(iterations + 1):
    if k == 0:
        continue
    # 计算出 xk 的先验估计值
    xk_prior = np.matmul(A, np.transpose(x_post_prev))

    prior.append(np.asarray(xk_prior))

    # 计算出 pk_prior，pk_prior 是先验估计值误差的协方差矩阵
    pk_prior = np.matmul(A, np.matmul(pk_post, np.transpose(A))) + Q

    # 下面三个式子计算出卡尔曼增益
    PK_PRIOR_H_T = np.matmul(pk_prior, np.transpose(H))
    tmp = np.linalg.inv(np.matmul(H, np.matmul(pk_prior, np.transpose(H))) + R)
    Kk = np.matmul(PK_PRIOR_H_T, tmp)

    # w1 是位置估计的不确定度
    w1 = np.random.normal(0, np.sqrt(Q)[0][0])
    # w2 是速度估计的不确定度
    w2 = np.random.normal(0, np.sqrt(Q)[1][1])
    # w 是位置和速度的估计的噪声，或者说不确定度。这里假设 w 符合正态分布，均值为 0，方差为协方差矩阵 Q
    # 不过这里假设 w1, w2 之间的协方差为 0
    w = np.array([w1, w2])

    # v1 是位置测量的不确定度
    v1 = np.random.normal(0, np.sqrt(R)[0][0])
    # v2 是速度测量的不确定度
    v2 = np.random.normal(0, np.sqrt(R)[1][1])
    # v 是位置和速度的测量的不确定度，这里也假设 v 符合正态分布，均值为 0，方差为协方差矩阵 R
    # 不过这里假设 v1, v2 之间的协方差为 0
    v = np.array([v1, v2])

    # 计算出物体的实际状态，xk = Axk-1 + wk-1
    xk = np.matmul(A, np.transpose(x_prev)) + np.transpose(w)
    real.append(np.asarray(xk))

    # 计算出物体的测量值，zk = Hxk + vk
    zk = np.matmul(H, xk) + np.transpose(v)

    measurement.append(np.asarray(zk))

    # 计算出物体状态 x 的后验估计值
    xk_post = xk_prior + np.matmul(Kk, (zk - np.matmul(H, xk_prior)))

    post.append(np.asarray(xk_post))

    # 计算出 pk_post，pk_post 是后验估计误差的协方差矩阵
    pk_post = np.matmul((I - np.matmul(Kk, H)), pk_prior)
    x_post_prev = xk_post
    x_prev = xk

# 生成速度比较的图像
x = np.linspace(1, 50, 50, dtype=int)
plt.title('velocity comparison')
velocity_real = np.asarray(real)[:, 1]
velocity_measurement = np.asarray(measurement)[:, 1]
velocity_prior = np.asarray(prior)[:, 1]
velocity_post = np.asarray(post)[:, 1]
plt.plot(x, velocity_real, label='real velocity')
plt.plot(x, velocity_measurement, label='measurement velocity')
plt.plot(x, velocity_prior, label='prior velocity')
plt.plot(x, velocity_post, label='post velocity')
plt.legend()
plt.grid()
plt.savefig("速度比较.png", dpi=900, bbox_inches='tight')
plt.show()

# 生成位置比较的曲线图
plt.title('position comparison')
position_real = np.asarray(real)[:, 0]
position_measurement = np.asarray(measurement)[:, 0]
position_prior = np.asarray(prior)[:, 0]
position_post = np.asarray(post)[:, 0]
plt.plot(x, position_real, label='real position')
plt.plot(x, position_measurement, label='measurement position')
plt.plot(x, position_prior, label='prior position')
plt.plot(x, position_post, label='post position')
plt.legend()
plt.grid()
plt.savefig("位置比较.png", dpi=900, bbox_inches='tight')
plt.show()