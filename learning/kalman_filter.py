import numpy as np
import matplotlib

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
# P0 是后验估计的协方差矩阵
pk_post = np.array([[1,0], [0,1]])
x_post_prev = np.array([0, 1])
x_prev = np.array([0, 1])

iterations = 50

for k in range(iterations):
    if k == 0:
        continue
    xk_prior = np.matmul(A, np.transpose(x_post_prev))
    pk_prior = np.matmul(A, np.matmul(pk_post, np.transpose(A))) + Q
    PK_PRIOR_H_T = np.matmul(pk_prior, np.transpose(H))
    tmp = np.linalg.inv(np.matmul(H, np.matmul(pk_prior, np.transpose(H))) + R)
    Kk = np.matmul(PK_PRIOR_H_T, tmp)

    w10 = np.random.normal(0, np.sqrt(Q)[0][0])
    w20 = np.random.normal(0, np.sqrt(Q)[1][1])
    v10 = np.random.normal(0, np.sqrt(R)[0][0])
    v11 = np.random.normal(0, np.sqrt(R)[1][1])

    xk = np.matmul(A, np.transpose(x_prev)) + np.transpose(np.array([w10, w20]))
    zk = np.matmul(H, xk) + np.transpose(np.array([v10, v11]))

    xk_post = xk_prior + np.matmul(Kk, (zk - np.matmul(H, xk_prior)))
    pk_post = np.matmul((I - np.matmul(Kk, H)), pk_prior)
    x_post_prev = xk_post
    x_prev = xk

    print("{0}".format(k), pk_post)