import numpy as np
import matplotlib.pyplot as plt

iterations = 100
sigma = np.array([0.02, 0.1, 0.6, 0.44])
sensor_num = 3
mu = 10
weight = np.zeros(sensor_num)
x_avg = np.zeros(sensor_num)

res = []
avg_res = []
weight_res = np.zeros((sensor_num, iterations))

for k in range(1, iterations + 1):
    vals = np.zeros(sensor_num)
    for _ in range(sensor_num):
        vals[_] = np.random.normal(mu, np.sqrt(sigma[_]))

    r_mat = np.zeros((sensor_num, sensor_num))
    for i in range(sensor_num):
        for j in range(sensor_num):
            r_mat[i][j] = ((k - 1.0) / k) * r_mat[i][j] + (1.0 / k) * vals[i] * vals[j]

    new_sigma = np.zeros(sensor_num)
    sigma_sum = 0
    for i in range(sensor_num):
        new_sigma[i] = r_mat[i][i] - (1.0 / (sensor_num - 1)) * (np.sum(r_mat[i, :]) - r_mat[i][i])
        new_sigma[i] = 1.19209e-09 if new_sigma[i] < 0 else new_sigma[i]
        x_avg[i] = ((k - 1.0) / k) * x_avg[i] + (1.0 / k) * vals[i]
        sigma_sum += 1.0 / new_sigma[i]

    weight = 1 / (new_sigma * sigma_sum)

    weight_res[:, k - 1] = weight
    avg_res.append(np.sum(vals) / sensor_num)
    res.append(np.sum(weight * x_avg))

# 把不同权重融合所得到的结果和
x = np.linspace(1, iterations, iterations, dtype=int)
plt.title('multi-sensor fusion')
plt.plot(x, res, label='adaptive weighted')
plt.plot(x, avg_res, label='avg')
plt.legend()
plt.grid()
plt.savefig("融合值.png", dpi=900, bbox_inches='tight')
plt.show()

# 生成各个传感器的权重图像
x = np.linspace(1, iterations, iterations, dtype=int)
plt.title('multi-sensor weights')
plt.plot(x, weight_res[0, :], label='sensor sigma=' + str(sigma[0]))
plt.plot(x, weight_res[1, :], label='sensor sigma=' + str(sigma[1]))
plt.legend()
plt.grid()
plt.savefig("权重比较.png", dpi=900)
plt.show()






