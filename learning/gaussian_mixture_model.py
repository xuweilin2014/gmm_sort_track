import matplotlib.pyplot as plt
import numpy as np
from sklearn.datasets import make_blobs


# 手动实现高斯混合模型，并且使用 E-M 算法迭代求解模型的参数
# 参考代码 https://blog.csdn.net/weixin_41566471/article/details/106221915
class GaussianMixtureModel:

    def __init__(self, K=3):
        """
        高斯混合模型，使用 E-M 算法求解其参数
        :param K: 超参数，分类类别，也就是高斯函数的个数
        :param N: 样本量，也就是样本的个数
        :param D: 一个样本有多少个维度
        :param alpha: 模型参数，高斯函数的系数，决定高斯函数的高度，维度（K）
        :param mu: 模型参数，高斯函数的均值，决定高斯函数的中型位置，维度为（K，D）
        :param sigma: 模型参数，高斯函数的方差矩阵，决定高斯函数的形状，维度为（K，D，D)
        :param gamma: 模型隐变量，决定单个样本具体属于哪一个高斯分布，维度为（N，K）
        """
        self.K = K
        self.sigma = None
        self.gama = None
        self.mu = None
        self.alpha = None
        self.N = None
        self.D = None

    def init_params(self):
        # 在 E-M 算法中，就是对参数取一个初始值，然后开始进行 e-step m-step 的迭代
        # 这里就是给 alpha, gama, mu, sigma 这四个参数赋一个随即初始值，然后再进行迭代
        # alpha 有一个约束条件，即 alpha 之和等于 1
        alpha = np.random.rand(self.K)
        self.alpha = alpha / sum(alpha)
        self.mu = np.random.rand(self.K, self.D)
        # 虽然 gama 有约束条件，但是第一步 e-step 的时候会对此重新赋值，所以可以随意初始化
        self.gama = np.random.rand(self.N, self.K)
        self.sigma = [np.identity(self.D) for _ in range(self.K)]

    def gaussian_function(self, sigma_k, y_j, mu_k):
        """
        计算高维度的高斯函数值，计算的时候，先取对数，然后再将取 exp 的指数抵消前面的对数操作
        log(gaussian) = -D/2 * log(2 * pi) - 1/2 * log(sigma) - 1/2 * [(y - mu).T * sigma逆矩阵 * (y - u)]
        :param sigma_k: 第 k 个 sigma 值，(N，N)
        :param y_j: 第 j 个观测值, (1,D)
        :param mu_k: 第 k 个 mu 值, (1,D)
        """

        # 先取对数
        param_1 = self.D * np.log(2 * np.pi)
        # 计算数组行列式的符号和（自然）对数
        _, param_2 = np.linalg.slogdet(sigma_k)
        # 计算矩阵的（乘法）逆矩阵
        param_3 = np.dot(np.dot(np.transpose(y_j - mu_k), np.linalg.inv(sigma_k)), (y_j - mu_k))

        # 返回是重新取指数抵消前面的取对数操作
        return np.exp(-0.5 * (param_1 + param_2 + param_3))

    def e_step(self):
        for j in range(self.N):
            gaussian_alpha_multi = []
            y_j = self.data[j]
            for k in range(self.K):
                alpha_k = self.alpha[k]
                sigma_k = self.sigma[k]
                mu_k = self.mu[k]
                gaussian_alpha_multi.append(alpha_k * self.gaussian_function(sigma_k, y_j, mu_k))

            multi_sum = sum(gaussian_alpha_multi)
            # 计算出 gama 矩阵 (N,K) 的值，对隐变量进行迭代更新
            self.gama[j, :] = [v / multi_sum for v in gaussian_alpha_multi]

    def m_step(self):
        for k in range(self.K):
            gama_k = self.gama[:, k]
            mu_k = self.mu[k]
            gama_k_sum = sum(gama_k)
            mu_k_part = 0
            sigma_k_part = 0
            alpha_k_part = gama_k_sum
            for j in range(self.N):
                gama_k_j = gama_k[j]
                y_j = self.data[j]
                # mu_k 的分子
                mu_k_part += y_j * gama_k_j
                # sigma_k 的分子
                sigma_k_part += gama_k_j * np.outer((y_j - mu_k), np.transpose(y_j - mu_k))

            # 对模型参数进行迭代更新
            self.alpha[k] = alpha_k_part / self.N
            self.sigma[k] = sigma_k_part / gama_k_sum
            self.mu[k] = mu_k_part / gama_k_sum

    def fit(self, y, iteration=1000):
        self.N, self.D = y.shape
        self.init_params()

        for _ in range(iteration):
            self.e_step()
            self.m_step()

    # noinspection PyAttributeOutsideInit
    def generate_samples(self, samples=100, classes=3, features=2, seed=None):
        # make_blobs: 函数是为聚类产生数据集，产生一个数据集和相应的标签
        # n_samples: 表示数据样本点个数，默认值为 100
        # n_features: 表示数据的维度，默认值为 2
        # centers: 产生的数据的中心点，默认值为 3
        self.data, self.label = make_blobs(n_samples=samples, centers=classes, n_features=features, random_state=seed)

    def run_model(self):
        if self.data.any():
            self.fit(self.data)
            class_index = np.argmax(self.gama, axis=1)
            self.show_classification(class_index)

    def show_classification(self, class_index):
        prediction_1 = []
        prediction_2 = []
        prediction_3 = []

        real_1 = []
        real_2 = []
        real_3 = []

        for y_i, label, index in zip(self.data, self.label, class_index):
            if index == 0:
                prediction_1.append(y_i)
            elif index == 1:
                prediction_2.append(y_i)
            else:
                prediction_3.append(y_i)

            if label == 0:
                real_1.append(y_i)
            elif label == 1:
                real_2.append(y_i)
            else:
                real_3.append(y_i)

        prediction_1 = np.array(prediction_1)
        prediction_2 = np.array(prediction_2)
        prediction_3 = np.array(prediction_3)
        real_1 = np.array(real_1)
        real_2 = np.array(real_2)
        real_3 = np.array(real_3)

        plt.figure(figsize=(10,5))

        plt.subplot(1, 2, 1)
        plt.scatter(prediction_1[:, 0], prediction_1[:, 1], c='pink')
        plt.scatter(prediction_2[:, 0], prediction_2[:, 1], c='maroon')
        plt.scatter(prediction_3[:, 0], prediction_3[:, 1], c='gold')
        plt.title('prediction')

        plt.subplot(1, 2, 2)
        plt.scatter(real_1[:, 0], real_1[:, 1], c='red')
        plt.scatter(real_2[:, 0], real_2[:, 1], c='blue')
        plt.scatter(real_3[:, 0], real_3[:, 1], c='green')
        plt.title('real')

        plt.show()


if __name__ == "__main__":
    gmm = GaussianMixtureModel()
    gmm.generate_samples()
    gmm.run_model()
