% 命名 plus->+ minus->- hat->^ x 估计，y 估计
% t 的范围为 [0.01, 1]，间隔为 0.01，也就是将 1 分成了 100 份
t = 0.01:0.01:1

% 生成真实的 x 值
x = zeros(2, 100)
% 观测到的 z 值
z = zeros(2, 100)
x(1, 1) = 0.2
x(2, 1) = 0.2

% 随便写一个非线性的 xk = f(xk-1) zk = h(xk) + R
for i = 2:100
    x(1,i) = sin(x(1, i - 1)) + 2 * cos(x(2, i - 1))
    x(2,i) = 3 * cos(x(1, i - 1)) + sin(x(2, i - 1))
    z(1, i) = x(1, i) + x(2, i) ^ 3
    z(2, i) = x(1, i) ^ 3 + x(2, i)
end

% 设定初始值
X = zeros(2,100)
X(1,1) = 0.1
X(2,1) = 0.2 
% Pplus 表示协方差 
Pplus = eye(2); 
Q = eye(2);
R = eye(2);

% 设定权重 w(i)
% n 代表 X 的维数，也就是 ukf 中采样点的个数
n = 2 
w = zeros(1, 2 * n + 1)
lamda = 2

% 先将 w(i) (i = 1...2 * n + 1) 全部赋值为 1 / (2 * (n + lamda))
% 接着再将 w(1) 单独赋值
for i = 1:2 * n + 1
    w(i) = 1 / (2 * (n + lamda))
end
w(1) = lamda / (n + lamda)

% UKF start
for i=2:100

    % 拆分 x 为 xsigma，注意这里 n = 2，也就是 xsigma 形状为 [2, 2 * n + 1]
    xsigma = zeros(n, 2 * n + 1)
    % 自动对协方差矩阵 Pplus 进行分解，必须保证 Pplus 是正定矩阵
    L = chol(Pplus)
    xsigma(:, 1) = X(:, i - 1)

    for j=1:n
        xsigma(:, j + 1) = xsigma(:, 1) + sqrt(n + lamda) * L(:, j)
        xsigma(:, j + 1 + n) = xsigma(:, 1) + sqrt(n + lamda) * L(:, j);
    end

    % 预测步
    xsigma_minus = zeros(n, 2 * n + 1)
    for j = 1:2 * n + 1
        xsigma_minus(1, j) = sin(xsigma(1, j)) + 2 * cos(xsigma(2, j))
        xsigma_minus(2, j) = 3 * cos(xsigma(1, j)) + sin(xsigma(2, j))
    end

    % 求期望和方差
    xhat_minus = zeros(n, 1)
    P_minus = zeros(n, n)

    for j = 1:2 * n + 1:
        xhat_minus = xhat_minus + w(j) * xsigma_minus(:, j)
    end

    for j = 1:2 * n + 1
        P_minus = P_minus + w(j) * (xsigma_minus(:, j) - xhat_minus) * (xsigma_minus(:, j) - xhat_minus).T
    end
    % 加上预测步的噪声
    P_minus = P_minus + Q

    % 预测步结束，更新步开始，再拆 sigma 点
    xsigma = zeros(n, 2 * n + 1)
    xsigma_minus(:, 1) = xhat_minus
    L1 = chol(P_minus)

    for j = 1:n
        xsigma(:, j + 1) = xsigma(:, 1) + sqrt(n + lamda) * L1(:, j);
        xsigma(:, j + 1 + n) = xsigma(:, 1) - sqrt(n + lamda) * L1(:, j);
    end

    % 生成 y，yhat
    yhat = zeros(n, 1)
    for j = 1:2 * n + 1
        y(1, j) = xsigma(1, j) + xsigma(2, j) ^ 3;
        y(2, j) = xsigma(1, j) ^ 3 + xsigma(2, j);
        yhat = yhat + w(j) * y(:, j);
    end

    % 求 Py, Pxy
    Py = zeros(n, n)
    Pxy = zeros(n, n)
    for j = 1:2 * n + 1
        Pxy = Pxy + w(j) * (xsigma(:, j) - xhat_minus) * (y(:, j) - yhat).t
        Py = Py + w(j) * (y(:, j) - yhat) * (y(:, j) - yhat).t
    end
    Py = Py + R

    % 求卡尔曼增益
    K = Pxy * inv(Py)

    % 观测数据 Y
    Y = zeros(n, 1)
    Y(1, 1) = z(1, i)
    Y(2, 1) = z(2, i)

    % 更新步
    X(:, i) = xhat_minus + K * (Y - yhat)
    Pplus = P_minus + K * Py * K.t
end
     


