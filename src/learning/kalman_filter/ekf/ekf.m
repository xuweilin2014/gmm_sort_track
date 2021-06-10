% ekf 代码
% x(k) = sin(3 * x(k - 1))
% y(k) = x(k) ^ 2
% 注意似然概率是多峰分布，具有强烈的非线性

% 生成真实信号与观测
t = 0.01:0.01:1;
n = length(t);
x = zeros(1, n);
y = zeros(1, n);
x(1) = 0.1;
y(1) = 0.1 ^ 2;

for i = 2:n
    % 生成 x 真实信号
    x(i) = sin(3 * x(i - 1));
    % 生成 y 的观测信号
    y(i) = x(i) ^ 2 + normrnd(0, 0.7);
end

% ekf
X_plus =  zeros(1, n)
% 设置初值
P_plus = 0.1
X_plus(1) = 0.1
Q = 0.1
R = 1

for i=2:n
    % 预测步
    A = 3 * cos(3 * X_plus(i - 1))
    X_minus = sin(3 * X_plus(i - 1))
    P_minus = A * P_plus * A.t + Q
    % 更新步
    C = 2 * X_minus
    K = P_minus * C * inv(C * P_minus * C.t + R)
    X_plus(i) = X_minus + K * (y(i) - X_minus ^ 2)
    P_plus = (eye(1) - K * C) * P_minus
end
