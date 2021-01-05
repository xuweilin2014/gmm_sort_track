import numpy as np
import cv2
from numba import jit

# constant
NUM_SECTOR = 9
FLT_EPSILON = 1e-07


'''
fhog 特征的处理代码，具体实现原理和思路请参考以下两个网址:
https://www.jianshu.com/p/69a3e39c51f9
https://zhuanlan.zhihu.com/p/56405827
'''

@jit
def func1(dx, dy, boundary_x, boundary_y, height, width, numChannels):
    # r.shape 为 [height, width]，height 为目标图像的高，width 为目标图像的宽度，r 也就是目标图像中每一个像素点的梯度值
    r = np.zeros((height, width), np.float32)
    # alfa.shape 为 [height, width, 2]，height, width 的意义同上，2 表示将每个像素点方向投影至 [0, 180) 和 [0, 360) 两个区间，
    # 从而每个像素点由两个不同的方向
    alfa = np.zeros((height, width, 2), np.int)

    for j in range(1, height - 1):
        for i in range(1, width - 1):
            '''
            step1.计算模板图像 RGB 三通道每个像素点的水平梯度被 dx 和垂直梯度 dy，计算各点的梯度幅值，并且以最大的梯度幅值所在通道为准
            '''
            c = 0
            x = dx[j, i, c]
            y = dy[j, i, c]
            # r[j][i] 就表示模板图像 z[j][i] 像素位置的梯度大小，不过是第 0 个通道的梯度幅值
            r[j, i] = np.sqrt(x * x + y * y)

            # 接下来遍历模板图像 z[j][i] 的剩下的 2 个通道，分别计算其梯度大小，最大的保存到 r[j][i] 中
            for ch in range(1, numChannels):
                tx = dx[j, i, ch]
                ty = dy[j, i, ch]
                magnitude = np.sqrt(tx * tx + ty * ty)
                if magnitude > r[j, i]:
                    r[j, i] = magnitude
                    c = ch
                    x = tx
                    y = ty

            '''
            step2.梯度的方向判定
            如果像素点最大幅值所在通道为 m，则利用该通道的水平梯度和垂直梯度计算该点的梯度方向，论文中将 [0,180) 分为了 9 个方向，
            同时还将 [0,360) 分为了 18 个方向，具体方向归属则是利用该像素点梯度在模板方向上的投影值确定，分别通过以 [0, 180] 和 [0, 360]
            为周期将每个像素点投影至这两个区间，从而每个像素点都有两种方向，公式为:
            使得 dx * cos(theta) + dy * sin(theta) 最大的 theta 即为我们所求的梯度方向
            '''
            mmax = boundary_x[0] * x + boundary_y[0] * y
            maxi = 0

            # 遍历从 0 到 9
            for kk in range(0, NUM_SECTOR):
                dotProd = boundary_x[kk] * x + boundary_y[kk] * y
                if dotProd > mmax:
                    mmax = dotProd
                    maxi = kk
                elif -dotProd > mmax:
                    mmax = -dotProd
                    maxi = kk + NUM_SECTOR

            alfa[j, i, 0] = maxi % NUM_SECTOR
            alfa[j, i, 1] = maxi
    return r, alfa


"""
step3.cell 的分割
cell 的大小默认为 k = 4 * 4, 假定目标图像的大小为 [32, 104], height = 32, width = 104,
因此，水平方向有 sizeX = 104 / 4 = 26, 竖直方向有 sizeY = 32 / 4 = 8 个 cell, 每个 cell 中的方向梯度直方图应该有 27 个方向, 因此 p = 27
stringSize = sizeX * p = 702, r, alfa 分别为目标图像的梯度幅值矩阵和梯度方向矩阵, 大小也为 [32, 104].

step4.cell 内像素点梯度幅值的加权方式. 
论文代码中将每个 cell 等分为 4 部分（左上、右上、左下和右下），每一个部分都包含 4 个像素，每一部分都是由包含该 cell 在内的相邻 4 个 cell 的同一部分加权平均得来的，
其权重即组合为: x^2, xy, yx, y^2.
比如 cell 的右下部分，则分别取该 cell 左上，正上，正左三个方向相邻的 cell 的右下部分以及该 cell 本身对应的梯度幅值进行加权平均组得到。 当然，对于边界 cell，
则只选取不超过边界的部分 cell 进行不完整加权。

step5.方向梯度直方图的计算
对于 cell 内每个像素点，将其梯度幅值分别以 [0,180) 和 [0,360) 两种投影区间累加至对应梯度方向直方图中，在按照上一步中提到的加权方式计算完 cell 特征之后，
每个 cell 保留了 9+18 个方向的梯度。
"""
@jit
def func2(r, alfa, nearest, w, k, height, width, sizeX, sizeY, p, stringSize):
    # mapp 是一个向量，相当于将所有 cell 的长度为 27 的向量拼接到一起形成
    # 不过 mapp 可以看成一个 shape 为 [sizeY, sizeX, P] 的矩阵，其中 p = 18 + 9 = 27
    mapp = np.zeros((sizeX * sizeY * p), np.float32)
    # i, j 表示的就是哪一个 cell，每个 cell 默认有 16 个像素点
    for i in range(sizeY):
        for j in range(sizeX):
            # ii, jj 表示就是一个 cell 中的哪一个像素点
            for ii in range(k):
                for jj in range(k):
                    # i * stringSize 表示 mapp 矩阵的下一行，j * p 表示 mapp 矩阵的下一列
                    # r[k * i + ii, j * k + jj] 表示目标图像 (i,j) 处 cell 中某一个像素点的梯度幅值
                    if (i * k + ii > 0) and (i * k + ii < height - 1) and (j * k + jj > 0) and (j * k + jj < width - 1):
                        mapp[i * stringSize + j * p + alfa[k * i + ii, j * k + jj, 0]] += r[k * i + ii, j * k + jj] * w[ii, 0] * w[jj, 0]
                        mapp[i * stringSize + j * p + alfa[k * i + ii, j * k + jj, 1] + NUM_SECTOR] += r[k * i + ii, j * k + jj] * w[ii, 0] * w[jj, 0]
                        if (i + nearest[ii] >= 0) and (i + nearest[ii] <= sizeY - 1):
                            mapp[(i + nearest[ii]) * stringSize + j * p + alfa[k * i + ii, j * k + jj, 0]] += r[k * i + ii, j * k + jj] * w[ii, 1] * w[jj, 0]
                            mapp[(i + nearest[ii]) * stringSize + j * p + alfa[k * i + ii, j * k + jj, 1] + NUM_SECTOR] += r[k * i + ii, j * k + jj] * w[ii, 1] * w[jj, 0]
                        if (j + nearest[jj] >= 0) and (j + nearest[jj] <= sizeX - 1):
                            mapp[i * stringSize + (j + nearest[jj]) * p + alfa[k * i + ii, j * k + jj, 0]] += r[k * i + ii, j * k + jj] * w[ii, 0] * w[jj, 1]
                            mapp[i * stringSize + (j + nearest[jj]) * p + alfa[k * i + ii, j * k + jj, 1] + NUM_SECTOR] += r[k * i + ii, j * k + jj] * w[ii, 0] * w[jj, 1]
                        if (i + nearest[ii] >= 0) and (i + nearest[ii] <= sizeY - 1) and (j + nearest[jj] >= 0) and (j + nearest[jj] <= sizeX - 1):
                            mapp[(i + nearest[ii]) * stringSize + (j + nearest[jj]) * p + alfa[k * i + ii, j * k + jj, 0]] += r[k * i + ii, j * k + jj] * w[ii, 1] * w[jj, 1]
                            mapp[(i + nearest[ii]) * stringSize + (j + nearest[jj]) * p + alfa[k * i + ii, j * k + jj, 1] + NUM_SECTOR] += r[k * i + ii, j * k + jj] * w[ii, 1] * w[jj, 1]
    return mapp


# 开始进行邻域归一化
# partOfNorm 的 shape 为 [sizeY, sizeX, p = 9]
# mappmap 的 shape 为 [sizeY, sizeX, xp = 27]
@jit
def func3(partOfNorm, mappmap, sizeX, sizeY, p, xp, pp):
    # newData 的 shape 为 [sizeY, sizeX, pp = 108]，其中 pp 就是每一个 cell 扩充之后的特征维度数目
    newData = np.zeros((sizeY * sizeX * pp), np.float32)
    # (i,j) 表示对 (i,j) 处的 cell 进行处理，由于处于边界处的 cell 不能进行 4 种不同的归一化处理方式，所以 i, j 都从 1 开始
    for i in range(1, sizeY + 1):
        for j in range(1, sizeX + 1):
            # pos1 代表的是 mappmap 矩阵中的哪个 cell
            pos1 = i * (sizeX + 2) * xp + j * xp
            # pos2 代表的是 newData 中的哪个 cell，注意 newData 中的一个 cell 有 108 维，而 mappmap 中的一个 cell 有 27 维
            pos2 = (i - 1) * sizeX * pp + (j - 1) * pp

            # 求包括 (i,j) cell 在内的正右、正下、右下处的四个 cell 的 L2 范数
            valOfNorm = np.sqrt(partOfNorm[i * (sizeX + 2) + j] +
                                partOfNorm[i * (sizeX + 2) + (j + 1)] +
                                partOfNorm[(i + 1) * (sizeX + 2) + j] +
                                partOfNorm[(i + 1) * (sizeX + 2) + (j + 1)]) + FLT_EPSILON
            # 将 mappmap 中 cell 内的 27 个方向梯度直方图除以 valOfNorm，得到规范化之后的 hog 特征，下同
            newData[pos2:pos2 + p] = mappmap[pos1:pos1 + p] / valOfNorm
            newData[pos2 + 4 * p:pos2 + 6 * p] = mappmap[pos1 + p:pos1 + 3 * p] / valOfNorm

            # 求包括 (i,j) cell 在内的正右、正上、右上处的四个 cell 的 L2 范数
            valOfNorm = np.sqrt(partOfNorm[i * (sizeX + 2) + j] +
                                partOfNorm[i * (sizeX + 2) + (j + 1)] +
                                partOfNorm[(i - 1) * (sizeX + 2) + j] +
                                partOfNorm[(i - 1) * (sizeX + 2) + (j + 1)]) + FLT_EPSILON
            newData[pos2 + p:pos2 + 2 * p] = mappmap[pos1:pos1 + p] / valOfNorm
            newData[pos2 + 6 * p:pos2 + 8 * p] = mappmap[pos1 + p:pos1 + 3 * p] / valOfNorm

            # 求包括 (i,j) cell 在内的正左、正下、左下处的四个 cell 的 L2 范数
            valOfNorm = np.sqrt(partOfNorm[i * (sizeX + 2) + j] +
                                partOfNorm[i * (sizeX + 2) + (j - 1)] +
                                partOfNorm[(i + 1) * (sizeX + 2) + j] +
                                partOfNorm[(i + 1) * (sizeX + 2) + (j - 1)]) + FLT_EPSILON
            newData[pos2 + 2 * p:pos2 + 3 * p] = mappmap[pos1:pos1 + p] / valOfNorm
            newData[pos2 + 8 * p:pos2 + 10 * p] = mappmap[pos1 + p:pos1 + 3 * p] / valOfNorm

            # 求包括 (i,j) cell 在内的正左、正上、左上处的四个 cell 的 L2 范数
            valOfNorm = np.sqrt(partOfNorm[i * (sizeX + 2) + j] +
                                partOfNorm[i * (sizeX + 2) + (j - 1)] +
                                partOfNorm[(i - 1) * (sizeX + 2) + j] +
                                partOfNorm[(i - 1) * (sizeX + 2) + (j - 1)]) + FLT_EPSILON
            newData[pos2 + 3 * p:pos2 + 4 * p] = mappmap[pos1:pos1 + p] / valOfNorm
            newData[pos2 + 10 * p:pos2 + 12 * p] = mappmap[pos1 + p:pos1 + 3 * p] / valOfNorm

            # 经过上面 4 个步骤的处理，将 cell 中 27 维的向量扩充为 108 维的向量

    return newData


@jit
def func4(mappmap, p, sizeX, sizeY, pp, yp, xp, nx, ny):
    # newData 矩阵的 shape 为 [sizeY, sizeX, pp = 31]
    newData = np.zeros((sizeX * sizeY * pp), np.float32)
    for i in range(sizeY):
        for j in range(sizeX):
            pos1 = (i * sizeX + j) * p
            pos2 = (i * sizeX + j) * pp

            """
            前面说过，108 维向量的组成为：[0,8),[9,18),[18,27),[27,36),[36,54),[54,72),[72,90),[90,108) 
            for-1 表示在 [36, 108) 这个区间中每隔 18 个选一个，也就是在 [36,54),[54,72),[72,90),[90,108) 四个区间中各选一个数，累加，然后乘以 1 / sqrt(4)
            for-2 表示在 [0, 36) 这个区间中每隔 9 个选一个，也就是在 [0,8),[9,18),[18,27),[27,36) 四个区间中各选一个数，累加，然后乘以 1 / sqrt(4)
            for-3 表示分别对 [36,54),[54,72),[72,90),[90,108) 这四个区间中的数进行累加，然后乘以 1 / sqrt(18)
            """
            for jj in range(2 * xp):  # for-1
                newData[pos2 + jj] = np.sum(mappmap[pos1 + yp * xp + jj: pos1 + 3 * yp * xp + jj: 2 * xp]) * ny
            for jj in range(xp):  # for-2
                newData[pos2 + 2 * xp + jj] = np.sum(mappmap[pos1 + jj: pos1 + jj + yp * xp: xp]) * ny
            for ii in range(yp):  # for-3
                newData[pos2 + 3 * xp + ii] = np.sum(mappmap[pos1 + yp * xp + ii * xp * 2: pos1 + yp * xp + ii * xp * 2 + 2 * xp]) * nx

    return newData


def getFeatureMaps(image, k, mapp):
    kernel = np.array([[-1., 0., 1.]], np.float32)

    height = image.shape[0]
    width = image.shape[1]
    assert (image.ndim == 3 and image.shape[2])
    numChannels = 3  # (1 if image.ndim == 2 else image.shape[2])

    sizeX = width // k
    sizeY = height // k
    px = 3 * NUM_SECTOR
    p = px
    stringSize = sizeX * p

    mapp['sizeX'] = sizeX
    mapp['sizeY'] = sizeY
    mapp['numFeatures'] = p
    mapp['map'] = np.zeros((mapp['sizeX'] * mapp['sizeY'] * mapp['numFeatures']), np.float32)

    # 进行卷积操作，分别得到 dx, dy 方向的梯度幅值
    dx = cv2.filter2D(np.float32(image), -1, kernel)
    dy = cv2.filter2D(np.float32(image), -1, kernel.T)

    # arg_vector 表示一个 [0, 1pi, 2pi, 3pi, ... , 9pi] / 9
    arg_vector = np.arange(NUM_SECTOR + 1).astype(np.float32) * np.pi / NUM_SECTOR
    boundary_x = np.cos(arg_vector)
    boundary_y = np.sin(arg_vector)

    # 得到目标图像的梯度幅值矩阵 r 和梯度方向矩阵 alfa，r 的 shape 为 [height, width]，alfa 的 shape 为 [height, width, 2]
    r, alfa = func1(dx, dy, boundary_x, boundary_y, height, width, numChannels)  # with @jit

    nearest = np.ones(k, np.int)
    nearest[0:k // 2] = -1

    # w 为一个权重矩阵，用于将 r 中某一个点的梯度按照相应的权重分配到和其相邻的 3 个 cell 中去
    w = np.zeros((k, 2), np.float32)
    a_x = np.concatenate((k / 2 - np.arange(k / 2) - 0.5, np.arange(k / 2, k) - k / 2 + 0.5)).astype(np.float32)
    b_x = np.concatenate((k / 2 + np.arange(k / 2) + 0.5, -np.arange(k / 2, k) + k / 2 - 0.5 + k)).astype(np.float32)
    w[:, 0] = 1.0 / a_x * ((a_x * b_x) / (a_x + b_x))
    w[:, 1] = 1.0 / b_x * ((a_x * b_x) / (a_x + b_x))

    # mapp['map'] 是一个 shape 为 [sizeX, sizeY, p] 的矩阵，其中 p = 18 + 9 = 27
    # 将 alfa 矩阵中的值保存到 mapp['map'] 矩阵中，并且权重为 r 矩阵中的梯度值，可以看成是按照 alfa 矩阵的值进行一个投票的过程
    mapp['map'] = func2(r, alfa, nearest, w, k, height, width, sizeX, sizeY, p, stringSize)  # with @jit

    return mapp


"""
step6.相对邻域归一化以及截断
对于每个 cell，分别取包含其在内的相邻 4 个 cell，因此有四个组合方式，每种组合方式都取该组合方式内四个 cell 的方向梯度直方图的前 9 个方向梯度的 L2 范数 val，
然后用该 cell 内 27 个方向的梯度直方图除以 val，即可得到规范化之后的 hog 特征。四个组合可以得到四组 hog 特征，即 9+9+9+9+18+18+18+18=108 个方向。也就是说，
对于每个 cell，在进行邻域归一化之后每个 cell 的特征维度会从 27 维变为 108 维。

108 维向量的生成过程如下：
每个 cell（27维）和包括这个 cell 相邻的 3 个 cell（比如正右，正下，右下）进行一个归一化，得到规范化之后的 27 维 hog 特征，这个 27 维的 hog 特征会保存到
108 维向量的 [0,8) 以及 [36,54)，也就是前 9 维和后 18 维在 108 维向量中是分开存储的。因此，4 种归一化的生成的 hog 特征向量在 108 维向量的区间如下：
[0,8]),[36,54)
[9,18),[54,72)
[18,27),[72,90)
[27,36),[90,108) 

另外，可以发现边界 cell 无法得到这么多方向，因此去掉边界 cell。所以 sizeX - 2，sizeY - 2
"""
def normalizeAndTruncate(mapp, alfa):
    sizeX = mapp['sizeX']
    sizeY = mapp['sizeY']

    p = NUM_SECTOR
    # xp = 3 * 9 = 27
    xp = NUM_SECTOR * 3
    # pp = 27 * 4 = 108，也就是上面所说的 108 个方向
    pp = NUM_SECTOR * 12

    # mapp['map'] 是一个 shape 为 [sizeX, sizeY, xp = 27] 大小的矩阵，而 partOfNorm 是一个 shape 为 [sizeX, sizeY, p = 9] 大小的矩阵
    # 并且 partOfNorm 中的值就是上面所说的 cell 方向梯度直方图中前 9 个方向梯度的平方值，不过这里是对应的一个 cell，具体的组合运算还是得等到
    # func3 中去进行
    idx = np.arange(0, sizeX * sizeY * mapp['numFeatures'], mapp['numFeatures']).reshape((sizeX * sizeY, 1)) + np.arange(p)
    partOfNorm = np.sum(mapp['map'][idx] ** 2, axis=1)

    # 排除掉边界的 cell
    sizeX, sizeY = sizeX - 2, sizeY - 2
    newData = func3(partOfNorm, mapp['map'], sizeX, sizeY, p, xp, pp)

    # 进行截断操作，也就是把 newData 中大于 alfa 的值设置为 alfa
    newData[newData > alfa] = alfa

    mapp['numFeatures'] = pp
    mapp['sizeX'] = sizeX
    mapp['sizeY'] = sizeY
    mapp['map'] = newData

    return mapp


"""
step7.PCA降维
在前面 step6 中，作者将一个 cell 的特征向量由 27 维扩充为 108 维，由于 108 维计算量太大，所以将 cell 的特征向量由 108 维转变为 27 + 4 = 31 维
"""
def PCAFeatureMaps(mapp):
    sizeX = mapp['sizeX']
    sizeY = mapp['sizeY']

    p = mapp['numFeatures']
    pp = NUM_SECTOR * 3 + 4
    yp = 4
    xp = NUM_SECTOR

    # nx = 1 / sqrt(18)
    nx = 1.0 / np.sqrt(xp * 2)
    # ny = 1 / sqrt(4)
    ny = 1.0 / np.sqrt(yp)

    newData = func4(mapp['map'], p, sizeX, sizeY, pp, yp, xp, nx, ny)

    mapp['numFeatures'] = pp
    mapp['map'] = newData

    return mapp
