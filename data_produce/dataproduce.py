import numpy as np
import odl
from art_fun import art_fun
from wgn_fun import wgn_fun
import matplotlib.pyplot as plt
w = np.loadtxt('E:/GH/AO/lungweight.txt')#投影矩阵

k = 5
r = 1
size = 256
space = odl.uniform_discr([-128, -128], [128, 128], [size, size], dtype='float32')

path2 = 'E:/GH/data_test/'


def random_ellipse():
    return ((np.random.rand() * 2 + 0.3) * (0.8 + np.random.rand()),
            np.random.rand() - 0.25, np.random.rand() - 0.25,
            np.random.rand() - 0.6, np.random.rand() - 0.6,
            np.random.rand() * 2 * np.pi)



def random_phantom(spc):
    n = max(1, np.random.poisson(2))
    ellipses = [random_ellipse() for _ in range(n)]
    return odl.phantom.ellipsoid_phantom(spc, ellipses)


# phantom = random_phantom(space)
# f = np.reshape(phantom, [size, size])
# f = np.array(f).T
# # if np.max(f) > 0:
# #     f = 1 / np.max(f) * f #归一化
# plt.imshow(f)

train_true_image = np.empty([2000, 256, 256, 1])
train_set_data = np.empty([2000, 15360, 1])  # 15360 =256*60
train_set = np.empty([2000, 256, 256, 1])
label_set = np.empty([2000, 256, 256, 1])
# plt.imshow((label_set[0, ..., 0]), cmap='gray')
"""生成1000个有伪影训练集"""
for i in range(0, 1000):
    noise_num = 0.1 * np.random.random()
    phantom = random_phantom(space)
    f = np.reshape(phantom, [size, size])
    f = np.array(f).T
    if np.max(f) > 0:
        f = 1 / np.max(f) * f #归一化
    train_true_image[i, ..., 0] = f #训练集真实图像
    p = np.dot(w, np.reshape(f, [256 * 256, 1]))
    b_noisy = wgn_fun(p, noise_num)
    train_set_data[i] = b_noisy #训练集投影数据（加噪声）
    rec, F = art_fun(w, b_noisy, k, r)
    rec = np.reshape(rec, [256, 256])
    train_set[i, ..., 0] = rec
    label_set[i, ..., 0] = np.abs(train_true_image[i, ..., 0] - train_set[i, ..., 0])
    print(i)


    """1000个无伪影的训练集"""
for i in range(1000, 2000):
    phantom = random_phantom(space)
    f = np.reshape(phantom, [size, size])
    f = np.array(f).T
    if np.max(f) > 0:
        f = 1 / np.max(f) * f  # 归一化
    train_true_image[i, ..., 0] = f  # 训练集真实图像
    train_set[i, ..., 0] = f
    p = np.dot(w, np.reshape(f, [256 * 256, 1]))
    train_set_data[i] = p
    print(i)

np.save(path2 + 'train_true_image.npy', train_true_image)
np.save(path2 + 'train_set_data.npy', train_set_data)
np.save(path2 + 'train_set.npy', train_set)
np.save(path2 + 'label_set.npy', label_set)


###############################################################
"""400个验证集"""
test_true_image = np.empty([400, 256, 256, 1])
test_set_data = np.empty([400, 15360, 1])  # 15360 =256*60
test_set = np.empty([400, 256, 256, 1])
test_label_set = np.empty([400, 256, 256, 1])
"""生成200个无伪影训练集"""
for i in range(0, 200):
    noise_num = 0.1 * np.random.random()
    phantom = random_phantom(space)
    f = np.reshape(phantom, [size, size])
    f = np.array(f).T
    if np.max(f) > 0:
        f = 1 / np.max(f) * f #归一化
    test_true_image[i, ..., 0] = f #训练集真实图像
    p = np.dot(w, np.reshape(f, [256 * 256, 1]))
    b_noisy = wgn_fun(p, noise_num)
    test_set_data[i] = b_noisy #训练集投影数据（加噪声）
    rec, F = art_fun(w, b_noisy, k, r)
    rec = np.reshape(rec, [256, 256])
    test_set[i, ..., 0] = rec
    test_label_set[i, ..., 0] = np.abs(test_true_image[i, ..., 0] - test_set[i, ..., 0])
    print(i)


    """200个无伪影的训练集"""
for i in range(200, 400):
    phantom = random_phantom(space)
    f = np.reshape(phantom, [size, size])
    f = np.array(f).T
    if np.max(f) > 0:
        f = 1 / np.max(f) * f  # 归一化
    test_true_image[i, ..., 0] = f  # 训练集真实图像
    test_set[i, ..., 0] = f
    p = np.dot(w, np.reshape(f, [256 * 256, 1]))
    test_set_data[i] = p
    print(i)


np.save(path2 + 'validation_true_image.npy', test_true_image)
np.save(path2 + 'validation_set_data.npy', test_set_data)
np.save(path2 + 'validation_set.npy', test_set)
np.save(path2 + 'validation_set_label.npy', test_label_set)
################################################################


"""200个测试集"""
predict_true_image = np.empty([200, 256, 256, 1])
predict_set_data = np.empty([200, 15360, 1])  # 155360 =256*60
predict_set = np.empty([200, 256, 256, 1])
predict_label_set = np.empty([200, 256, 256, 1])

"""生成100个测试集"""
for i in range(100):
    noise_num = 0.05
    phantom = random_phantom(space)
    f = np.reshape(phantom, [size, size])
    f = np.array(f).T
    if np.max(f) > 0:
        f = 1 / np.max(f) * f  # 归一化
    predict_true_image[i, ..., 0] = f  # 验证集真实图像
    p = np.dot(w, np.reshape(f, [256 * 256, 1]))
    b_noisy = wgn_fun(p, noise_num)
    predict_set_data[i] = b_noisy #训练集投影数据（加噪声）
    rec, F = art_fun(w, b_noisy, k, r)
    rec = np.reshape(rec, [256, 256])
    predict_set[i, ..., 0] = rec
    predict_label_set[i, ..., 0] = np.abs(predict_true_image[i, ..., 0] - predict_set[i, ..., 0])
    print(i)

"""200个有伪影的测试集"""
for i in range(100, 200):
    phantom = random_phantom(space)
    f = np.reshape(phantom, [size, size])
    f = np.array(f).T
    if np.max(f) > 0:
        f = 1 / np.max(f) * f  # 归一化
    predict_true_image[i, ..., 0] = f  # 验证集真实图像
    predict_set[i, ..., 0] = f
    p = np.dot(w, np.reshape(f, [256 * 256, 1]))
    predict_set_data[i] = p

np.save(path2 + 'predict_true_image.npy', predict_true_image)
np.save(path2 + 'predict_set_data.npy', predict_set_data)
np.save(path2 + 'predict_set.npy', predict_set)
np.save(path2 + 'predict_set_label.npy', predict_label_set)
