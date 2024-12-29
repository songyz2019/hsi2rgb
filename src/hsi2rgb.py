import numpy as np
import skimage

def _cie_xyz_weights(n):
    """

    n 波长
    """

    def g(x, mu, sigma1, sigma2):
        """分段高斯函数"""
        return np.where(
            x < mu,
            np.exp(-(x - mu) ** 2 / (sigma1 ** 2) / 2),
            np.exp(-(x - mu) ** 2 / (sigma2 ** 2) / 2),
        )
    x = (
        0.362   * g(n, 442.0 , 16.0 , 26.7)
        -0.065  * g(n, 501.1, 20.4 , 26.2 )
        + 1.056 * g(n, 599.8, 37.9 , 31.0 )
    )
    y = (
        0.286   * g(n, 530.9 , 16.3 , 31.1)
        + 0.821 * g(n, 568.8, 46.9 , 40.5 )
    )
    z = (
        0.980   * g(n, 437.0 , 11.8 , 36.0)
        + 0.681 * g(n, 459.0, 26.0 , 13.8 )
    )
    return x,y,z


def hsi2rgb(hsi, wavelength, raw=False):
    """真彩色"""
    x,y,z = _cie_xyz_weights(wavelength)
    X = np.sum(hsi*x, axis=2)
    Y = np.sum(hsi*y, axis=2)
    Z = np.sum(hsi*z, axis=2)

    transform_matrix = [
        [3.1338561, -1.6168667, -0.4906146],
        [-0.9787684, 1.9161415, 0.0334540],
        [0.0719453, -0.2289914, 1.4052427]
    ]

    XYZ = np.stack([X, Y, Z])
    rgb = np.einsum('C c, c h w -> h w C', transform_matrix, XYZ)
    rgb = skimage.exposure.rescale_intensity(rgb,out_range=(0,1))
    return rgb

# from sklearn.decomposition import PCA
# def hsi2rgb_pca(hsi):
#     """PCA降维"""
#     hsi = hsi.transpose(1,2,0)
#     # 使用PCA降维
#     pca = PCA(n_components=3)
#     rgb = pca.fit_transform(hsi.reshape(-1, hsi.shape[-1]))

#     # 标准化PCA结果到[0, 255]
#     rgb -= np.min(rgb, axis=0)
#     rgb /= np.max(rgb, axis=0)
#     rgb = (rgb * 255).astype(np.uint8)

#     # 调整形状以匹配图像尺寸
#     rgb = rgb.reshape(hsi.shape[0], hsi.shape[1], 3)

#     return rgb


# def hsi2rgb_pesudo(hsi):
#     """伪菜色"""
#     rgb_bands = hsi[(hsi.shape[0]//5*2, hsi.shape[0]//5*3, hsi.shape[0]//5*4),:,:]
#     rgb_image = rgb_bands.transpose(1,2,0)

#     # 归一化均衡化
#     rgb_image = (rgb_image - np.min(rgb_image)) / (np.max(rgb_image) - np.min(rgb_image))
#     rgb_image = skimage.exposure.equalize_hist(rgb_image)

#     return rgb_image

