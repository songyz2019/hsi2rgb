import einops
import matplotlib.pyplot as pl

def hsi2cubeside(hsi):
    """
    Convert HSI (Hyperspectral Imaging) data to two cube side images for visualization.
    
    Parameters:
    hsi : numpy.ndarray
        The input HSI data, expected to be a 3D array of shape (height, width, spectral bands) where the last dimension represents different spectral bands.
        The values are expected to be in the range [0, 1].

            Returns:
    numpy.ndarray
        The RGB representation of the input HSI data, with shape (height, width, 3). The values are rescaled to the range [0, 1] if raw is False.
    numpy.ndarray
        The RGB representation of the input HSI data, with shape (height, width, 3). The values are rescaled to the range [0, 1] if raw is False
    """

    wc = einops.reduce(hsi, 'h w c -> w c', 'mean')
    hc = einops.reduce(hsi, 'h w c -> h c', 'mean')
    wc_colormap = pl.cm.viridis(wc)
    hc_colormap = pl.cm.viridis(hc)
    return wc_colormap, hc_colormap