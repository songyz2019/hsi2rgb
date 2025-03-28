import unittest
from pathlib import Path
import numpy as np
from hsi2rgb import hsi2rgb
from fetch_houston2013 import fetch_muufl, fetch_houston2013, fetch_trento

class TestHsi2Rgb(unittest.TestCase):
    def setUp(self):
        return super().setUp()
        Path("tests/runs/").mkdir(parents=True, exist_ok=True)

    def test_batch_tensor(self):
        import torch
        B, C,H,W = 16, 144, 11, 11
        hsi = torch.rand(B, C, H, W)
        rgb = hsi2rgb(hsi, wavelength_range=(400, 700), input_format='CHW')
        self.assertEqual(rgb.shape, (B, 3, H, W))
        self.assertGreaterEqual(rgb.max(), 0)
        self.assertLessEqual(rgb.min(), 1)

    def test_manybatch_tensor(self):
        import torch
        B1, B2, C,H,W = 16, 5, 144, 11, 11
        hsi = torch.rand(B1, B2, C, H, W)
        rgb = hsi2rgb(hsi, wavelength_range=(400, 700), input_format='CHW')
        self.assertEqual(rgb.shape, (B1, B2, 3, H, W))

    def test_torch_tensor(self):
        C,H,W = 144, 11, 11
        import torch
        hsi = torch.rand(C, H, W)
        rgb = hsi2rgb(hsi, wavelength_range=(400, 700))
        self.assertEqual(rgb.shape, (3, H, W))

    def test_hsi2rgb_bhwc(self):
        hsi = np.random.rand(16, 11, 11, 144)
        rgb = hsi2rgb(hsi, wavelength_range=(400, 700), input_format='HWC')
        self.assertEqual(rgb.shape, (16, 11, 11, 3))

    def test_hsi2rgb(self):
        hsi = np.random.rand(11, 11, 144)
        rgb = hsi2rgb(hsi, wavelength_range=(400, 700), input_format='HWC')
        self.assertEqual(rgb.shape, (11, 11, 3))

    def test_muufl(self):
        from matplotlib.pyplot import imsave
        hsi, _, _, _ = fetch_muufl()
        rgb = hsi2rgb(hsi, wavelength_range=(350, 1000), input_format='CHW', output_format='HWC')
        self.assertEqual(rgb.shape, (hsi.shape[1], hsi.shape[2], 3))
        self.assertGreaterEqual(rgb.max(), 0)
        self.assertLessEqual(rgb.min(), 1)
        imsave('tests/runs/test_muufl.png', rgb)

    def test_trento(self):
        from matplotlib.pyplot import imsave
        hsi, _, _, _ = fetch_trento()
        rgb = hsi2rgb(hsi, wavelength_range=(350, 1000), input_format='CHW', output_format='HWC')
        self.assertEqual(rgb.shape, (hsi.shape[1], hsi.shape[2], 3))
        self.assertGreaterEqual(rgb.max(), 0)
        self.assertLessEqual(rgb.min(), 1)
        imsave('tests/runs/test_trento.png', rgb)

    def test_houston2013(self):
        from matplotlib.pyplot import imsave
        hsi, _, _, _, info = fetch_houston2013()
        rgb = hsi2rgb(hsi, wavelength=info["wavelength"], input_format='CHW', output_format='HWC')
        self.assertEqual(rgb.shape, (hsi.shape[1], hsi.shape[2], 3))
        self.assertGreaterEqual(rgb.max(), 0)
        self.assertLessEqual(rgb.min(), 1)
        imsave('tests/runs/test_houston2013.png', rgb)  
    
    def test_jax_tensor(self):
        import jax
        import jax.numpy as jnp
        C,H,W = 144, 11, 11
        hsi = jax.random.uniform(jax.random.PRNGKey(0), (C, H, W))
        rgb = hsi2rgb(hsi, wavelength_range=(400, 700))
        self.assertEqual(rgb.shape, (3, H, W))

if __name__ == "__main__":
    unittest.main()