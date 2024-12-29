import gradio as gr
import numpy as np
import skimage
from hsi2rgb import hsi2rgb
from scipy.io import loadmat
from einops import rearrange

def hsi2rgb_app(hsi_file, wavelength_from=380, wavelength_to=1050, gamma=1/1.5):

    match = hsi_file.split('.')[-1]
    if match == 'mat':
        mat = loadmat(
            hsi_file,
            squeeze_me=True,
            mat_dtype=True,
            struct_as_record=False
        )
        key = list(mat.keys())[3]
        hsi = mat[key]
    elif match in ['tiff','tif']:
        hsi = skimage.io.imread(hsi_file)
    else:
        return

    n_channel = hsi.shape[-1]
    wavelength = np.linspace(wavelength_from, wavelength_to, n_channel)
    rgb = hsi2rgb(hsi, wavelength, raw=False) 
    
    img = skimage.exposure.adjust_gamma(rgb, gamma)
    return img
    

demo = gr.Interface(
    fn=hsi2rgb_app,
    inputs=[
        gr.File(file_types=['.mat','.tiff','.tif'], type='filepath'), 
        gr.Number(380), 
        gr.Number(1050), 
        gr.Slider(0.5, 2, 1/1.5)
    ],
    outputs=[gr.Image()],
)

demo.launch(share=True)