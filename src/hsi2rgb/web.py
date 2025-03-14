import gradio as gr
import numpy as np
import skimage
from hsi2rgb.hsi2rgb import hsi2rgb
from scipy.io import loadmat
from einops import rearrange
import logging
import json

logger = logging.getLogger(__name__)


def hsi2rgb_app(hsi_file, wavelength_from=380, wavelength_to=1050, gamma=1/1.5, is_chw_input=False):
    ext = hsi_file.split('.')[-1]
    if ext == 'mat':
        mat = loadmat(
            hsi_file,
            squeeze_me=True,
            mat_dtype=True,
            struct_as_record=False
        )
        key = list(mat.keys())[3]
        hsi = mat[key]

        # For capbility of reading .mat file with different structure, like the offical MUUFL dataset
        if not isinstance(hsi, np.ndarray):
            hsi = hsi.Data
        
    elif ext in ['tiff','tif']:
        hsi = skimage.io.imread(hsi_file)
    else:
        return

    if is_chw_input:
        hsi = rearrange(hsi, 'c h w -> h w c')

    n_channel = hsi.shape[-1]
    wavelength = np.linspace(wavelength_from, wavelength_to, n_channel)
    rgb = hsi2rgb(hsi, wavelength, raw=False) 
    
    img = skimage.exposure.adjust_gamma(rgb, gamma)

    extra = json.dumps({
        'n_channel': n_channel,
        'mat_file_key': key if ext == 'mat' else None,
    })
    return img, extra

def main():  
    logger.setLevel(logging.INFO)
    demo = gr.Interface(
        fn=hsi2rgb_app,
        inputs=[
            gr.File(file_types=['.mat','.tiff','.tif'], type='filepath'), 
            gr.Number(380), 
            gr.Number(1050), 
            gr.Slider(0.5, 2, 1/1.5),
            gr.Checkbox(False, label='is CHW input', show_label=True)
        ],
        outputs=[gr.Image(format='png', label='RGB Image'), gr.Textbox(label='Extra Information')],
        title='hsi2rgb: Easily convert HSI image to RGB image',
        article='''You can download a sample [HSI image from GatorSense/MUUFLGulfport](https://github.com/GatorSense/MUUFLGulfport/raw/refs/heads/master/MUUFLGulfportSceneLabels/muufl_gulfport_campus_1_hsi_220_label.mat) <br/> for more information, please visit https://github.com/songyz2019/hsi2rgb''',
        theme=gr.themes.Citrus(),
    )

    demo.launch(share=True)

if __name__ == '__main__':
    main()