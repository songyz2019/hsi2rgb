from typing import Literal
import einops
import gradio as gr
from matplotlib import pyplot as pl
import numpy as np
import skimage
from hsi2rgb import hsi2rgb
from scipy.io import loadmat
import logging
import json
import os

logger = logging.getLogger(__name__)


def hsi2rgb_app(hsi_file, input_format:Literal['CHW', 'HWC']='CHW', wavelength_from=380, wavelength_to=1050, crop=(0,0,None,None), gamma=1.5):
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

    if input_format == 'CHW':
        hsi = einops.rearrange(hsi, 'c h w -> h w c')

    rgb = hsi2rgb(hsi, wavelength_range=(wavelength_from, wavelength_to), input_format='HWC', output_format='HWC', gamma=1) 
    
    img = skimage.exposure.adjust_gamma(rgb, 1/gamma)
    
    pl.figure()
    histo,bin_centers = skimage.exposure.histogram(img)
    n, bins, patches = pl.hist(histo, bins=256, facecolor='black', edgecolor='black',alpha=1, histtype='bar')
    histo_fig = pl.gcf()

    extra = json.dumps({
        'mat_file_key': key if ext == 'mat' else None,
    })
    return img, histo_fig, extra

def main():  
    logger.setLevel(logging.INFO)
    demo = gr.Interface(
        fn=hsi2rgb_app,
        inputs=[
            gr.File(file_types=['.mat','.tiff','.tif'], type='filepath'), 
            gr.Radio(['CHW', 'HWC'], value='CHW', label='Input Format'),
            gr.Number(380), 
            gr.Number(1050), 
            gr.Textbox(label='Crop (x1, y1, x2, y2)', placeholder='0, 0, None, None', value='0, 0, None, None', lines=1),
            gr.Slider(0.1, 10, 2.2),
        ],
        outputs=[gr.Image(format='png', label='RGB Image'), gr.Plot(label='Histogram'), gr.JSON(label='Extra Information')],
        title='hsi2rgb: Easily convert HSI image to RGB image',
        article='''You can download a sample [HSI image from GatorSense/MUUFLGulfport](https://github.com/GatorSense/MUUFLGulfport/raw/refs/heads/master/MUUFLGulfportSceneLabels/muufl_gulfport_campus_1_hsi_220_label.mat) <br/> for more information, please visit https://github.com/songyz2019/hsi2rgb''',
        theme=gr.themes.Citrus(),
    )


    demo.launch(share = True )

if __name__ == '__main__':
    main()