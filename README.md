# hsi2rgb

![Docker Pulls](https://img.shields.io/docker/pulls/songyz2019/hsi2rgb?logo=docker&style=flat-square&logoColor=white)
[![Hugging Face](https://img.shields.io/badge/HuggingFace-Demo-yellow?logo=huggingface&style=flat-square&logoColor=white)](https://huggingface.co/spaces/songyz2019/hsi2rgb)
![License](https://img.shields.io/github/license/songyz2019/hsi2rgb?style=flat-square)
[![PyPI - Version](https://img.shields.io/pypi/v/hsi2rgb.svg)](https://pypi.org/project/hsi2rgb)
[![PyPI - Python Version](https://img.shields.io/pypi/pyversions/hsi2rgb.svg)](https://pypi.org/project/hsi2rgb)

Easily convert a hyperspectral image to an RGB image. hsi2rgb is:
1. A python package that supports numpy, torch and jax, with gamma correction, automatically convertions, XYZ and RGB colorspace suppots, etc.
2. A docker container that can be easily deployed as a GUI application

![ui-demo.jpg](asset/ui-demo.jpg)

# Usage
## Python Package
1. Install hsi2rgb
```bash
pip install hsi2rgb
```
2. Import and use hsi2rgb
```python
hsi = np.random.rand(144, 11, 11)
rgb = hsi2rgb(hsi) # rgb: (3, 11, 11)
```
3. More examples can be found in [test.py](core/tests/test.py)


## GUI
[Online HuggingFace Demo](https://huggingface.co/spaces/songyz2019/hsi2rgb)

1. Run the container: `docker run -it -p 7860:7860 songyz2019/hsi2rgb`
2. Open `http://localhost:7860` in your browser
3. Upload your HSI image (`.mat` or `.tif`)
4. Set the wave length range
5. Submit and see the result

# Build
## uv
1. Install [uv](https://docs.astral.sh/uv/)
2. Simply sync dependicies and build
```bash
uv sync
uv build
```

## Docker
> The docker app is under refactor, please use [this branch](https://github.com/songyz2019/hsi2rgb/tree/f2c76629b49659e7d16ec62958c8a972e97fde3c) to build currently.
1. Build the container: `docker compose build`
2. Run the container: `docker compose up -d`

# License
```text
Copyright 2025 songyz2023

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
```
