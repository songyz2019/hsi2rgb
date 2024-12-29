# hsi2rgb

![Docker Pulls](https://img.shields.io/docker/pulls/songyz2019/hsi2rgb?logo=docker&style=flat-square&logoColor=white)
[![Hugging Face](https://img.shields.io/badge/HuggingFace-Demo-yellow?logo=huggingface&style=flat-square&logoColor=white)](https://huggingface.co/spaces/songyz2019/hsi2rgb)
![License](https://img.shields.io/github/license/songyz2019/hsi2rgb?style=flat-square)

Easily convert a hyperspectral image to an RGB image.

![ui-demo.jpg](asset/ui-demo.jpg)

# Usage
## HuggingFace
[Online HuggingFace Demo](https://huggingface.co/spaces/songyz2019/hsi2rgb)
## Docker
1. Run the container: `docker run -it -p 7860:7860 songyz2019/hsi2rgb`
2. Open `http://localhost:7860` in your browser
3. Upload your HSI image (`.mat` or `.tif`)
4. Set the wave length range
5. Submit and see the result


# Build
1. Build the container: `docker compose build`
2. Run the container: `docker compose up -d`

# License

```text
This program is free software: you can redistribute it and/or modify it under the terms of the GNU Affero General Public License as published by the Free Software Foundation, version 3.

This program is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU Affero General Public License for more details.

You should have received a copy of the GNU Affero General Public License along with this program. If not, see <https://www.gnu.org/licenses/>.
```
