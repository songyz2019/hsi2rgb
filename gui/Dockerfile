FROM ghcr.io/astral-sh/uv:debian-slim

RUN mkdir /app
WORKDIR /app

# For better layer caching, src should be add after uv sync
ADD ./ /app/
# Avoid unnecessary uv sync
RUN rm -rf /app/src 
# The --no-cache for uv make huggingface buggy, with errors like permission denied for /uv/.cache
RUN uv sync --frozen


# Use these commands in alphine linux since there's no manylinux musl for scikit-image
# The image size of alphine can be reduced to 650MB without aggressive optimization. 
# So, if the size of debian-slim is not a important problem, we can use debian-slim.
# ADD . /app/
# RUN rm -rf /app/src 
# RUN apk add --no-cache gcc g++ musl-dev libffi-dev libgomp && \
# uv sync --no-cache && \
# apk del gcc g++ musl-dev libffi-dev
# ADD ./src /app/src

ADD ./src /app/src

EXPOSE 7860
ENV GRADIO_SERVER_NAME="0.0.0.0"

# We should make Dockerfile less coupled with the project and focusing on environment setup
CMD ["uv", "run", "hsi2rgb_gui"]