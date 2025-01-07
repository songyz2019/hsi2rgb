FROM ghcr.io/astral-sh/uv:python3.12-alpine



RUN mkdir /app
WORKDIR /app
ADD ./pyproject.toml /app/pyproject.toml

RUN apk add --no-cache gcc g++ musl-dev libffi-dev libgomp && \
uv sync --no-cache && \
apk del gcc g++ musl-dev libffi-dev

EXPOSE 7860
ADD . /app/
ENV GRADIO_SERVER_NAME="0.0.0.0"
CMD ["uv", "run", "start"]