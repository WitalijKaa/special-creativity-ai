# syntax=docker/dockerfile:1.7

FROM pytorch/pytorch:2.8.0-cuda12.9-cudnn9-runtime as app

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PIP_NO_CACHE_DIR=1

ARG TZ
ENV TZ=$TZ
RUN ln -snf /usr/share/zoneinfo/$TZ /etc/localtime && echo $TZ > /etc/timezone

RUN apt-get update \
 && apt-get install -y --no-install-recommends tzdata \
 && apt-get clean \
 && rm -rf /var/lib/apt/lists/*

WORKDIR /app

RUN --mount=type=cache,target=/root/.cache/pip \
    python -m pip install --upgrade pip \
 && python -m pip install \
    transformers \
    pydantic \
    "fastapi[standard]" \
    "python-dotenv" \
    "rich" \
    "openai" \
    "accelerate" \
    "bitsandbytes==0.47.*" \
    "sentencepiece==0.2.*" \
    "protobuf==6.32.*"

COPY . .

ENTRYPOINT ["uvicorn"]
CMD ["main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]
