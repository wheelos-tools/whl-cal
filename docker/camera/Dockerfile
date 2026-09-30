FROM python:3.10-slim-bookworm

LABEL org.opencontainers.image.title="whl-cal camera intrinsic calibration"
LABEL org.opencontainers.image.source="https://github.com/wheelos-tools/whl-cal"

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONPATH=/opt/whl-cal \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        libgl1 \
        libglib2.0-0 \
        libsm6 \
        libxext6 \
        libxrender1 \
    && rm -rf /var/lib/apt/lists/*

COPY docker/camera/requirements.txt /tmp/requirements.txt
RUN python -m pip install --disable-pip-version-check -r /tmp/requirements.txt

COPY camera /opt/whl-cal/camera
COPY common /opt/whl-cal/common

COPY docker/camera/config.yaml /opt/whl-cal/config/camera.yaml

RUN useradd --create-home --uid 1000 calibrator \
    && mkdir -p /data /output \
    && chown -R calibrator:calibrator /output

USER calibrator
WORKDIR /output

ENTRYPOINT ["python", "-m", "camera.cli", "--config", "/opt/whl-cal/config/camera.yaml"]
CMD ["--help"]
