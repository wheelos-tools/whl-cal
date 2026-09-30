ARG PYTHON_RUNTIME_CACHE=whl-cal-lidar2imu:slim
FROM ${PYTHON_RUNTIME_CACHE} AS dependency-cache

FROM python:3.10-slim-bookworm

LABEL org.opencontainers.image.title="whl-cal ROS-free native GRIL"
LABEL org.opencontainers.image.source="https://github.com/wheelos-tools/whl-cal"
LABEL io.wheelos.release-status="review-only"

ENV LD_LIBRARY_PATH=/opt/whl-cal/lib \
    MPLBACKEND=Agg \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONPATH=/opt/whl-cal \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

COPY --from=dependency-cache /usr/local/lib/python3.10/site-packages/ \
     /usr/local/lib/python3.10/site-packages/

COPY common /opt/whl-cal/common
COPY gril /opt/whl-cal/gril
COPY lidar2lidar /opt/whl-cal/lidar2lidar

COPY docker/lidar2imu/runtime/bin/ /opt/whl-cal/bin/
COPY docker/lidar2imu/runtime/lib/ /opt/whl-cal/lib/
COPY docker/lidar2imu/runtime/manifest.yaml /opt/whl-cal/runtime-manifest.yaml
COPY .agents/skills/gril-calib-validation/resources/vanjeelidar16.yaml \
     /opt/whl-cal/config/vanjeelidar16.yaml

RUN test -x /opt/whl-cal/bin/gril_native_full_frontend \
    && test -x /opt/whl-cal/bin/gril_native_batch \
    && useradd --create-home --uid 1000 calibrator \
    && mkdir -p \
        /data \
        /output \
        /home/wfh/01code/whl-cal/third_party/gril_native/build/Log \
    && chown -R calibrator:calibrator \
        /output \
        /home/wfh/01code/whl-cal/third_party/gril_native/build/Log

USER calibrator
WORKDIR /output

ENTRYPOINT ["python", "-m", "gril.cli", "run-native", "--config", "/opt/whl-cal/config/vanjeelidar16.yaml", "--executable", "/opt/whl-cal/bin/gril_native_full_frontend"]
CMD ["--help"]
