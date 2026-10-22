# syntax=docker/dockerfile:1
# Intel's runtime image carries OpenVINO *and* the Intel GPU userspace (OpenCL /
# Level-Zero) that the Arc needs; a slim python base fails with libOpenCL missing.
# Pinned by digest: no :latest, ever.
FROM openvino/ubuntu24_runtime:2026.4.1@sha256:ef00731944b89db0148ff41bae9ef5f581dc9a1a2de90b1dd9a8ce1af69dc7b5

USER root
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    RESERV_CONFIG=/etc/reserv/models.yaml

# No pip at runtime: dependencies and weights are baked, so a pod starts offline.
RUN /opt/venv/bin/pip install --no-cache-dir \
      "starlette>=0.41" "uvicorn>=0.30" "numpy>=2" "prometheus-client>=0.21" \
      "pyyaml>=6" "transformers>=4.46"

COPY models.yaml /etc/reserv/models.yaml
COPY models/ /models/
COPY src/ /app/src
# Keep OpenVINO's own PYTHONPATH entries; only prepend ours.
ENV PYTHONPATH=/app/src:${PYTHONPATH}

RUN useradd -u 10001 -m -s /usr/sbin/nologin reserv \
 && chown -R reserv:reserv /models /app /etc/reserv \
 && /opt/venv/bin/python -c "import openvino, starlette, uvicorn, numpy, yaml, prometheus_client"

USER reserv
EXPOSE 8080
HEALTHCHECK --interval=30s --timeout=5s --start-period=120s --retries=3 \
  CMD /opt/venv/bin/python -c "import urllib.request,sys; sys.exit(0 if urllib.request.urlopen('http://127.0.0.1:8080/health', timeout=3).status==200 else 1)"
CMD ["/opt/venv/bin/python", "-m", "reserv"]
