# syntax=docker/dockerfile:1.7-labs
FROM python:3.14-slim

# ===== Project Variables =====
ARG APP_NAME=vision2mqtt
ENV APP_NAME=${APP_NAME}
ARG SERVICE_DESC="YOLO object detection service for MQTT camera events"
ARG VERSION=0.0.0
ENV APP_VERSION=${VERSION}
ARG USER_ID=1000
ARG GROUP_ID=1000

# ===== Base Setup =====
WORKDIR /app
ENV DEBIAN_FRONTEND=noninteractive

# Generic pretend version variables (used by setuptools-scm)
ENV SETUPTOOLS_SCM_PRETEND_VERSION=${VERSION}
ENV APP_PRETEND_VERSION=${VERSION}

# ===== System Dependencies =====
RUN apt-get update && \
    apt-get upgrade -y && \
    apt-get install -y --no-install-recommends git gosu && \
    pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir uv && \
    rm -rf /var/lib/apt/lists/*

# ===== Copy Project Metadata =====
COPY pyproject.toml uv.lock ./

# ===== Build & Install =====
# 1. Create isolated virtual environment
RUN uv venv
ENV PATH="/app/.venv/bin:${PATH}"

# 2. Export locked dependencies (with pretend version active)
RUN SETUPTOOLS_SCM_PRETEND_VERSION=${VERSION} uv export --no-dev --format=requirements-txt > /tmp/reqs.all.txt

# 3. Strip the local project from deps list so setuptools-scm isn't triggered during deps install
RUN grep -v -E "(^-e\s+(\.|file://)|@\s+file://|^file://|/app)" /tmp/reqs.all.txt > /tmp/reqs.deps.txt || true

# 4. Install dependencies
RUN uv pip install --no-cache-dir -r /tmp/reqs.deps.txt

# ===== Copy Application Source =====
COPY . .

# 5. Install the app itself (pretend version visible, no deps)
RUN SETUPTOOLS_SCM_PRETEND_VERSION=${VERSION} uv pip install --no-cache-dir . --no-deps

# 6. Install axengine for AX8850 NPU support (optional, pure-python wheel from AXERA-TECH)
ARG ENABLE_NPU=false
ARG AXENGINE_TAG=0.1.3.rc2
ARG AXENGINE_WHEEL=axengine-0.1.3-py3-none-any.whl
RUN if [ "${ENABLE_NPU}" = "true" ]; then \
        echo "Installing axengine (tag=${AXENGINE_TAG}) for AX8850 NPU support..."; \
        uv pip install --no-cache-dir \
            "https://github.com/AXERA-TECH/pyaxengine/releases/download/${AXENGINE_TAG}/${AXENGINE_WHEEL}" \
        || { echo >&2 "ERROR: Failed to install axengine while ENABLE_NPU=true."; exit 1; }; \
    else \
        echo "Skipping axengine installation (ENABLE_NPU=${ENABLE_NPU})."; \
    fi

# 7. Cleanup
RUN rm -f /tmp/reqs.all.txt /tmp/reqs.deps.txt .git || true

# ===== Non-root Runtime User =====
RUN groupadd -g "${GROUP_ID}" appuser && \
    useradd -u "${USER_ID}" -g "${GROUP_ID}" --create-home --shell /bin/bash appuser && \
    mkdir -p /config /models && chown -R appuser:appuser /app /config /models

# Pre-register AXCL library path for axengine's ctypes.util.find_library
RUN echo '/usr/lib/axcl' > /etc/ld.so.conf.d/axcl.conf

# ===== Runtime =====
ENV SERVICE=${APP_NAME}
LABEL org.opencontainers.image.title=${APP_NAME} \
      org.opencontainers.image.description=${SERVICE_DESC} \
      org.opencontainers.image.version=${VERSION}

COPY entrypoint.sh /app/entrypoint.sh
ENTRYPOINT ["/app/entrypoint.sh"]
CMD ["-c", "/config"]
