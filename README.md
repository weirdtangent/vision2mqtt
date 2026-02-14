# weirdtangent/vision2mqtt

YOLO object detection service for MQTT camera events — subscribes to motion event
images from camera bridges, runs inference via [YOLO11](https://docs.ultralytics.com/models/yolo11/),
and publishes detection results back to MQTT.

[![Deploy Status](https://github.com/weirdtangent/vision2mqtt/actions/workflows/deploy.yaml/badge.svg)](https://github.com/weirdtangent/vision2mqtt/actions/workflows/deploy.yaml)

Designed to work with [amcrest2mqtt](https://github.com/weirdtangent/amcrest2mqtt)
and [blink2mqtt](https://github.com/weirdtangent/blink2mqtt), but any MQTT client
can publish vision requests in the expected format.

## How It Works

```
Camera bridges (Synology)              vision2mqtt (Raspberry Pi 5 + LLM-8850)
┌─────────────┐                        ┌─────────────────────┐
│amcrest2mqtt │──vision/request──┐     │ subscribe to        │
│blink2mqtt   │──vision/request──┼────►│ +/vision/request    │
└─────────────┘                  │     │                     │
                              MQTT     │ YOLO11 inference    │
                            (Mosquitto)│ (AX8850 NPU or CPU) │
                                 │     │                     │
                                 │◄────│ publish results     │
                                       └─────────────────────┘
```

1. Camera bridges detect motion and publish a JSON message with a base64-encoded image
2. vision2mqtt picks up the message, decodes the image, runs YOLO11 object detection
3. Results (detected objects, summary, optional presence) are published back to MQTT

## Detection Backends

| Backend | Config value | Hardware | Speed | Use case |
|---------|-------------|----------|-------|----------|
| AX8850 NPU | `axcl` | [M5Stack LLM-8850](https://docs.m5stack.com/en/ai_hardware/LLM-8850_Card) on Pi 5 | ~8ms/frame | Production |
| Ultralytics CPU | `ultralytics` | Any machine | ~200-500ms/frame | Development/testing |

## Docker

For `docker-compose`, use the [configuration included](https://github.com/weirdtangent/vision2mqtt/blob/main/docker-compose.yaml) in this repository.

Using the [docker image](https://hub.docker.com/repository/docker/graystorm/vision2mqtt/general), mount your configuration volume at `/config` and include a `config.yaml` file (see the included [config.yaml.sample](config.yaml.sample) file as a template).

For the `axcl` backend, also mount your model directory at `/models`:
```yaml
volumes:
  - ./config:/config
  - ./models:/models
```

## Configuration

The recommended way to configure vision2mqtt is via the `config.yaml` file. See [config.yaml.sample](config.yaml.sample) for a complete example with all available options.

### MQTT Settings

```yaml
mqtt:
  host: 10.10.10.1
  port: 1883
  username: mqtt
  password: password
  qos: 0
  protocol_version: "5"
  prefix: vision2mqtt
  # TLS settings (optional)
  tls_enabled: false
  tls_ca_cert: /config/ca.crt
  tls_cert: /config/client.crt
  tls_key: /config/client.key
```

### Vision Settings

```yaml
vision:
  backend: ultralytics         # "axcl" for AX8850 NPU, "ultralytics" for CPU
  model: yolo11n.pt            # model path (.axmodel) or name (.pt)
  subscribe_topics:
    - "+/vision/request"
  labels:
    - person
    - vehicle
    - animal
    - bird
  min_confidence: 0.45
  concurrency: 1
  max_queue: 20
  retain_presence: false
```

### Environment Variables

While the config file is recommended, environment variables are also supported. See [ENVIRONMENT_VARIABLES.md](ENVIRONMENT_VARIABLES.md) for the full list of available environment variables.

## MQTT Topics

### Input (subscribed)

- `+/vision/request` — JSON with `camera_id`, `camera_name`, `event_id`, `image_b64`, `timestamp`, `source`

### Output (published)

- `vision2mqtt/{camera_id}/{event_id}/objects` — JSON array of detected objects
- `vision2mqtt/{camera_id}/{event_id}/summary` — JSON summary with label counts and timing
- `vision2mqtt/{camera_id}/presence/{label}` — retained `ON`/`OFF` per camera per label (optional)

### Example Output

**Objects:**
```json
[
  {"label": "person", "raw_label": "person", "confidence": 0.87, "bbox": [0.12, 0.34, 0.45, 0.89]},
  {"label": "vehicle", "raw_label": "car", "confidence": 0.72, "bbox": [0.56, 0.10, 0.98, 0.55]}
]
```

**Summary:**
```json
{
  "camera_id": "2BEFD0C907BB6BF2",
  "camera_name": "Front Yard",
  "event_id": "20260214-153045",
  "timestamp": "2026-02-14T15:30:45",
  "labels": {"person": 1, "vehicle": 1},
  "object_count": 2,
  "processing_time_ms": 8.2,
  "source": "recording_snapshot"
}
```

## Label Mapping

COCO classes are simplified to categories useful for home security:

| Simplified | COCO classes |
|-----------|-------------|
| person | person |
| vehicle | car, truck, bus, motorcycle, bicycle |
| animal | cat, dog, horse, cow, sheep, bear, elephant, zebra, giraffe |
| bird | bird |

## Raspberry Pi 5 + M5Stack LLM-8850 Setup

The `axcl` backend is specifically tested on a **Raspberry Pi 5** with the **[M5Stack LLM-8850 Pi HAT](https://docs.m5stack.com/en/ai_hardware/LLM-8850_Card)** kit (AXera AX8850 NPU, 24 TOPS @ INT8, 8GB LPDDR4x).

### Quick start (fresh Raspberry Pi OS 64-bit Lite)

```bash
# 1. Install build deps
sudo apt update && sudo apt upgrade -y
sudo apt install -y gcc make patch dkms linux-headers-$(uname -r)

# 2. Enable PCIe Gen 3 — add to /boot/firmware/config.txt under [all]:
#    dtparam=pciex1_gen=3

# 3. Install AXCL driver from M5Stack APT repo
sudo wget -qO /etc/apt/keyrings/StackFlow.gpg https://repo.llm.m5stack.com/m5stack-apt-repo/key/StackFlow.gpg
echo 'deb [signed-by=/etc/apt/keyrings/StackFlow.gpg] https://repo.llm.m5stack.com/m5stack-apt-repo axclhost main' \
  | sudo tee /etc/apt/sources.list.d/axclhost.list
sudo apt update && sudo apt install -y axclhost

# 4. Install Docker
sudo install -m 0755 -d /etc/apt/keyrings
sudo curl -fsSL https://download.docker.com/linux/debian/gpg -o /etc/apt/keyrings/docker.asc
echo "deb [arch=arm64 signed-by=/etc/apt/keyrings/docker.asc] https://download.docker.com/linux/debian bookworm stable" \
  | sudo tee /etc/apt/sources.list.d/docker.list
sudo apt update && sudo apt install -y docker-ce docker-ce-cli containerd.io docker-compose-plugin
sudo usermod -aG docker $USER

# 5. COLD BOOT (power-cycle required — soft reboot won't reset the AX8850 PCIe link)
sudo shutdown -h now
# Unplug and replug power

# 6. Verify
axcl-smi          # should show AX650N with temp and memory
docker --version   # should show Docker CE
```

### Deploy vision2mqtt

```bash
mkdir -p ~/vision2mqtt/config ~/vision2mqtt/models

# Download YOLO11s model
wget https://huggingface.co/AXERA-TECH/YOLO11/resolve/main/ax650/yolo11s.axmodel \
  -P ~/vision2mqtt/models/
```

Create `config/config.yaml` with `backend: axcl` and `model: /models/yolo11s.axmodel` (see [config.yaml.sample](config.yaml.sample)).

For the Pi 5 with LLM-8850, the `docker-compose.yaml` needs NPU device passthrough:
```yaml
services:
  vision2mqtt:
    image: graystorm/vision2mqtt:latest
    container_name: vision2mqtt
    restart: unless-stopped
    network_mode: host
    devices:
      - /dev/axcl_host:/dev/axcl_host
      - /dev/ax_mmb_dev:/dev/ax_mmb_dev
    volumes:
      - ./config:/config
      - ./models:/models
      - /usr/lib/axcl:/usr/lib/axcl:ro
    environment:
      - TZ=America/New_York
      - LD_LIBRARY_PATH=/usr/lib/axcl
```

Start:
```bash
cd ~/vision2mqtt && docker compose up -d
```

The container auto-starts on boot via `restart: unless-stopped`.

### Updating to a new image

```bash
cd ~/vision2mqtt
docker compose pull        # pull latest image
docker compose up -d       # recreate container with new image
docker image prune -f      # clean up old images
```

### Resources

- [M5Stack LLM-8850 software setup guide](https://docs.m5stack.com/en/guide/ai_accelerator/llm-8850/m5_llm_8850_software_install)
- [AXCL Pi 5 examples](https://github.com/AXERA-TECH/axcl-pi5-examples)
- [AXERA-TECH models on Hugging Face](https://huggingface.co/AXERA-TECH)

## Running the app

For Docker Compose, see the included [docker-compose.yaml](docker-compose.yaml).

The app expects the config directory to be mounted at `/config`:
```
CMD [ "python", "./app.py", "-c", "/config" ]
```

## Out of Scope

### Non-Docker Environments

Docker is the only supported way of deploying the application. The app should run directly via Python but this is not supported.

## See also
* [amcrest2mqtt](https://github.com/weirdtangent/amcrest2mqtt)
* [blink2mqtt](https://github.com/weirdtangent/blink2mqtt)
* [govee2mqtt](https://github.com/weirdtangent/govee2mqtt)

## Buy Me A Coffee

A few people have kindly requested a way to donate a small amount of money. If you feel so inclined I've set up a "Buy Me A Coffee"
page where you can donate a small sum. Please do not feel obligated to donate in any way - I work on the app because it's
useful to myself and others, not for any financial gain - but any token of appreciation is much appreciated :)

<a href="https://buymeacoffee.com/weirdtangent">Buy Me A Coffee</a>

---

### Build & Quality Status

![Build & Release](https://img.shields.io/github/actions/workflow/status/weirdtangent/vision2mqtt/deploy.yaml?branch=main&label=build%20%26%20release&logo=githubactions)
![Lint](https://img.shields.io/github/actions/workflow/status/weirdtangent/vision2mqtt/deploy.yaml?branch=main&label=lint%20(ruff%2Fblack%2Fmypy)&logo=python)
![Docker Build](https://img.shields.io/github/actions/workflow/status/weirdtangent/vision2mqtt/deploy.yaml?branch=main&label=docker%20build&logo=docker)
![Python](https://img.shields.io/badge/python-3.12%20|%203.13%20|%203.14-blue?logo=python)
![Release](https://img.shields.io/github/v/release/weirdtangent/vision2mqtt?sort=semver)
![Docker Image Tag](https://img.shields.io/github/v/release/weirdtangent/vision2mqtt?label=docker%20tag&sort=semver&logo=docker)
![Docker Pulls](https://img.shields.io/docker/pulls/graystorm/vision2mqtt?logo=docker)
![License](https://img.shields.io/github/license/weirdtangent/vision2mqtt)

### Security

![SBOM](https://img.shields.io/badge/SBOM-included-green?logo=docker)
![Provenance](https://img.shields.io/badge/provenance-attested-green?logo=sigstore)
![Signed](https://img.shields.io/badge/cosign-signed-green?logo=sigstore)
![Trivy](https://img.shields.io/badge/trivy-scanned-green?logo=aquasecurity)
