# Environment Variables

While using a config.yaml file is the recommended approach, vision2mqtt also supports configuration via environment variables.

## MQTT Settings

- `MQTT_HOST` (optional, default = 'localhost') - MQTT broker hostname or IP
- `MQTT_PORT` (optional, default = 1883) - MQTT broker port
- `MQTT_USERNAME` (required) - MQTT username
- `MQTT_PASSWORD` (optional, default = empty password) - MQTT password
- `MQTT_QOS` (optional, default = 0) - Quality of Service (0-2)
- `MQTT_PROTOCOL` (optional, default = '5') - MQTT protocol version: '3.1.1' or '5'
- `MQTT_PREFIX` (optional, default = 'vision2mqtt') - MQTT topic prefix
- `MQTT_DISCOVERY_PREFIX` (optional, default = 'homeassistant') - MQTT discovery prefix for Home Assistant

## MQTT TLS Settings

- `MQTT_TLS_ENABLED` (required if using TLS) - set to `true` to enable
- `MQTT_TLS_CA_CERT` (required if using TLS) - path to the CA certificate
- `MQTT_TLS_CERT` (required if using TLS) - path to the client certificate
- `MQTT_TLS_KEY` (required if using TLS) - path to the client private key

## Vision Settings

- `VISION_BACKEND` (optional, default = 'ultralytics') - detection backend: 'ultralytics' or 'axcl'
- `VISION_MODEL` (optional, default = 'yolo11n.pt') - model path (.axmodel) or model name (.pt)
- `VISION_MIN_CONFIDENCE` (optional, default = 0.45) - minimum confidence threshold for detections
- `VISION_CONCURRENCY` (optional, default = 1) - number of worker tasks processing the queue
- `VISION_MAX_QUEUE` (optional, default = 20) - maximum queued vision requests (oldest dropped when full)
- `VISION_RETAIN_PRESENCE` (optional, default = false) - publish retained ON/OFF presence per camera per label
- `VISION_DEBUG_SAVE` (optional, default = false) - save annotated images to /tmp for debugging

## Home Assistant Settings

- `HOME_ASSISTANT` (optional, default = true) - enable Home Assistant MQTT discovery

## Other Settings

- `DEBUG` (optional, default = false) - enable debug logging
- `APP_VERSION` (optional) - override the version string (normally read from VERSION file)
- `APP_TIER` (optional, default = 'prod') - set to 'dev' to append :DEV suffix to version
- `READY_FILE` (optional, default = '/tmp/vision2mqtt.ready') - path to the healthcheck ready file
