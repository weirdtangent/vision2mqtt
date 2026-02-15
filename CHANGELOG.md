# [0.3.0](https://github.com/weirdtangent/vision2mqtt/compare/v0.2.2...v0.3.0) (2026-02-15)


### Bug Fixes

* decode raw YOLO feature maps from axmodel output ([055dd14](https://github.com/weirdtangent/vision2mqtt/commit/055dd1458b4e8e12ca99b19bf2a13a628c18c9ef))
* only trigger Claude review on [@claude](https://github.com/claude) mention ([a801c20](https://github.com/weirdtangent/vision2mqtt/commit/a801c207dbb04e1ffe478f738ac0ccd62d3f71ea))


### Features

* upgrade Claude review with auto-PR review and GitHub App token ([a9ce4af](https://github.com/weirdtangent/vision2mqtt/commit/a9ce4afec87ad00bb764984bb81ef7a2dd448478))

## [0.2.2](https://github.com/weirdtangent/vision2mqtt/compare/v0.2.1...v0.2.2) (2026-02-15)


### Bug Fixes

* add checkout step to claude-code-review workflow ([9ece56c](https://github.com/weirdtangent/vision2mqtt/commit/9ece56ccaf58b7dc8563d0a043eb09d2b1e627b1))
* add ldconfig for axengine provider detection ([6ffaa12](https://github.com/weirdtangent/vision2mqtt/commit/6ffaa12a2754018b730b352b1b97c5017c66b7be))

## [0.2.1](https://github.com/weirdtangent/vision2mqtt/compare/v0.2.0...v0.2.1) (2026-02-15)


### Bug Fixes

* add id-token: write permission for claude-code-action ([d57ff36](https://github.com/weirdtangent/vision2mqtt/commit/d57ff36255e4d744b72c5fe0c24f4ae7943a8691))
* replace pyaxcl with axengine for AX8850 NPU inference ([1d8e460](https://github.com/weirdtangent/vision2mqtt/commit/1d8e4602d9caf77bbe597609c070661ac1c9dace))

# [0.2.0](https://github.com/weirdtangent/vision2mqtt/compare/v0.1.2...v0.2.0) (2026-02-14)


### Bug Fixes

* include pyaxcl wheel in Docker image for AX8850 NPU support ([6d53adf](https://github.com/weirdtangent/vision2mqtt/commit/6d53adf01708a8a992f3544c38663034a24f3dc5))


### Features

* add Home Assistant MQTT discovery for auto-registering entities ([3658ac2](https://github.com/weirdtangent/vision2mqtt/commit/3658ac2737a0a111b8d804d1f3654419fe772ca3))

## [0.1.2](https://github.com/weirdtangent/vision2mqtt/compare/v0.1.1...v0.1.2) (2026-02-14)


### Bug Fixes

* add missing publish stubs required by BaseMqttMixin ([f0008fb](https://github.com/weirdtangent/vision2mqtt/commit/f0008fb7bcdd6a17de43752f141652ba9b5d0b91))

## [0.1.1](https://github.com/weirdtangent/vision2mqtt/compare/v0.1.0...v0.1.1) (2026-02-14)


### Bug Fixes

* scope mypy to src/ and simplify strict settings ([7753fe5](https://github.com/weirdtangent/vision2mqtt/commit/7753fe5de3d8349c7ce8554bc4dedf89ae182ec2))
