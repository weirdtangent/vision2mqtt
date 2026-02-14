# SPDX-License-Identifier: MIT
# Copyright (c) 2025 Jeff Culverhouse
import asyncio
import argparse
from json_logging import setup_logging, get_logger
from mqtt_helper import MqttError
from .core import Vision2Mqtt
from .mixins.helpers import ConfigError


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="vision2mqtt", exit_on_error=True)
    p.add_argument(
        "-c",
        "--config",
        default="/config",
        help="Directory or file path for config.yaml (defaults to /config/config.yaml)",
    )
    return p


async def async_main() -> int:
    setup_logging()
    logger = get_logger(__name__)

    parser = build_parser()
    args = parser.parse_args()

    try:
        async with Vision2Mqtt(args=args) as vision2mqtt:
            await vision2mqtt.main_loop()
    except ConfigError as err:
        logger.error(f"fatal config error was found: {err!r}")
        return 1
    except MqttError as err:
        logger.error(f"mqtt service problems: {err!r}")
        return 1
    except KeyboardInterrupt:
        logger.warning("shutdown requested (Ctrl+C). exiting gracefully...")
        return 1
    except asyncio.CancelledError:
        logger.warning("main loop cancelled.")
        return 1
    except Exception as err:
        logger.error(f"unhandled exception: {err!r}", exc_info=True)
        return 1
    finally:
        logger.info("vision2mqtt stopped.")

    return 0


def main() -> int:
    try:
        return asyncio.run(async_main())
    except RuntimeError as err:
        if "asyncio.run() cannot be called from a running event loop" in str(err):
            loop = asyncio.get_event_loop()
            return loop.run_until_complete(async_main())
        raise
