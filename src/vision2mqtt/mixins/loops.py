# SPDX-License-Identifier: MIT
# Copyright (c) 2025 Jeff Culverhouse
from __future__ import annotations

import asyncio
import signal
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from vision2mqtt.interface import VisionServiceProtocol as Vision2Mqtt


class LoopsMixin:
    async def worker(self: Vision2Mqtt, worker_id: int) -> None:
        self.logger.info(f"worker-{worker_id} started")
        while self.running:
            try:
                event = await asyncio.wait_for(self.queue.get(), timeout=2.0)
            except asyncio.TimeoutError:
                continue
            except asyncio.CancelledError:
                self.logger.debug(f"worker-{worker_id} cancelled")
                break

            try:
                self.logger.debug(f"worker-{worker_id} processing '{event.camera_name}' ({event.event_id})")
                result = await self.detect_objects(event)
                await self.publish_vision_result(event, result)
            except Exception as err:
                self.logger.error(f"worker-{worker_id} failed processing '{event.camera_name}' ({event.event_id}): {err!r}", exc_info=True)
            finally:
                self.queue.task_done()

        self.logger.info(f"worker-{worker_id} stopped")

    async def heartbeat(self: Vision2Mqtt) -> None:
        while self.running:
            try:
                await asyncio.sleep(60)
                self.heartbeat_ready()
            except asyncio.CancelledError:
                self.logger.debug("heartbeat cancelled during sleep")
                break

    async def main_loop(self: Vision2Mqtt) -> None:
        for sig in (signal.SIGTERM, signal.SIGINT):
            try:
                signal.signal(sig, self.handle_signal)
            except Exception:
                self.logger.debug(f"cannot install handler for {sig}")

        self.running = True
        self.mark_ready()

        concurrency = self.vision_config.get("concurrency", 1)
        self.logger.info(f"starting {concurrency} worker(s), listening on {self.vision_config['subscribe_topics']}")

        tasks = [
            *[asyncio.create_task(self.worker(i), name=f"worker-{i}") for i in range(concurrency)],
            asyncio.create_task(self.heartbeat(), name="heartbeat"),
        ]

        try:
            await asyncio.gather(*tasks)
        except asyncio.CancelledError:
            self.logger.warning("main loop cancelled — shutting down...")
        except Exception as err:
            self.logger.exception(f"unhandled exception in main loop: {err!r}")
            self.running = False
        finally:
            self.logger.info("all loops terminated — cleanup complete.")
