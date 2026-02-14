# SPDX-License-Identifier: MIT
# Copyright (c) 2025 Jeff Culverhouse
from .mixins.helpers import HelpersMixin
from .mixins.labels import LabelsMixin
from .mixins.detector import DetectorMixin
from .mixins.publish import PublishMixin
from .mixins.loops import LoopsMixin
from .mixins.mqtt import MqttMixin
from .base import Base


class Vision2Mqtt(
    HelpersMixin,
    LabelsMixin,
    DetectorMixin,
    PublishMixin,
    LoopsMixin,
    MqttMixin,
    Base,
):
    pass
