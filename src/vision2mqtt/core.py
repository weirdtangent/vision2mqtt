# SPDX-License-Identifier: MIT
# Copyright (c) 2025 Jeff Culverhouse
from .base import Base
from .mixins.composites import CompositesMixin
from .mixins.detector import DetectorMixin
from .mixins.helpers import HelpersMixin
from .mixins.labels import LabelsMixin
from .mixins.loops import LoopsMixin
from .mixins.mqtt import MqttMixin
from .mixins.publish import PublishMixin
from .mixins.system_stats import SystemStatsMixin


class Vision2Mqtt(
    HelpersMixin,
    LabelsMixin,
    CompositesMixin,
    DetectorMixin,
    SystemStatsMixin,
    PublishMixin,
    LoopsMixin,
    MqttMixin,
    Base,
):
    pass
