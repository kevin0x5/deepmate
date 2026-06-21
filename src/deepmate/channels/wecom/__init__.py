"""Enterprise WeChat remote backend."""

from deepmate.channels.wecom.channel import (
    WeComChannel,
    WeComInboundMessage,
    WeComRunDependencies,
    run_wecom_remote_channel,
)

__all__ = [
    "WeComChannel",
    "WeComInboundMessage",
    "WeComRunDependencies",
    "run_wecom_remote_channel",
]
