from __future__ import annotations


class PassthroughRestorer:
    def restore(self, pixels: bytes) -> bytes:
        return pixels
