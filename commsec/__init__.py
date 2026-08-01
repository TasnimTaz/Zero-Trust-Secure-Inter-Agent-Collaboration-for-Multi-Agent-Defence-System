"""Communication Security utilities for inter-agent messaging (prototype).

Provides simple HMAC-based signing/verification and an anti-replay store.
Replace with PQC primitives and proper KMS in production.
"""
from .comms import Signer, verify_message, MessageReplayStore

__all__ = ["Signer", "verify_message", "MessageReplayStore"]
