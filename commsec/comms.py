# commsec/comms.py
"""
Zero-Trust Communication Security Layer (CSL) for inter-agent messages.

Per-hop properties:
    - Authentication         -> per-agent AES-256-GCM key (only the holder
                                 of that agent's key -- ideally PQC-derived,
                                 see commsec/pqc.py -- can produce a valid tag)
    - Encryption              -> AES-256-GCM (confidentiality of the payload)
    - Integrity Verification  -> AES-GCM authentication tag
    - Anti-Replay Protection  -> unique msg_id + timestamp window (MessageReplayStore)
    - Trust Verification      -> TrustStore: a longitudinal trust score per
                                 agent, not just a single-message pass/fail.
                                 An agent with repeated verification
                                 failures is distrusted outright, even if a
                                 later message happens to verify correctly
                                 -- this is what stops "malicious message
                                 propagation" from a partially-compromised
                                 agent rather than just catching one bad
                                 message.
"""

import base64
import hashlib
import json
import os
import time

try:
    from cryptography.hazmat.primitives.ciphers.aead import AESGCM
except ImportError:  # pragma: no cover
    AESGCM = None


REPLAY_WINDOW_SECONDS = 300
CLOCK_SKEW_TOLERANCE_SECONDS = 30


class MessageReplayStore:
    """In-memory anti-replay tracker. One instance per pipeline/session."""

    def __init__(self):
        self._seen = set()

    def check_and_record(self, msg_id: str, timestamp: float) -> bool:
        now = time.time()
        if timestamp < now - REPLAY_WINDOW_SECONDS:
            return False
        if timestamp > now + CLOCK_SKEW_TOLERANCE_SECONDS:
            return False
        if msg_id in self._seen:
            return False
        self._seen.add(msg_id)
        return True


class TrustStore:
    """
    Per-agent longitudinal trust score (the "Trust Verification" component
    of the Zero-Trust proposal, as distinct from per-message crypto checks).

    - Starts every agent at full trust (1.0).
    - Each failed CSL verification decays the score.
    - Each successful verification slowly recovers it (capped at 1.0), so
      a single transient failure (e.g. clock skew) doesn't permanently
      blacklist an agent, but a *pattern* of failures does.
    - is_trusted() gates whether a message from this agent is even
      considered, independent of whether that specific message's
      signature verifies.
    """

    def __init__(self, threshold: float = 0.4, decay: float = 0.3, recovery: float = 0.05):
        self.scores: dict[str, float] = {}
        self.threshold = threshold
        self.decay = decay
        self.recovery = recovery

    @staticmethod
    def _fix(score: float) -> float:
        # Round to kill float accumulation drift, e.g. 1.0 - 0.3 - 0.3
        # otherwise yields 0.39999... which is <.4 after only two failures.
        return round(score, 10)

    def _get(self, agent: str) -> float:
        return self.scores.setdefault(agent, 1.0)

    def record_success(self, agent: str) -> float:
        score = self._fix(min(1.0, self._get(agent) + self.recovery))
        self.scores[agent] = score
        return score

    def record_failure(self, agent: str) -> float:
        score = self._fix(max(0.0, self._get(agent) - self.decay))
        self.scores[agent] = score
        return score

    def is_trusted(self, agent: str) -> bool:
        # Epsilon tolerance so an exact 0.4 equal to threshold isn't treated
        # as untrusted due to a rounding tail (drift is already prevented above,
        # this guards ties reached through other score sources).
        return self._get(agent) >= self.threshold - 1e-9

    def score_of(self, agent: str) -> float:
        return self._get(agent)


class Signer:
    """
    Encrypts + authenticates one agent's outgoing message with that
    agent's session key (ideally PQC-derived via commsec/pqc.py; falls
    back to the classical pre-shared AGENT_KEYS key otherwise).
    """

    def __init__(self, key: bytes, sender: str = None):
        if AESGCM is None:
            raise RuntimeError(
                "cryptography package not installed. Run: "
                "pip install cryptography --break-system-packages"
            )
        self._key = hashlib.sha256(key).digest()
        self._aead = AESGCM(self._key)
        self.sender = sender

    def sign(self, payload: dict, sender: str = None) -> dict:
        sender_name = sender or payload.get("agent") or payload.get("sender") or self.sender or "unknown"
        nonce = os.urandom(12)
        msg_id = base64.urlsafe_b64encode(os.urandom(16)).decode()
        timestamp = time.time()

        plaintext = json.dumps(payload).encode("utf-8")
        aad = json.dumps({"sender": sender_name, "msg_id": msg_id, "timestamp": timestamp}).encode("utf-8")
        ciphertext = self._aead.encrypt(nonce, plaintext, aad)

        return {
            "sender": sender_name,
            "msg_id": msg_id,
            "timestamp": timestamp,
            "nonce": base64.b64encode(nonce).decode(),
            "ciphertext": base64.b64encode(ciphertext).decode(),
        }


def verify_message(signed: dict, key: bytes, replay_store: MessageReplayStore = None) -> bool:
    if AESGCM is None:
        return False
    try:
        aead = AESGCM(hashlib.sha256(key).digest())
        nonce = base64.b64decode(signed["nonce"])
        ciphertext = base64.b64decode(signed["ciphertext"])
        aad = json.dumps(
            {
                "sender": signed["sender"],
                "msg_id": signed["msg_id"],
                "timestamp": signed["timestamp"],
            }
        ).encode("utf-8")

        plaintext = aead.decrypt(nonce, ciphertext, aad)
        payload = json.loads(plaintext.decode("utf-8"))

        if replay_store is not None:
            if not replay_store.check_and_record(signed["msg_id"], signed["timestamp"]):
                return False

        signed["payload"] = payload
        return True
    except Exception:
        return False