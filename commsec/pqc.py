# commsec/pqc.py
"""
Post-Quantum Key Encapsulation (KEM) layer for the Zero-Trust CSL.

Uses ML-KEM-768 (FIPS 203, formerly CRYSTALS-Kyber) via liboqs-python for
KEY AGREEMENT only. The resulting shared secret becomes the AES-256-GCM
key used for actual message encryption -- this is the standard "hybrid
PQC" pattern (PQC protects the key exchange; AES-256 itself already has
~128-bit security against Grover's algorithm, so it doesn't need to be
replaced, only the classical key-agreement step does).

Install: pip install liboqs-python --break-system-packages
(needs the liboqs C library; see https://github.com/open-quantum-safe/liboqs-python)

If liboqs-python isn't installed, PQC_AVAILABLE is False and callers must
fall back to the classical pre-shared AGENT_KEYS (config.py) -- this
fallback is logged loudly, never silent, so a report can't accidentally
overclaim "PQC-secured" when it wasn't actually active for a given run.
"""

import hashlib

try:
    import oqs

    PQC_AVAILABLE = True
except ImportError:
    PQC_AVAILABLE = False


KEM_ALG = "ML-KEM-768"


class PQCIdentity:
    """
    One agent's long-term ML-KEM keypair. The agent keeps the private key
    (inside this object) and exposes only `public_key`. The orchestrator
    uses that public key to encapsulate a fresh session secret; the agent
    decapsulates with its private key to arrive at the same secret --
    neither side ever transmits the secret itself, only KEM ciphertext.
    """

    def __init__(self, agent_name: str):
        self.agent_name = agent_name
        if not PQC_AVAILABLE:
            self._kem = None
            self.public_key = None
            return
        self._kem = oqs.KeyEncapsulation(KEM_ALG)
        self.public_key = self._kem.generate_keypair()

    def decapsulate(self, kem_ciphertext: bytes) -> bytes:
        """Agent-side: derive the AES-256 session key from the KEM ciphertext."""
        if not PQC_AVAILABLE:
            raise RuntimeError("liboqs-python not installed -- PQC unavailable")
        shared_secret = self._kem.decap_secret(kem_ciphertext)
        return hashlib.sha256(shared_secret).digest()


def orchestrator_encapsulate(peer_public_key: bytes) -> tuple[bytes, bytes]:
    """
    Orchestrator-side: encapsulate against an agent's ML-KEM public key.
    Returns (kem_ciphertext_to_send_to_agent, session_key_32_bytes).
    """
    if not PQC_AVAILABLE:
        raise RuntimeError("liboqs-python not installed -- PQC unavailable")
    with oqs.KeyEncapsulation(KEM_ALG) as kem:
        kem_ciphertext, shared_secret = kem.encap_secret(peer_public_key)
    return kem_ciphertext, hashlib.sha256(shared_secret).digest()