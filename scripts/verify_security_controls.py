from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def _load_groq_api_key() -> str:
    secrets_path = PROJECT_ROOT / ".streamlit" / "secrets.toml"
    if not secrets_path.exists():
        raise RuntimeError("Missing .streamlit/secrets.toml")

    try:
        import tomllib

        with open(secrets_path, "rb") as file_handle:
            data = tomllib.load(file_handle)
        key = data.get("GROQ_API_KEY")
    except Exception:
        key = None
        for line in secrets_path.read_text(encoding="utf-8").splitlines():
            stripped = line.strip()
            if stripped.startswith("GROQ_API_KEY") and "=" in stripped:
                _, raw_value = stripped.split("=", 1)
                key = raw_value.strip().strip('"').strip("'")
                break

    if not key:
        raise RuntimeError("GROQ_API_KEY not found in .streamlit/secrets.toml")
    return key


def trust_decay_demo() -> None:
    from commsec.comms import TrustStore

    store = TrustStore()
    agent = "pattern_expert"
    print("Trust decay demo:")
    print(f"  baseline: {store.score_of(agent):.2f}")
    for idx in range(1, 5):
        score = store.record_failure(agent)
        print(f"  failure {idx}: {score:.2f}")


def duplicate_key_hard_fail_demo() -> None:
    env = os.environ.copy()
    env["GROQ_API_KEY"] = _load_groq_api_key()
    env["AGENT_KEY_PATTERN"] = "0123456789abcdef0123456789abcdef"
    env["AGENT_KEY_INTENT"] = "0123456789abcdef0123456789abcdef"
    env["AGENT_KEY_CATEGORY"] = "fedcba9876543210fedcba9876543210"
    env["AGENT_KEY_JUDGE"] = "00112233445566778899aabbccddeeff"
    env.pop("ALLOW_SHARED_KEYS", None)

    code = "import config"
    proc = subprocess.run(
        [sys.executable, "-c", code],
        cwd=str(PROJECT_ROOT),
        env=env,
        capture_output=True,
        text=True,
    )

    print("Duplicate-key hard-fail demo:")
    print(f"  return code: {proc.returncode}")
    if proc.returncode == 0:
        raise RuntimeError("Expected config import to fail on duplicate AGENT_KEY_* values")
    if "same value" not in (proc.stderr + proc.stdout).lower():
        raise RuntimeError("config.py failed for a different reason than duplicate AGENT_KEY_* values")
    print("  config.py refused to start as expected")


def main() -> None:
    trust_decay_demo()
    duplicate_key_hard_fail_demo()


if __name__ == "__main__":
    main()