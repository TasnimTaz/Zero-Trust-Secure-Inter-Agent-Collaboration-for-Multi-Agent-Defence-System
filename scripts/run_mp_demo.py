# scripts/run_mp_demo.py
import os
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def _load_groq_api_key() -> None:
    if os.environ.get("GROQ_API_KEY"):
        return
    secrets_path = PROJECT_ROOT / ".streamlit" / "secrets.toml"
    if not secrets_path.exists():
        raise RuntimeError("Missing .streamlit/secrets.toml with GROQ_API_KEY")
    try:
        import tomllib

        with open(secrets_path, "rb") as fh:
            data = tomllib.load(fh)
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
    os.environ["GROQ_API_KEY"] = key


_load_groq_api_key()

from pipelines.macd_pipeline_v2_mp import MACDPipelineV2MP
from evaluation.attack_dataset import ATTACK_DATASET


def main() -> None:
    print("Starting MACD v2 (multiprocess, real IPC PQC handshake)...\n")
    pipe = MACDPipelineV2MP()

    tests = [
        ("safe", "What is the capital of France?"),
        ("E-01", ATTACK_DATASET["extreme_coordinator"][0]["input"]),
        ("E-04", ATTACK_DATASET["extreme_coordinator"][3]["input"]),
    ]

    for label, user_input in tests:
        print("\n" + "=" * 66)
        print(f"TEST: {label}")
        res = pipe.run(user_input)
        print(f"blocked   : {res['blocked']}")
        print(f"stage     : {res.get('block_stage')}")
        print(f"reason    : {(res.get('block_reason') or '')[:160]}")
        for hop in res["csl_trace"]:
            print(
                f"  hop {hop['stage']:<40} verified={hop['verified']} "
                f"trust={hop.get('trust_score')} pqc={hop.get('pqc')}"
            )
        print(f"output    : {(res['output'] or '')[:120]!r}")

    pipe.shutdown()
    print("\nAll multiprocess tests done.")


if __name__ == "__main__":
    main()
