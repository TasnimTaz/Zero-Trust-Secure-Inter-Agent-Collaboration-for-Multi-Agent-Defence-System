import argparse
import os
import subprocess
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from evaluation.adaptive_attackagent_bridge import STRATEGY_TO_DEFENSE

# ---------------------------------------------------------------------------
# TEMPORARY: only run GCG for now (fast sanity check). To run all 4 strategies
# again later, change ONLY_STRATEGIES back to None.
# ---------------------------------------------------------------------------
ONLY_STRATEGIES = {"GCG": "InstructionalPrevention"}


def _resolve_python_executable() -> str:
    """Prefer the active venv's python.exe (via VIRTUAL_ENV) over sys.executable,
    since on some Windows venv setups sys.executable incorrectly reports the base install."""
    venv_dir = os.environ.get("VIRTUAL_ENV")
    if venv_dir:
        candidate = Path(venv_dir) / ("Scripts" if os.name == "nt" else "bin") / ("python.exe" if os.name == "nt" else "python")
        if candidate.exists():
            return str(candidate)
    return sys.executable


def run_cmd(cmd: list, cwd: str) -> None:
    proc = subprocess.run(cmd, cwd=cwd, check=False)
    if proc.returncode != 0:
        raise RuntimeError(f"Command failed ({proc.returncode}): {' '.join(cmd)}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-path", required=True, help="Path to cloned AdaptiveAttackAgent repo")
    parser.add_argument("--model", required=True, help="Model path/name expected by AdaptiveAttackAgent run.py")
    parser.add_argument("--data-setting", default="base_subset", help="Data setting (base, base_subset, enhanced)")
    parser.add_argument("--limit", type=int, default=None, help="Limit number of test cases used for GCG training")
    args = parser.parse_args()

    repo_path = str(Path(args.repo_path).resolve())
    run_py = Path(repo_path) / "run.py"
    if not run_py.exists():
        raise FileNotFoundError(f"run.py not found under {repo_path}")

    python_exe = _resolve_python_executable()
    print(f"[AdaptiveAttackAgent] Using Python executable: {python_exe}")

    strategies_to_run = ONLY_STRATEGIES if ONLY_STRATEGIES else STRATEGY_TO_DEFENSE

    for strategy, defense in strategies_to_run.items():
        print(f"[AdaptiveAttackAgent] Running strategy={strategy} via defense={defense}")
        cmd = [
            python_exe,
            "run.py",
            "--model",
            args.model,
            "--defense",
            defense,
            "--data_setting",
            args.data_setting,
        ]
        if args.limit:
            cmd += ["--limit", str(args.limit)]
        run_cmd(cmd, cwd=repo_path)

    print("Adaptive attack generation finished.")
    print("Generated files are in InjecAgent/adaptive_attack_results under the AdaptiveAttackAgent repo.")


if __name__ == "__main__":
    main()