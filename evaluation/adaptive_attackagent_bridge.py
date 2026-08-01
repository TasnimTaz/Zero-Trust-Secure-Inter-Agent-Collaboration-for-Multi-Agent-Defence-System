import glob
import json
import os
from pathlib import Path


STRATEGY_TO_DEFENSE = {
    "GCG": "InstructionalPrevention",
    "MGCG_ST": "LLMDetector",
    "MGCG_DT": "FinetunedDetector",
    "TGCG": "Paraphrasing",
}

STRATEGY_TO_CATEGORY = {
    "GCG": "adaptive_gcg",
    "MGCG_ST": "adaptive_mgcg_st",
    "MGCG_DT": "adaptive_mgcg_dt",
    "TGCG": "adaptive_tgcg",
}


def _read_json_file(path: str) -> list:
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        if isinstance(data, list):
            return data
    except Exception:
        return []
    return []


def _find_latest_generated_files(repo_root: str, defense: str, data_setting: str) -> list:
    base_dir = Path(repo_root) / "InjecAgent" / "adaptive_attack_results"
    if not base_dir.exists():
        return []

    pattern = str(base_dir / f"*_{defense}_*_{data_setting}*_*.json")
    files = glob.glob(pattern)
    if not files:
        return []

    files.sort(key=lambda p: os.path.getmtime(p), reverse=True)

    grouped = {}
    for fp in files:
        name = Path(fp).name
        key = name.replace("_dh_data.json", "").replace("_ds_data.json", "")
        grouped.setdefault(key, []).append(fp)

    latest_key = next(iter(grouped.keys()))
    chosen = grouped.get(latest_key, [])
    return chosen


def _load_strategy_inputs(repo_root: str, strategy: str, data_setting: str, prefer_generated: bool) -> list:
    defense = STRATEGY_TO_DEFENSE[strategy]
    inputs = []

    if prefer_generated:
        generated_files = _find_latest_generated_files(repo_root, defense, data_setting)
        for fp in generated_files:
            for case in _read_json_file(fp):
                text = case.get("Attacker Input") or case.get("Attacker Instruction")
                if text:
                    inputs.append(text)

    if inputs:
        return inputs

    data_dir = Path(repo_root) / "InjecAgent" / "data"
    raw_files = [
        data_dir / f"test_cases_dh_{data_setting}.json",
        data_dir / f"test_cases_ds_{data_setting}.json",
    ]
    for fp in raw_files:
        for case in _read_json_file(str(fp)):
            text = case.get("Attacker Instruction")
            if text:
                inputs.append(text)
    return inputs


def build_adaptive_attackagent_suite(
    repo_root: str,
    data_setting: str = "base_subset",
    per_strategy: int = 3,
    prefer_generated: bool = True,
    require_generated: bool = False,
) -> list:
    """
    Build an exact AdaptiveAttackAgent-derived suite.
    - If generated adaptive results exist, uses Attacker Input from those files.
    - Otherwise falls back to Attacker Instruction from InjecAgent test cases.
    - If require_generated=True, strategies without generated files are skipped.
    """
    attacks = []
    for strategy in ["GCG", "MGCG_ST", "MGCG_DT", "TGCG"]:
        category = STRATEGY_TO_CATEGORY[strategy]
        generated_files = _find_latest_generated_files(repo_root, STRATEGY_TO_DEFENSE[strategy], data_setting)
        has_generated = len(generated_files) > 0
        if require_generated and not has_generated:
            continue

        inputs = _load_strategy_inputs(repo_root, strategy, data_setting, prefer_generated)
        if not inputs:
            continue

        for i, text in enumerate(inputs[:per_strategy], start=1):
            attacks.append(
                {
                    "id": f"AAG-{strategy}-{i:02d}",
                    "category": category,
                    "input": text,
                    "failure_mode": "indirect_prompt_injection",
                    "source": "AdaptiveAttackAgent",
                    "strategy": strategy,
                    "generated": has_generated,
                }
            )
    return attacks
