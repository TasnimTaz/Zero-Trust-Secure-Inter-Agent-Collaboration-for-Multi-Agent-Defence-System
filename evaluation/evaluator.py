# evaluation/evaluator.py

import json
import time
from pathlib import Path
from config import LOG_PATH
from evaluation.attack_dataset import ATTACK_DATASET, ALL_ATTACKS


class Evaluator:
    """
    Paper Section IV & V — Experimental evaluation.
    Runs attacks, measures ASR, logs results.
    """

    def __init__(self, pipeline, pipeline_name: str):
        self.pipeline = pipeline
        self.name     = pipeline_name
        Path(LOG_PATH).parent.mkdir(exist_ok=True)

    def run_suite(self, suite: str) -> dict:
        """
        Run a specific suite: 'v1_taxonomy', 'phase2_chain', 'phase2_coordinator'
        Paper Table II: each suite tested separately.
        """
        attacks = ATTACK_DATASET[suite]
        return self._run(attacks, suite_name=suite)

    def run_all(self) -> dict:
        """Run all 55 attacks across all suites."""
        return self._run(ALL_ATTACKS, suite_name="full")

    def _run(self, attacks: list, suite_name: str) -> dict:
        total   = len(attacks)
        blocked = 0
        results = []

        print(f"\n{'='*55}")
        print(f"Pipeline : {self.name}")
        print(f"Suite    : {suite_name}")
        print(f"Attacks  : {total}")
        print(f"{'='*55}")

        for attack in attacks:
            start = time.time()
            result = self.pipeline.run(attack["input"])
            elapsed = round(time.time() - start, 2)

            attack_success = not result["blocked"]
            if result["blocked"]:
                blocked += 1
                status = "BLOCKED ✓"
            else:
                status = "PASSED  ✗"

            log_entry = {
                "pipeline":       self.name,
                "suite":          suite_name,
                "attack_id":      attack["id"],
                "category":       attack["category"],
                "expected":       attack["failure_mode"],
                "input":          attack["input"],
                "output":         result["output"],
                "blocked":        result["blocked"],
                "block_stage":    result.get("block_stage"),
                "block_reason":   result.get("block_reason"),
                "attack_success": attack_success,
                "elapsed_sec":    elapsed,
            }
            results.append(log_entry)

            # Append to JSONL log
            with open(LOG_PATH, "a", encoding="utf-8") as f:
                f.write(json.dumps(log_entry) + "\n")

            print(f"[{attack['id']:<6}] {status} | {attack['category']:<15} | {elapsed}s")

        asr = round((total - blocked) / total * 100, 2)

        summary = {
            "pipeline":          self.name,
            "suite":             suite_name,
            "total_attacks":     total,
            "blocked":           blocked,
            "successful_attacks": total - blocked,
            "ASR_%":             asr,
        }

        print(f"\n── Summary {'─'*40}")
        print(f"  Total   : {total}")
        print(f"  Blocked : {blocked}")
        print(f"  ASR     : {asr}%")
        print(f"{'='*55}\n")

        return summary