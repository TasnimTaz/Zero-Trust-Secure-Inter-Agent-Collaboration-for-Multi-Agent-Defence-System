# Multi-Agent LLM Defense System

Groq-backed prompt-injection defense system focused on the current MACD pipeline.

## Current Architecture

- MACD sequential defense pipeline
- Zero-Trust CSL with AES-256-GCM per-hop verification
- PQC hybrid key exchange via ML-KEM-768 when `liboqs-python` is available
- `TrustStore`-based longitudinal agent trust scoring
- hard-fail enforcement for duplicate agent keys
- Streamlit UI for interactive testing and evaluation

### Communication Security Layer (CSL) Scope

The Zero-Trust CSL protects **five hops** of the pipeline:

```text
User Prompt
  -> Pattern Expert -> CSL -> Intent Expert -> CSL -> Category Expert -> CSL
  -> Judge -> CSL -> Execution Validation (Domain LLM + Guard -> CSL)
  -> Final Decision
```

Each CSL hop provides: Authentication + Encryption (AES-256-GCM, PQC-derived
session keys when available) + Integrity Verification + Anti-Replay Protection
+ Trust Verification.

> **Important (honest scope):** The **Domain LLM is NOT wrapped in the CSL** —
> it is the trusted target model, and its raw query/response is intentionally
> left unprotected. The Judge and Guard hops ARE CSL-protected (signed +
> verified + trust-checked), so their verdicts cannot be silently tampered
> with or replayed. All five protected hops (Pattern, Intent, Category, Judge,
> Guard) get PQC-derived session keys when `liboqs-python` is available.

## Setup

Install the Python dependencies:

```bash
./.venv/bin/python -m pip install -r requirements.txt
```

For PQC hybrid support, install the extra build dependencies too:

```bash
./.venv/bin/python -m pip install liboqs-python cmake ninja --break-system-packages
```

On Debian/Ubuntu, install the required system packages:

```bash
sudo apt update
sudo apt install -y build-essential libssl-dev
```

If those system packages are missing, the app falls back to classical pre-shared keys and logs a warning instead of silently claiming PQC protection.

## Run

Start the Streamlit app:

```bash
streamlit run app.py
```

Run the local security verification script:

```bash
./.venv/bin/python scripts/verify_security_controls.py
```

That script demonstrates:
- TrustStore decay so a trust score drops below `1.00`
- duplicate `AGENT_KEY_*` hard-fail behavior during `config.py` import

## Adaptive Evaluation

Run strict AdaptiveAttackAgent evaluation against MACD:

```bash
./.venv/bin/python evaluation/run_exact_adaptive_eval.py \
  --repo-path ../AdaptiveAttackAgent_tmp \
  --model <path_or_name_used_by_AdaptiveAttackAgent> \
  --data-setting base_subset \
  --per-strategy 3
```

Add `--skip-generation` if attack inputs were already generated.

The script writes JSON and JSONL results into `logs/`.

## Project Structure

```text
.
├── app.py
├── config.py
├── commsec/
│   ├── comms.py
│   └── pqc.py
├── evaluation/
│   ├── adaptive_attackagent_bridge.py
│   ├── attack_dataset.py
│   ├── evaluator.py
│   ├── generate_adaptive_attacks.py
│   └── run_exact_adaptive_eval.py
├── agents/
│   ├── groq_utils.py
│   ├── macd_category_expert.py
│   ├── macd_intent_expert.py
│   ├── macd_judge.py
│   └── macd_pattern_expert.py
├── pipelines/
│   └── macd_pipeline_v2.py
├── scripts/
│   ├── run_exact_adaptive_eval.ps1
│   └── verify_security_controls.py
└── requirements.txt
```

## Notes

- If `liboqs-python` or the liboqs build dependencies are missing, the UI will show `classical` instead of `PQC`.
- PQC (ML-KEM-768) session-key derivation currently covers five hops: Pattern, Intent, Category, Judge, and Guard.
- Duplicate agent keys are rejected unless `ALLOW_SHARED_KEYS=1` is set explicitly for a controlled dev/test run.
