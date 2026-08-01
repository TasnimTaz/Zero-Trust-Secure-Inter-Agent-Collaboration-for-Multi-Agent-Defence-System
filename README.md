# Multi-Agent LLM Defence System

Groq-backed prompt-injection defence system. Five specialised LLM agents
collaborate to detect prompt-injection attacks, and a **Zero-Trust
Communication Security Layer (CSL)** protects the messages flowing between them.

## What this system does

1. **Multi-agent detection.** A `Pattern Expert`, `Intent Expert` and
   `Category Expert` independently analyse the user prompt; a `Judge`
   synthesises their verdicts into one `is_safe` decision; a `Domain LLM`
   (the trusted target model) produces the answer; and a `Guard` re-validates
   the final response before it is returned.

2. **Zero-Trust inter-agent communication (CSL).** Every message an agent
   sends to the next agent is signed, encrypted and verified at the receiving
   hop. A hop accepts a message only if it passes **all five** checks:
   Authentication + Encryption (AES-256-GCM) + Integrity + Anti-Replay +
   Trust.

3. **Post-quantum key agreement.** Session keys are derived with an
   **ML-KEM-768** handshake over a real IPC pipe when `liboqs-python` is
   available, so the private key never leaves the worker process.

4. **Longitudinal trust scoring.** A `TrustStore` per agent decays the trust
   score on every failed verification and recovers it slowly on success.
   A hop is refused once an agent's trust drops below threshold.

5. **Provable communication-layer robustness.** A CSL attack harness runs 16
   checks against the five communication-level attack classes from the
   problem statement plus a live pipeline hop-injection, and the UI supports
   a **CSL ON vs OFF ablation** that quantifies what the CSL actually
   contributes.

## Current Architecture

- MACD sequential defence pipeline, single-process `macd_pipeline_v2.py`
- **Distributed multiprocess deployment** `macd_pipeline_v2_mp.py`
  (default in the UI): each agent runs in its own OS process over real IPC
  pipes; session keys are derived via an ML-KEM-768 PQC handshake
- Zero-Trust CSL with AES-256-GCM per-hop verification
- `TrustStore`-based longitudinal agent trust scoring
- hard-fail enforcement for duplicate agent keys
- Streamlit UI for interactive testing, evaluation, harness and ablation

### Protected hops

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

> **Honest scope:** the **Domain LLM is NOT wrapped in the CSL** — it is the
> trusted target model and its raw query/response is intentionally left
> unprotected. The Judge and Guard hops **are** CSL-protected (signed +
> verified + trust-checked), so their verdicts cannot be silently tampered
> with or replayed. All five protected hops (Pattern, Intent, Category, Judge,
> Guard) get PQC-derived session keys when `liboqs-python` is available.

## How CSL protects against the five communication-level attacks

| Attack class | What an attacker does | CSL response |
|---|---|---|
| Message tampering | Flips bytes in an agent's ciphertext | AES-256-GCM integrity check fails -> hop refused |
| Agent impersonation | Signs a message with an attacker key claiming to be `judge` | Message key != hop's session key -> auth fails |
| Replay attacks | Re-sends a previously captured message | `MessageReplayStore` rejects duplicate `msg_id` / stale timestamp |
| Integrity violations | Swaps nonce, truncates ciphertext, tampers AAD fields (`sender`, `msg_id`, `timestamp`) | GCM authentication fails -> hop refused |
| Malicious propagation | A compromised agent keeps sending bad messages | `TrustStore` decays its trust below threshold -> hop refused |

## Why "CSL OFF" lets attacks succeed (the ablation story)

The CSL is a **communication-layer** control. It does **not** change whether an
LLM agent *detects* a malicious prompt — it guarantees that the message the
next agent reads is the message the previous agent actually sent.

- **Input-level ASR is the same with CSL ON or OFF.** A prompt-injection
  attack sent straight to the pipeline is detected by the same LLM agents
  either way. Measuring ASR this way tells you about detection, not about CSL.
- **Under an active wire forgery the difference is decisive.** The ablation
  mode simulates an attacker who intercepts the Judge's message on the pipe
  and replaces it with a forged `{"is_safe": True}` verdict:
  - **CSL ON:** the forged envelope fails AES-256-GCM authentication/integrity
    at the `Judge Synthesis` hop and is **blocked** (`comms_security_layer`).
  - **CSL OFF (plaintext baseline):** the forged payload is accepted verbatim,
    the pipeline believes the attack is safe, and the attack **bypasses** the
    defence.
  - Measured result (5-attack sample): **ASR drops from 80% (CSL OFF) to 0%
    (CSL ON)**. The remaining 20% on the CSL-OFF side is the Guard catching the
    forged path's output — a separate, independent layer (defence-in-depth).
- The same contrast is visible in the harness's CSL ON/OFF toggle: with CSL ON
  all 16 checks hold (0 breaches); with CSL OFF the identical attacker-signed
  envelope is **accepted** at the pipeline hop and counted as a breach.

So CSL exists not to make detection "smarter" but to make the detection
**trustworthy** — a tampered or forged verdict can never reach the decision
point.

## Evaluation

The dataset (`evaluation/attack_dataset.py`) contains **300 hand-crafted
attack prompts** across three suites:

- `moderate_intermediate` (100)
- `hard_advanced` (100)
- `extreme_coordinator` (100)

The Streamlit UI (`📊 Evaluate — ...`) runs any suite or a custom range and
reports **ASR** (Attack Success Rate), blocked/stage breakdown, category
breakdown and a per-hop Zero-Trust security trace. Results can be downloaded
as `.jsonl`.

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

If those system packages are missing, the app falls back to classical
pre-shared keys and logs a warning instead of silently claiming PQC protection.

## Run

Start the Streamlit app (set `GROQ_API_KEY` first, e.g. in
`.streamlit/secrets.toml`):

```bash
streamlit run app.py
```

UI modes:
- `💬 Interactive` — chat with the distributed MACD pipeline, per-hop CSL trace
- `📊 Evaluate — ...` — run attack suites (Moderate / Hard / Extreme / Full 300)
- `🔬 CSL Attack Harness` — 16 checks, with a **CSL ON/OFF** toggle
- `🧪 CSL Ablation (ON vs OFF)` — single-prompt or batch ASR comparison under
  an active forged-judge-verdict attack

Standalone scripts:

```bash
./.venv/bin/python scripts/verify_security_controls.py        # TrustStore + duplicate-key hard-fail
./.venv/bin/python scripts/run_mp_demo.py                     # distributed pipeline demo
./.venv/bin/python scripts/run_csl_attack_harness.py          # 16-check CSL harness
```

## Project Structure

```text
.
├── app.py
├── config.py
├── commsec/
│   ├── comms.py              # Signer, verify_message, MessageReplayStore, TrustStore
│   └── pqc.py                # PQCIdentity, orchestrator_encapsulate, ML-KEM-768
├── evaluation/
│   ├── attack_dataset.py     # 300 attacks, 3 suites
│   └── evaluator.py
├── agents/
│   ├── groq_utils.py
│   ├── macd_category_expert.py
│   ├── macd_intent_expert.py
│   ├── macd_judge.py
│   └── macd_pattern_expert.py
├── pipelines/
│   ├── macd_pipeline_v2.py        # single-process
│   └── macd_pipeline_v2_mp.py     # multiprocess, real IPC + PQC handshake (default)
├── scripts/
│   ├── run_csl_attack_harness.py
│   ├── run_mp_demo.py
│   └── verify_security_controls.py
└── requirements.txt
```

## Notes

- If `liboqs-python` or the liboqs build dependencies are missing, the UI will
  show `classical` instead of `PQC`.
- PQC (ML-KEM-768) session-key derivation currently covers five hops: Pattern,
  Intent, Category, Judge, and Guard.
- Duplicate agent keys are rejected unless `ALLOW_SHARED_KEYS=1` is set
  explicitly for a controlled dev/test run.
