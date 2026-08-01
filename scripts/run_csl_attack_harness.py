# scripts/run_csl_attack_harness.py
import base64
import json
import os
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from commsec.comms import Signer, verify_message, MessageReplayStore, TrustStore

KEY_A = bytes(range(32))
KEY_B = bytes(range(32, 64))
KEY_ATTACKER = b"attacker-controlled-key" * 2

RESULTS = []


def report(name: str, ok: bool, expect_block: bool = True) -> None:
    passed = ok
    if passed and expect_block:
        label = "BLOCKED"
    elif passed:
        label = "OK"
    else:
        label = "FAILED"
    icon = "✅" if passed else "❌"
    RESULTS.append({"attack": _CURRENT_ATTACK, "test": name, "passed": passed, "label": label})
    print(f"  {icon} {label:8} | {name}")


_CURRENT_ATTACK = "General"


def section(title: str) -> None:
    global _CURRENT_ATTACK
    _CURRENT_ATTACK = title
    print(f"\n{'=' * 66}\n{title}\n{'-' * 66}")


def _tamper_ciphertext(signed: dict) -> dict:
    out = dict(signed)
    raw = bytearray(base64.b64decode(signed["ciphertext"]))
    raw[0] ^= 0xFF
    out["ciphertext"] = base64.b64encode(bytes(raw)).decode()
    return out


def _tamper_field(signed: dict, field: str, value) -> dict:
    out = dict(signed)
    out[field] = value
    return out


def test_message_tampering() -> None:
    section("ATTACK 1: Message Tampering")
    signer = Signer(key=KEY_A, sender="pattern_expert")
    good = signer.sign({"agent": "pattern_expert", "sender": "pattern_expert", "v": "ok"})
    report("baseline sign->verify roundtrip works", verify_message(good, KEY_A), expect_block=False)

    tampered = _tamper_ciphertext(good)
    report("ciphertext byte-flip is rejected", not verify_message(tampered, KEY_A))


def test_agent_impersonation() -> None:
    section("ATTACK 2: Agent Impersonation")
    signer_a = Signer(key=KEY_A, sender="pattern_expert")
    judge_signer = Signer(key=KEY_B, sender="judge")

    forged = signer_a.sign({"agent": "judge", "sender": "judge", "v": "imposter"})
    report("forged 'judge' message signed with wrong key rejected", not verify_message(forged, KEY_B))

    honest = judge_signer.sign({"agent": "judge", "sender": "judge", "v": "real"})
    report("genuine judge message verifies", verify_message(honest, KEY_B), expect_block=False)

    claimed = signer_a.sign({"agent": "pattern_expert", "sender": "pattern_expert", "v": "ok"})
    replayed_as_judge = _tamper_field(claimed, "sender", "judge")
    report("sender field spoof (AAD binding) rejected", not verify_message(replayed_as_judge, KEY_B))


def test_replay_attacks() -> None:
    section("ATTACK 3: Replay Attacks")
    signer = Signer(key=KEY_A, sender="intent_expert")
    env = signer.sign({"agent": "intent_expert", "sender": "intent_expert", "v": "ok"})
    store = MessageReplayStore()

    first = verify_message(env, KEY_A, replay_store=store)
    report("first delivery accepted", first, expect_block=False)
    replay = verify_message(env, KEY_A, replay_store=store)
    report("same msg_id replayed is rejected", not replay)

    stale = dict(env)
    stale["timestamp"] = time.time() - 10_000
    stale["msg_id"] = base64.urlsafe_b64encode(os.urandom(16)).decode()
    report("stale/out-of-window timestamp rejected", not verify_message(stale, KEY_A, replay_store=store))


def test_integrity_violations() -> None:
    section("ATTACK 4: Communication Integrity Violations")
    signer = Signer(key=KEY_A, sender="category_expert")
    env = signer.sign({"agent": "category_expert", "sender": "category_expert", "v": "ok"})

    report("timestamp field tampered (AAD) rejected", not verify_message(_tamper_field(env, "timestamp", 12345), KEY_A))
    report("msg_id field tampered (AAD) rejected", not verify_message(_tamper_field(env, "msg_id", "forged"), KEY_A))

    bad_nonce = dict(env)
    bad_nonce["nonce"] = base64.b64encode(b"\x00" * 12).decode()
    report("nonce swapped rejected", not verify_message(bad_nonce, KEY_A))

    trunc = dict(env)
    trunc["ciphertext"] = base64.b64encode(base64.b64decode(env["ciphertext"])[:8]).decode()
    report("truncated ciphertext rejected", not verify_message(trunc, KEY_A))


def test_malicious_propagation_trust() -> None:
    section("ATTACK 5: Malicious Message Propagation (TrustStore)")
    store = TrustStore()
    agent = "judge"
    print(f"  baseline trust      : {store.score_of(agent):.2f}")

    ok = True
    for i in range(5):
        ok = store.record_failure(agent)
        print(f"  failure {i + 1:<3} trust={store.score_of(agent):.2f} is_trusted={store.is_trusted(agent)}")
    report("repeated failures decay below threshold", not store.is_trusted(agent))

    store.record_success(agent)
    store.record_success(agent)
    store.record_success(agent)
    print(f"  after recoveries   : trust={store.score_of(agent):.2f} is_trusted={store.is_trusted(agent)}")
    report("slow recovery does not jump back to trusted", not store.is_trusted(agent))


def test_pipeline_hop_gate(csl_enabled: bool = True) -> None:
    section("PIPELINE INTEGRATION: forged envelope at the CSL hop")
    from pipelines.macd_pipeline_v2_mp import MACDPipelineV2MP

    pipe = MACDPipelineV2MP()
    try:
        attacker = Signer(key=KEY_ATTACKER, sender="pattern_expert")
        forged = attacker.sign({"agent": "pattern_expert", "sender": "pattern_expert", "v": "injected"})

        pipe.csl_enabled = csl_enabled
        payload, ok, trace = pipe._hop(forged, "pattern_expert", "Input Analysis (Pattern Expert)")
        if csl_enabled:
            report("pipeline _hop rejects attacker-signed envelope", not ok)
        else:
            report("attacker-signed envelope accepted without CSL (BREACH)", not ok)
        print(f"  trace: verified={trace.get('verified')} csl={trace.get('csl')} reason={trace.get('reason') or 'auth/integrity failure'}")

        pipe.csl_enabled = not csl_enabled
        payload, ok, trace = pipe._hop(forged, "pattern_expert", "Input Analysis (Pattern Expert)")
        if csl_enabled:
            report("baseline: SAME forged envelope would be ACCEPTED with CSL off", ok, expect_block=False)
        else:
            report("contrast: SAME forged envelope REJECTED when CSL enabled", not ok, expect_block=False)
        print(f"  trace: verified={trace.get('verified')} csl={trace.get('csl')} reason={trace.get('reason')}")
        pipe.csl_enabled = True
    finally:
        pipe.shutdown()


def run_attack_tests(include_pipeline: bool = True, csl_enabled: bool = True) -> list:
    RESULTS.clear()
    test_message_tampering()
    test_agent_impersonation()
    test_replay_attacks()
    test_integrity_violations()
    test_malicious_propagation_trust()
    if include_pipeline:
        test_pipeline_hop_gate(csl_enabled=csl_enabled)
    return list(RESULTS)


def main() -> None:
    print("\n══════════════════════════════════════════════════════")
    print("  CSL Attack Harness — Zero-Trust Communication Layer")
    print("  Problem statement: tamper / impersonation / replay /")
    print("  malicious propagation / integrity violations")
    print("══════════════════════════════════════════════════════")

    results = run_attack_tests()
    blocked = sum(1 for r in results if r["label"] == "BLOCKED")
    ok = sum(1 for r in results if r["label"] == "OK")
    failed = len(results) - blocked - ok

    print(f"\n{'=' * 66}")
    print(f"SUMMARY: {blocked} blocked  |  {ok} baseline-OK  |  {failed} breached")
    print(f"{'=' * 66}")
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
