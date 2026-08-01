import json
import warnings

from agents.macd_pattern_expert import PatternExpertAgent
from agents.macd_intent_expert import IntentExpertAgent
from agents.macd_category_expert import CategoryExpertAgent
from agents.macd_judge import MACDJudgeAgent
from config import (
    GROQ_CLIENT,
    TARGET_MODEL,
    DEFENSE_MODEL,
    SYSTEM_PROMPT,
    GUARD_SYSTEM_PROMPT,
    SAFE_REFUSAL_MSG,
    MACD_V2_PATTERN_MODEL,
    MACD_V2_INTENT_MODEL,
    MACD_V2_CATEGORY_MODEL,
    MACD_V2_JUDGE_MODEL,
    AGENT_KEYS,
)
from agents.groq_utils import safe_completion
from commsec.comms import verify_message, MessageReplayStore, TrustStore
from commsec.pqc import PQCIdentity, orchestrator_encapsulate, PQC_AVAILABLE


class DomainLLM:
    """Current Groq-backed domain model used for legitimate user queries."""

    def __init__(self):
        self.client = GROQ_CLIENT
        self.model = TARGET_MODEL
        self.system_prompt = SYSTEM_PROMPT
        print(f"[DomainLLM] Ready via Groq — model: {self.model}")

    def generate(self, user_input: str) -> str:
        try:
            completion = safe_completion(
                self.client,
                model=self.model,
                messages=[
                    {"role": "system", "content": self.system_prompt},
                    {"role": "user", "content": user_input},
                ],
                temperature=0.7,
                max_completion_tokens=1024,
                stream=False,
            )
            return completion.choices[0].message.content
        except Exception as e:
            return f"DomainLLM Error: {str(e)}"


class GuardAgent:
    """Current Groq-backed output validator used after the domain model."""

    def __init__(self):
        self.client = GROQ_CLIENT
        self.model = DEFENSE_MODEL
        self.system_prompt = GUARD_SYSTEM_PROMPT
        print(f"[GuardAgent] Ready via Groq using {self.model}")

    def get_refusal(self) -> str:
        return SAFE_REFUSAL_MSG

    def validate(self, response_text: str) -> tuple[bool, str, str]:
        try:
            completion = safe_completion(
                self.client,
                model=self.model,
                messages=[
                    {"role": "system", "content": self.system_prompt},
                    {"role": "user", "content": f"Validate this AI response:\n\n{response_text}"},
                ],
                temperature=0.1,
                max_completion_tokens=1024,
                stream=False,
            )
            raw = completion.choices[0].message.content.strip()

            if "</think>" in raw:
                raw = raw.split("</think>")[-1].strip()

            if raw.startswith("```"):
                raw = raw.split("```")[1]
                if raw.startswith("json"):
                    raw = raw[4:]
            raw = raw.strip()

            result = json.loads(raw)
            is_safe = bool(result.get("is_safe", False))
            reason = result.get("reason", "")
            cleaned = result.get("cleaned_response", "")

            return is_safe, cleaned, reason
        except (json.JSONDecodeError, KeyError, Exception) as e:
            return False, "", f"Guard error/fail-safe: {str(e)}"


class MACDPipelineV2:
    """
    MACD-v2 — Zero-Trust Sequential Pipeline with PQC-derived session keys
    and longitudinal trust scoring.

    User Prompt
      -> Input Analysis Agent (Pattern Expert)
      -> Communication Security Layer (PQC-derived AES-256-GCM: auth + encrypt + integrity + anti-replay + trust)
      -> Intent Analysis Agent (Intent Expert)
      -> Communication Security Layer
      -> Prompt Injection Detection Agent (Category Expert)
      -> Communication Security Layer
      -> Judge Agent
      -> Execution Validation (Domain LLM + Guard)
      -> Final Decision

    At init, the orchestrator performs an ML-KEM-768 handshake with each
    expert agent (if liboqs-python is installed) to derive a fresh AES-256
    session key -- replacing the static pre-shared AGENT_KEYS for that run.
    If PQC isn't available, it falls back to the classical AGENT_KEYS key
    and logs a warning so this is never silently overclaimed.

    Each hop's message is also checked against TrustStore before its
    signature is even verified: an agent with too many past failures is
    distrusted outright, not just re-checked message-by-message.
    """

    def __init__(self):
        self.replay_store = MessageReplayStore()
        self.trust_store = TrustStore()

        self.pqc_active = PQC_AVAILABLE
        if not PQC_AVAILABLE:
            warnings.warn(
                "liboqs-python not installed -- MACD-v2 CSL is running on classical pre-shared AES-256-GCM keys only, NOT PQC-derived session keys. Run: pip install liboqs-python --break-system-packages",
                stacklevel=2,
            )

        session_keys = {}
        for name in ("pattern_expert", "intent_expert", "category_expert"):
            if PQC_AVAILABLE:
                identity = PQCIdentity(name)
                kem_ciphertext, orch_side_key = orchestrator_encapsulate(identity.public_key)
                agent_side_key = identity.decapsulate(kem_ciphertext)
                assert agent_side_key == orch_side_key, "PQC KEM handshake mismatch"
                session_keys[name] = agent_side_key
            else:
                session_keys[name] = AGENT_KEYS.get(name)

        self.pattern_expert = PatternExpertAgent(model=MACD_V2_PATTERN_MODEL, signing_key=session_keys["pattern_expert"])
        self.intent_expert = IntentExpertAgent(model=MACD_V2_INTENT_MODEL, signing_key=session_keys["intent_expert"])
        self.category_expert = CategoryExpertAgent(model=MACD_V2_CATEGORY_MODEL, signing_key=session_keys["category_expert"])
        self.judge = MACDJudgeAgent(model=MACD_V2_JUDGE_MODEL)
        self.llm = DomainLLM()
        self.guard = GuardAgent()
        self._session_keys = session_keys

    def _hop(self, signed_msg: dict, key_name: str, stage_name: str):
        """
        One CSL hop: trust-gate, then verify+decrypt a signed message.

        Returns (payload, ok, trace_entry).
        `trace_entry` is only for UI/paper visualization of the security
        layer -- it is never used to make the actual block/pass decision,
        `ok` is the sole source of truth for that.
        """
        trust_score = round(self.trust_store.score_of(key_name), 2)
        if not self.trust_store.is_trusted(key_name):
            trace_entry = {
                "stage": stage_name,
                "sender": signed_msg.get("sender", key_name),
                "msg_id": signed_msg.get("msg_id", ""),
                "timestamp": signed_msg.get("timestamp"),
                "verified": False,
                "trust_score": trust_score,
                "pqc": self.pqc_active,
                "reason": "trust_score_below_threshold",
            }
            return None, False, trace_entry

        key = self._session_keys.get(key_name) or AGENT_KEYS.get(key_name)
        ok = verify_message(signed_msg, key, replay_store=self.replay_store)
        if ok:
            self.trust_store.record_success(key_name)
        else:
            self.trust_store.record_failure(key_name)
        trace_entry = {
            "stage": stage_name,
            "sender": signed_msg.get("sender", "unknown"),
            "msg_id": signed_msg.get("msg_id", ""),
            "timestamp": signed_msg.get("timestamp"),
            "verified": ok,
            "trust_score": round(self.trust_store.score_of(key_name), 2),
            "pqc": self.pqc_active,
        }
        if not ok:
            return None, False, trace_entry
        return signed_msg["payload"], True, trace_entry

    def _buffer(self, response: str, stage: int) -> str:
        """Coordinator/MACD-v1 এর সাথে সামঞ্জস্যপূর্ণ 3-bullet rule buffer।"""
        lines = response.split('\n')
        bullet_count = 0
        result = []
        for line in lines:
            stripped = line.strip()
            if stripped.startswith(('-', '*', '•', '·', '1.', '2.', '3.')):
                bullet_count += 1
                if bullet_count > 3:
                    continue
            result.append(line)
        return '\n'.join(result)

    def _csl_block(self, stage_name: str, reason: str = None):
        return {
            "pipeline": "macd_v2",
            "blocked": True,
            "block_stage": "comms_security_layer",
            "block_reason": reason or f"Message failed authentication/encryption/integrity/anti-replay/trust check at hop: {stage_name}",
            "category": "comms_violation",
            "confidence": 1.0,
            "raw_response": None,
        }

    def run(self, user_input: str) -> dict:
        agent_verdicts = {}
        csl_trace = []

        # Hop 1: Input Analysis Agent (Pattern Expert) -> CSL
        signed_pattern = self.pattern_expert.analyze(user_input)
        pattern_payload, ok1, t1 = self._hop(signed_pattern, "pattern_expert", "Input Analysis (Pattern Expert)")
        csl_trace.append(t1)
        if not ok1:
            out = self._csl_block("pattern_expert", t1.get("reason"))
            out.update({
                "input": user_input,
                "output": self.guard.get_refusal(),
                "agent_verdicts": agent_verdicts,
                "csl_trace": csl_trace,
            })
            return out
        agent_verdicts["pattern_expert"] = pattern_payload

        # Hop 2: Intent Analysis Agent, informed by verified Hop-1 verdict -> CSL
        signed_intent = self.intent_expert.analyze(user_input, context=pattern_payload)
        intent_payload, ok2, t2 = self._hop(signed_intent, "intent_expert", "Intent Analysis")
        csl_trace.append(t2)
        if not ok2:
            out = self._csl_block("intent_expert", t2.get("reason"))
            out.update({
                "input": user_input,
                "output": self.guard.get_refusal(),
                "agent_verdicts": agent_verdicts,
                "csl_trace": csl_trace,
            })
            return out
        agent_verdicts["intent_expert"] = intent_payload

        # Hop 3: Prompt Injection Detection Agent, informed by verified Hop-2 verdict -> CSL
        signed_category = self.category_expert.analyze(user_input, context=intent_payload)
        category_payload, ok3, t3 = self._hop(signed_category, "category_expert", "Prompt Injection Detection")
        csl_trace.append(t3)
        if not ok3:
            out = self._csl_block("category_expert", t3.get("reason"))
            out.update({
                "input": user_input,
                "output": self.guard.get_refusal(),
                "agent_verdicts": agent_verdicts,
                "csl_trace": csl_trace,
            })
            return out
        agent_verdicts["category_expert"] = category_payload

        # Judge Agent — synthesizes the three verified verdicts
        is_safe, category, reason, confidence = self.judge.synthesize(
            user_input, pattern_payload, intent_payload, category_payload
        )

        if not is_safe:
            return {
                "pipeline": "macd_v2",
                "input": user_input,
                "output": self.guard.get_refusal(),
                "blocked": True,
                "block_stage": "macd_judge",
                "block_reason": reason,
                "category": category,
                "confidence": confidence,
                "raw_response": None,
                "agent_verdicts": agent_verdicts,
                "csl_trace": csl_trace,
            }

        # Execution Validation: Domain LLM + Guard
        raw_response = self.llm.generate(user_input)
        is_safe, cleaned, guard_reason = self.guard.validate(raw_response)

        if not is_safe:
            return {
                "pipeline": "macd_v2",
                "input": user_input,
                "output": self.guard.get_refusal(),
                "blocked": True,
                "block_stage": "guard",
                "block_reason": guard_reason,
                "category": category,
                "confidence": confidence,
                "raw_response": raw_response,
                "agent_verdicts": agent_verdicts,
                "csl_trace": csl_trace,
            }

        buffered = self._buffer(cleaned, stage=1)
        buffered = self._buffer(buffered, stage=2)

        # Final Decision
        return {
            "pipeline": "macd_v2",
            "input": user_input,
            "output": buffered,
            "blocked": False,
            "block_stage": None,
            "block_reason": None,
            "category": category,
            "confidence": confidence,
            "raw_response": raw_response,
            "agent_verdicts": agent_verdicts,
            "csl_trace": csl_trace,
        }