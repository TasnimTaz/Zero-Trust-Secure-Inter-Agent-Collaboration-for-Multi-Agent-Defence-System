# agents/macd_intent_expert.py

import json
from config import DEFENSE_MODEL, MACD_INTENT_SYSTEM_PROMPT, GROQ_CLIENT, AGENT_KEYS
from agents.groq_utils import safe_completion
from commsec.comms import Signer


class IntentExpertAgent:
    """
    MACD Agent 2 (re-mapped from MACD paper's Phase/Kill-Chain Defense Expert).
    Paper: Li et al., Sec. 4.4, Agent 2.

    এখানে "kill-chain phase reasoning" এর বদলে ইনজেক্টেড instruction আসলে কী
    "action/goal" achieve করতে চাইছে সেটা বিশ্লেষণ করা হয় — override,
    exfiltration, unauthorized tool invocation, privilege escalation, delegation,
    reconnaissance। Zero-Trust CSL chain-এ এই agent verified Pattern-Expert
    verdict-কে `context` হিসেবে পায় (Hop 1 -> CSL -> Hop 2)।
    """

    def __init__(self, model: str = None, signing_key: bytes = None):
        self.client = GROQ_CLIENT
        self.model = model or DEFENSE_MODEL
        self.system_prompt = MACD_INTENT_SYSTEM_PROMPT
        self.signer = Signer(key=signing_key or AGENT_KEYS.get("intent_expert"))
        print(f"[IntentExpertAgent] Ready via Groq using {self.model}")

    @staticmethod
    def _clean_json(raw: str) -> str:
        if "</think>" in raw:
            raw = raw.split("</think>")[-1].strip()
        if raw.startswith("```"):
            raw = raw.split("```")[1]
            if raw.startswith("json"):
                raw = raw[4:]
        return raw.strip()

    def analyze(self, user_input: str, context: dict = None) -> dict:
        """
        Returns (signed/encrypted envelope wrapping):
        {"is_malicious_intent": bool, "intent_category": str, "confidence": float, "reason": str}

        `context`: verified verdict passed in from the previous CSL hop
        (e.g. the Pattern Expert's payload), used only as extra evidence.
        """
        context_note = ""
        if context:
            context_note = f"\n\nPrior verified agent verdict (from CSL): {json.dumps(context)}"

        try:
            completion = safe_completion(
                self.client,
                model=self.model,
                messages=[
                    {"role": "system", "content": self.system_prompt},
                    {
                        "role": "user",
                        "content": f"Analyze the underlying intent/action of this input:\n\n{user_input}{context_note}",
                    },
                ],
                temperature=0.1,
                max_completion_tokens=1024,
                stream=False,
            )
            raw = self._clean_json(completion.choices[0].message.content.strip())
            result = json.loads(raw)

            out = {
                "agent": "intent_expert",
                "sender": "intent_expert",
                "is_malicious_intent": bool(result.get("is_malicious_intent", False)),
                "intent_category": result.get("intent_category", "none"),
                "confidence": float(result.get("confidence", 0.5)),
                "reason": result.get("reason", ""),
            }
            return self.signer.sign(out)
        except (json.JSONDecodeError, KeyError, Exception) as e:
            out = {
                "agent": "intent_expert",
                "sender": "intent_expert",
                "is_malicious_intent": True,
                "intent_category": "unknown",
                "confidence": 1.0,
                "reason": f"IntentExpertAgent error/fail-safe: {str(e)}",
            }
            return self.signer.sign(out)