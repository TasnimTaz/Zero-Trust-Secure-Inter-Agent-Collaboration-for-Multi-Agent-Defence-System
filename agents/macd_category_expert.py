# agents/macd_category_expert.py

import json
from config import DEFENSE_MODEL, MACD_CATEGORY_SYSTEM_PROMPT, MACD_KB, GROQ_CLIENT, AGENT_KEYS
from agents.groq_utils import safe_completion
from commsec.comms import Signer


class CategoryExpertAgent:
    """
    MACD Agent 3 (re-mapped from MACD paper's APT Defense Expert).
    Paper: Li et al., Sec. 4.4, Agent 3 + Sec. 4.3 (RAG-enhanced retrieval).

    এখানে "APT group / TTP profiling" এর বদলে input-টা আমাদের নিজস্ব IPI
    taxonomy (override, obfuscation, roleplay, cta, recon, exfiltration,
    delegation, signal) এর কোন ক্যাটাগরির সাথে ম্যাচ করে সেটা বের করা হয়।
    Phase A হিসেবে একটা lightweight keyword-based KB retrieval (RAG-lite)
    ব্যবহার করা হয়েছে — vector DB ছাড়াই, top-K similar known attack category
    LLM-কে context হিসেবে দেওয়া হয় (paper Fig. 1: Phase 1-2 KB + RAG এর
    সরলীকৃত সংস্করণ)।

    Zero-Trust CSL chain-এ এটি Hop 3 (Prompt Injection Detection Agent) —
    verified Intent-Expert verdict (Hop 2) `context` হিসেবে পায়।
    """

    def __init__(self, kb=None, top_k: int = 3, model: str = None, signing_key: bytes = None):
        self.client = GROQ_CLIENT
        self.model = model or DEFENSE_MODEL
        self.system_prompt = MACD_CATEGORY_SYSTEM_PROMPT
        self.kb = kb if kb is not None else MACD_KB
        self.top_k = top_k
        self.signer = Signer(key=signing_key or AGENT_KEYS.get("category_expert"))
        print(f"[CategoryExpertAgent] Ready via Groq using {self.model} (KB size={len(self.kb)})")

    def _retrieve(self, user_input: str) -> list[dict]:
        """Phase A: keyword-overlap based retrieval (RAG-lite), returns top_k KB entries."""
        text = user_input.lower()
        scored = []
        for entry in self.kb:
            hits = [kw for kw in entry["keywords"] if kw in text]
            if hits:
                scored.append((len(hits), entry["id"], entry["category"], hits))
        scored.sort(key=lambda x: x[0], reverse=True)
        return [
            {"id": kb_id, "category": cat, "matched_keywords": hits}
            for _, kb_id, cat, hits in scored[: self.top_k]
        ]

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
        Returns: {"matches_known_attack": bool, "category": str, "confidence": float,
                  "reason": str, "retrieved": [...]}

        `context`: verified verdict from the previous CSL hop (Intent Expert),
        passed in as extra evidence alongside the KB retrieval.
        """
        retrieved = self._retrieve(user_input)
        kb_context = (
            "\n".join(f"- {r['id']} ({r['category']}): matched {r['matched_keywords']}" for r in retrieved)
            or "No similar known attacks retrieved."
        )
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
                        "content": (
                            f"User input:\n{user_input}\n\n"
                            f"Similar known attacks retrieved from KB (RAG):\n{kb_context}"
                            f"{context_note}"
                        ),
                    },
                ],
                temperature=0.1,
                max_completion_tokens=1024,
                stream=False,
            )
            raw = self._clean_json(completion.choices[0].message.content.strip())
            result = json.loads(raw)

            out = {
                "agent": "category_expert",
                "sender": "category_expert",
                "matches_known_attack": bool(result.get("matches_known_attack", False)),
                "category": result.get("category", "unknown"),
                "confidence": float(result.get("confidence", 0.5)),
                "reason": result.get("reason", ""),
                "retrieved": retrieved,
            }
            return self.signer.sign(out)
        except (json.JSONDecodeError, KeyError, Exception) as e:
            out = {
                "agent": "category_expert",
                "sender": "category_expert",
                "matches_known_attack": bool(retrieved),
                "category": retrieved[0]["category"] if retrieved else "unknown",
                "confidence": 1.0 if retrieved else 0.5,
                "reason": f"CategoryExpertAgent error/fail-safe: {str(e)}",
                "retrieved": retrieved,
            }
            return self.signer.sign(out)