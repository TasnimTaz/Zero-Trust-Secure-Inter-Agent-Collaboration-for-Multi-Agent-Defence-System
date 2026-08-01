import json
import logging
from config import DEFENSE_MODEL, MACD_JUDGE_SYSTEM_PROMPT, GROQ_CLIENT
from agents.groq_utils import safe_completion


class MACDJudgeAgent:
    """
    MACD Agent 4 (re-mapped from MACD paper's Coordinator Agent).
    """

    def __init__(self, model: str = None):
        self.client = GROQ_CLIENT
        self.model = model or DEFENSE_MODEL
        self.system_prompt = MACD_JUDGE_SYSTEM_PROMPT
        print(f"[MACDJudgeAgent] Ready via Groq using {self.model}")

    @staticmethod
    def _clean_json(raw: str) -> str:
        if "</think>" in raw:
            raw = raw.split("</think>")[-1].strip()
        if raw.startswith("```"):
            raw = raw.split("```")[1]
            if raw.startswith("json"):
                raw = raw[4:]
        return raw.strip()

    def synthesize(self, user_input: str, pattern_verdict: dict, intent_verdict: dict, category_verdict: dict) -> tuple[bool, str, str, float]:
        """
        Returns: (is_safe, category, reason, confidence)
        """
        # টাইপ সেফটি নিশ্চিত করতে ডিকশনারি না হলে খালি ডিকশনারি সেট করা
        pattern_verdict = pattern_verdict if isinstance(pattern_verdict, dict) else {}
        intent_verdict = intent_verdict if isinstance(intent_verdict, dict) else {}
        category_verdict = category_verdict if isinstance(category_verdict, dict) else {}

        verdicts_summary = (
            f"Agent 1 (Pattern/Syntax): is_suspicious={pattern_verdict.get('is_suspicious')}, "
            f"patterns={pattern_verdict.get('patterns_found')}, "
            f"confidence={pattern_verdict.get('confidence')}, reason={pattern_verdict.get('reason')}\n"
            f"Agent 2 (Intent/Action): is_malicious_intent={intent_verdict.get('is_malicious_intent')}, "
            f"intent_category={intent_verdict.get('intent_category')}, "
            f"confidence={intent_verdict.get('confidence')}, reason={intent_verdict.get('reason')}\n"
            f"Agent 3 (Attack-Category): matches_known_attack={category_verdict.get('matches_known_attack')}, "
            f"category={category_verdict.get('category')}, "
            f"confidence={category_verdict.get('confidence')}, reason={category_verdict.get('reason')}"
        )
        try:
            completion = safe_completion(
                self.client,
                model=self.model,
                messages=[
                    {"role": "system", "content": self.system_prompt},
                    {
                        "role": "user",
                        "content": (
                            f"Original user input:\n{user_input}\n\n"
                            f"Expert verdicts:\n{verdicts_summary}"
                        ),
                    },
                ],
                temperature=0.1,
                max_completion_tokens=1024,
                stream=False,
            )
            raw = self._clean_json(completion.choices[0].message.content.strip())
            result = json.loads(raw)

            # স্ট্রিং বা বুলিয়ান যাই আসুক, তা সঠিকভাবে হ্যান্ডেল করা
            is_safe_raw = result.get("is_safe", False)
            is_safe = is_safe_raw if isinstance(is_safe_raw, bool) else str(is_safe_raw).lower() == "true"
            
            category = result.get("category", "unknown")
            reason = result.get("reason", "")
            confidence = float(result.get("confidence", 1.0))

            return is_safe, category, reason, confidence

        except (json.JSONDecodeError, KeyError, Exception) as e:
            logging.warning(f"[MACDJudgeAgent Exception Catch]: Parsing failed, falling back to strict check. Error: {e}")
            
            # স্ট্রিং বনাম বুলিয়ান সেফটি চেকার হেল্পার লজিক
            def is_flagged(verdict, key):
                val = verdict.get(key, False)
                if isinstance(val, bool):
                    return val
                return str(val).lower() in ["true", "yes", "1", "suspicious", "malicious"]

            # ৩টি এক্সপার্টের কড়া ও নিরাপদ মূল্যায়ন
            p_flag = is_flagged(pattern_verdict, "is_suspicious")
            i_flag = is_flagged(intent_verdict, "is_malicious_intent")
            c_flag = is_flagged(category_verdict, "matches_known_attack")
            
            any_flagged = p_flag or i_flag or c_flag
            
            # সিকিউরিটি ফ্রেমওয়ার্কের নিয়ম অনুযায়ী: জাজ ফেইল করলে এবং কোনো এক্সপার্ট ফ্লাগ দিলে 
            # সেটিকে অনিরাপদ (is_safe = False) ধরাটাই শ্রেয়।
            return (
                not any_flagged,
                category_verdict.get("category", "unknown") if any_flagged else "safe",
                f"MACDJudgeAgent error/fail-safe strict validation triggered: {str(e)}",
                0.5 if any_flagged else 1.0,
            )