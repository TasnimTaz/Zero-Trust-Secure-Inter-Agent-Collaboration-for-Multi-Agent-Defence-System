# agents/groq_utils.py

import re
import time
import random

def safe_completion(client, max_retries: int = 8, base_delay: float = 2.0, max_wait: float = 90.0, **kwargs):
    last_err = None
    for attempt in range(max_retries):
        try:
            return client.chat.completions.create(**kwargs)
        except Exception as e:
            last_err = e
            msg = str(e)
            is_rate_limit = "429" in msg or "rate limit" in msg.lower()

            if not is_rate_limit:
                raise

            # মিনিট + সেকেন্ড দুটোই ধরার জন্য regex আপডেট
            wait_match = re.search(r"try again in (?:(\d+)m)?([\d.]+)s", msg, re.IGNORECASE)
            if wait_match:
                minutes = float(wait_match.group(1) or 0)
                seconds = float(wait_match.group(2))
                wait_time = minutes * 60 + seconds + 0.5
            else:
                wait_time = base_delay * (2 ** attempt) + random.uniform(0, 1)

            wait_time = min(wait_time, max_wait)  # কোনো একটা call-এ অনেকক্ষণ আটকে থাকা এড়াতে cap

            if attempt < max_retries - 1:
                time.sleep(wait_time)
            else:
                raise last_err