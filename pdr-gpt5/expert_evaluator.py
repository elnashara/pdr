import openai
import json
import re
from typing import Optional, Dict, Any, List


class ExpertEvaluator:
    """
    Calls a model (default: GPT-4o) to produce:
      - correctness_score (0-5)
      - style_score (0-5)
      - notes (string)
    Returns a dict: {"correctness_score": float, "style_score": float, "notes": str}
    """

    _NO_TEMPERATURE_MODELS = {"gpt-4o", "gpt-5-mini", "gpt-4o-mini"}

    def __init__(self, model: str = "gpt-4o", temperature: Optional[float] = 0):
        self.model = model
        self.temperature = temperature

    @staticmethod
    def _clamp_0_5(x: float) -> float:
        try:
            return max(0.0, min(5.0, float(x)))
        except Exception:
            return 0.0

    @staticmethod
    def _fallback_extract_scores(text: str) -> Dict[str, Optional[float]]:
        """
        Fallback parsing for non-JSON replies; tries to find 'correctness' and 'style'
        numbers in common phrasings like 'correctness: 4.5/5' or 'style 3 out of 5'.
        """
        def find_score(label: str) -> Optional[float]:
            # Examples matched: 'correctness 4.5/5', 'correctness: 4 out of 5', 'correctness = 3.0'
            pattern = rf"{label}\s*[:=]?\s*(\d+(?:\.\d+)?)\s*(?:/|out of)?\s*5"
            m = re.search(pattern, text, flags=re.IGNORECASE)
            if m:
                return ExpertEvaluator._clamp_0_5(float(m.group(1)))
            # Looser: just the first number after the label
            pattern2 = rf"{label}\s*[:=]?\s*(\d+(?:\.\d+)?)"
            m2 = re.search(pattern2, text, flags=re.IGNORECASE)
            if m2:
                return ExpertEvaluator._clamp_0_5(float(m2.group(1)))
            return None

        correctness = find_score("correctness")
        style = find_score("style|clarity|style/clarity")
        return {"correctness_score": correctness, "style_score": style}

    @staticmethod
    def _parse_json_scores(raw: str) -> Optional[Dict[str, Any]]:
        raw = (raw or "").strip()
        if not raw:
            return None

        # Strip code fences if present
        if raw.startswith("```"):
            raw = re.sub(r"^```[a-zA-Z0-9_+-]*\n", "", raw)
            raw = re.sub(r"\n```$", "", raw)

        try:
            obj = json.loads(raw)
            c = ExpertEvaluator._clamp_0_5(obj.get("correctness_score", 0))
            s = ExpertEvaluator._clamp_0_5(obj.get("style_score", 0))
            notes = str(obj.get("notes", "")).strip()
            return {"correctness_score": c, "style_score": s, "notes": notes}
        except Exception:
            return None

    def _make_kwargs(
        self,
        messages: List[Dict[str, str]],
        max_comp_tokens: int,
        # reasoning_effort: str = "minimal",
    ) -> Dict[str, Any]:
        kwargs = {
            "model": self.model,
            "messages": messages,
            "max_completion_tokens": max_comp_tokens,
            # "reasoning_effort": reasoning_effort,
        }
        if self.temperature is not None and self.model not in self._NO_TEMPERATURE_MODELS:
            kwargs["temperature"] = self.temperature
        return kwargs

    def evaluate_as_expert(
        self,
        output_text: str,
        domain: str = "general",
        max_tokens: int = 400,
        max_retries: int = 1,
        debug: bool = False,
    ) -> dict:
        """
        Evaluate with compact JSON-only output. Retries once with a stronger nudge / bigger budget if needed.
        """
        system_msg = (
            "You are a strict domain expert grader. "
            "Return ONLY a single compact JSON object with keys: "
            "correctness_score (0-5), style_score (0-5), notes (string). "
            "No markdown, no code fences, no extra text."
        )
        user_msg = (
            f"As an expert in {domain}, assess the following TEXT.\n"
            "Scores must be numbers from 0 to 5 (allow halves, e.g., 3.5). "
            "Keep notes concise (<= 60 words).\n\n"
            "TEXT START\n"
            f"{output_text}\n"
            "TEXT END"
        )

        base_messages = [
            {"role": "system", "content": system_msg},
            {"role": "user", "content": user_msg},
        ]

        try:
            # Attempt 1
            kwargs = self._make_kwargs(
                messages=base_messages,
                max_comp_tokens=max_tokens,
                # reasoning_effort="minimal",
            )
            resp = openai.ChatCompletion.create(**kwargs)
            if debug:
                print("EVAL REQUEST 1:", kwargs)
                print("EVAL RESPONSE 1:", resp)

            choice = resp["choices"][0]
            content = (choice.get("message") or {}).get("content") or ""
            parsed = self._parse_json_scores(content)

            if parsed:
                return parsed

            # Fallback parse from freeform (if model ignored JSON instruction)
            fb = self._fallback_extract_scores(content)
            if fb["correctness_score"] is not None and fb["style_score"] is not None:
                return {**fb, "notes": content.strip()}

            # Decide whether to retry (empty/length/heavy reasoning/no JSON)
            finish_reason = choice.get("finish_reason")
            usage = resp.get("usage", {})
            details = usage.get("completion_tokens_details", {})
            reasoning_used = details.get("reasoning_tokens", 0)
            should_retry = (
                not content.strip() or
                finish_reason == "length" or
                reasoning_used >= max(64, int(0.8 * max_tokens))
            )

            if should_retry and max_retries > 0:
                # Stronger nudge + bigger budget on retry
                retry_messages = [
                    {
                        "role": "system",
                        "content": (
                            "Return ONLY valid JSON with keys "
                            '{"correctness_score": number, "style_score": number, "notes": "string"} '
                            "— no markdown, no backticks, no extra text."
                        ),
                    },
                    *base_messages
                ]
                bigger = max(600, max_tokens * 2)
                kwargs2 = self._make_kwargs(
                    messages=retry_messages,
                    max_comp_tokens=bigger,
                    # reasoning_effort="minimal",
                )
                resp2 = openai.ChatCompletion.create(**kwargs2)
                if debug:
                    print("EVAL REQUEST 2:", kwargs2)
                    print("EVAL RESPONSE 2:", resp2)

                content2 = (resp2["choices"][0]["message"] or {}).get("content") or ""

                parsed2 = self._parse_json_scores(content2)
                if parsed2:
                    return parsed2

                fb2 = self._fallback_extract_scores(content2)
                if fb2["correctness_score"] is not None and fb2["style_score"] is not None:
                    return {**fb2, "notes": content2.strip()}

                raise RuntimeError(
                    f"Expert evaluation returned no parsable scores after retry. "
                    f"finish_reason={resp2['choices'][0].get('finish_reason')}, "
                    f"content_len={len(content2)}"
                )

            # No retry warranted → raise with diagnostic
            raise RuntimeError(
                f"Expert evaluation returned no parsable scores. "
                f"finish_reason={finish_reason}, content_len={len(content)}"
            )

        except Exception as e:
            raise RuntimeError(f"Error calling {self.model} API: {e}") from e
