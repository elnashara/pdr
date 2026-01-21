import openai
import os
from typing import Optional, List, Dict, Any, Tuple

# uses the helper we created earlier
from retry_helpers import chat_with_retries


class Participant:
    """
    Represents a simulated participant (persona).
    """

    # Models that DO NOT support temperature (keep list easy to extend)
    _NO_TEMPERATURE_MODELS = {"gpt-5", "gpt-5-mini", "gpt-4o-mini"}

    # Models that accept `verbosity` (best-effort guard; won't include param for others)
    _HAS_VERBOSITY_MODELS = {"gpt-5", "gpt-5-mini"}

    def __init__(self, name: str, persona_description: str, model: str = "gpt-5"):
        self.name = name
        self.persona_description = persona_description
        self.model = model

    def _make_kwargs(
        self,
        messages: List[Dict[str, Any]],
        max_comp_tokens: int,
        temperature: Optional[float],
        reasoning_effort: str = "minimal",
        verbosity: Optional[str] = "low",
    ) -> Dict[str, Any]:
        kwargs: Dict[str, Any] = {
            "model": self.model,
            "messages": messages,
            "max_completion_tokens": max_comp_tokens,
            "reasoning_effort": reasoning_effort,  # GPT-5 reasoning models
        }
        # Only pass temperature for models that support it
        if temperature is not None and self.model not in self._NO_TEMPERATURE_MODELS:
            kwargs["temperature"] = temperature
        # Only pass verbosity for models that support it
        if verbosity is not None and self.model in self._HAS_VERBOSITY_MODELS:
            kwargs["verbosity"] = verbosity  # keeps answers short & to-the-point
        return kwargs

    def _call(
        self,
        kwargs: Dict[str, Any],
        network_attempts: int,
        request_timeout: int,
        debug: bool,
        label: str,
    ):
        resp = chat_with_retries(
            max_attempts=network_attempts,
            request_timeout=request_timeout,
            **kwargs
        )
        if debug:
            print(f"{label} KWARGS:", {k: v for k, v in kwargs.items() if k != "messages"})
            print(f"{label} RAW:", resp)
        return resp

    def generate_output(
        self,
        user_instruction: str,
        temperature: float = 0.7,
        max_tokens: int = 1000,
        max_retries: int = 1,                 # content-based retry (empty/length)
        network_attempts: int = 6,            # network/5xx/429 retries
        request_timeout: int = 90,            # per-request timeout (seconds)
        allow_model_fallback: bool = True,    # try a non-reasoning model if GPT-5 still empty
        fallback_models: Tuple[str, ...] = ("gpt-4o",),
        debug: bool = False,
    ) -> str:
        """
        Simulates how this participant would respond to a given user instruction.
        Strategy:
          1) GPT-5, minimal reasoning, normal budget.
          2) If empty/length, GPT-5 again with strong code-only nudge + larger budget.
          3) If still empty and allowed, fall back to a non-reasoning model (e.g., gpt-4o).
        """
        base_messages = [
            {"role": "system", "content": f"You are {self.name}. {self.persona_description}"},
            {"role": "user", "content": user_instruction},
        ]

        def extract(resp):
            choice = resp["choices"][0]
            content = (choice.get("message") or {}).get("content") or ""
            finish_reason = choice.get("finish_reason")
            usage = resp.get("usage", {})
            comp_details = usage.get("completion_tokens_details", {})
            reasoning_used = comp_details.get("reasoning_tokens", 0)
            return content.strip(), finish_reason, reasoning_used

        try:
            # ---- Attempt 1: GPT-5, minimal reasoning, low verbosity
            kwargs1 = self._make_kwargs(
                messages=base_messages,
                max_comp_tokens=max_tokens,
                temperature=temperature,
                reasoning_effort="minimal",
                verbosity="low",
            )
            resp1 = self._call(kwargs1, network_attempts, request_timeout, debug, "REQUEST 1")
            content, finish_reason, reasoning_used = extract(resp1)
            if content:
                return content

            # Should we do a second attempt with a larger budget & stronger instruction?
            should_retry = (
                finish_reason == "length"
                or reasoning_used >= max(128, int(0.8 * max_tokens))
                or not content
            )

            if should_retry and max_retries > 0:
                retry_messages = [
                    {
                        "role": "system",
                        "content": (
                            "Return ONLY the final answer as a single Python code block. "
                            "Do NOT include explanations, commentary, or prose. "
                            "Avoid hidden chain-of-thought; emit the code immediately."
                        ),
                    },
                    *base_messages,
                ]
                bigger_budget = max(4096, int(max_tokens * 2.5))  # larger than before to avoid back-to-back truncation

                kwargs2 = self._make_kwargs(
                    messages=retry_messages,
                    max_comp_tokens=bigger_budget,
                    temperature=temperature,
                    reasoning_effort="minimal",
                    verbosity="low",
                )
                resp2 = self._call(kwargs2, network_attempts, request_timeout, debug, "REQUEST 2")
                content2, finish2, reasoning2 = extract(resp2)
                if content2:
                    return content2

                # Optional model fallback if GPT-5 still burned all tokens on reasoning
                if allow_model_fallback:
                    for fb_model in fallback_models:
                        prev_model = self.model
                        try:
                            self.model = fb_model
                            # For non-reasoning models, we can include temperature.
                            kwargs_fb = self._make_kwargs(
                                messages=retry_messages,
                                max_comp_tokens=max_tokens,  # smaller again; non-reasoning models typically emit faster
                                temperature=temperature if fb_model not in self._NO_TEMPERATURE_MODELS else None,
                                reasoning_effort="minimal",   # safe no-op for non-reasoning models
                                verbosity=None,               # don't pass verbosity unless supported
                            )
                            resp_fb = self._call(kwargs_fb, network_attempts, request_timeout, debug, f"FALLBACK {fb_model}")
                            content_fb, _, _ = extract(resp_fb)
                            if content_fb:
                                return content_fb
                        finally:
                            self.model = prev_model

                # If we reach here, we got nothing useful back
                raise RuntimeError(
                    f"Empty completion from {self.model} after retry. "
                    f"finish_reason={finish2}, reasoning_tokens={reasoning2}, "
                    f"max_completion_tokens={bigger_budget}"
                )

            # No retry warranted → raise
            raise RuntimeError(
                f"Empty completion from {self.model}. "
                f"finish_reason={finish_reason}, reasoning_tokens={reasoning_used}, "
                f"max_completion_tokens={max_tokens}"
            )

        except Exception as e:
            # Bubble up so the caller can decide to break/abort the run
            raise RuntimeError(f"Error calling {self.model} API: {e}") from e
