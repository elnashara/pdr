import openai
from typing import List, Dict, Any, Optional, Tuple

from retry_helpers import chat_with_retries


class LLMCritic:
    """
    LLM-based critic that labels each output given instructions.
    """

    _NO_TEMPERATURE_MODELS = {"gpt-5", "gpt-5-mini", "gpt-4o-mini"}
    _HAS_VERBOSITY_MODELS = {"gpt-5", "gpt-5-mini"}

    def __init__(self, model: str = "gpt-5", temperature: Optional[float] = None):
        self.model = model
        self.temperature = temperature

    def _make_kwargs(
        self,
        messages: List[Dict[str, Any]],
        max_comp_tokens: int,
        reasoning_effort: str = "minimal",
        verbosity: Optional[str] = "low",
    ) -> Dict[str, Any]:
        kwargs: Dict[str, Any] = {
            "model": self.model,
            "messages": messages,
            "max_completion_tokens": max_comp_tokens,
            "reasoning_effort": reasoning_effort,
        }
        if self.temperature is not None and self.model not in self._NO_TEMPERATURE_MODELS:
            kwargs["temperature"] = self.temperature
        if verbosity is not None and self.model in self._HAS_VERBOSITY_MODELS:
            kwargs["verbosity"] = verbosity
        return kwargs

    def _format_outputs(self, outputs: List[str]) -> str:
        text = ""
        for i, out in enumerate(outputs):
            text += f"Output #{i+1}:\n{out}\n\n"
        return text

    def _extract(self, resp):
        choice = resp["choices"][0]
        content = (choice.get("message") or {}).get("content") or ""
        finish_reason = choice.get("finish_reason")
        usage = resp.get("usage", {})
        comp_details = usage.get("completion_tokens_details", {})
        reasoning_used = comp_details.get("reasoning_tokens", 0)
        return content.strip(), finish_reason, reasoning_used

    def critique_outputs(
        self,
        outputs: List[str],
        instructions: str,
        *,
        max_tokens: int = 1200,
        max_retries: int = 1,               # content-based retry
        network_attempts: int = 6,          # 5xx/429/network retries
        request_timeout: int = 90,          # seconds
        allow_model_fallback: bool = True,  # optionally switch model if GPT-5 still empty
        fallback_models: Tuple[str, ...] = ("gpt-4o",),
        debug: bool = False,
    ) -> str:
        """
        Returns a textual 'critic report'. If the first attempt is empty/truncated,
        retries with stronger constraints and a larger token budget.
        """

        messages = [
            {
                "role": "system",
                "content": (
                    "You are an objective, detailed critic. For each output, provide:"
                    "\n- Strengths"
                    "\n- Weaknesses"
                    "\n- Concrete improvements"
                    "\nBe concise and structured."
                )
            },
            {
                "role": "user",
                "content": (
                    f"Instructions:\n{instructions}\n\n"
                    "Below are multiple outputs. For each output, label strengths, weaknesses, "
                    "and an overall recommendation for improvement.\n\n"
                    + self._format_outputs(outputs)
                )
            }
        ]

        try:
            # --- Attempt 1: minimal reasoning, low verbosity
            kwargs1 = self._make_kwargs(
                messages=messages,
                max_comp_tokens=max_tokens,
                reasoning_effort="minimal",
                verbosity="low",
            )
            resp1 = chat_with_retries(
                max_attempts=network_attempts,
                request_timeout=request_timeout,
                **kwargs1
            )
            if debug:
                print("CRITIC REQUEST 1 (no messages shown):", {k: v for k, v in kwargs1.items() if k != "messages"})
                print("CRITIC RESPONSE 1:", resp1)

            content, finish_reason, reasoning_used = self._extract(resp1)
            if content:
                return content

            # Decide on retry if empty/length/heavy reasoning
            should_retry = (
                finish_reason == "length" or
                reasoning_used >= max(128, int(0.8 * max_tokens)) or
                not content
            )

            if should_retry and max_retries > 0:
                retry_messages = [
                    {
                        "role": "system",
                        "content": (
                            "Return ONLY the critic report now. Be concise, bullet the strengths/weaknesses, "
                            "and provide concrete improvements for each output. Do NOT include chain-of-thought."
                        ),
                    },
                    *messages,
                ]
                bigger_budget = max(3000, int(max_tokens * 2))

                kwargs2 = self._make_kwargs(
                    messages=retry_messages,
                    max_comp_tokens=bigger_budget,
                    reasoning_effort="minimal",
                    verbosity="low",
                )
                resp2 = chat_with_retries(
                    max_attempts=network_attempts,
                    request_timeout=request_timeout,
                    **kwargs2
                )
                if debug:
                    print("CRITIC REQUEST 2 (no messages shown):", {k: v for k, v in kwargs2.items() if k != "messages"})
                    print("CRITIC RESPONSE 2:", resp2)

                content2, finish2, reasoning2 = self._extract(resp2)
                if content2:
                    return content2

                # Optional: fall back to a non-reasoning model (e.g., gpt-4o)
                if allow_model_fallback:
                    for fb_model in fallback_models:
                        prev_model = self.model
                        try:
                            self.model = fb_model
                            kwargs_fb = self._make_kwargs(
                                messages=retry_messages,
                                max_comp_tokens=max_tokens,
                                reasoning_effort="minimal",  # harmless for non-reasoning models
                                verbosity=None,              # only pass where supported
                            )
                            resp_fb = chat_with_retries(
                                max_attempts=network_attempts,
                                request_timeout=request_timeout,
                                **kwargs_fb
                            )
                            if debug:
                                print(f"CRITIC FALLBACK {fb_model}:", {k: v for k, v in kwargs_fb.items() if k != "messages"})
                                print(f"CRITIC FALLBACK {fb_model} RESP:", resp_fb)

                            content_fb, _, _ = self._extract(resp_fb)
                            if content_fb:
                                return content_fb
                        finally:
                            self.model = prev_model

                # Exhausted retries & fallbacks
                raise RuntimeError(
                    f"Empty critic completion from {self.model} after retry. "
                    f"finish_reason={finish2}, reasoning_tokens={reasoning2}, "
                    f"max_completion_tokens={bigger_budget}"
                )

            # No retry warranted, but empty → raise
            raise RuntimeError(
                f"Empty critic completion from {self.model}. "
                f"finish_reason={finish_reason}, reasoning_tokens={reasoning_used}, "
                f"max_completion_tokens={max_tokens}"
            )

        except Exception as e:
            raise RuntimeError(f"Error calling {self.model} API: {e}") from e
