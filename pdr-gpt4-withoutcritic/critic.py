import openai

class LLMCritic:
    """
    A separate LLM-based critic that automatically labels each output chunk
    for alignment with a given style, correctness, or other criteria.
    This can be used to automate preference identification in PDR or baseline.
    """

    def __init__(self, model="gpt-4o", temperature=0):
        """
        model: GPT-4o model to use for criticism (default: gpt-4o)
        temperature=0 for more deterministic feedback.
        """
        self.model = model
        self.temperature = temperature

    def critique_outputs(self, outputs, instructions):
        """
        Given a list of output strings and some textual instructions describing
        what to check (style, correctness, tone, etc.), returns a list of
        'critic reports' for each output.

        Each 'critic report' could be a dict with fields like:
          - 'strengths'
          - 'weaknesses'
          - 'overall_label' or 'score'
        or just a textual summary from GPT-4o.
        """
        # We'll create a single conversation with GPT-4o that sees
        # all the outputs. You can also do multiple calls, one per output.
        messages = [
            {
                "role": "system",
                "content": (
                    "You are an objective, detailed critic. "
                    "You will label each output for its strengths and weaknesses "
                    "based on the instructions."
                )
            },
            {
                "role": "user",
                "content": (
                    f"Instructions:\n{instructions}\n\n"
                    "Below are multiple outputs. For each output, label its strengths, "
                    "weaknesses, and provide an overall recommendation for improvement.\n\n"
                    + self._format_outputs(outputs)
                )
            }
        ]

        response = openai.ChatCompletion.create(
            model=self.model,
            messages=messages,
            temperature=self.temperature,
            max_tokens=1500
        )

        # The returned text might contain a structured or semi-structured
        # breakdown for each output. You can decide how to parse it.
        # For now, we'll just return the entire textual response as a string.
        critic_report = response["choices"][0]["message"]["content"]

        return critic_report

    def _format_outputs(self, outputs):
        """
        Helper method to present multiple outputs in a single user message.
        """
        # e.g. number them:
        text = ""
        for i, out in enumerate(outputs):
            text += f"Output #{i+1}:\n{out}\n\n"
        return text
