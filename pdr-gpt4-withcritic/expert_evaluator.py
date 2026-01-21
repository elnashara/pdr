import openai

class ExpertEvaluator:
    """
    An optional tool that provides domain-expert judgments on correctness,
    style, or other specialized criteria. Can be real humans or GPT-4o.
    """

    def __init__(self, model="gpt-4o", temperature=0):
        self.model = model
        self.temperature = temperature

    def evaluate_as_expert(self, output_text: str, domain: str = "general") -> dict:
        """
        Calls GPT-4o or uses specialized logic to produce:
          - correctness_score (0-5)
          - style_score (0-5)
          - notes: any textual feedback from the 'expert'
        'domain' can be something like 'educational', 'technical', etc.
        """
        # Example with GPT-4o, you can adapt to a real human process or more complex logic
        prompt = (
            f"As a domain expert in {domain}, rate the following text on:\n"
            "1) correctness (0-5)\n"
            "2) style/clarity (0-5)\n"
            "Provide some brief commentary.\n\n"
            f"OUTPUT TEXT:\n{output_text}\n"
        )

        try:
            response = openai.ChatCompletion.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                temperature=self.temperature,
                max_tokens=500
            )
            eval_text = response["choices"][0]["message"]["content"]
            # Here, you'd parse 'eval_text' to extract numeric scores and notes
            # For simplicity, let's just return the raw text in 'notes'
            return {
                "correctness_score": None,  # you could parse
                "style_score": None,        # you could parse
                "notes": eval_text
            }
        except Exception as e:
            return {
                "correctness_score": None,
                "style_score": None,
                "notes": f"Error calling GPT-4o: {e}"
            }
