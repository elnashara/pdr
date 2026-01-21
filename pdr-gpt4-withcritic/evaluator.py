import openai

class Evaluator:
    """
    Evaluates a participant's output against a given rubric,
    returning a numeric score and optional GPT-4o analysis.
    """
    def __init__(self, use_gpt4_for_eval: bool = True, model: str = "gpt-4o"):
        self.use_gpt4_for_eval = use_gpt4_for_eval
        self.model = model

    def evaluate_output(self, output_text: str, rubric: dict) -> dict:
        """
        Returns a dict with keys:
          - 'word_count_ok': bool
          - 'must_include_ok': bool
          - 'score': int
          - 'analysis': str (optional, GPT-4o analysis)
        """
        # 1) Word count check
        words = output_text.split()
        word_count = len(words)
        min_count, max_count = rubric["word_count_range"]
        word_count_ok = (min_count <= word_count <= max_count)

        # 2) Must-include check
        all_keywords_present = True
        for kw in rubric.get("must_include", []):
            if kw.lower() not in output_text.lower():
                all_keywords_present = False
                break

        # Simple scoring approach
        base_score = 50
        if word_count_ok:
            base_score += 25
        if all_keywords_present:
            base_score += 25

        results = {
            "word_count_ok": word_count_ok,
            "must_include_ok": all_keywords_present,
            "score": base_score
        }

        # 3) Optional GPT-4o analysis
        if self.use_gpt4_for_eval:
            analysis_text = self._gpt4_qualitative_eval(output_text, rubric["evaluation_instructions"])
            results["analysis"] = analysis_text
        else:
            results["analysis"] = "No GPT-4o evaluation performed."

        return results

    def _gpt4_qualitative_eval(self, output_text: str, instructions: str) -> str:
        """
        Calls GPT-4o with instructions to produce a qualitative analysis.
        """
        messages = [
            {"role": "system", "content": "You are a strict evaluator of text quality."},
            {
                "role": "user",
                "content": (
                    f"Text to evaluate:\n---\n{output_text}\n---\n\n"
                    f"Rubric:\n{instructions}\n\n"
                    "Please provide a concise analysis on how well this text meets each rubric point. "
                    "End with a short summary of strengths and weaknesses."
                )
            }
        ]

        try:
            response = openai.ChatCompletion.create(
                model=self.model,
                messages=messages,
                temperature=0,
                max_tokens=500
            )
            return response["choices"][0]["message"]["content"]
        except Exception as e:
            return f"Error calling GPT-4o API for qualitative evaluation: {e}"
