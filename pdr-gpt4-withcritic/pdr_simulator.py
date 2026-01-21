import time

class PDRSimulator:
    """
    Implements the Preference-Driven Refinement (PDR) approach for a (participant, task) pair.
    Each iteration:
      1. Generate multiple outputs.
      2. Evaluate & pick best output.
      3. Identify preferred and non-preferred elements (based on evaluation or GPT-4o analysis).
      4. Refine the prompt to embed preferences and avoid non-preferred elements.
    """

    def __init__(self, evaluator, max_iterations=5, score_threshold=85, num_outputs_per_iter=3):
        self.evaluator = evaluator
        self.max_iterations = max_iterations
        self.score_threshold = score_threshold
        self.num_outputs_per_iter = num_outputs_per_iter

    def simulate(self, participant, task):
        """
        Runs the PDR simulation for a single participant-task pair.
        Returns a dictionary with iteration count, time spent, final score, etc.
        """

        start_time = time.time()
        iteration_count = 0
        final_output = ""
        final_score = 0

        # Start with the raw target_spec as the participant's initial prompt
        current_prompt = (
            f"Your task:\n{task.target_spec}\n\n"
            "Generate multiple distinct outputs.\n"
            "We'll pick the best and refine from there."
        )

        for _ in range(self.max_iterations):
            iteration_count += 1

            # Step 1: Generate multiple outputs
            outputs = []
            for i in range(self.num_outputs_per_iter):
                # Slight variation: we can label each request or just re-call
                output_text = participant.generate_output(
                    user_instruction=f"{current_prompt}\n\n(Version #{i+1})",
                    temperature=0.7
                )
                outputs.append(output_text)

            # Step 2: Evaluate each output & pick the best
            best_index = -1
            best_score = -1
            best_eval = None
            for i, out in enumerate(outputs):
                eval_results = self.evaluator.evaluate_output(out, task.rubric)
                if eval_results["score"] > best_score:
                    best_score = eval_results["score"]
                    best_eval = eval_results
                    best_index = i
            
            best_output = outputs[best_index]
            final_output = best_output
            final_score = best_score

            # If the best output meets threshold, we stop
            if best_score >= self.score_threshold:
                break

            # Step 3: Identify preferences from the best output (preferred vs. non-preferred)
            preference_instructions = self._extract_preferences(best_output, best_eval)

            # Step 4: Refine the prompt with the new preferences
            # We embed a "Preferred Elements" vs. "Non-preferred" section.
            # In a real scenario, these might be bullet points or examples.
            current_prompt += (
                "\n\n[PDR REFINEMENT]\n"
                f"{preference_instructions}\n"
                "Based on these preferences, please refine future outputs."
            )

        end_time = time.time()
        total_time_sec = end_time - start_time
        satisfaction_score = self._simulate_satisfaction(final_score)

        return {
            "participant_name": participant.name,
            "task_name": task.name,
            "iteration_count": iteration_count,
            "time_spent_sec": total_time_sec,
            "final_score": final_score,
            "final_output": final_output,
            "satisfaction_score": satisfaction_score
        }

    def _extract_preferences(self, output_text, eval_results):
        """
        Simple method to derive 'preferred' and 'non-preferred' elements from the best output.
        For demonstration, we rely on:
          - Word count alignment
          - Missing keywords
          - GPT-4o analysis
        In a real scenario, you might do more advanced parsing or prompt GPT-4o to identify
        positive vs. negative style elements, tone, etc.
        """
        lines = []

        # If word count was OK, treat that as a "preferred element"
        if eval_results["word_count_ok"]:
            lines.append("Preferred: The word count is within the specified range. Keep that length/style.")
        else:
            lines.append("Non-preferred: The word count is out of the specified range. Adjust accordingly.")

        # If required keywords are missing, we mark them as non-preferred
        if not eval_results["must_include_ok"]:
            lines.append("Non-preferred: Missing required keywords or phrases. Add them in next version.")

        # We can also incorporate a snippet of GPT-4o analysis
        analysis_text = eval_results.get("analysis", "")
        # Let’s just include a short excerpt
        truncated_analysis = analysis_text[:200] + "..." if len(analysis_text) > 200 else analysis_text
        lines.append("GPT-4o Analysis (excerpt): " + truncated_analysis)

        # You might also choose to highlight certain lines from `output_text` that you like
        # For brevity, we'll keep it simple here.

        return "\n".join(lines)

    def _simulate_satisfaction(self, final_score):
        return round(min(5, max(1, (final_score / 20))), 1)
    