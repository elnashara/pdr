import time

from pdr_simulator import PDRSimulator
from critic import LLMCritic

class PDRSimulatorWithCritic(PDRSimulator):
    """
    Inherits from the PDRSimulator, but uses a separate LLM-based critic
    to label each output chunk for style, correctness, etc.

    The general flow:
    1) Generate multiple outputs per iteration.
    2) Evaluate each with the main evaluator (for a numeric score).
    3) Also call the Critic to get more qualitative labeling.
    4) Use those labels to refine the prompt more effectively.
    """

    def __init__(self, evaluator, max_iterations=5, score_threshold=85,
                 num_outputs_per_iter=3, critic=None):
        super().__init__(evaluator, max_iterations, score_threshold, num_outputs_per_iter)
        # If no critic is provided, create a default one
        self.critic = critic if critic else LLMCritic()

    def simulate(self, participant, task):
        start_time = time.time()
        iteration_count = 0
        final_output = ""
        final_score = 0

        current_prompt = (
            f"Your task:\n{task.target_spec}\n\n"
            "Generate multiple distinct outputs.\n"
            "We'll pick the best, but also have a 'critic' assess each output.\n"
        )

        for _ in range(self.max_iterations):
            iteration_count += 1

            # Step 1: Generate multiple outputs
            outputs = []
            for i in range(self.num_outputs_per_iter):
                out = participant.generate_output(
                    user_instruction=f"{current_prompt}\n\n(Version #{i+1})"
                )
                outputs.append(out)

            # Step 2: Evaluate each output for a numeric score
            best_index, best_score, best_eval = -1, -1, None
            for i, out in enumerate(outputs):
                eval_results = self.evaluator.evaluate_output(out, task.rubric)
                if eval_results["score"] > best_score:
                    best_score = eval_results["score"]
                    best_eval = eval_results
                    best_index = i

            best_output = outputs[best_index]
            final_output = best_output
            final_score = best_score

            # Step 3: Critic evaluates all outputs for deeper labeling
            instructions_for_critic = (
                "Evaluate each output for stylistic alignment, correctness, etc. "
                "Label strengths/weaknesses. Provide short improvement suggestions."
            )
            critic_report = self.critic.critique_outputs(outputs, instructions_for_critic)

            # If best output meets threshold, stop
            if best_score >= self.score_threshold:
                break

            # Step 4: Identify preferences from the best output and/or the critic report
            preference_instructions = self._extract_preferences_with_critic(
                best_output, best_eval, critic_report
            )

            # Step 5: Refine prompt
            current_prompt += (
                "\n\n[PDR WITH CRITIC REFINEMENT]\n"
                f"{preference_instructions}\n"
                "Based on these preferences and critic's feedback, please refine future outputs."
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

    def _extract_preferences_with_critic(self, best_output, best_eval, critic_report):
        """
        Incorporate the main evaluator's numeric analysis and
        the critic's labels to produce more refined preferences.
        """
        lines = []

        # From the main evaluator:
        if best_eval["word_count_ok"]:
            lines.append("Preferred: Word count is within range.")
        else:
            lines.append("Non-preferred: Word count out of range.")

        if not best_eval["must_include_ok"]:
            lines.append("Non-preferred: Missing required keywords.")

        # From the critic:
        # The critic_report is a single string containing feedback for all outputs.
        # You might parse it for the best output specifically, but let's keep it simple:
        # We'll just include a short excerpt from the critic's overall feedback.
        excerpt = critic_report[:300] + "..." if len(critic_report) > 300 else critic_report
        lines.append("Critic Summary (excerpt): " + excerpt)

        return "\n".join(lines)

    def _simulate_satisfaction(self, final_score):
        return round(min(5, max(1, (final_score / 20))), 1)
    