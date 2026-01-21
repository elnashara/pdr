import time
from measures import ObjectiveMeasures, SubjectiveMeasures, ExpertEvaluation

class BaselineAdHocSimulator:
    """
    Ad hoc approach. We incorporate measure-tracking here.
    """

    def __init__(self, evaluator, max_iterations=5, score_threshold=85, expert_evaluator=None):
        self.evaluator = evaluator
        self.max_iterations = max_iterations
        self.score_threshold = score_threshold
        # Optional domain expert evaluator (e.g., GPT-4o in expert mode or real human)
        self.expert_evaluator = expert_evaluator

    def simulate(self, participant, task):
        start_time = time.time()
        iteration_count = 0
        final_output = ""
        final_score = 0

        current_prompt = (
            f"Your task:\n{task.target_spec}\n\n"
            "Please produce your best final output."
        )

        for _ in range(self.max_iterations):
            iteration_count += 1
            output_text = participant.generate_output(user_instruction=current_prompt)
            eval_results = self.evaluator.evaluate_output(output_text, task.rubric)
            score = eval_results["score"]

            final_output = output_text
            final_score = score

            if score >= self.score_threshold:
                break

            feedback_summary = self._extract_feedback(eval_results)
            current_prompt += (
                "\n\n[AD HOC FEEDBACK] Please refine the output based on:\n"
                f"{feedback_summary}\n"
                "Try again and improve your answer."
            )

        end_time = time.time()
        total_time_sec = end_time - start_time
        satisfaction_score = self._simulate_satisfaction(final_score)

        # Now gather objective measures
        obj_measures = ObjectiveMeasures(
            iteration_count=iteration_count,
            time_spent_sec=total_time_sec,
            final_score=final_score,
            satisfaction_score=satisfaction_score
        )

        # Optional: If an expert evaluator is provided, we can do an expert assessment
        expert_eval_data = None
        if self.expert_evaluator is not None:
            domain = "technical"  # or "educational", "business", etc.
            expert_dict = self.expert_evaluator.evaluate_as_expert(final_output, domain)
            expert_eval_data = ExpertEvaluation(
                correctness_score=expert_dict["correctness_score"],
                style_score=expert_dict["style_score"],
                notes=expert_dict["notes"]
            )

        # Subjective measures are not collected in a simulation. If real participants
        # exist, they'd fill in a survey. For now, we can set them to None or placeholders.
        subj_measures = SubjectiveMeasures(
            perceived_quality=None,
            usability_score=None
        )

        # Return a single dictionary. We embed measure dictionaries with a prefix for clarity.
        result = {
            "participant_name": participant.name,
            "task_name": task.name,
            **obj_measures.to_dict(),
            **subj_measures.to_dict()
        }

        if expert_eval_data:
            result.update(expert_eval_data.to_dict())

        # Optionally include the final output text
        result["final_output"] = final_output

        return result

    def _extract_feedback(self, eval_results):
        lines = []
        if not eval_results["word_count_ok"]:
            lines.append("Word count is out of the specified range.")
        if not eval_results["must_include_ok"]:
            lines.append("You missed one or more required keywords or phrases.")
        analysis_text = eval_results.get("analysis", "")
        truncated_analysis = analysis_text[:300] + "..." if len(analysis_text) > 300 else analysis_text
        lines.append(f"Analysis says: {truncated_analysis}")
        return "\n".join(lines)

    def _simulate_satisfaction(self, final_score):
        """Assigns a simulated satisfaction score based on final output quality."""
        return round(min(5, max(1, (final_score / 20))), 1)
    