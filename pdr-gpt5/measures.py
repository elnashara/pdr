class ObjectiveMeasures:
    """
    Tracks objective measures, such as:
      - iteration_count
      - time_spent (seconds)
      - final_score
      - satisfaction_score (simulated measure of participant satisfaction)
    You can expand this with additional metrics (token usage, cost, etc.).
    """
    def __init__(self, iteration_count: int, time_spent_sec: float, final_score: float, satisfaction_score: float = None):
        self.iteration_count = iteration_count
        self.time_spent_sec = time_spent_sec
        self.final_score = final_score
        self.satisfaction_score = satisfaction_score

    def to_dict(self):
        return {
            "iteration_count": self.iteration_count,
            "time_spent_sec": self.time_spent_sec,
            "final_score": self.final_score,
            "satisfaction_score": self.satisfaction_score
        }



class SubjectiveMeasures:
    """
    Holds subjective/user-reported measures (if real participants are involved).
    For instance:
      - perceived_quality (1-5 or 1-7 Likert)
      - usability_score (1-5 or 1-7 Likert)
      - other questionnaire data
    In a real study, you'd collect these from a user form or survey.
    """
    def __init__(self, perceived_quality: float = None, usability_score: float = None):
        self.perceived_quality = perceived_quality
        self.usability_score = usability_score

    def to_dict(self):
        return {
            "perceived_quality": self.perceived_quality,
            "usability_score": self.usability_score
        }


class ExpertEvaluation:
    """
    Optionally captures domain-expert or advanced GPT-4o-based evaluation
    of correctness, style, completeness, etc. for specialized tasks.
    You could store multiple ratings, e.g., correctness_score, style_score, etc.
    """
    def __init__(self, correctness_score: float = None, style_score: float = None, notes: str = ""):
        self.correctness_score = correctness_score
        self.style_score = style_score
        self.notes = notes

    def to_dict(self):
        return {
            "expert_correctness_score": self.correctness_score,
            "expert_style_score": self.style_score,
            "expert_notes": self.notes
        }
