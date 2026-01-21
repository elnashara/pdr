
class Task:
    """
    Represents a single task with:
      - name: a short label (e.g. 'Creative_Writing')
      - target_spec: a text describing the desired output (the "gold standard")
      - rubric: a dictionary of evaluation criteria (e.g., word_count_range, must_include, etc.)
    """
    def __init__(self, name: str, target_spec: str, rubric: dict):
        self.name = name
        self.target_spec = target_spec
        self.rubric = rubric

    def __repr__(self):
        return f"Task(name={self.name}, target_spec={self.target_spec[:30]}...)"

def get_all_tasks():
    """
    Returns a list of Task objects for the experiment.
    Modify or expand as needed.
    """
    return [
        Task(
            name="Creative_Writing",
            target_spec=(
                "Write a 300-500 word story in the style of Jane Austen, "
                "featuring polite society dialogue and a moral dilemma."
            ),
            rubric={
                "word_count_range": (300, 500),
                "must_include": ["moral dilemma", "dialogue"],
                "evaluation_instructions": (
                    "1) Is the text between 300-500 words?\n"
                    "2) Does it use language reminiscent of Jane Austen?\n"
                    "3) Does it include polite society dialogue?\n"
                    "4) Is there a clear moral dilemma?"
                )
            }
        ),
        Task(
            name="Technical_Explanation",
            target_spec=(
                "Provide a short technical summary of how convolutional neural networks work, "
                "targeted at advanced undergraduates. Mention convolution, filters/kernels, "
                "and backpropagation."
            ),
            rubric={
                "word_count_range": (150, 300),
                "must_include": ["convolution", "filters", "kernels", "backpropagation"],
                "evaluation_instructions": (
                    "1) Is the text between 150-300 words?\n"
                    "2) Does it correctly explain CNNs for advanced undergraduates?\n"
                    "3) Are convolution, filters/kernels, and backpropagation mentioned?\n"
                    "4) Is it suitably technical yet understandable?"
                )
            }
        ),
        Task(
            name="Business_Executive_Summary",
            target_spec=(
                "Write a concise, one-page summary containing Q2 revenue ($2M), costs ($1.2M), "
                "and profit ($0.8M) with a formal executive tone. Highlight key growth areas and "
                "challenges, and end with a recommendation for next quarter."
            ),
            rubric={
                "word_count_range": (200, 400),
                "must_include": ["$2M", "$1.2M", "$0.8M", "growth", "challenges", "recommendation"],
                "evaluation_instructions": (
                    "1) Is it 200-400 words?\n"
                    "2) Does it maintain a formal, executive-level tone?\n"
                    "3) Does it include revenue=$2M, costs=$1.2M, profit=$0.8M?\n"
                    "4) Are growth areas, challenges, and a recommendation stated?"
                )
            }
        )
    ]
