
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
    Returns a list of Task objects for the experiment (ChatGPT-5 run).
    All tasks are software-development focused.
    """
    return [
        # 1) Unit Test Generation (pytest)
        Task(
            name="Unit_Test_Generation_Pytest",
            target_spec=(
                "Write pytest unit tests for a Python function:\n"
                "def normalize_email(s: str) -> str:\n"
                "    \"\"\"Trim whitespace, lowercase the local/domain parts, and collapse multiple spaces.\n"
                "    If input is invalid (no '@'), raise ValueError.\n"
                "    \"\"\"\n"
                "    # (Implementation hidden)\n\n"
                "Requirements:\n"
                "- Provide a complete pytest module in a Python code block.\n"
                "- Cover success cases (whitespace, mixed case, multiple spaces), edge cases (empty string),\n"
                "  and error handling (missing '@' -> ValueError).\n"
                "- Use descriptive test names (test_*) and >= 6 asserts total."
            ),
            rubric={
                "word_count_range": (120, 500),
                "must_include": ["```python", "def test_", "assert", "ValueError", "pytest"],
                "evaluation_instructions": (
                    "1) Is a single Python code block present (```python ... ```)?\n"
                    "2) Are there multiple tests with names starting with test_?\n"
                    "3) Do tests assert correct normalization (trim, lowercase, collapse spaces)?\n"
                    "4) Is error handling covered (missing '@' raises ValueError)?\n"
                    "5) Are there at least 6 total asserts?"
                )
            }
        ),

        # 2) REST API Endpoint Documentation
        Task(
            name="API_Documentation_REST",
            target_spec=(
                "Document a REST endpoint for creating a project issue.\n\n"
                "Endpoint: POST /api/v1/projects/{projectId}/issues\n"
                "Auth: Bearer token\n"
                "Request JSON: { \"title\": str (required, 1-120 chars), \"description\": str (optional),\n"
                "               \"labels\": [str], \"assigneeId\": str (optional) }\n"
                "Responses:\n"
                "  201 Created -> returns issue object with id, title, description, labels, assigneeId, createdAt\n"
                "  400 Bad Request -> validation error details\n"
                "  401 Unauthorized\n"
                "  409 Conflict -> duplicate title within project\n\n"
                "Write API docs with the following sections and headings exactly: "
                "'Endpoint', 'Method', 'Auth', 'Request', 'Response Example (201)', 'Status Codes', 'Notes'. "
                "Include a JSON request and a JSON 201 response example."
            ),
            rubric={
                "word_count_range": (250, 600),
                "must_include": [
                    "Endpoint", "Method", "Auth", "Request", "Response Example (201)",
                    "Status Codes", "Notes", "/api/v1/projects/{projectId}/issues", "POST", "Bearer"
                ],
                "evaluation_instructions": (
                    "1) Are all required headings present exactly as specified?\n"
                    "2) Does the request schema include title, description, labels, assigneeId with constraints?\n"
                    "3) Is a 201 JSON example provided with id/title/description/labels/assigneeId/createdAt?\n"
                    "4) Are 400/401/409 status codes listed with explanations?\n"
                    "5) Is Auth documented as Bearer?"
                )
            }
        ),

        # 3) Refactoring Rationale (Code Smell -> Plan)
        Task(
            name="Refactoring_Rationale_SRP",
            target_spec=(
                "Given a module that violates Single Responsibility Principle (SRP) by mixing HTTP routing,\n"
                "domain validation, and database persistence in one 400-line class, write a refactoring rationale.\n"
                "Include the following subsections (use these exact headings):\n"
                "1) Context, 2) Code Smells, 3) Proposed Refactor, 4) Risks & Mitigations, 5) Acceptance Criteria.\n"
                "Mention SRP, cyclomatic complexity, and separation of concerns. Propose extracting Router,\n"
                "Service (validation/business rules), and Repository layers, and adding unit tests around boundaries."
            ),
            rubric={
                "word_count_range": (300, 600),
                "must_include": [
                    "Context", "Code Smells", "Proposed Refactor", "Risks & Mitigations", "Acceptance Criteria",
                    "Single Responsibility Principle", "SRP", "cyclomatic complexity", "separation of concerns",
                    "Repository", "Service", "Router"
                ],
                "evaluation_instructions": (
                    "1) Are all five required headings present?\n"
                    "2) Does it explicitly mention SRP, cyclomatic complexity, and separation of concerns?\n"
                    "3) Does the plan propose extracting Router/Service/Repository layers?\n"
                    "4) Are risks and mitigations specific (e.g., regression risk, migration plan)?\n"
                    "5) Are acceptance criteria testable (e.g., reduced complexity thresholds, passing tests)?"
                )
            }
        ),

        # 4) Commit Message Normalization (Conventional Commits)
        Task(
            name="Commit_Message_Normalization",
            target_spec=(
                "Normalize the following raw commit notes into 5 Conventional Commit messages (one per line),\n"
                "including a scope, a JIRA ticket, and a concise body wrapped at ~72 chars:\n"
                "- add retry and timeouts for client\n"
                "- fix npe on user svc when roles missing\n"
                "- change api to v2 and update docs\n"
                "- tidy code + reformat\n"
                "- temp disable flaky test\n\n"
                "Rules:\n"
                "- Use types: feat, fix, refactor, docs, test, chore as appropriate.\n"
                "- Include a scope in parentheses (e.g., client, usersvc, api, build).\n"
                "- Append the ticket [JIRA-1234] in the subject.\n"
                "- Provide a one-paragraph body for each, explaining the change and rationale."
            ),
            rubric={
                "word_count_range": (150, 500),
                "must_include": ["feat(", "fix(", "docs(", "refactor(", "test(", "[JIRA-1234]"],
                "evaluation_instructions": (
                    "1) Are there exactly 5 commit messages, one per line, in Conventional Commit style?\n"
                    "2) Do messages include a scope in parentheses and the ticket [JIRA-1234]?\n"
                    "3) Is type selection appropriate for each raw note (feat/fix/docs/refactor/test/chore)?\n"
                    "4) Does each include a short wrapped body (~72 chars/line) explaining rationale?\n"
                    "5) Are subjects concise (<= 72 chars) and imperative mood?"
                )
            }
        ),

        # 5) Bug Report Triage (Structured Template)
        Task(
            name="Bug_Report_Triage_Template",
            target_spec=(
                "Create a structured bug report for a login flow regression.\n"
                "Symptoms: users intermittently get 401 after successful OAuth callback.\n"
                "Context: rollout of session store from in-memory to Redis cluster last night.\n"
                "Add the following sections with exact headings:\n"
                "Title, Environment, Severity, Steps to Reproduce, Expected, Actual, Logs, Root Cause Hypothesis,\n"
                "Proposed Fix, Test Plan. Use Severity: High. Include reproducible steps and a minimal log snippet."
            ),
            rubric={
                "word_count_range": (200, 500),
                "must_include": [
                    "Title", "Environment", "Severity", "Steps to Reproduce", "Expected", "Actual",
                    "Logs", "Root Cause Hypothesis", "Proposed Fix", "Test Plan",
                    "Severity: High", "401", "OAuth", "Redis"
                ],
                "evaluation_instructions": (
                    "1) Are all 10 required headings present exactly?\n"
                    "2) Is Severity set to High?\n"
                    "3) Do steps reproduce intermittent 401 after OAuth callback?\n"
                    "4) Do logs plausibly implicate session storage/Redis?\n"
                    "5) Is the proposed fix and test plan concrete and verifiable?"
                )
            }
        ),
    ]
