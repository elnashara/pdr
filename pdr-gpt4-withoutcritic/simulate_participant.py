import openai
import os

class Participant:
    """
    Represents a simulated participant (persona).
    Each participant has:
      - A name (e.g., Participant_A)
      - A persona_description (e.g., 'expert in business writing...')
      - A preferred GPT-4o model (default 'gpt-4o').

    The generate_output method simulates how the participant
    responds to a user instruction (prompt) using GPT-4o or another model.
    """

    def __init__(self, name: str, persona_description: str, model: str = "gpt-4o"):
        self.name = name
        self.persona_description = persona_description
        self.model = model

    def generate_output(self, user_instruction: str, temperature: float = 0.7, max_tokens: int = 1000) -> str:
        """
        Simulates how this participant would respond to a given user instruction
        using the GPT-4o model with the participant's persona as the 'system' role.
        """
        messages = [
            {
                "role": "system",
                "content": f"You are {self.name}. {self.persona_description}"
            },
            {
                "role": "user",
                "content": user_instruction
            }
        ]

        try:
            response = openai.ChatCompletion.create(
                model=self.model,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens
            )
            return response["choices"][0]["message"]["content"]
        except Exception as e:
            return f"Error calling GPT-4o API: {e}"
