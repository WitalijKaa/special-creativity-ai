from src.models.llm.prompt_service import PromptService
from src.models.llm.ai_provider import AiProvider

def action_translate_eng(text: str) -> str:
    prompt = PromptService.prompt_translate_slavic_to_english(text)
    ai = AiProvider()
    return ai.answer_vs_prompt(prompt)
