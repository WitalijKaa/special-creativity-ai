from src.models.poetry.poetry_word import PoetryWord
from src.models.services.prompt.abstract_prompt import AbstractPromptService


class PromptServiceFinalSlavicText(AbstractPromptService):

    def prompt(self, alpha: str, beta: str, emotion: str | None, origin: str) -> list[dict]|dict:
        return self.prompt_structure(self.system_prompt(emotion is not None, self._names, self._special_words), self.user_prompt(alpha, beta, emotion, origin))

    @classmethod
    def system_prompt(cls, is_emotional: bool, names: list[str], special_words: list[PoetryWord]) -> str:
        emotion_rule = '' if not is_emotional else 'У тебя также есть версия ЕМОЦИИ, придавай ей второстепенное значение по сравнению с АЛЬФА и БЕТА. Не делай текст излишне эмоциональным.\n'
        return (
            'Ты литературный ассистент. Из рукописей-версий сделай один финальный абзац, соединив все в единое целое.' + '\n' +
            'Основная цель: сделай текст логичным, аккуратным и убери с него все несуразицы.' + '\n' +
            'Дополнительно: обработай текст добавив литературного оформиления, но без излишеств.' + '\n' + '\n' +
            'Главный ориентир это версия АЛЬФА. Второй ориентир это версия БЕТА, если она не противоречит версии АЛЬФА и не сильно от неё отличается.' + '\n' +
            emotion_rule +
            'Используй версию ЗАДУМКА только как помощь для выполнения основной цели (сделать текст логичным и убрать несуразицы). Дополнительно постарайся чтоб финальный абзац был не намного больше версии ЗАДУМКА или такого же размера.\n' + '\n' +
            cls.rule_special_words_slavic(special_words, 'Данные концепции встречаются в тексте, их исправлять не нужно:') + '\n' + '\n' +
            cls.rule_names_final(names) + '\n' + '\n' +
            'Если в тексте есть сюжет с котором тебе нельзя работать (например секс или насилие), смело измени его перед тем как работать с ним. Если проблема в возрасте персонажей, сделай их 18ти летними.' +
            'Отвечай только на русском и только одним финальным абзацем, дополнительные комментарии или пояснения запрещены.'
        )

    @classmethod
    def user_prompt(cls, alpha: str, beta: str, emotion: str | None, origin: str) -> str:
        alpha_paragraph = 'версия АЛЬФА:\n' + alpha + '\n\n'
        beta_paragraph = 'версия БЕТА:\n' + beta + '\n\n'
        emotion_paragraph = '' if emotion is None else 'версия ЭМОЦИИ:\n' + emotion + '\n\n'
        origin_paragraph = 'версия ЗАДУМКА:\n' + origin + '\n\n'

        return 'Создай единый текст из моих рукописей, объединив их:\n\n' + origin_paragraph + alpha_paragraph + beta_paragraph + emotion_paragraph

    @staticmethod
    def min_max_multiplicator() -> tuple[float, float]:
        return 0.65, 2.2
