from src.requests.request_ai import RequestAi
from src.models.poetry.chapter import Chapter


class RequestTranslate(RequestAi):
    content: Chapter
