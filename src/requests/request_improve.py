from src.requests.request_ai import RequestAi
from src.models.poetry.chapter import Chapter


class RequestImprove(RequestAi):
    content: Chapter
    separator: str
