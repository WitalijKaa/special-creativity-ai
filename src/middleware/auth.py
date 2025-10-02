import os
from dotenv import load_dotenv
load_dotenv()


class AuthMiddleware:
    @staticmethod
    def hugging_face_token() -> str:
        return os.getenv("HF_TOKEN")

    @staticmethod
    def open_ai_token() -> str:
        return os.getenv("OA_TOKEN")