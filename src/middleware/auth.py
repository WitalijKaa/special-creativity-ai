import os

class AuthMiddleware:
    @staticmethod
    def hugging_face_token() -> str:
        return os.getenv("HF_TOKEN")