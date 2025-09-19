from dotenv import load_dotenv
load_dotenv()

from fastapi import FastAPI
from src.routes.web import web_routes_init

app = FastAPI()

def main():
    web_routes_init(app)

main()
