import sys; sys.dont_write_bytecode = True
from dotenv import load_dotenv
load_dotenv()
from src.models.basic_logger import aLog
from pathlib import Path

from fastapi import FastAPI
from src.routes.web import web_routes_init

app = FastAPI()

def main():
    aLog.init(Path(__file__).resolve().parents[0] / 'logs')
    web_routes_init(app)

main()
