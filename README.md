# SPECIAL CREATIVITY is an engine for my hobby — writing science fiction in a very specific genre.

I have created a calculator + accounting notebook to track lives and events of personages of my book. Its a very simple project, but it has good code examples and integration with LLM that u can run on local PC. Very personal stuff in the fact.

----

# special-creativity-python

## install python 3

#### pip install pydantic fastapi[standard] python-dotenv rich openai transformers accelerate bitsandbytes==0.47.* sentencepiece==0.2.* protobuf==6.32.*

#### for video_card 4060
#### pip install torch torchvision --index-url https://download.pytorch.org/whl/cu129

#### py -m venv .ex_env --system-site-packages
#### .\.ex_env\Scripts\Activate.ps1

#### fastapi dev .\main.py
#### uvicorn main:app --host 127.0.0.1 --port 8011 --reload

## additional info

#### for france
#### pip install sentencepiece protobuf
