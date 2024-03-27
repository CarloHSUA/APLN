from fastapi import FastAPI
from fastapi.responses import HTMLResponse
import os
from utility.response import get_response 


# App
app = FastAPI()
front_path = "../frontend/"


# Routes 
@app.get("/", response_class=HTMLResponse)
def read_html():
    with open(front_path + "index.html", "r") as file:
        html_content = file.read()
    with open(front_path + "style.css", "r") as css_file:
        css_content = css_file.read()

    # Insertar el contenido del CSS en el HTML
    html_content = html_content.replace('</head>', f'<style>{css_content}</style></head>')
    return HTMLResponse(content=html_content, status_code=200)


@app.get("/response")
def read_response(query: str):
    response = get_response(query, num_resposnes=1)
    return {'msg': response}


if __name__ == "__main__":
    import uvicorn
    os.system("uvicorn --host 127.0.0.1 --port 8000 main:app --reload")