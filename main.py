import os
from flask import Flask, request
from huggingface_hub import InferenceClient

app = Flask(__name__)

# Inicializa el cliente de Hugging Face
client = InferenceClient(
    provider="fireworks-ai",
    api_key=os.environ.get("HF_TOKEN"),  # Asegúrate de tener HF_TOKEN en variables de entorno
)

HTML_PAGE = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <title>Prueba Hugging Face</title>
</head>
<body>
    <h1>Prueba tu modelo Hugging Face</h1>
    <form method="POST">
        <input type="text" name="question" placeholder="Escribe tu pregunta aquí" style="width:300px;">
        <button type="submit">Enviar</button>
    </form>
    {answer_block}
</body>
</html>
"""

@app.route("/", methods=["GET", "POST"])
def index():
    answer_block = ""
    if request.method == "POST":
        question = request.form.get("question")
        if question:
            completion = client.chat.completions.create(
                model="deepseek-ai/DeepSeek-V3.1",
                messages=[{"role": "user", "content": question}],
            )
            answer = completion.choices[0].message["content"]
            answer_block = f"<h2>Respuesta:</h2><p>{answer}</p>"
    return HTML_PAGE.format(answer_block=answer_block)

if __name__ == "__main__":
    app.run(debug=True)
