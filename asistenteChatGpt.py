from flask import Flask, request, jsonify, render_template_string
from openai import OpenAI

app = Flask(__name__)

# Configuración del cliente OpenAI/Hugging Face
client = OpenAI(
    base_url="https://router.huggingface.co/v1",
    api_key="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"
)

# Función que llama al modelo
def obtener_respuesta(mensaje_usuario):
    try:
        completion = client.chat.completions.create(
            model="deepseek-ai/DeepSeek-V3.1:fireworks-ai",
            messages=[{"role": "user", "content": mensaje_usuario}]
        )
        return completion.choices[0].message
    except Exception as e:
        return f"Error al obtener respuesta: {e}"

# Endpoint API que recibe JSON
@app.route("/preguntar", methods=["POST"])
def preguntar():
    data = request.json
    if not data or "mensaje" not in data:
        return jsonify({"error": "Falta el campo 'mensaje'"}), 400
    
    respuesta = obtener_respuesta(data["mensaje"])
    return jsonify({"respuesta": respuesta})

# Página web simple para probar el modelo
HTML_FORM = """
<!DOCTYPE html>
<html>
<head>
    <title>Chat con Modelo AI</title>
</head>
<body>
    <h2>Prueba el modelo</h2>
    <form method="post" action="/preguntar_web">
        <input type="text" name="mensaje" placeholder="Escribe tu pregunta" size="50"/>
        <input type="submit" value="Enviar"/>
    </form>
</body>
</html>
"""

@app.route("/")
def index():
    return render_template_string(HTML_FORM)

@app.route("/preguntar_web", methods=["POST"])
def preguntar_web():
    mensaje = request.form.get("mensaje")
    if not mensaje:
        return "Por favor escribe un mensaje", 400
    respuesta = obtener_respuesta(mensaje)
    return f"<p><b>Pregunta:</b> {mensaje}</p><p><b>Respuesta:</b> {respuesta}</p><a href='/'>Volver</a>"

if __name__ == "__main__":
    app.run(debug=True)
