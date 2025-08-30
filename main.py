import os
from flask import Flask, request, render_template_string
from PIL import Image
import requests
from transformers import BlipProcessor, BlipForConditionalGeneration
from huggingface_hub import InferenceClient

app = Flask(__name__)

# Inicializamos BLIP
processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-large")
model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-large")

# Inicializamos Hugging Face InferenceClient
hf_client = InferenceClient(
    provider="fireworks-ai",
    api_key=os.environ.get("HF_TOKEN"),  # asegurate de tener la variable de entorno HF_TOKEN
)

# HTML simple para subir imágenes
HTML_PAGE = """
<!doctype html>
<title>Sube una imagen</title>
<h1>Sube una imagen para obtener la descripción y calorías</h1>
<form method=post enctype=multipart/form-data>
  <input type=file name=file>
  <input type=submit value=Upload>
</form>
{% if description %}
<h2>Descripción generada:</h2>
<p>{{ description }}</p>
<h2>Calorías estimadas:</h2>
<p>{{ calories }}</p>
{% endif %}
"""

@app.route("/", methods=["GET", "POST"])
def upload_image():
    description = None
    calories = None

    if request.method == "POST":
        file = request.files["file"]
        if file:
            image = Image.open(file).convert("RGB")
            
            # 1️⃣ Generamos descripción con BLIP
            text_prompt = "a photography of"
            inputs = processor(image, text_prompt, return_tensors="pt")
            out = model.generate(**inputs)
            description = processor.decode(out[0], skip_special_tokens=True)

            # 2️⃣ Preguntamos calorías con Hugging Face
            prompt = f"How many calories does this contain? {description}"
            completion = hf_client.chat.completions.create(
                model="deepseek-ai/DeepSeek-V3.1",
                messages=[{"role": "user", "content": prompt}],
            )
            calories = completion.choices[0].message["content"]

    return render_template_string(HTML_PAGE, description=description, calories=calories)

if __name__ == "__main__":
    app.run(debug=True)
