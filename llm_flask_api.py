from flask import Flask, request, jsonify
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

# Load the model and tokenizer
model_name = "gpt2"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)

# Move model to GPU if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# Create Flask app
app = Flask(__name__)

# GET route to show a simple welcome message
@app.route('/', methods=['GET'])
def home():
    return "Welcome to the LLM API. Send a POST request with a prompt."


@app.route('/generate', methods=['GET'])
def generate():
    prompt = request.args.get("prompt", "Once upon a time")
    # max_length = request.args.get("max_length", 50)
    max_length = 50

    try:
        max_length = int(max_length)
    except ValueError:
        return jsonify({"error": "max_length must be an integer"}), 400

    # Encode prompt and generate
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    outputs = model.generate(**inputs, max_length=max_length)
    generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)

    return jsonify({"generated_text": generated_text})


# # POST route to generate text from prompt
# @app.route('/generate', methods=['GET'])
# def generate():
#     data = request.json
#     if data is None:
#         return jsonify({"error": "No JSON data provided"}), 400

#     prompt = data.get("prompt", "Once upon a time")
#     max_length = data.get("max_length", 50)

#     # Encode prompt and generate
#     inputs = tokenizer(prompt, return_tensors="pt").to(device)
#     outputs = model.generate(**inputs, max_length=max_length)
#     generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)

#     return jsonify({"generated_text": generated_text})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8000)
