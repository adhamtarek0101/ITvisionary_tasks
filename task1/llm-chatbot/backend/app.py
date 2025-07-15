from flask import Flask, request, jsonify
import time
import torch
import os
from dotenv import load_dotenv
from transformers import AutoTokenizer, AutoModelForCausalLM

# we load Hugging Face token
load_dotenv()
HF_TOKEN = os.getenv("HF_TOKEN")

app = Flask(__name__)

MODEL_NAME = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"

# then we load tokenizer and model
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, token=HF_TOKEN)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    device_map="auto",  
    torch_dtype=torch.float32,  # For better compatibility for my pc
    token=HF_TOKEN
)

@app.route("/")
def home():
    return jsonify({"message": "TinyLlama backend is running."})

@app.route("/chat", methods=["POST"])
def chat():
    data = request.get_json()
    user_input = data.get("user_input", "").strip()

    if not user_input:
        return jsonify({"response": "No input provided.", "time": 0})

    # few-shot prompt (in-context learning i added it for better understanding)
    few_shot_prompt = """<|system|>
You are a helpful assistant.
<|user|>
Who are you?
<|assistant|>
I am a helpful virtual assistant created to answer your questions and help you with tasks.
<|user|>
How are you?
<|assistant|>
I'm doing great! Thanks for asking. How can I assist you today?
<|user|>
Can you help me with something?
<|assistant|>
Absolutely! Just tell me what you need help with."""

    # the prompt we use with TinyLlama instruction format
    prompt = f"{few_shot_prompt}\n<|user|>\n{user_input}\n<|assistant|>\n"

    try:
        start = time.time()
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
        outputs = model.generate(
            **inputs,
            max_new_tokens=150,
            temperature=0.0,
            top_p=0.9,
            repetition_penalty=1.2,
            pad_token_id=tokenizer.eos_token_id
        )
        decoded = tokenizer.decode(outputs[0], skip_special_tokens=True)
        reply = decoded.split("<|assistant|>")[-1].strip()
        elapsed = round(time.time() - start, 2)

        return jsonify({"response": reply, "time": elapsed})

    except Exception as e:
        return jsonify({"response": f"Error: {str(e)}"}), 500

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8500, debug=False)
