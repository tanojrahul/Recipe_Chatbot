from flask import Flask, render_template, request
from huggingface_hub import InferenceClient

app = Flask(__name__)

# Hugging Face API Key and Model Setup
HF_API_KEY = "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"
MODEL_NAME = "meta-llama/Llama-3.2-11B-Vision-Instruct"
client = InferenceClient(api_key=HF_API_KEY)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/ask', methods=['POST'])
def ask():
    user_question = request.form.get('question')
    if not user_question:
        return render_template('index.html', error="Please enter a question.")

    # Send the question to the Hugging Face model
    messages = [{"role": "user", "content": user_question}]
    try:
        response = client.chat.completions.create(
            model=MODEL_NAME, messages=messages, max_tokens=500
        )
        answer = response.choices[0].message["content"]
    except Exception as e:
        answer = f"Error occurred: {e}"

    return render_template('result.html', question=user_question, answer=answer)

if __name__ == '__main__':
    app.run(debug=True)
