import os
import numpy as np
import pickle
from flask import Flask, request, render_template
from dotenv import load_dotenv
import requests

load_dotenv()
AZURE_ENDPOINT = os.getenv("AZURE_ENDPOINT")
AZURE_KEY = os.getenv("AZURE_KEY")
AZURE_DEPLOYMENT = os.getenv("AZURE_DEPLOYMENT")
AZURE_API_VERSION = os.getenv("AZURE_API_VERSION", "2024-02-15")

app = Flask(__name__)

with open("model.pkl", "rb") as f:
    model = pickle.load(f)

def azure_chat(messages, temperature=0.5, max_tokens=1000):
    """Call Azure OpenAI chat completion API."""
    url = f"{AZURE_ENDPOINT}"
    headers = {"Content-Type": "application/json", "api-key": AZURE_KEY}
    payload = {"messages": messages, "temperature": temperature, "max_tokens": max_tokens}
    try:
        resp = requests.post(url, headers=headers, json=payload, timeout=30)
        resp.raise_for_status()
        return resp.json()["choices"][0]["message"]["content"]
    except requests.exceptions.RequestException as e:
        return f"Sorry, I couldn’t reach the assistant service. ({e})"

def handle_agent(user_msg, numeric_features=None, conversation_history=None):
    """
    Handles user messages and numeric data.
    - numeric_features: list of 7 numbers from the form
    """
    if conversation_history is None:
        conversation_history = []

    nums = [float(s) for s in user_msg.replace(',', ' ').split() if s.replace('.', '', 1).isdigit()]
    if len(nums) == 7:
        features = [np.array(nums)]
        pred = model.predict(features)
        crop = pred[0]

        response = (
            f"The predicted crop is **{crop}**.\n"
            "- Optimal sowing time\n"
            "- Irrigation schedule\n"
            "- Fertilizer recommendations\n"
            "- Pest management tips\n\n"
            "Ask me for more details on any step!"
        )
        return response, conversation_history

    elif len(nums) > 0:
        return f"Please provide exactly 7 numeric features for prediction; you provided {len(nums)}.", conversation_history

    system_prompt = {"role": "system", "content": "You are a helpful agricultural assistant. Give the best ever solution in a concise way"}
    user_prompt = {"role": "user", "content": user_msg}
    messages = conversation_history + [system_prompt, user_prompt] if conversation_history else [system_prompt, user_prompt]
    answer = azure_chat(messages)
    conversation_history.append(user_prompt)
    conversation_history.append({"role": "assistant", "content": answer})
    return answer, conversation_history


@app.route("/")
def home():
    return render_template("index.html")

@app.route("/agent", methods=["POST"])
def agent():
    features = request.form.getlist("feature") 
    features = [float(x) for x in features if x.strip() != ""]

    user_query = request.form.get("query", "")

    response, _ = handle_agent(user_query, features)
    return render_template("index.html", agent_response=response)

@app.route("/predict", methods=["POST"])
def predict():
    float_features = [float(x) for x in request.form.values()]
    if len(float_features) != 7:
        return render_template("index.html", predicted_text=f"Please provide exactly 7 features, you provided {len(float_features)}.")
    
    features = [np.array(float_features)]
    pred = model.predict(features)
    return render_template("index.html", predicted_text=f"The Predicted Crop is {pred[0]}")

if __name__ == "__main__":
    app.run(debug=True)
