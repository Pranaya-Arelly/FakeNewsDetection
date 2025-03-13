from flask import Flask, render_template, request, session
import torch
from transformers import BertTokenizer, BertForSequenceClassification

# 🔹 Change model path here for future models
MODEL_PATH = "models/fine_tuned_tinybert/"

# Load Tokenizer and Model
tokenizer = BertTokenizer.from_pretrained(MODEL_PATH)
model = BertForSequenceClassification.from_pretrained(MODEL_PATH)
model.eval()  # Set to evaluation mode

app = Flask(__name__)
app.secret_key = "super_secret_key"  # Used for session storage

# Prediction Function
def predict_news(text):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True)
    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits
        prediction = torch.argmax(logits, dim=1).item()
    
    return "Real News" if prediction == 0 else "Fake News"

@app.route("/", methods=["GET", "POST"])
def index():
    result = None
    text = ""
    bg_color = "white"

    # Ensure session history is initialized
    if "history" not in session:
        session["history"] = []

    if request.method == "POST":
        text = request.form["news_text"]
        result = predict_news(text)

        # Change background color based on prediction
        bg_color = "green" if result == "Real News" else "red"

        # Store in session history
        session["history"].append({"text": text, "result": result})
        session.modified = True  # Ensure Flask updates session

    return render_template("index.html", result=result, text=text, bg_color=bg_color, history=session["history"])

if __name__ == "__main__":
    app.run(debug=True)
