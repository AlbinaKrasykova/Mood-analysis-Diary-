from flask import Flask, request, render_template
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch

# Initialize Flask app
app = Flask(__name__)

# Load tokenizer and model
model_name = "michellejieli/emotion_text_classifier"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(model_name)

# Emotion labels
emotion_labels = ['anger', 'fear', 'joy', 'love', 'sadness', 'surprise']

def classify_emotion(text):
    # Tokenize input
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True)

    # Perform inference
    with torch.no_grad():
        outputs = model(**inputs)

    # Get logits and apply softmax
    logits = outputs.logits
    probabilities = torch.softmax(logits, dim=-1)

    # Map class indices to emotion labels
    predicted_emotion = emotion_labels[torch.argmax(probabilities).item()]
    confidence = torch.max(probabilities).item()
    print(text, predicted_emotion, confidence)
    return predicted_emotion, confidence

@app.route('/', methods=['GET', 'POST'])
def index():
    predicted_emotion, confidence = None, None
    if request.method == 'POST':
        user_text = request.form['user_text']
        print('user_text', user_text)
        predicted_emotion, confidence = classify_emotion(user_text)
        print(user_text, predicted_emotion, confidence)
    
    return render_template('index.html', predicted_emotion=predicted_emotion, confidence=confidence)

if __name__ == '__main__':
    app.run(debug=True)
