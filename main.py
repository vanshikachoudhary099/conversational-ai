from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from google import genai

from intents import intents
from config import GEMINI_API_KEY

# Configure Gemini (NEW SDK)
client = genai.Client(api_key=GEMINI_API_KEY)

# ---------------- MEMORY ----------------
chat_history = []
MAX_HISTORY = 6
# ---------------------------------------

# Prepare training data
sentences = []
labels = []

for intent, data in intents.items():
    for example in data["examples"]:
        sentences.append(example)
        labels.append(intent)

# Vectorize text
vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(sentences)

# Train model
model = LogisticRegression()
model.fit(X, labels)

print("Chatbot: Hi! Ask me anything (type 'bye' to exit)")

while True:
    user_input = input("You: ").lower()

    # EXIT FIRST
    if user_input == "bye":
        print("Chatbot: Goodbye!")
        break

    user_vec = vectorizer.transform([user_input])
    probs = model.predict_proba(user_vec)[0]
    confidence = max(probs)
    predicted_intent = model.classes_[probs.argmax()]

    if confidence < 0.7:
        # -------- GEMINI WITH MEMORY --------
        chat_history.append(f"User: {user_input}")

        prompt = "\n".join(chat_history)

        response = client.models.generate_content(
            model="models/gemini-flash-lite-latest",
            contents=prompt
        )

        bot_reply = response.text
        print("Chatbot:", bot_reply)

        chat_history.append(f"Bot: {bot_reply}")

        # Keep memory short
        if len(chat_history) > MAX_HISTORY:
            chat_history = chat_history[-MAX_HISTORY:]
        # -----------------------------------

    else:
        responses = intents[predicted_intent]["responses"]
        print("Chatbot:", responses[0])
