import streamlit as st
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from google import genai

from intents import intents
import os


# ---------------- PAGE SETUP ----------------
st.set_page_config(page_title="Conversational AI", layout="centered")
st.title("🤖 Conversational AI Chatbot")

# ---------------- GEMINI SETUP ----------------
client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))


# ---------------- MEMORY ----------------
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

MAX_HISTORY = 6

# ---------------- TRAIN INTENT MODEL ----------------
sentences = []
labels = []

for intent, data in intents.items():
    for example in data["examples"]:
        sentences.append(example)
        labels.append(intent)

vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(sentences)

model = LogisticRegression()
model.fit(X, labels)

# ---------------- CHAT UI ----------------
user_input = st.text_input("You:", placeholder="Ask me anything...")

if user_input:
    if user_input.lower() == "bye":
        st.write("👋 Goodbye!")
    else:
        user_vec = vectorizer.transform([user_input.lower()])
        probs = model.predict_proba(user_vec)[0]
        confidence = max(probs)
        predicted_intent = model.classes_[probs.argmax()]

        if confidence < 0.7:
            st.session_state.chat_history.append(f"User: {user_input}")
            prompt = "\n".join(st.session_state.chat_history)

            response = client.models.generate_content(
                model="models/gemini-flash-lite-latest",
                contents=prompt
            )

            bot_reply = response.text
            st.session_state.chat_history.append(f"Bot: {bot_reply}")

            if len(st.session_state.chat_history) > MAX_HISTORY:
                st.session_state.chat_history = st.session_state.chat_history[-MAX_HISTORY:]

        else:
            bot_reply = intents[predicted_intent]["responses"][0]

        # DISPLAY CHAT
        for msg in st.session_state.chat_history:
            if msg.startswith("User:"):
                st.markdown(f"**You:** {msg[5:]}")
            else:
                st.markdown(f"**Bot:** {msg[4:]}")
