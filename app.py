import streamlit as st
import os
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from google import genai

from intents import intents

# ---------------- PAGE CONFIG ----------------
st.set_page_config(page_title="Conversational AI", layout="centered")

# ---------------- SIDEBAR ----------------
st.sidebar.title("🤖 Conversational AI")
st.sidebar.markdown("""
**Features**
- Intent classification (ML)
- Gemini LLM fallback
- Conversation memory
- Free-tier safe model

Built with ❤️ using Python
""")

if st.sidebar.button("🧹 Clear Chat"):
    st.session_state.chat_history = []
    st.experimental_rerun()

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

# ---------------- MAIN UI ----------------
st.title("💬 Chat with AI")

user_input = st.text_input("Type your message and press Enter", key="input")

if user_input:
    if user_input.lower() == "bye":
        st.session_state.chat_history.append(("bot", "👋 Goodbye!"))
    else:
        user_vec = vectorizer.transform([user_input.lower()])
        probs = model.predict_proba(user_vec)[0]
        confidence = max(probs)
        predicted_intent = model.classes_[probs.argmax()]

        if confidence < 0.7:
            st.session_state.chat_history.append(("user", user_input))

            prompt = ""
            for role, msg in st.session_state.chat_history:
                prefix = "User" if role == "user" else "Bot"
                prompt += f"{prefix}: {msg}\n"

            response = client.models.generate_content(
                model="models/gemini-flash-lite-latest",
                contents=prompt
            )

            bot_reply = response.text
            st.session_state.chat_history.append(("bot", bot_reply))

            if len(st.session_state.chat_history) > MAX_HISTORY:
                st.session_state.chat_history = st.session_state.chat_history[-MAX_HISTORY:]
        else:
            st.session_state.chat_history.append(("user", user_input))
            st.session_state.chat_history.append(
                ("bot", intents[predicted_intent]["responses"][0])
            )

# ---------------- DISPLAY CHAT ----------------
for role, message in st.session_state.chat_history:
    if role == "user":
        st.markdown(f"🧑 **You:** {message}")
    else:
        st.markdown(f"🤖 **Bot:** {message}")
