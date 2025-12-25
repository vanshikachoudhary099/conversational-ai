\# 🤖 Conversational AI Chatbot



This is a \*\*Conversational AI chatbot\*\* that combines traditional Machine Learning with a Large Language Model (Google Gemini).



The chatbot first tries to understand user intent using an ML classifier.  

If the confidence is low, it intelligently falls back to Gemini to answer any open-ended question.



---



\## 🚀 Features



\- Intent classification using \*\*TF-IDF + Logistic Regression\*\*

\- Fallback to \*\*Google Gemini API\*\* for general conversations

\- Conversation memory (context-aware replies)

\- Web-based UI built with \*\*Streamlit\*\*

\- Secure API key handling (environment variables)

\- Graceful handling of API rate limits



---



\## 🧠 How It Works

User Input

↓

Intent Classifier (ML)

↓

High confidence → Predefined response

Low confidence → Gemini LLM

↓

Conversation Memory

↓

Response to User





---



\## 🛠 Tech Stack



\- Python

\- scikit-learn

\- Google Gemini API (google-genai)

\- Streamlit

\- Git \& GitHub



---



\## 🔑 Gemini API Key Setup



This project requires a \*\*Google Gemini API key\*\*.



\### Step 1: Get an API key

1\. Go to https://aistudio.google.com/

2\. Sign in with your Google account

3\. Create a new API key



\### Step 2: Set the API key as an environment variable



\#### Windows (PowerShell)

```powershell

setx GEMINI\_API\_KEY "your\_api\_key\_here"





Restart the terminal after running this command.



macOS / Linux

export GEMINI\_API\_KEY="your\_api\_key\_here"





⚠️ Do NOT commit API keys to GitHub



▶️ How to Run the Project

pip install -r requirements.txt

streamlit run app.py





Open the browser link shown in the terminal.

conversational\_ai/

├── app.py          # Streamlit web chatbot

├── main.py         # Terminal-based chatbot

├── intents.py      # Intent definitions

├── requirements.txt

├── README.md





