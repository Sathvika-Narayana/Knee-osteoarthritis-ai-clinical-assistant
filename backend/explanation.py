'''
import requests

OLLAMA_URL = "http://localhost:11434/api/generate"
OLLAMA_MODEL = "tinyllama"


def generate_explanation(prediction):

    prompt = f"""
Explain knee osteoarthritis severity level {prediction}.
Include:
- Clinical meaning
- Common symptoms
- Treatment suggestions
Keep it professional and structured.
"""

    try:
        response = requests.post(
            OLLAMA_URL,
            json={
                "model": OLLAMA_MODEL,
                "prompt": prompt,
                "stream": False
            },
            timeout=120
        )

        print("Ollama status:", response.status_code)

        if response.status_code == 200:

            data = response.json()

            print("Ollama raw:", data)

            if "response" in data:
                return data["response"]

            else:
                return "AI explanation unavailable"

        else:
            print("Ollama error:", response.text)
            return "AI explanation unavailable"

    except Exception as e:
        print("Ollama connection error:", str(e))
        return "AI explanation unavailable"

'''
# Gemini api
import os
import google.generativeai as genai
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Gemini API setup
API_KEY = os.getenv("GEMINI_API_KEY")

genai.configure(api_key=API_KEY)

# Fast + good model
MODEL_NAME = "gemini-2.5-flash"


def generate_explanation(prediction):

    prompt = f"""
You are an orthopedic clinical AI assistant.

Generate a PROFESSIONAL medical explanation for:

Knee Osteoarthritis severity level {prediction}.

Format the output EXACTLY like a clinical report:

Title: KOA Clinical Summary

Sections:
1. Clinical Meaning
2. Typical Symptoms
3. Recommended Management

Rules:
- Do NOT use special bullet symbols like • or ***
- Use clean paragraphs or numbered headings.
- Keep language medical and professional.
- Avoid unnecessary decoration.
"""

    try:

        model = genai.GenerativeModel(MODEL_NAME)

        response = model.generate_content(prompt)

        if response and response.text:
            return response.text

        else:
            return "AI explanation unavailable"

    except Exception as e:

        print("Gemini connection error:", str(e))
        return "AI explanation unavailable"