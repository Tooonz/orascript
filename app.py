import os
import traceback

import requests
from flask import Flask, jsonify, request, send_from_directory
from flask_cors import CORS

app = Flask(__name__)
CORS(app)

# Hosted AI settings (set these on Render under Environment, never in the code)
API_KEY = os.environ.get("GEMINI_API_KEY", "")
MODEL = os.environ.get("GEMINI_MODEL", "gemini-3.8-flash")

# Memory buffer for the teacher's lecture (resets when the server restarts)
STORED_LESSON = ""


@app.route("/")
def home_page():
    return send_from_directory(os.path.dirname(os.path.abspath(__file__)), "index.html")


@app.route("/store-lesson", methods=["POST"])
def store_lesson():
    global STORED_LESSON
    try:
        data = request.json or {}
        lesson_text = data.get("lesson", "").strip()
        if not lesson_text:
            return jsonify({"success": False, "error": "No lesson text received"}), 400
        STORED_LESSON = lesson_text[-8000:]  # keep the most recent part
        return jsonify({"success": True, "message": "Lesson synchronized successfully!"})
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500


@app.route("/ask-question", methods=["POST"])
def ask_question():
    try:
        data = request.json or {}
        question = data.get("question", "").strip()
        if not question:
            return jsonify({"success": False, "error": "No question received"}), 400
        if not API_KEY:
            return jsonify({"success": False, "error": "Server has no GEMINI_API_KEY set"}), 500

        prompt = f"""
You are an expert multilingual AI tutor assisting students in a paperless classroom setup.
Use the provided Lesson Context to answer the student's question accurately and clearly.

Context (Teacher's Stored Lecture):
\"\"\"
{STORED_LESSON if STORED_LESSON else "No lesson context recorded yet. Answer using general knowledge."}
\"\"\"

Student Question:
{question}

RULES:
- Automatically detect the language and respond in the EXACT same language.
- If it is a math problem, show clear step-by-step working and state the final answer clearly at the end.
- Keep responses well-structured, clear, and encouraging.
"""
        response = requests.post(
            f"https://generativelanguage.googleapis.com/v1beta/models/{MODEL}:generateContent",
            headers={"x-goog-api-key": API_KEY, "Content-Type": "application/json"},
            json={
                "contents": [{"parts": [{"text": prompt}]}],
                "generationConfig": {"temperature": 0.3, "maxOutputTokens": 800},
            },
            timeout=60,
        )
        ai_data = response.json()
        try:
            parts = ai_data["candidates"][0]["content"]["parts"]
            reply = "".join(p.get("text", "") for p in parts)
            return jsonify({"success": True, "ai_response": reply})
        except (KeyError, IndexError):
            print("AI ERROR:", ai_data)
            msg = (ai_data.get("error") or {}).get("message") if isinstance(ai_data, dict) else None
            return jsonify({"success": False, "error": msg or "AI returned no answer (it may have been blocked or empty)", "details": ai_data}), 500
    except Exception as e:
        print(traceback.format_exc())
        return jsonify({"success": False, "error": str(e)}), 500


@app.route("/transcribe", methods=["POST"])
def transcribe():
    """Backup speech-to-text: browser sends a short WAV clip (base64), Gemini returns the words."""
    try:
        data = request.json or {}
        audio = data.get("audio", "")
        if not audio:
            return jsonify({"success": False, "error": "No audio received"}), 400
        if not API_KEY:
            return jsonify({"success": False, "error": "Server has no GEMINI_API_KEY set"}), 500
        response = requests.post(
            f"https://generativelanguage.googleapis.com/v1beta/models/{MODEL}:generateContent",
            headers={"x-goog-api-key": API_KEY, "Content-Type": "application/json"},
            json={
                "contents": [{"parts": [
                    {"text": "Transcribe this audio exactly as spoken, in the language spoken. "
                             "Output ONLY the transcript text. If there is no clear speech, output nothing."},
                    {"inline_data": {"mime_type": "audio/wav", "data": audio}},
                ]}],
                "generationConfig": {"temperature": 0},
            },
            timeout=60,
        )
        ai_data = response.json()
        try:
            parts = ai_data["candidates"][0]["content"]["parts"]
            text = "".join(p.get("text", "") for p in parts).strip()
            return jsonify({"success": True, "text": text})
        except (KeyError, IndexError):
            print("TRANSCRIBE ERROR:", ai_data)
            msg = (ai_data.get("error") or {}).get("message") if isinstance(ai_data, dict) else None
            return jsonify({"success": False, "error": msg or "No transcript returned"}), 500
    except Exception as e:
        print(traceback.format_exc())
        return jsonify({"success": False, "error": str(e)}), 500


if __name__ == "__main__":
    app.run(debug=True, port=int(os.environ.get("PORT", 5000)))
