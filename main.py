import os
import tempfile
from datetime import datetime
from pathlib import Path

import httpx
import edge_tts
from anthropic import AnthropicVertex
from fastapi import FastAPI, UploadFile, Form
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

app = FastAPI()
app.mount("/static", StaticFiles(directory="static"), name="static")

HTML_PATH = Path(__file__).parent / "templates" / "index.html"

GROQ_API_KEY = os.environ["GROQ_API_KEY"]
GCP_PROJECT_ID = os.environ.get("GCP_PROJECT_ID", "prj-six-aa2bbe69")
GCP_REGION = os.environ.get("GCP_REGION", "us-east5")

history = []

VOICE_MAP = {
    "boy": {"voice": "zh-CN-YunxiaNeural"},
    "girl": {"voice": "zh-CN-XiaoyiNeural", "rate": "-5%", "pitch": "+15Hz"},
}

STYLE_PROMPTS = {
    "direkt": "直接、清楚地回答问题。",
    "entdecken": "用提问的方式引导孩子一起思考和发现答案。",
    "geschichten": "用一个简短的小故事来回答。",
    "emotional": "先关心孩子的感受，再温柔地回答问题。",
}

AGE_PROMPTS = {
    "2-4": "用最简单的词语，只说1到2句话，不超过30个字。",
    "5-10": "可以稍微详细一点，用2到3句话回答，不超过60个字。",
}

LANGUAGE_PROMPTS = {
    "zh": ("用中文回答。", "zh"),
    "de": ("用德语回答。", "de"),
    "en": ("用英语回答。", "en"),
    "fr": ("用法语回答。", "fr"),
    "it": ("用意大利语回答。", "it"),
}


def build_system_prompt(language="zh", age="2-4", style="direkt"):
    if language in LANGUAGE_PROMPTS:
        lang_prompt, _ = LANGUAGE_PROMPTS[language]
    else:
        lang_prompt = f"用{language}回答。"
    age_prompt = AGE_PROMPTS.get(age, AGE_PROMPTS["2-4"])
    style_prompt = STYLE_PROMPTS.get(style, STYLE_PROMPTS["direkt"])
    return (
        f"你是一个温柔的AI助手，专门回答小朋友的问题。"
        f"{lang_prompt}"
        f"{age_prompt}"
        f"语气亲切温暖，不要自称任何身份。"
        f"回答风格：{style_prompt}"
        f"不要用列举、不要用比喻堆叠，直接简单回答。"
    )


def _setup_gcp_credentials():
    creds_path = os.environ.get("GOOGLE_APPLICATION_CREDENTIALS", "")
    if creds_path and os.path.isfile(creds_path):
        return
    creds_json = os.environ.get("GCP_SA_KEY_JSON", "")
    if creds_json:
        tmp = tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False)
        tmp.write(creds_json)
        tmp.close()
        os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = tmp.name


_setup_gcp_credentials()


async def speech_to_text(audio_bytes: bytes) -> str:
    async with httpx.AsyncClient(timeout=30) as client:
        resp = await client.post(
            "https://api.groq.com/openai/v1/audio/transcriptions",
            headers={"Authorization": f"Bearer {GROQ_API_KEY}"},
            files={"file": ("audio.webm", audio_bytes, "audio/webm")},
            data={"model": "whisper-large-v3", "language": "zh"},
        )
        resp.raise_for_status()
    return resp.json()["text"]


def generate_answer(question: str, language="zh", age="2-4", style="direkt") -> str:
    client = AnthropicVertex(project_id=GCP_PROJECT_ID, region=GCP_REGION)
    system_prompt = build_system_prompt(language, age, style)
    max_tok = 80 if age == "2-4" else 150
    message = client.messages.create(
        model="claude-sonnet-4-5",
        max_tokens=max_tok,
        system=system_prompt,
        messages=[{"role": "user", "content": question}],
    )
    return message.content[0].text


async def text_to_speech(text: str, voice_key="boy") -> str:
    output_path = tempfile.mktemp(suffix=".mp3")
    v = VOICE_MAP.get(voice_key, VOICE_MAP["boy"])
    tts = edge_tts.Communicate(text, voice=v["voice"], rate=v.get("rate", "+0%"), pitch=v.get("pitch", "+0Hz"))
    await tts.save(output_path)
    return output_path


@app.get("/")
async def index():
    return HTMLResponse(HTML_PATH.read_text())


@app.post("/ask")
async def ask(
    audio: UploadFile,
    language: str = Form("zh"),
    age: str = Form("2-4"),
    style: str = Form("direkt"),
    voice: str = Form("boy"),
):
    try:
        audio_bytes = await audio.read()
        print(f"[1/3] Audio: {len(audio_bytes)} bytes")
        question = await speech_to_text(audio_bytes)
        print(f"[2/3] Frage: {question}")
        answer = generate_answer(question, language, age, style)
        print(f"[3/3] Antwort: {answer}")
        audio_path = await text_to_speech(answer, voice)
        history.append({
            "question": question,
            "answer": answer,
            "time": datetime.now().strftime("%H:%M"),
            "date": datetime.now().strftime("%Y-%m-%d"),
        })
        return FileResponse(audio_path, media_type="audio/mpeg")
    except Exception as e:
        print(f"[ERROR] {type(e).__name__}: {e}")
        return JSONResponse({"error": str(e)}, status_code=500)


@app.post("/ask-text")
async def ask_text(
    question: str = Form(...),
    language: str = Form("zh"),
    age: str = Form("2-4"),
    style: str = Form("direkt"),
    voice: str = Form("boy"),
):
    try:
        print(f"[1/2] Frage: {question}")
        answer = generate_answer(question, language, age, style)
        print(f"[2/2] Antwort: {answer}")
        audio_path = await text_to_speech(answer, voice)
        history.append({
            "question": question,
            "answer": answer,
            "time": datetime.now().strftime("%H:%M"),
            "date": datetime.now().strftime("%Y-%m-%d"),
        })
        return JSONResponse({"answer": answer, "audio": f"/audio/{Path(audio_path).name}"})
    except Exception as e:
        print(f"[ERROR] {type(e).__name__}: {e}")
        return JSONResponse({"error": str(e)}, status_code=500)


@app.get("/audio/{filename}")
async def get_audio(filename: str):
    path = Path(tempfile.gettempdir()) / filename
    if path.exists():
        return FileResponse(path, media_type="audio/mpeg")
    return JSONResponse({"error": "not found"}, status_code=404)


@app.get("/history")
async def get_history():
    return JSONResponse(list(reversed(history)))
