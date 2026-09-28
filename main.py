import asyncio
import json
import os
import tempfile
from pathlib import Path

import httpx
import edge_tts
from anthropic import AnthropicFoundry
from fastapi import FastAPI, UploadFile, Form
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

app = FastAPI()
app.mount("/static", StaticFiles(directory="static"), name="static")

HTML_PATH = Path(__file__).parent / "templates" / "index.html"

GROQ_API_KEY = os.environ["GROQ_API_KEY"]

def normalize_question(question: str) -> str:
    """Interpunktion und Leerzeichen entfernen, damit '为什么天是蓝的?' und
    '为什么天是蓝的' als dieselbe Frage erkannt werden."""
    return "".join(ch for ch in question.lower() if ch.isalnum())


def find_previous_answers(question: str, conversation: list) -> list:
    """Frühere Antworten auf dieselbe Frage aus dem laufenden Gespräch holen.
    Das Gespräch schickt das Gerät mit, deshalb braucht es keine Datenbank."""
    question_norm = normalize_question(question)
    return [
        turn["answer"] for turn in conversation
        if normalize_question(turn.get("question", "")) == question_norm
    ]


VOICE_MAP = {
    "zh": {
        "boy": {"voice": "zh-CN-YunxiaNeural", "pitch": "+15Hz"},
        "girl": {"voice": "zh-CN-XiaoyiNeural", "rate": "-5%", "pitch": "+15Hz"},
    },
    "de": {
        "boy": {"voice": "de-DE-ConradNeural", "pitch": "+18Hz"},
        "girl": {"voice": "de-DE-AmalaNeural"},
    },
    "en": {
        "boy": {"voice": "en-US-GuyNeural", "pitch": "+18Hz"},
        "girl": {"voice": "en-US-AnaNeural"},
    },
    "fr": {
        "boy": {"voice": "fr-FR-HenriNeural", "pitch": "+18Hz"},
        "girl": {"voice": "fr-FR-DeniseNeural"},
    },
    "it": {
        "boy": {"voice": "it-IT-DiegoNeural", "pitch": "+18Hz"},
        "girl": {"voice": "it-IT-ElsaNeural"},
    },
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

# Bei wiederholten Fragen braucht eine Analogie oder Geschichte etwas mehr Platz.
AGE_PROMPTS_REPEAT = {
    "2-4": "用最简单的词语，3到4句话，不超过60个字。",
    "5-10": "用4到5句话回答，不超过120个字。",
}

LANGUAGE_PROMPTS = {
    "zh": ("用中文回答。", "zh"),
    "de": ("用德语回答。", "de"),
    "en": ("用英语回答。", "en"),
    "fr": ("用法语回答。", "fr"),
    "it": ("用意大利语回答。", "it"),
}


REPEAT_PROMPTS = {
    "analogie": (
        "孩子已经问过这个问题一次了，说明上次的解释没听懂。"
        "这次换一种说法：用孩子每天都能看到、摸到的东西打一个比方"
        "（吃的东西、玩具、小动物、自己的身体）。"
        "比方要贴近孩子的想法，但事实必须依然科学正确，不能为了简单就说错。"
    ),
    "geschichte": (
        "孩子已经反复问过这个问题好几次了，说明前几次都还没听懂。"
        "这次用一个很短的小故事，或者一个他自己动手就能试试看的小例子来解释。"
        "说法要比上次更接近孩子的思维方式，但内容必须依然科学正确，不可以编造。"
    ),
}


def pick_strategy(ask_count: int, style: str) -> str:
    """Beim ersten Mal der von den Eltern gewählte Stil, danach Eskalation:
    Erklärung -> Analogie -> Geschichte."""
    if ask_count >= 3:
        return "geschichte"
    if ask_count == 2:
        return "analogie"
    return style


def build_system_prompt(language="zh", age="2-4", style="direkt", strategy=None, previous_answers=None):
    if language in LANGUAGE_PROMPTS:
        lang_prompt, _ = LANGUAGE_PROMPTS[language]
    else:
        lang_prompt = f"用{language}回答。"
    is_repeat = strategy in REPEAT_PROMPTS
    age_table = AGE_PROMPTS_REPEAT if is_repeat else AGE_PROMPTS
    age_prompt = age_table.get(age, age_table["2-4"])
    style_prompt = STYLE_PROMPTS.get(style, STYLE_PROMPTS["direkt"])
    prompt = (
        f"你是一个温柔的AI助手，专门回答小朋友的问题。"
        f"{lang_prompt}"
        f"{age_prompt}"
        f"语气亲切温暖，不要自称任何身份，不要用表情符号。"
        f"回答风格：{style_prompt}"
    )
    if is_repeat:
        prompt += REPEAT_PROMPTS[strategy]
        if previous_answers:
            vorher = "；".join(previous_answers[-2:])
            prompt += f"之前已经这样回答过了：{vorher}。不要再用同样的说法。"
    else:
        prompt += "不要用列举、不要用比喻堆叠，直接简单回答。"
    return prompt


async def speech_to_text(audio_bytes: bytes, filename: str = "audio.webm",
                         content_type: str = "audio/webm") -> str:
    # Keine feste Sprache: Whisper erkennt selbst, ob das Kind Deutsch oder
    # Chinesisch spricht. Die Antwortsprache kommt weiterhin aus der App.
    async with httpx.AsyncClient(timeout=30) as client:
        resp = await client.post(
            "https://api.groq.com/openai/v1/audio/transcriptions",
            headers={"Authorization": f"Bearer {GROQ_API_KEY}"},
            files={"file": (filename, audio_bytes, content_type)},
            data={"model": "whisper-large-v3"},
        )
        resp.raise_for_status()
    return resp.json()["text"]


def generate_answer(question: str, language="zh", age="2-4", style="direkt", conversation=None,
                    strategy=None, previous_answers=None) -> str:
    # Claude über Azure Foundry. Liest ANTHROPIC_FOUNDRY_API_KEY und
    # ANTHROPIC_FOUNDRY_RESOURCE automatisch aus der Umgebung.
    client = AnthropicFoundry()
    system_prompt = build_system_prompt(language, age, style, strategy, previous_answers)
    if strategy in REPEAT_PROMPTS:
        max_tok = 160 if age == "2-4" else 300
    else:
        max_tok = 80 if age == "2-4" else 150
    messages = []
    if conversation:
        for turn in conversation[-5:]:
            messages.append({"role": "user", "content": turn["question"]})
            messages.append({"role": "assistant", "content": turn["answer"]})
    messages.append({"role": "user", "content": question})
    message = client.messages.create(
        model="claude-sonnet-5",
        max_tokens=max_tok,
        # Sonnet 5 denkt sonst manchmal zuerst nach und braucht dafür die kurzen max_tokens auf
        thinking={"type": "disabled"},
        system=system_prompt,
        messages=messages,
    )
    return "".join(block.text for block in message.content if block.type == "text")


async def text_to_speech(text: str, voice_key="boy", language="zh") -> str:
    output_path = tempfile.mktemp(suffix=".mp3")
    lang_voices = VOICE_MAP.get(language, VOICE_MAP["zh"])
    v = lang_voices.get(voice_key, lang_voices["boy"])
    tts = edge_tts.Communicate(text, voice=v["voice"], rate=v.get("rate", "+0%"), pitch=v.get("pitch", "+0Hz"))
    await tts.save(output_path)
    return output_path


@app.get("/")
async def index():
    return HTMLResponse(HTML_PATH.read_text())


@app.post("/ask")
def ask(
    audio: UploadFile,
    language: str = Form("zh"),
    age: str = Form("2-4"),
    style: str = Form("direkt"),
    voice: str = Form("boy"),
    conversation: str = Form("[]"),
):
    try:
        conv = json.loads(conversation)
        audio_bytes = asyncio.run(audio.read())
        print(f"[1/3] Audio: {len(audio_bytes)} bytes")
        # Dateiname und Format vom Gerät übernehmen (Web: .webm, Android: .m4a)
        question = asyncio.run(speech_to_text(
            audio_bytes,
            audio.filename or "audio.webm",
            audio.content_type or "audio/webm",
        ))
        previous = find_previous_answers(question, conv)
        ask_count = len(previous) + 1
        strategy = pick_strategy(ask_count, style)
        print(f"[2/3] Frage: {question} ({ask_count}. Mal, Strategie: {strategy})")
        answer = generate_answer(question, language, age, style, conv, strategy, previous)
        print(f"[3/3] Antwort: {answer}")
        audio_path = asyncio.run(text_to_speech(answer, voice, language))
        return JSONResponse({
            "question": question,
            "answer": answer,
            "ask_count": ask_count,
            "strategy": strategy,
            "audio": f"/audio/{Path(audio_path).name}",
        })
    except Exception as e:
        print(f"[ERROR] {type(e).__name__}: {e}")
        return JSONResponse({"error": str(e)}, status_code=500)


@app.post("/ask-text")
def ask_text(
    question: str = Form(...),
    language: str = Form("zh"),
    age: str = Form("2-4"),
    style: str = Form("direkt"),
    voice: str = Form("boy"),
    conversation: str = Form("[]"),
):
    try:
        conv = json.loads(conversation)
        previous = find_previous_answers(question, conv)
        ask_count = len(previous) + 1
        strategy = pick_strategy(ask_count, style)
        print(f"[1/2] Frage: {question} ({ask_count}. Mal, Strategie: {strategy})")
        answer = generate_answer(question, language, age, style, conv, strategy, previous)
        print(f"[2/2] Antwort: {answer}")
        audio_path = asyncio.run(text_to_speech(answer, voice, language))
        return JSONResponse({
            "answer": answer,
            "ask_count": ask_count,
            "strategy": strategy,
            "audio": f"/audio/{Path(audio_path).name}",
        })
    except Exception as e:
        print(f"[ERROR] {type(e).__name__}: {e}")
        return JSONResponse({"error": str(e)}, status_code=500)


@app.get("/audio/{filename}")
async def get_audio(filename: str):
    path = Path(tempfile.gettempdir()) / filename
    if path.exists():
        return FileResponse(path, media_type="audio/mpeg")
    return JSONResponse({"error": "not found"}, status_code=404)
