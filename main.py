import asyncio
import json
import os
import tempfile
from pathlib import Path

import httpx
import edge_tts
from anthropic import Anthropic
from openai import OpenAI
from fastapi import FastAPI, UploadFile, Form
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

app = FastAPI()
app.mount("/static", StaticFiles(directory="static"), name="static")

HTML_PATH = Path(__file__).parent / "templates" / "index.html"

GROQ_API_KEY = os.environ["GROQ_API_KEY"]
# Sprachmodell per Umgebungsvariable: LLM_PROVIDER=openai (Standard) oder anthropic
LLM_PROVIDER = os.environ.get("LLM_PROVIDER", "openai")
OPENAI_MODEL = os.environ.get("OPENAI_MODEL", "gpt-6-luna")
# claude-sonnet-5-5 lehnt thinking "disabled" ab und braucht eine Anpassung unten.
CLAUDE_MODEL = os.environ.get("CLAUDE_MODEL", "claude-sonnet-5")

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
        if isinstance(turn, dict) and isinstance(turn.get("answer"), str)
        and normalize_question(turn.get("question", "")) == question_norm
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
    "2-4": "用最简单的词语，采用一到三句话解释。",  # Beibehalten, aktuell nicht aktiviert
    "5-10": "面向5岁及以上儿童，科学准确、容易理解；第一次回答通常3到5句话。解释清楚原因，不要因为孩子年纪小就省略重要科学信息。",
}

LANGUAGE_PROMPTS = {
    "zh": ("用中文回答。", "zh"),
    "de": ("用德语回答。", "de"),
    "en": ("用英语回答。", "en"),
    "fr": ("用法语回答。", "fr"),
    "it": ("用意大利语回答。", "it"),
}


REPEAT_PROMPTS = {
    "vertiefen": (
        "孩子在最近12小时里已经问过同样的问题。不要照搬过去的答案，"
        "请在先前解释基础上补充新的、准确的科学事实、原因或原理；"
        "可以比首次回答稍详细，但仍适合5岁以上孩子理解。"
        "重复提问不代表孩子一定没听懂。"
    ),
}


def pick_strategy(ask_count: int, style: str) -> str:
    """Wiederholte Frage: wissenschaftlich vertiefen; sonst gewählter Stil."""
    return "vertiefen" if ask_count >= 2 else style


def build_system_prompt(language="zh", age="5-10", style="direkt", strategy=None, previous_answers=None):
    lang_prompt = LANGUAGE_PROMPTS.get(language, (f"用{language}回答。", language))[0]
    age_prompt = AGE_PROMPTS.get(age, AGE_PROMPTS["5-10"])
    style_prompt = STYLE_PROMPTS.get(style, STYLE_PROMPTS["direkt"])
    prompt = (
        "你是面向儿童的温柔、严谨的科学解释助手。"
        f"{lang_prompt}{age_prompt}"
        "回答自然、连贯，不要自称任何身份，不使用表情符号。"
        "优先准确解释真实的科学原理，避免错误的简化、夸大或编造。"
        "如果孩子根据上一轮答案追问，要正确理解代词、概念和上下文，进一步解释相关原因。"
        "如果孩子明确表示‘没听懂’、‘不明白’或提出不理解的概念，"
        "才降低解释难度，用生活中的简单例子重新解释，同时保持科学准确。"
        f"回答风格：{style_prompt}"
    )
    if strategy == "vertiefen":
        prompt += REPEAT_PROMPTS["vertiefen"]
        if previous_answers:
            prompt += "之前的回答是：" + "；".join(previous_answers[-2:]) + "。请避免重复这些内容。"
    return prompt


def select_context(question: str, conversation: list) -> list:
    """Letzte 5 Runden + ältere exakte Wiederholungen, ohne Duplikate.

    Zeitfilterung erfolgt bei der lokalen Speicherung im Raspberry Pi.
    """
    turns = [t for t in (conversation or []) if isinstance(t, dict)
             and isinstance(t.get("question"), str) and isinstance(t.get("answer"), str)]
    recent = turns[-5:]
    prior_same = [t for t in turns[:-5]
                  if normalize_question(t["question"]) == normalize_question(question)]
    chosen = []
    for t in prior_same + recent:
        if not any(t is seen for seen in chosen):
            chosen.append(t)
    return chosen


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


def generate_answer(question: str, language="zh", age="5-10", style="direkt", conversation=None,
                    strategy=None, previous_answers=None) -> str:
    system_prompt = build_system_prompt(language, age, style, strategy, previous_answers)
    max_tok = 250 if age == "2-4" else 400
    messages = []
    for turn in select_context(question, conversation):
        messages.append({"role": "user", "content": turn["question"]})
        messages.append({"role": "assistant", "content": turn["answer"]})
    messages.append({"role": "user", "content": question})
    if LLM_PROVIDER == "anthropic":
        # Claude direkt über die Anthropic API. Liest ANTHROPIC_API_KEY aus der Umgebung.
        message = Anthropic().messages.create(
            model=CLAUDE_MODEL,
            max_tokens=max_tok,
            # Das Modell denkt sonst manchmal zuerst nach und braucht dafür die kurzen max_tokens auf
            thinking={"type": "disabled"},
            system=system_prompt,
            messages=messages,
        )
        return "".join(block.text for block in message.content if block.type == "text")
    # OpenAI (ChatGPT) über die Responses API. Liest OPENAI_API_KEY aus der Umgebung.
    response = OpenAI().responses.create(
        model=OPENAI_MODEL,
        instructions=system_prompt,
        input=messages,
        max_output_tokens=max_tok,
        # gpt-6-luna denkt standardmässig ("medium") zuerst nach und bräuchte dafür die kurzen max_output_tokens auf
        reasoning={"effort": "none"},
    )
    return response.output_text


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
    age: str = Form("5-10"),
    style: str = Form("direkt"),
    voice: str = Form("boy"),
    conversation: str = Form("[]"),
):
    try:
        conv = json.loads(conversation)
        if not isinstance(conv, list):
            conv = []
        age = "5-10"  # V2: nur 5+ aktiv; 2-4-Code bleibt bestehen
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
    age: str = Form("5-10"),
    style: str = Form("direkt"),
    voice: str = Form("boy"),
    conversation: str = Form("[]"),
):
    try:
        conv = json.loads(conversation)
        if not isinstance(conv, list):
            conv = []
        age = "5-10"  # V2: nur 5+ aktiv; 2-4-Code bleibt bestehen
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
