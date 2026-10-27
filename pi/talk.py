"""MaxiBox auf dem Raspberry Pi: gemeinsame Funktionen und ein Test mit der Enter-Taste.

Die Aufnahme geht an das lokale Backend (/ask), die Antwort kommt über den
USB-Lautsprecher. Geräte und Einstellungen stehen in ../.env (legt pi/setup.sh an).
Die Box selbst mit Knopf läuft über pi/box.py.
"""
import json
import os
import signal
import subprocess
import tempfile
from pathlib import Path

import httpx

REPO_DIR = Path(__file__).resolve().parent.parent


def load_env(path: Path) -> None:
    """KEY=VALUE-Zeilen übernehmen, ohne gesetzte Umgebungsvariablen zu überschreiben."""
    if not path.exists():
        return
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line and not line.startswith("#") and "=" in line:
            key, value = line.split("=", 1)
            os.environ.setdefault(key.strip(), value.strip())


load_env(REPO_DIR / ".env")

BACKEND = os.environ.get("MAXIBOX_URL", "http://localhost:8000")
MIC = os.environ.get("MIC_DEVICE", "default")
SPEAKER = os.environ.get("SPEAKER_DEVICE", "default")
MAX_SECONDS = int(os.environ.get("MAX_SECONDS", "30"))  # längste Frage in Sekunden
SETTINGS = {
    "language": os.environ.get("MAXIBOX_LANGUAGE", "zh"),
    "age": "5-10",  # V2: 5+ only; younger-age logic stays in main.py
    "style": os.environ.get("MAXIBOX_STYLE", "direkt"),
    "voice": os.environ.get("MAXIBOX_VOICE", "boy"),
}


def start_recording(path: str) -> subprocess.Popen:
    """Aufnahme starten; endet mit stop_recording() oder nach MAX_SECONDS."""
    return subprocess.Popen(
        ["arecord", "-q", "-D", MIC, "-f", "S16_LE", "-r", "16000", "-c", "1",
         "-d", str(MAX_SECONDS), path]
    )


def stop_recording(proc: subprocess.Popen) -> None:
    if proc.poll() is None:
        # SIGINT statt kill: arecord schliesst die WAV-Datei dann sauber ab
        proc.send_signal(signal.SIGINT)
    proc.wait()


def ask(path: str, conversation: list) -> dict:
    with open(path, "rb") as f:
        resp = httpx.post(
            f"{BACKEND}/ask",
            files={"audio": ("frage.wav", f, "audio/wav")},
            data={**SETTINGS, "conversation": json.dumps(conversation, ensure_ascii=False)},
            timeout=90,
        )
    return resp.json()


def play(audio_url: str) -> None:
    resp = httpx.get(f"{BACKEND}{audio_url}", timeout=30)
    resp.raise_for_status()
    with tempfile.NamedTemporaryFile(suffix=".mp3", delete=False) as f:
        f.write(resp.content)
    try:
        subprocess.run(["mpg123", "-q", "-o", "alsa", "-a", SPEAKER, f.name], check=False)
    finally:
        os.unlink(f.name)


def answer(wav: str, conversation: list) -> list:
    """Aufnahme schicken, Antwort zeigen und abspielen; gibt den neuen Verlauf zurück.

    Die lokale 12-Stunden-Historie wird an das Backend weitergereicht.
    """
    print("思考中……", flush=True)
    try:
        result = ask(wav, conversation)
    except (httpx.HTTPError, ValueError) as e:
        print(f"后端出问题了：{e}", flush=True)
        return conversation
    if "error" in result:
        print(f"出错了：{result['error']}", flush=True)
        return conversation
    print(f"问：{result.get('question')}", flush=True)
    print(f"答：{result.get('answer')}", flush=True)
    play(result["audio"])
    return conversation + [{"question": result["question"], "answer": result["answer"]}]


def main() -> None:
    print(f"MaxiBox 测试：后端 {BACKEND}，麦克风 {MIC}，喇叭 {SPEAKER}")
    print("按 Enter 开始提问，Ctrl+C 退出。")
    conversation = []
    wav = os.path.join(tempfile.gettempdir(), "maxibox_frage.wav")
    while True:
        input()
        proc = start_recording(wav)
        input("录音中……说完按 Enter ")
        stop_recording(proc)
        conversation = answer(wav, conversation)
        print("\n按 Enter 再问一个问题，Ctrl+C 退出。")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print()
