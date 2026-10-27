"""Die MaxiBox selbst: Knopf gedrückt halten, Frage stellen, loslassen.

Der Knopf hängt an GPIO17 (Pin 11) und GND (z. B. Pin 39) und nutzt den internen
Pull-up-Widerstand. Die LED an GPIO27 (Pin 13, GND an Pin 14) leuchtet, solange
der Knopf gedrückt ist und die Box zuhört. Pausen beim Sprechen beenden die Frage
nicht: Sie endet erst beim Loslassen oder nach MAX_SECONDS. Läuft als Dienst
maxibox-box (pi/setup.sh). Auf dem Pi 5 funktioniert RPi.GPIO nicht mehr, deshalb gpiozero.
"""
import math
import json
import os
import struct
import subprocess
import tempfile
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path
import wave

from gpiozero import LED, Button

from talk import MAX_SECONDS, SPEAKER, answer, start_recording, stop_recording

BUTTON_GPIO = int(os.environ.get("BUTTON_GPIO", "17"))
LED_GPIO = int(os.environ.get("LED_GPIO", "27"))
HISTORY_HOURS = 12
HISTORY_FILE = Path.home() / ".maxibox" / "conversation_history.json"
MIN_SECONDS = 0.5  # kürzeres Drücken gilt als versehentlich


def make_beep(path: str) -> None:
    """Kurzer Ton (880 Hz, 0.15 s), damit das Kind hört, dass die Box zuhört."""
    rate = 16000
    frames = b"".join(
        struct.pack("<h", int(8000 * math.sin(2 * math.pi * 880 * i / rate)))
        for i in range(int(rate * 0.15))
    )
    with wave.open(path, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(rate)
        w.writeframes(frames)


def load_history() -> list:
    """Lokal gespeicherte Fragen/Antworten; nur die letzten 12 Stunden."""
    try:
        turns = json.loads(HISTORY_FILE.read_text(encoding="utf-8"))
        return prune_history(turns)
    except (OSError, ValueError, TypeError):
        return []


def prune_history(turns: list) -> list:
    threshold = datetime.now(timezone.utc) - timedelta(hours=HISTORY_HOURS)
    valid = []
    for turn in turns if isinstance(turns, list) else []:
        if not isinstance(turn, dict):
            continue
        if not isinstance(turn.get("question"), str) or not isinstance(turn.get("answer"), str):
            continue
        try:
            ts = datetime.fromisoformat(turn["timestamp"])
            if ts.tzinfo is None:
                continue
            if ts.astimezone(timezone.utc) >= threshold:
                valid.append(turn)
        except (KeyError, ValueError, TypeError):
            continue
    return valid


def save_history(turns: list) -> None:
    HISTORY_FILE.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    tmp = HISTORY_FILE.with_suffix(".tmp")
    tmp.write_text(json.dumps(turns, ensure_ascii=False, indent=2), encoding="utf-8")
    tmp.chmod(0o600)
    tmp.replace(HISTORY_FILE)


def merge_result(old: list, result) -> list:
    """talk.answer() kann die komplette Conversation oder nur neue Turns liefern."""
    if not isinstance(result, list):
        print("Warnung: talk.answer hat keine Gesprächsliste zurückgegeben", flush=True)
        return old
    newer = [t for t in result if isinstance(t, dict)
             and isinstance(t.get("question"), str) and isinstance(t.get("answer"), str)]
    if not newer:
        return old
    # Search for the longest overlap of old tail with returned prefix.
    overlap = 0
    for n in range(min(len(old), len(newer)), 0, -1):
        if [(t["question"], t["answer"]) for t in old[-n:]] == [(t["question"], t["answer"]) for t in newer[:n]]:
            overlap = n
            break
    additions = newer[overlap:]
    # If returned conversation already contains all old turns, only append the extra turns.
    timestamp = datetime.now(timezone.utc).isoformat()
    for t in additions:
        old.append({"question": t["question"], "answer": t["answer"], "timestamp": timestamp})
    return old


def main() -> None:
    button = Button(BUTTON_GPIO, pull_up=True, bounce_time=0.05)
    led = LED(LED_GPIO)
    beep = os.path.join(tempfile.gettempdir(), "maxibox_beep.wav")
    wav = os.path.join(tempfile.gettempdir(), "maxibox_frage.wav")
    make_beep(beep)
    conversation = load_history()
    print(f"MaxiBox 准备好了：按钮 GPIO{BUTTON_GPIO}，灯 GPIO{LED_GPIO}", flush=True)
    while True:
        button.wait_for_press()
        conversation = prune_history(conversation)
        save_history(conversation)
        subprocess.run(["aplay", "-q", "-D", SPEAKER, beep], check=False)
        led.on()
        started = time.monotonic()
        proc = start_recording(wav)
        button.wait_for_release(timeout=MAX_SECONDS)
        stop_recording(proc)
        led.off()
        button.wait_for_release()
        if time.monotonic() - started < MIN_SECONDS:
            continue
        result = answer(wav, [dict(t) for t in conversation])
        conversation = prune_history(merge_result(conversation, result))
        save_history(conversation)


if __name__ == "__main__":
    main()
