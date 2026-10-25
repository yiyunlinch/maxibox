"""Die MaxiBox selbst: Knopf gedrückt halten, sprechen, loslassen.

Der Knopf hängt an GPIO17 (Pin 11) und GND (Pin 9) und nutzt den internen
Pull-up-Widerstand. Beim Drücken ertönt ein kurzer Ton, dann läuft die Aufnahme,
bis der Knopf losgelassen wird. Läuft als Dienst maxibox-box (pi/setup.sh).
Auf dem Pi 5 funktioniert RPi.GPIO nicht mehr, deshalb gpiozero.
"""
import math
import os
import struct
import subprocess
import tempfile
import time
import wave

from gpiozero import Button

from talk import MAX_SECONDS, SPEAKER, answer, start_recording, stop_recording

BUTTON_GPIO = int(os.environ.get("BUTTON_GPIO", "17"))
NEW_CHAT_AFTER = 5 * 60  # nach 5 Minuten Pause beginnt ein neues Gespräch
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


def main() -> None:
    button = Button(BUTTON_GPIO, pull_up=True, bounce_time=0.05)
    beep = os.path.join(tempfile.gettempdir(), "maxibox_beep.wav")
    wav = os.path.join(tempfile.gettempdir(), "maxibox_frage.wav")
    make_beep(beep)
    conversation = []
    last_answer = 0.0
    print(f"MaxiBox 准备好了：按钮接在 GPIO{BUTTON_GPIO}", flush=True)
    while True:
        button.wait_for_press()
        if time.monotonic() - last_answer > NEW_CHAT_AFTER:
            conversation = []
        subprocess.run(["aplay", "-q", "-D", SPEAKER, beep], check=False)
        started = time.monotonic()
        proc = start_recording(wav)
        button.wait_for_release(timeout=MAX_SECONDS)
        stop_recording(proc)
        if time.monotonic() - started < MIN_SECONDS:
            continue
        conversation = answer(wav, conversation)
        last_answer = time.monotonic()


if __name__ == "__main__":
    main()
