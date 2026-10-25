# MaxiBox auf dem Raspberry Pi

Der Pi ist die Box: Das Backend (`main.py`) und das Knopf-Programm laufen direkt darauf.
Render wird nicht mehr gebraucht. Aufbau: Raspberry Pi 5, USB-Mikrofon, USB-Lautsprecher,
ein Knopf zwischen GPIO17 (Pin 11) und GND (Pin 9) und eine LED an GPIO27 (Pin 13) und
GND (Pin 14). Eine einzelne LED braucht einen Vorwiderstand von 220–330 Ω: langes Bein
über den Widerstand an Pin 13, kurzes Bein an Pin 14.

Pins: Liegt der Pi mit den USB-Buchsen nach rechts, ist die untere Reihe der Stiftleiste
von links nach rechts Pin 1, 3, 5, 7, 9, 11, 13, … (die obere Reihe 2, 4, 6, …, 14, …).
Pin 9 ist also der 5., Pin 11 der 6. und Pin 13 der 7. Stift unten von links; Pin 14 ist
der 7. Stift oben.

## Einrichten (einmalig)

Auf dem Pi, zum Beispiel per SSH von einem anderen Computer:

```bash
sudo apt install -y git
git clone https://github.com/yiyunlinch/maxibox.git ~/maxibox
bash ~/maxibox/pi/setup.sh
```

Das Skript
1. installiert die Pakete,
2. sucht USB-Mikrofon und -Lautsprecher,
3. macht einen Aufnahme-Test,
4. richtet die Python-Umgebung ein (`venv/`),
5. fragt nach `GROQ_API_KEY` und `ANTHROPIC_API_KEY` (nur in `~/maxibox/.env` auf dem Pi gespeichert),
6. startet Backend (`maxibox`) und Knopf-Programm (`maxibox-box`) als Dienste, auch nach jedem Neustart.

## Benutzen

Knopf einmal drücken: Ein kurzer Ton kommt, die LED leuchtet und die Box hört zu. Pausen
beim Sprechen sind kein Problem. Zum Schluss nochmals drücken: Die LED geht aus und die
Antwort kommt. Ohne zweiten Druck endet die Frage nach 30 Sekunden (`MAX_SECONDS` in `.env`).
Nach 5 Minuten Pause beginnt ein neues Gespräch. Die Elternseite ist im gleichen WLAN unter
`http://<IP des Pi>:8000` erreichbar.

Ohne Knopf testen: `sudo systemctl stop maxibox-box`, dann
`~/maxibox/venv/bin/python ~/maxibox/pi/talk.py` (Enter drücken, sprechen, nochmals Enter).

## Nützlich

- Schlüssel oder Modell ändern: `nano ~/maxibox/.env`, danach `sudo systemctl restart maxibox`
- Modell: `CLAUDE_MODEL=claude-sonnet-5` (Standard) oder `claude-haiku-5-5` (rund 20-mal günstiger)
- Logs: `journalctl -u maxibox-box -f` (Knopf) und `journalctl -u maxibox -f` (Backend)
- Aktualisieren: `cd ~/maxibox && git pull && sudo systemctl restart maxibox maxibox-box`
