# MaxiBox

### Eine interaktive KI-Box für Kinder

## 1. Projektbeschreibung

MaxiBox ist eine interaktive Sprachbox für Kinder, die ihre Neugier und ihr selbstständiges Lernen fördern soll.

Kinder können per Knopfdruck Fragen stellen und erhalten gesprochene Antworten von einer künstlichen Intelligenz. Die Antworten sollen wissenschaftlich korrekt, altersgerecht und verständlich sein.

MaxiBox funktioniert ohne Bildschirm. Die Kinder können sich dadurch ganz auf das Zuhören, Nachdenken und Nachfragen konzentrieren.

## 2. Technische Umsetzung

MaxiBox basiert auf einem Raspberry Pi 5 mit Mikrofon, Lautsprecher und einem physischen Knopf.

Die Verarbeitung erfolgt in mehreren Schritten:

1. **Spracheingabe:** Das Kind stellt eine Frage über das Mikrofon.
2. **Spracherkennung:** Groq Whisper API wandelt die Sprache in Text um.
3. **KI-Antwort:** OpenAI API generiert eine passende Antwort.
4. **Sprachausgabe:** Edge TTS (Microsoft) wandelt den Text in Sprache um.
5. **Wiedergabe:** Die Antwort wird über den Lautsprecher ausgegeben.

Das Backend läuft lokal auf dem Raspberry Pi mit Python und FastAPI. Externe APIs werden über das Internet aufgerufen.

<img width="542" height="184" alt="Bildschirmfoto 2026-10-26 um 13 56 17" src="https://github.com/user-attachments/assets/4cd6bfbd-82e7-4ba5-9d6f-6ac2ea0a4fae" />


## 3. Version 1 – Grundfunktion (Frühjahrssemester 2026)

In der ersten Version stand die technische Umsetzung im Mittelpunkt.

- Fragen per Spracheingabe stellen
- Antworten mit KI generieren
- Antworten über einen Lautsprecher wiedergeben
- Alter, Sprache und Stimme auswählen
- Unterschiedliche Antwortstile anbieten
- Die letzten fünf Gesprächsrunden als Kontext berücksichtigen

**Technische Architektur V1:**

Das Backend wurde mit Python und FastAPI auf Render gehostet. Für die Generierung der Antworten wurde die Claude API von Anthropic verwendet.

## 4. Version 2 – Weiterentwicklung (Herbstsemester 2026)

In diesem Semester liegt der Schwerpunkt auf der Qualität der Antworten und auf natürlichen Folgegesprächen.

### 4.1 Technische Umstellung

Die technische Architektur wurde vereinfacht:

- **Backend-Migration:** Von Render auf den Raspberry Pi
- **Lokaler Server:** Python und FastAPI laufen direkt auf dem Raspberry Pi.
- **KI-Modell:** Wechsel von Claude API (Anthropic) zu OpenAI API
- **Cloud-Dienste:** Nur Spracherkennung, KI-Antworten und Sprachsynthese benötigen externe Dienste.

Dadurch wird kein externer Hosting-Dienst wie Render mehr benötigt. Für die KI-Funktionen bleibt eine Internetverbindung notwendig.


### 4.2 Wissenschaftlich fundierte Antworten

MaxiBox soll nicht nur kurze und einfache Antworten geben, sondern Kindern helfen, wissenschaftliche Zusammenhänge zu verstehen.

Geplante Anpassungen:

- Standardalter: ab 5 Jahren
- Die Altersgruppe 2–4 Jahre bleibt im Code erhalten, wird aber vorerst nicht verwendet.
- Erhöhung der maximalen Antwortlänge von 150 auf 400 Tokens
- Wissenschaftlich korrekte, altersgerechte Erklärungen mit normalerweise 3–5 Sätzen

<img width="1448" height="1086" alt="450760d7-93a5-453b-a1a1-26df426d0e2b" src="https://github.com/user-attachments/assets/97149ef3-b9f7-4975-9263-6e62a6f0b1db" />


### 4.3 Vier Arten von Fragen

**1. Erste Frage – Erklären**

Eine Frage wird wissenschaftlich korrekt und verständlich beantwortet.

**2. Wiederholte Frage – Vertiefen**

Wenn ein Kind dieselbe Frage erneut stellt, ergänzt MaxiBox neue wissenschaftliche Informationen, anstatt die bisherige Antwort zu wiederholen.

**3. Folgefrage – Kontext verstehen**

MaxiBox berücksichtigt vorherige Antworten und erkennt Zusammenhänge zwischen verschiedenen Fragen.

**4. Nicht verstanden – Vereinfachen**

Wenn ein Kind ausdrücklich sagt, dass es etwas nicht verstanden hat, erklärt MaxiBox den Inhalt einfacher und mit Beispielen.

### 4.4 Gesprächsgedächtnis

Für Version 2 ist ein lokales Gesprächsgedächtnis vorgesehen.

- Fragen und Antworten werden 12 Stunden auf dem Raspberry Pi gespeichert.
- Bei einer neuen Frage werden die letzten fünf Gesprächsrunden berücksichtigt.
- Frühere gleiche Fragen innerhalb von 12 Stunden können zusätzlich einbezogen werden.
- Doppelte Gesprächseinträge werden beim Senden an die OpenAI API vermieden.
- Nach 12 Stunden werden die gespeicherten Einträge gelöscht.

Dadurch können Kinder ein Thema über einen längeren Zeitraum weiterverfolgen, auch wenn sie zwischendurch andere Fragen stellen.

### Beispiel: Vier Arten von Fragen

**1. Erste Frage – Wissenschaftlich erklären**

**Kind:** Warum ist die Kiwi die Königin der Früchte?

**MaxiBox:** Die Kiwi wird manchmal so genannt, weil sie viele wichtige Nährstoffe enthält. Sie hat besonders viel Vitamin C, das unser Immunsystem unterstützt. Ausserdem enthält sie Ballaststoffe, die gut für unsere Verdauung sind. Aber eigentlich gibt es keine offizielle Königin der Früchte!

**2. Wiederholte Frage – Wissen vertiefen**

**Kind:** Warum ist die Kiwi die Königin der Früchte?

**MaxiBox:** Die Kiwi enthält besonders viel Vitamin C. Dieses Vitamin unterstützt unser Immunsystem und hilft bei der Heilung von Wunden. Eine grüne Kiwi enthält pro 100 Gramm sogar mehr Vitamin C als eine Orange. Trotzdem sind auch andere Früchte gesund, weil sie unterschiedliche Nährstoffe enthalten.

**3. Folgefrage – Kontext verstehen**

**Kind:** Und wer ist dann der König der Tiere?

**MaxiBox:** Der Löwe wird oft als König der Tiere bezeichnet. Das liegt an seiner mächtigen Erscheinung und seiner grossen Mähne, die wie eine Krone aussieht. Aber genau wie bei der Kiwi ist das nur ein Spitzname. In der Natur gibt es keinen echten König!

**4. Nicht verstanden – Einfacher erklären**

**Kind:** Ich verstehe nicht, was das Immunsystem ist.

**MaxiBox:** Das Immunsystem ist wie eine Schutzmannschaft in deinem Körper. Es hilft dir, dich gegen Krankheitserreger wie Viren und Bakterien zu wehren. Stell dir vor, kleine Beschützer passen auf deinen Körper auf. Vitamin C aus der Kiwi hilft deinem Immunsystem, gut zu arbeiten.

## 5. Ziel der Weiterentwicklung

MaxiBox soll sich von einer einfachen Frage-Antwort-Box zu einem interaktiven Lernbegleiter entwickeln.

Dabei stehen drei Ziele im Mittelpunkt:

- **Wissenschaftliche Qualität:** Kinder erhalten korrekte und informative Antworten.
- **Vertiefendes Lernen:** Wiederholte Fragen führen zu neuen Erkenntnissen.
- **Natürliche Kommunikation:** MaxiBox versteht Folgefragen und passt seine Erklärungen an.

Der Schwerpunkt liegt nicht darauf, Kindern möglichst schnell eine Antwort zu geben, sondern ihre Neugier zu unterstützen und sie zum Weiterfragen anzuregen.

## 6. Projektstatus

**Bereits umgesetzt:**
- Raspberry Pi mit physischem Knopf, Mikrofon und Lautsprecher
- Sprachaufnahme und Sprachausgabe
- Groq Whisper API
- OpenAI API
- Edge TTS
- Migration des FastAPI-Backends von Render auf den Raspberry Pi
- Funktionierende sprachbasierte Frage-Antwort-Interaktion

**Für Version 2 geplant bzw. noch umzusetzen:**
- Erweiterte wissenschaftliche Erklärungen
- Anpassung der Antwortlänge
- Intelligente Behandlung wiederholter Fragen
- Kontextbezogene Folgefragen
- Lokales Gesprächsgedächtnis über 12 Stunden
- LED-Statusanzeige

---

*Studienprojekt – FH Graubünden, Multimedia Production*

*Version 1: Frühjahrssemester 2026*

*Version 2: Herbstsemester 2026*
