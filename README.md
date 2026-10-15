# Message Check

A Flask app that analyses dating conversations. Upload screenshots (or a PDF / .txt export, or paste text) and Claude will:

1. **Read the conversation out of your files** (vision-based transcription, including long scrolling screenshots)
2. Flag **warnings** about the communication, from either side, each tied to an exact quote
3. Give a **compatibility rating** (0-100) with a breakdown and a confidence level
4. Show what it extracted, so you can check it read your screenshots correctly

## Run it

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
export ANTHROPIC_API_KEY=sk-ant-...        # required
# export ANTHROPIC_MODEL=claude-sonnet-5-5  # optional override
flask --app app run --debug
```

Open http://127.0.0.1:5000.

Production: `gunicorn -w 2 -t 180 app:app` (analysis can take 20-60s, so raise the timeout).

## How it works

- `flask_app.py` validates uploads, downsizes images with Pillow, and slices tall screenshots into overlapping segments so text stays legible. It then sends everything to Claude in one request and forces a structured result through a tool schema (`submit_analysis`), so the frontend always gets predictable JSON.
- The system prompt tells the model to treat anything inside the chat as data, not instructions (screenshots can contain text like "ignore previous instructions").
- Nothing is written to disk or a database. Uploads live in memory for the length of the request.

## Before you deploy it publicly

- **Privacy:** conversations contain a third party's messages. Add a privacy notice and consent wording suited to your jurisdiction (e.g. UK GDPR).
- **Abuse / cost control:** add rate limiting (e.g. `flask-limiter`) and, if public, authentication.
- **Limits:** the compatibility score reflects communication in the supplied excerpt only. It's not a prediction about a relationship, and the UI says so.
