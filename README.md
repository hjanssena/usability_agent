# Persona Tester — LLM Vision Agent for Web Apps

A Python agent that uses a local **Ollama Qwen-VL** vision model to test a web app by *looking at screenshots* and acting as a configurable user persona.

## How it works

```
Screenshot → Qwen-VL (local) → JSON action → Playwright executes → repeat
```

The LLM receives a base64 screenshot and outputs a normalised bounding box
(`[ymin, xmin, ymax, xmax]` on a 1000×1000 grid). The agent converts that to
pixel coordinates and performs the physical click / type / scroll via Playwright.
No DOM inspection, no accessibility tree — purely visual.

---

## Prerequisites

| Tool | Install |
|------|---------|
| **Python ≥ 3.11** | Your preferred method |
| **Ollama** | <https://ollama.com/download> |
| **Flutter SDK** | <https://docs.flutter.dev/get-started/install> |

---

## Setup

### 1. Pull the Qwen vision model

```bash
ollama pull qwen3.5:cloud 
```

> The model name in `config.py` defaults to `qwen3.5:cloud`. Change it if you
> pulled a different tag (e.g. `qwen2.5-vl:7b`).

### 2. Install Python dependencies

```bash
cd persona-tester
pip install -r requirements.txt
```

### 3. Install Playwright browsers

```bash
playwright install chromium
```

### 4. Start your Flutter web app

```bash
cd /path/to/your/flutter/project
flutter run -d web-server --web-port 8080
```

The tester expects the app at `http://localhost:8080` (configurable in
`config.py`).

---

## Run

```bash
python main.py --persona personas/example.json
```

Add `-v` for debug-level logging:

```bash
python main.py --persona personas/example.json -v
```

---

## Persona format

Create any JSON file with the following schema:

```json
{
  "name": "Maria",
  "description": "A 60-year-old retiree unfamiliar with technology...",
  "goal": "Successfully add an item to cart and reach checkout",
  "traits": [
    "reads all visible text before clicking",
    "avoids anything that looks like an ad"
  ]
}
```

---

## Session output

Each run creates a directory under `sessions/`:

```
sessions/Maria_20260420T173200Z/
├── step_001.png
├── step_002.png
├── ...
└── session.json      # full log with reasoning, actions, coordinates
```

---

## Configuration

All tunables live in `config.py`:

| Variable | Default | Description |
|----------|---------|-------------|
| `FLUTTER_URL` | `http://localhost:8080` | Target app URL |
| `OLLAMA_MODEL` | `qwen2.5-vl` | Ollama model name |
| `OLLAMA_URL` | `http://localhost:11434` | Ollama API base |
| `MAX_STEPS` | `15` | Maximum agent steps |
| `HEADLESS` | `False` | Show/hide browser |
| `VIEWPORT_WIDTH` | `1280` | Browser viewport width |
| `VIEWPORT_HEIGHT` | `800` | Browser viewport height |
| `OLLAMA_TIMEOUT` | `120` | API timeout (seconds) |

---

## Exit codes

| Code | Meaning |
|------|---------|
| `0` | Goal completed (`done`) |
| `1` | Max steps reached |
| `2` | Agent got stuck (`stuck`) |
