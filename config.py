"""
Configuration constants for the persona tester.
"""

# Flutter web app URL
FLUTTER_URL: str = "https://duckduckgo.com/"

# Ollama model name — must be a Qwen-VL variant pulled locally
OLLAMA_MODEL: str = "qwen3.5:cloud"

# Ollama API base URL
OLLAMA_URL: str = "http://localhost:11434"

# Maximum number of agent steps before stopping
MAX_STEPS: int = 30

# Directory for session logs and screenshots
SCREENSHOT_DIR: str = "sessions/"

# Show browser window for debugging (set True to watch live)
HEADLESS: bool = False

# Viewport dimensions for Playwright
VIEWPORT_WIDTH: int = 1280
VIEWPORT_HEIGHT: int = 800

# Delay (ms) after each action for UI to settle
ACTION_DELAY_MS: int = 1500

# Number of recent history entries to include in the LLM context
HISTORY_WINDOW: int = 20

# Timeout (seconds) for Ollama API calls
OLLAMA_TIMEOUT: int = 120
