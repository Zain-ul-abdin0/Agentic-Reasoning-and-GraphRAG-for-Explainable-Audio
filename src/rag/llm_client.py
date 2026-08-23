from dataclasses import dataclass
import json
from urllib.error import URLError, HTTPError
from urllib.request import Request, urlopen


class LLMUnavailableError(RuntimeError):
    pass


@dataclass
class LocalLLMConfig:
    provider: str = "ollama"
    model: str = "gemma3"
    base_url: str = "http://localhost:11434"
    timeout_seconds: int = 60
    temperature: float = 0.2


def generate_with_ollama(prompt: str, config: LocalLLMConfig) -> str:
    endpoint = config.base_url.rstrip("/") + "/api/generate"
    payload = {
        "model": config.model,
        "prompt": prompt,
        "stream": False,
        "options": {
            "temperature": config.temperature
        }
    }
    request = Request(
        endpoint,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST"
    )

    try:
        with urlopen(request, timeout=config.timeout_seconds) as response:
            body = json.loads(response.read().decode("utf-8"))
    except (HTTPError, URLError, TimeoutError, OSError) as exc:
        raise LLMUnavailableError(str(exc)) from exc

    text = body.get("response", "").strip()
    if not text:
        raise LLMUnavailableError("Ollama returned an empty response.")

    return text


def generate_text(prompt: str, config: LocalLLMConfig) -> str:
    if config.provider == "none":
        raise LLMUnavailableError("LLM provider is disabled.")

    if config.provider == "ollama":
        return generate_with_ollama(prompt, config)

    raise LLMUnavailableError(f"Unsupported LLM provider: {config.provider}")
