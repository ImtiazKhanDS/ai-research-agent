import os
from openai import OpenAI

# LLM_MODE=api   → Groq (default)
# LLM_MODE=local → Ollama (gemma4:e2b)
LLM_MODE = os.environ.get("LLM_MODE", "api").lower()

_GROQ_MODEL = os.environ.get("GROQ_MODEL", "llama-3.3-70b-versatile")
_OLLAMA_MODEL = os.environ.get("OLLAMA_MODEL", "gemma4:e2b")
_OLLAMA_BASE_URL = os.environ.get("OLLAMA_BASE_URL", "http://localhost:11434/v1")


def _make_client() -> tuple[OpenAI, str]:
    if LLM_MODE == "local":
        return OpenAI(api_key="ollama", base_url=_OLLAMA_BASE_URL), _OLLAMA_MODEL
    return OpenAI(api_key=os.environ["GROQ_API_KEY"], base_url="https://api.groq.com/openai/v1"), _GROQ_MODEL


def complete(messages: list, tools: list | None = None, tool_choice: str | None = "auto") -> object:
    client, model = _make_client()
    kwargs = {"model": model, "messages": messages}
    if tools:
        kwargs["tools"] = tools
        kwargs["tool_choice"] = tool_choice or "auto"
    return client.chat.completions.create(**kwargs)


def current_provider() -> str:
    return f"{LLM_MODE} ({'Ollama/' + _OLLAMA_MODEL if LLM_MODE == 'local' else 'Groq/' + _GROQ_MODEL})"
