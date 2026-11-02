
import os
import requests
import time

# Configuration comes from environment variables (see .env.example)
API_KEY  = os.getenv("OPENAI_API_KEY", "")
API_BASE = os.getenv("API_BASE", "")
MODEL    = os.getenv("MODEL_NAME", "bens_model")

def call_model_chat_completions(prompt: str,
                                system: str = "You are a helpful assistant. Reply with only the final answer—no explanation.",
                                model: str = MODEL,
                                temperature: float = 0.0,
                                timeout: int = 60,
                                max_retries: int = 3) -> dict:
    """
    Calls an OpenAI-style /v1/chat/completions endpoint and returns:
    { 'ok': bool, 'text': str or None, 'raw': dict or None, 'status': int, 'error': str or None, 'headers': dict }
    """
    if not API_BASE:
        return {"ok": False, "text": None, "raw": None, "status": -1,
                "error": "API_BASE is not set. Export API_BASE (see .env.example).", "headers": {}}

    url = f"{API_BASE}/chat/completions"
    headers = {
        "Authorization": f"Bearer {API_KEY}",
        "Content-Type":  "application/json",
    }
    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": system},
            {"role": "user",   "content": prompt}
        ],
        "temperature": temperature,
        "max_tokens": 2048, # Increased token limit for CoT
    }

    last_error = None
    for attempt in range(max_retries):
        try:
            resp = requests.post(url, headers=headers, json=payload, timeout=timeout)
            status = resp.status_code
            hdrs   = dict(resp.headers)
            if status == 200:
                data = resp.json()
                text = data.get("choices", [{}])[0].get("message", {}).get("content", "")
                return {"ok": True, "text": text, "raw": data, "status": status, "error": None, "headers": hdrs}
            else:
                # try best-effort to surface error text
                err_text = None
                try:
                    err_text = resp.json()
                except Exception:
                    err_text = resp.text
                last_error = f"HTTP {status}: {err_text}"
                print(f"Attempt {attempt+1} failed: {last_error}")
                time.sleep(1) # Basic backoff
        except requests.RequestException as e:
            last_error = str(e)
            print(f"Attempt {attempt+1} exception: {last_error}")
            time.sleep(1)

    return {"ok": False, "text": None, "raw": None, "status": -1, "error": str(last_error), "headers": {}}
