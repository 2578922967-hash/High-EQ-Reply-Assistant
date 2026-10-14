import os
import re
import json
import time
import requests
from dotenv import load_dotenv

load_dotenv(override=True)

HKBU_BASE_URL = "https://genai.hkbu.edu.hk/api/v0/rest"
MODEL_API_VERSIONS = {
    "gpt-5": "2024-12-01-preview",
    "gpt-5-mini": "2024-12-01-preview",
    "gpt-4.1": "2024-12-01-preview",
    "gpt-4.1-mini": "2024-12-01-preview",
    "o1": "2024-12-01-preview",
    "o3-mini": "2024-12-01-preview",
    "deepseek-r1": "2024-05-01-preview",
    "deepseek-v3": "2024-05-01-preview",
    "qwen3-max": "v1",
    "qwen-plus": "v1",
    "gemini-2.5-pro": "v1",
    "gemini-2.5-flash": "v1",
}


def call_llm(system_prompt: str, user_message: str, api_key: str = None, model_name: str = "gpt-4.1",
             temperature: float = 0.7, max_tokens: int = 6000, retries: int = 2, timeout: int = 180,
             thinking: bool = False) -> str:
    """调用大模型并返回文本。

    默认走 HKBU GenAI；若设置了环境变量 LLM_BASE_URL，则改走任意 OpenAI 兼容接口
    （如 DeepSeek / 通义千问官方 API），此时 api_key 取 LLM_API_KEY。
    阿里云百炼上的新模型默认开启深度思考，单次调用会慢 3-4 倍，因此默认关闭。
    """
    compat_base = os.environ.get("LLM_BASE_URL")
    if compat_base:
        final_key = api_key or os.environ.get("LLM_API_KEY")
        url = f"{compat_base.rstrip('/')}/chat/completions"
        headers = {"Content-Type": "application/json", "Authorization": f"Bearer {final_key}"}
        payload = {"model": model_name}
        if "dashscope" in compat_base:
            payload["enable_thinking"] = thinking
    else:
        final_key = api_key or os.environ.get("OPENAI_API_KEY")
        api_version = MODEL_API_VERSIONS.get(model_name, "2024-12-01-preview")
        url = f"{HKBU_BASE_URL}/deployments/{model_name}/chat/completions?api-version={api_version}"
        headers = {"accept": "application/json", "Content-Type": "application/json", "api-key": final_key}
        payload = {}

    if not final_key:
        raise Exception("API key is missing. Please provide it in request or .env file.")

    payload.update({
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_message},
        ],
        "temperature": temperature,
        "max_tokens": max_tokens,
        "top_p": 0.95,
    })

    last_error = None
    for attempt in range(retries + 1):
        try:
            resp = requests.post(url, json=payload, headers=headers, timeout=timeout)
            if resp.status_code == 200:
                return resp.json()["choices"][0]["message"]["content"]
            last_error = Exception(f"API Error {resp.status_code}: {resp.text[:500]}")
            if resp.status_code < 500 and resp.status_code != 429:
                break
        except requests.RequestException as e:
            last_error = e
        time.sleep(2 * (attempt + 1))
    raise last_error


# 兼容旧名称
call_hkbu_api = call_llm


def parse_json_response(raw_text: str):
    """鲁棒性：清洗并解析大模型返回的 JSON"""
    md_match = re.search(r'```(?:json)?\s*(\{.*?\})\s*```', raw_text, re.DOTALL | re.IGNORECASE)
    if md_match:
        try:
            return json.loads(md_match.group(1), strict=False)
        except json.JSONDecodeError:
            pass

    cleaned = re.sub(r'<think>.*?</think>', '', raw_text, flags=re.DOTALL)

    start_idx = cleaned.find('{')
    end_idx = cleaned.rfind('}')

    if start_idx != -1 and end_idx != -1 and end_idx >= start_idx:
        json_str = cleaned[start_idx:end_idx + 1]
        try:
            return json.loads(json_str, strict=False)
        except json.JSONDecodeError as e:
            raise ValueError(f"JSON 格式格式有误: {str(e)}\n提取的文本片段: {json_str[:200]}...")

    raise ValueError("在大模型的返回结果中没有找到任何有效的 {...} 数据。可能是回答被截断。")


def call_llm_json(system_prompt: str, user_message: str, api_key: str = None, model_name: str = "gpt-4.1",
                  temperature: float = 0.7, json_retries: int = 1, thinking: bool = False):
    """调用模型并解析 JSON；解析失败时重新生成一次。返回 (parsed, raw_text)。"""
    last_error = None
    for _ in range(json_retries + 1):
        raw = call_llm(system_prompt, user_message, api_key, model_name, temperature, thinking=thinking)
        try:
            return parse_json_response(raw), raw
        except ValueError as e:
            last_error = e
    raise last_error
