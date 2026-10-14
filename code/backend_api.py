from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
import os
import json
from dotenv import load_dotenv
from pydantic import BaseModel
from typing import Optional

from prompts import get_instant_reply_prompt, get_sandbox_simulation_prompt
from llm_client import call_hkbu_api, parse_json_response
from agent_runtime import run_agent_reply

# 加载 .env 中的环境变量并强制覆盖终端缓存
load_dotenv(override=True)

app = FastAPI(title="High-EQ Reply Assistant API", description="AI沟通助手项目核心后端接口")

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
STATIC_DIR = os.path.join(BASE_DIR, "static")

app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")


# ================================
# 数据模型定义 (Request / Response validation)
# ================================
DEFAULT_MODEL = os.environ.get("LLM_DEFAULT_MODEL", "qwen3.8-max")

class InstantReplyRequest(BaseModel):
    message: str
    tone: str
    context: Optional[str] = ""
    model: Optional[str] = DEFAULT_MODEL
    api_key: Optional[str] = None  # 允许由前端随意透传新的临时API Key过来

class SandboxRequest(BaseModel):
    scenario: str
    my_role: str
    target_role: str
    my_plan: str
    model: Optional[str] = DEFAULT_MODEL
    api_key: Optional[str] = None  # 允许由前端随意透传新的临时API Key过来


# ================================
# 路由接口定义 
# ================================
@app.get("/")
def read_root():
    return FileResponse(os.path.join(STATIC_DIR, "index.html"))

@app.post("/api/instant-reply")
def instant_reply(req: InstantReplyRequest):
    """即时高情商回复生成接口"""
    prompt = get_instant_reply_prompt(req.message, req.tone, req.context)
    raw_response = ""
    try:
        raw_response = call_hkbu_api(prompt, "请直接输出符合我要求的JSON格式的回复方案。", req.api_key, req.model)
        parsed = parse_json_response(raw_response)
        return {"status": "success", "data": parsed}
    except json.JSONDecodeError as je:
        raise HTTPException(status_code=500, detail=f"LLM JSON Error: {str(je)}\nRaw: {raw_response}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/agent-reply")
def agent_reply(req: InstantReplyRequest):
    """多阶段 Agent 回复接口：感知 → 策略 → 撰写 → 评审（必要时重写一轮）"""
    try:
        result = run_agent_reply(req.message, req.tone, req.context, req.model, req.api_key)
        return {"status": "success", "data": result}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/sandbox-simulation")
def sandbox_simulation(req: SandboxRequest):
    """场景模拟推演接口"""
    prompt = get_sandbox_simulation_prompt(req.scenario, req.my_role, req.target_role, req.my_plan)
    raw_response = ""
    try:
        raw_response = call_hkbu_api(prompt, "请不要客气，直接开始锐利的沙盘推演，并仅返回严格的JSON。", req.api_key, req.model)
        parsed = parse_json_response(raw_response)
        return {"status": "success", "data": parsed}
    except json.JSONDecodeError as je:
        raise HTTPException(status_code=500, detail=f"LLM JSON Error: {str(je)}\nRaw: {raw_response}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    # 为了方便热重载调试，终端启动建议使用：uvicorn backend_api:app --reload
    uvicorn.run("backend_api:app", host="0.0.0.0", port=8000, reload=True)
