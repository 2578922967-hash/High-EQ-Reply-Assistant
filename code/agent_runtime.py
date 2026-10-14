"""多阶段回复 Agent：感知 → 策略 → 撰写 → 评审，评审不达标时带着意见重写一轮。"""

import time

from llm_client import call_llm_json
from prompts import get_instant_reply_prompt
from agent_prompts import (
    get_perception_prompt,
    get_strategy_prompt,
    get_writer_prompt,
    get_critic_prompt,
)

DIMENSIONS = ["empathy", "boundary", "clarity", "context_fit", "naturalness"]
MAX_REVISIONS = 1
STAGE_TEMPERATURE = {"perception": 0.2, "strategy": 0.4, "writer": 0.8, "critic": 0.1}


def _is_attack_tone(tone: str) -> bool:
    return any(k in tone for k in ("阴阳怪气", "攻击", "Sarcastic", "Aggressive"))


def _run_stage(trace, name, system_prompt, user_message, api_key, model):
    start = time.time()
    parsed, _ = call_llm_json(system_prompt, user_message, api_key, model, STAGE_TEMPERATURE[name.split("#")[0]])
    trace.append({"stage": name, "latency_ms": int((time.time() - start) * 1000), "output": parsed})
    return parsed


def _total(score: dict) -> int:
    return sum(int(score.get(d, 0) or 0) for d in DIMENSIONS)


def run_agent_reply(message: str, tone: str, context: str = "", model: str = "qwen3.8-max", api_key: str = None) -> dict:
    """返回结构兼容旧版 /api/instant-reply（reply_1~3 + explanation），并附带分析、策略、评分与执行记录。"""
    trace = []
    start = time.time()

    # 攻击模式的目标与"高情商"评审标准相反，沿用原有单次生成逻辑
    if _is_attack_tone(tone):
        parsed = _run_stage(trace, "writer", get_instant_reply_prompt(message, tone, context),
                            "请直接输出符合我要求的JSON格式的回复方案。", api_key, model)
        parsed.update({"mode": "single_shot", "trace": trace,
                       "total_latency_ms": int((time.time() - start) * 1000)})
        return parsed

    go = "请严格按要求输出 JSON。"
    perception = _run_stage(trace, "perception", get_perception_prompt(message, context), go, api_key, model)
    strategy = _run_stage(trace, "strategy", get_strategy_prompt(message, context, tone, perception), go, api_key, model)

    feedback = None
    for round_idx in range(MAX_REVISIONS + 1):
        suffix = f"#{round_idx}" if round_idx else ""
        written = _run_stage(trace, f"writer{suffix}",
                             get_writer_prompt(message, context, tone, perception, strategy, feedback), go, api_key, model)
        replies = written.get("replies", [])[:3]
        critique = _run_stage(trace, f"critic{suffix}",
                              get_critic_prompt(message, context, tone, perception, replies), go, api_key, model)
        if not critique.get("needs_revision") or round_idx == MAX_REVISIONS:
            break
        feedback = {"advice": critique.get("revision_advice"),
                    "issues": [s.get("issue") for s in critique.get("scores", [])]}

    scores = {s.get("id"): s for s in critique.get("scores", [])}
    best_id = critique.get("best_id")
    if best_id not in scores and scores:
        best_id = max(scores, key=lambda k: _total(scores[k]))
    ordered = sorted(replies, key=lambda r: (r.get("id") != best_id, -_total(scores.get(r.get("id"), {}))))

    result = {
        "mode": "agent",
        "explanation": critique.get("reason", ""),
        "analysis": perception,
        "strategy": strategy,
        "candidates": [
            {"id": r.get("id"), "text": r.get("text", ""), "scores": scores.get(r.get("id"), {}),
             "total": _total(scores.get(r.get("id"), {}))}
            for r in ordered
        ],
        "best_id": best_id,
        "revised": round_idx > 0,
        "trace": trace,
        "total_latency_ms": int((time.time() - start) * 1000),
    }
    for i, r in enumerate(ordered, 1):
        result[f"reply_{i}"] = r.get("text", "")
    return result
