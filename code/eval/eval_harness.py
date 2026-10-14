"""三方对照评测：原生 AI（zero-shot） vs 当前版本（单次结构化 Prompt） vs 多阶段 Agent。

用法（在 code/ 目录下）：
    python eval/eval_harness.py --limit 3                       # 先跑 3 条冒烟测试
    python eval/eval_harness.py                                  # 跑完整测试集
    python eval/eval_harness.py --judge-only eval/results/<run>  # 复用已生成的回复，只重新评分

评分方式：同一案例下三个系统各 3 条回复，共 9 条，打乱顺序并隐去来源后交给评审模型统一打分。
评审模型默认与生成模型不同，以降低"自己给自己打分"的偏差。
"""

import argparse
import csv
import json
import os
import random
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from statistics import mean

CODE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, CODE_DIR)

from llm_client import call_llm_json  # noqa: E402
from prompts import get_instant_reply_prompt  # noqa: E402
from agent_runtime import run_agent_reply, DIMENSIONS  # noqa: E402
from agent_prompts import EVAL_DIMENSIONS  # noqa: E402

SYSTEMS = ["baseline", "v1_prompt", "v2_agent"]
SYSTEM_LABELS = {"baseline": "原生 AI（zero-shot）", "v1_prompt": "当前版本（单次结构化 Prompt）",
                 "v2_agent": "多阶段 Agent"}


def gen_baseline(case, model):
    system = "You are a helpful assistant."
    user = (f"对方发来消息：{case['message']}\n场景：{case['context']}\n"
            f"请用「{case['tone']}」的语气帮我写 3 条回复。只返回 JSON："
            '{"reply_1": "...", "reply_2": "...", "reply_3": "..."}')
    parsed, _ = call_llm_json(system, user, None, model, 0.7)
    return [parsed.get(f"reply_{i}", "") for i in (1, 2, 3)], 1, {}


def gen_v1(case, model):
    prompt = get_instant_reply_prompt(case["message"], case["tone"], case["context"])
    parsed, _ = call_llm_json(prompt, "请直接输出符合我要求的JSON格式的回复方案。", None, model, 0.7)
    return [parsed.get(f"reply_{i}", "") for i in (1, 2, 3)], 1, {}


def gen_v2(case, model):
    result = run_agent_reply(case["message"], case["tone"], case["context"], model)
    extra = {"revised": result.get("revised"), "best_id": result.get("best_id"),
             "stage_latency": {t["stage"]: t["latency_ms"] for t in result.get("trace", [])}}
    return [result.get(f"reply_{i}", "") for i in (1, 2, 3)], len(result.get("trace", [])), extra


GENERATORS = {"baseline": gen_baseline, "v1_prompt": gen_v1, "v2_agent": gen_v2}


def generate_case(case, model):
    record = {"case": case, "systems": {}}
    for name in SYSTEMS:
        start = time.time()
        try:
            replies, calls, extra = GENERATORS[name](case, model)
            record["systems"][name] = {"replies": replies, "latency_ms": int((time.time() - start) * 1000),
                                       "llm_calls": calls, "error": None, **extra}
        except Exception as e:
            record["systems"][name] = {"replies": [], "latency_ms": int((time.time() - start) * 1000),
                                       "llm_calls": 0, "error": str(e)[:300]}
    print(f"  generated {case['id']}", flush=True)
    return record


def judge_prompt(case, items):
    listing = "\n".join(f'- [{it["rid"]}] {it["text"]}' for it in items)
    return f"""你是一位严格、中立的沟通质量评审。下面是同一条消息的多条候选回复，来源已被隐去且顺序随机。

### [场景]
- 对方消息：{case['message']}
- 场景补充：{case['context']}
- 用户期望语气：{case['tone']}

### [候选回复]
{listing}

### [评分维度]（每项 1-5 分，5 分最好；请充分拉开差距，同等水平给同等分）
{EVAL_DIMENSIONS}

### [输出 JSON]
{{"scores": [{{"rid": "R1", "empathy": 0, "boundary": 0, "clarity": 0, "context_fit": 0, "naturalness": 0}}],
  "best_rid": "你认为最适合用户直接发送的一条"}}
请为每一条候选回复都给出评分。"""


def judge_case(record, judge_model, seed):
    case = record["case"]
    items = []
    for name in SYSTEMS:
        for idx, text in enumerate(record["systems"][name]["replies"], 1):
            if text:
                items.append({"system": name, "idx": idx, "text": text})
    rng = random.Random(f"{seed}-{case['id']}")
    rng.shuffle(items)
    for i, it in enumerate(items, 1):
        it["rid"] = f"R{i}"

    parsed, _ = call_llm_json(judge_prompt(case, items), "请严格按要求输出 JSON。", None, judge_model, 0.0, thinking=True)
    by_rid = {s.get("rid"): s for s in parsed.get("scores", [])}
    for it in items:
        s = by_rid.get(it["rid"], {})
        it["scores"] = {d: int(s.get(d, 0) or 0) for d in DIMENSIONS}
        it["total"] = sum(it["scores"].values())
        it["judge_best"] = parsed.get("best_rid") == it["rid"]
    print(f"  judged {case['id']}", flush=True)
    return {"case_id": case["id"], "lang": case["lang"], "category": case["category"], "items": items}


def summarize(records, judgments, gen_model, judge_model):
    rows, lines = [], []
    by_case = {r["case"]["id"]: r for r in records}

    stats = {}
    for name in SYSTEMS:
        all_items = [it for j in judgments for it in j["items"] if it["system"] == name and it["total"] > 0]
        first = [it for j in judgments for it in j["items"] if it["system"] == name and it["idx"] == 1 and it["total"] > 0]
        best_of_3 = []
        for j in judgments:
            mine = [it["total"] for it in j["items"] if it["system"] == name and it["total"] > 0]
            if mine:
                best_of_3.append(max(mine))
        gens = [r["systems"][name] for r in records]
        stats[name] = {
            "dims_all": {d: mean(it["scores"][d] for it in all_items) if all_items else 0 for d in DIMENSIONS},
            "total_all": mean(it["total"] for it in all_items) if all_items else 0,
            "total_first": mean(it["total"] for it in first) if first else 0,
            "total_best3": mean(best_of_3) if best_of_3 else 0,
            "judge_pick_rate": sum(1 for j in judgments for it in j["items"]
                                   if it["system"] == name and it["judge_best"]) / max(len(judgments), 1),
            "latency_s": mean(g["latency_ms"] for g in gens if not g["error"]) / 1000 if any(not g["error"] for g in gens) else 0,
            "calls": mean(g["llm_calls"] for g in gens if not g["error"]) if any(not g["error"] for g in gens) else 0,
            "fail_rate": sum(1 for g in gens if g["error"]) / max(len(gens), 1),
        }

    wins = {"v2_vs_baseline": [0, 0, 0], "v2_vs_v1": [0, 0, 0]}
    for j in judgments:
        firsts = {it["system"]: it["total"] for it in j["items"] if it["idx"] == 1}
        for key, other in (("v2_vs_baseline", "baseline"), ("v2_vs_v1", "v1_prompt")):
            if "v2_agent" in firsts and other in firsts:
                a, b = firsts["v2_agent"], firsts[other]
                wins[key][0 if a > b else (1 if a == b else 2)] += 1

    for j in judgments:
        for it in j["items"]:
            rows.append({"case_id": j["case_id"], "lang": j["lang"], "category": j["category"],
                         "system": it["system"], "reply_idx": it["idx"], **it["scores"], "total": it["total"],
                         "judge_best": it["judge_best"], "text": it["text"]})

    dim_names = {"empathy": "共情礼貌", "boundary": "边界守护", "clarity": "清晰有效",
                 "context_fit": "场景适配", "naturalness": "自然度"}
    lines.append("# 评测报告：原生 AI vs 当前版本 vs 多阶段 Agent\n")
    lines.append(f"- 运行时间：{datetime.now().strftime('%Y-%m-%d %H:%M')}")
    lines.append(f"- 测试集：{len(records)} 条（中文 {sum(1 for r in records if r['case']['lang'] == 'zh')} / "
                 f"英文 {sum(1 for r in records if r['case']['lang'] == 'en')}）")
    lines.append(f"- 生成模型：{gen_model}；评审模型：{judge_model}（盲评，9 条回复打乱顺序、隐去来源）")
    lines.append("- 评分：5 个维度各 1-5 分，总分 25 分\n")

    lines.append("## 总分对比\n")
    lines.append("| 系统 | 首条回复 | 3 条平均 | 3 条最佳 | 被评审选为最佳 | 平均耗时 | 模型调用次数 | 失败率 |")
    lines.append("| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |")
    for name in SYSTEMS:
        s = stats[name]
        lines.append(f"| {SYSTEM_LABELS[name]} | {s['total_first']:.2f} | {s['total_all']:.2f} | {s['total_best3']:.2f} | "
                     f"{s['judge_pick_rate']:.0%} | {s['latency_s']:.1f}s | {s['calls']:.1f} | {s['fail_rate']:.0%} |")

    lines.append("\n## 分维度平均分（全部回复）\n")
    lines.append("| 维度 | " + " | ".join(SYSTEM_LABELS[n] for n in SYSTEMS) + " |")
    lines.append("| :--- |" + " :---: |" * len(SYSTEMS))
    for d in DIMENSIONS:
        lines.append(f"| {dim_names[d]} | " + " | ".join(f"{stats[n]['dims_all'][d]:.2f}" for n in SYSTEMS) + " |")

    lines.append("\n## 首条回复逐案对比（Agent 胜 / 平 / 负）\n")
    for key, label in (("v2_vs_baseline", "Agent vs 原生 AI"), ("v2_vs_v1", "Agent vs 当前版本")):
        w, t, l = wins[key]
        lines.append(f"- {label}：{w} 胜 / {t} 平 / {l} 负")

    revised = [r["systems"]["v2_agent"].get("revised") for r in records if not r["systems"]["v2_agent"]["error"]]
    if revised:
        lines.append(f"- Agent 触发评审重写的比例：{sum(1 for x in revised if x) / len(revised):.0%}")

    lines.append("\n## 说明与局限\n")
    lines.append("- 评审为大模型自动打分（LLM-as-judge），可能与真人偏好存在偏差，结论需结合真人盲测验证。")
    lines.append("- \"首条回复\"指用户界面上第一个看到的回复：Agent 为评审选出的最优条，另外两个系统为模型输出的第一条。")
    lines.append("- 测试集规模较小，单次运行存在随机波动，差距小于 1 分时不宜下强结论。")
    return rows, "\n".join(lines) + "\n"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--gen-model", default="gpt-4.1")
    parser.add_argument("--judge-model", default="deepseek-v3")
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--seed", default="hera")
    parser.add_argument("--judge-only", default="")
    args = parser.parse_args()

    if args.judge_only:
        out_dir = args.judge_only
        with open(os.path.join(out_dir, "generations.jsonl"), encoding="utf-8") as f:
            records = [json.loads(line) for line in f]
    else:
        with open(os.path.join(os.path.dirname(__file__), "testset.json"), encoding="utf-8") as f:
            cases = json.load(f)
        if args.limit:
            cases = cases[:args.limit]
        out_dir = os.path.join(os.path.dirname(__file__), "results", datetime.now().strftime("%Y%m%d-%H%M%S"))
        os.makedirs(out_dir, exist_ok=True)
        print(f"Generating {len(cases)} cases with {args.gen_model} ...", flush=True)
        with ThreadPoolExecutor(args.workers) as pool:
            records = list(pool.map(lambda c: generate_case(c, args.gen_model), cases))
        with open(os.path.join(out_dir, "generations.jsonl"), "w", encoding="utf-8") as f:
            for r in records:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")

    print(f"Judging with {args.judge_model} ...", flush=True)

    def safe_judge(r):
        try:
            return judge_case(r, args.judge_model, args.seed)
        except Exception as e:
            print(f"  judge failed {r['case']['id']}: {e}", flush=True)
            return None

    with ThreadPoolExecutor(args.workers) as pool:
        judgments = [j for j in pool.map(safe_judge, records) if j]
    with open(os.path.join(out_dir, "judgments.jsonl"), "w", encoding="utf-8") as f:
        for j in judgments:
            f.write(json.dumps(j, ensure_ascii=False) + "\n")

    rows, report = summarize(records, judgments, args.gen_model, args.judge_model)
    with open(os.path.join(out_dir, "scores.csv"), "w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    with open(os.path.join(out_dir, "report.md"), "w", encoding="utf-8") as f:
        f.write(report)
    print(report)
    print(f"Results saved to {out_dir}")


if __name__ == "__main__":
    main()
