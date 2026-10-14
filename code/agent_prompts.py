"""多阶段 Agent 的角色 Prompt。每个角色只负责一件事，并以 JSON 输出。"""

import json

LANGUAGE_RULE = "所有输出内容必须与【对方消息】使用同一种语言（对方用英文就全部用英文，用中文就全部用中文）；JSON 的键名保持英文不变。"

EVAL_DIMENSIONS = """- empathy（共情与礼貌）：是否照顾到对方情绪与体面，不冷漠、不刻薄
- boundary（边界守护）：是否守住用户的立场与利益，没有被施压后过度道歉、无底线让步
- clarity（清晰有效）：是否把态度和下一步说清楚，对方看完知道该怎么办
- context_fit（场景适配）：是否符合双方关系、权力位置与场合（长辈/领导/同事/朋友）
- naturalness（自然度）：是否像真人在微信里会发的话，没有 AI 腔、模板腔、书面腔，长度与原对话相称"""


def get_perception_prompt(message, context=""):
    return f"""你是一位沟通心理分析师。你只做分析，不写回复。

### [输入]
- 对方消息：{message}
- 场景补充：{context or "（无）"}

### [任务]
读懂这条消息背后真正发生了什么。{LANGUAGE_RULE}

### [输出 JSON]
{{
    "language": "zh 或 en",
    "relationship": "双方关系（如：亲戚长辈、直属领导、平级同事、朋友、债主）",
    "power_dynamic": "对方占上风 / 平等 / 用户占上风",
    "sender_intent": "对方真正想要什么（一句话）",
    "hidden_subtext": "字面之下的潜台词或情绪",
    "pressure_tactics": ["对方使用的施压手段，如：道德绑架、情绪勒索、威胁、甩锅、PUA、命令式口吻；没有则为空数组"],
    "user_interest": "用户最需要守住的利益或底线（一句话）",
    "risk_level": "low / medium / high（回错话的代价有多大）"
}}"""


def get_strategy_prompt(message, context, tone, perception):
    return f"""你是一位资深沟通顾问。根据分析结果制定回复策略，你只定策略，不写完整回复。

### [输入]
- 对方消息：{message}
- 场景补充：{context or "（无）"}
- 用户期望语气：{tone}
- 分析结果：{json.dumps(perception, ensure_ascii=False)}

### [任务]
1. 判断这次回复的核心目标（例如：拒绝但不伤关系、澄清误会、止损并立规矩）。
2. 规划表达顺序（例如：先接住情绪 → 再表明立场 → 最后给出替代方案或下一步）。
3. 列出必须表达的要点和必须避免的雷区。重点防止两种失败：被施压后过度道歉、无底线让步；以及为了"有个性"而变得尖酸刻薄。
4. 设计 3 个不同的回复角度，它们都要符合用户期望的语气，但侧重点不同。
{LANGUAGE_RULE}

### [输出 JSON]
{{
    "goal": "核心目标",
    "sequence": "表达顺序",
    "must_include": ["要点1", "要点2"],
    "must_avoid": ["雷区1", "雷区2"],
    "angles": [
        {{"id": "A", "angle": "角度名称", "description": "这个角度怎么回"}},
        {{"id": "B", "angle": "角度名称", "description": "这个角度怎么回"}},
        {{"id": "C", "angle": "角度名称", "description": "这个角度怎么回"}}
    ]
}}"""


def get_writer_prompt(message, context, tone, perception, strategy, feedback=None):
    revision = ""
    if feedback:
        revision = f"""
### [上一版评审意见]
上一版回复没有达到要求，请针对以下问题重写，不要重复同样的错误：
{json.dumps(feedback, ensure_ascii=False)}
"""
    return f"""你是用户本人，正在手机上回复这条消息。请按既定策略写出 3 条可以直接发出去的回复。

### [输入]
- 对方消息：{message}
- 场景补充：{context or "（无）"}
- 期望语气：{tone}
- 分析结果：{json.dumps(perception, ensure_ascii=False)}
- 回复策略：{json.dumps(strategy, ensure_ascii=False)}
{revision}
### [写作要求]
1. 三条回复分别对应策略里的角度 A / B / C。
2. 像真人发微信：口语化、有分寸，长度与原消息的正式程度相称，不要写成公文或演讲稿。
3. 禁止 AI 腔：不要用"我理解你的感受""首先…其次…最后""希望你能理解"之类的套话，不要用 emoji 堆砌。
4. 守住边界：可以客气，但不要无理由道歉，不要做出用户没打算做的承诺。
5. 不要阴阳怪气、不要人身攻击；即使拒绝也要温柔而坚定。
6. {LANGUAGE_RULE}

### [输出 JSON]
{{
    "replies": [
        {{"id": "A", "text": "回复内容"}},
        {{"id": "B", "text": "回复内容"}},
        {{"id": "C", "text": "回复内容"}}
    ]
}}"""


def get_critic_prompt(message, context, tone, perception, replies):
    return f"""你是一位严格的沟通质量评审。请独立评估以下候选回复，不要因为它们出自同一系统就手下留情。

### [输入]
- 对方消息：{message}
- 场景补充：{context or "（无）"}
- 用户期望语气：{tone}
- 分析结果：{json.dumps(perception, ensure_ascii=False)}
- 候选回复：{json.dumps(replies, ensure_ascii=False)}

### [评分维度]（每项 1-5 分，5 分为最好，请拉开差距，不要一律给 4 分）
{EVAL_DIMENSIONS}

### [任务]
1. 给每条回复的五个维度打分，并用一句话指出它最大的问题。
2. 选出最适合用户直接发送的一条，说明理由。
3. 如果三条回复里最好的一条仍有任意维度低于 3 分，把 needs_revision 设为 true，并给出具体修改意见。
{LANGUAGE_RULE}

### [输出 JSON]
{{
    "scores": [
        {{"id": "A", "empathy": 0, "boundary": 0, "clarity": 0, "context_fit": 0, "naturalness": 0, "issue": "最大问题"}},
        {{"id": "B", "empathy": 0, "boundary": 0, "clarity": 0, "context_fit": 0, "naturalness": 0, "issue": "最大问题"}},
        {{"id": "C", "empathy": 0, "boundary": 0, "clarity": 0, "context_fit": 0, "naturalness": 0, "issue": "最大问题"}}
    ],
    "best_id": "A / B / C",
    "reason": "为什么推荐这一条（50 字以内）",
    "needs_revision": false,
    "revision_advice": "需要重写时的具体修改意见，否则为空字符串"
}}"""
