# -*- coding: utf-8 -*-
"""aidaily 子命令的提示词模板。"""
import hashlib

from .util import md5

# 四个主题分类标识符及其含义，供 system prompt 与编排器共用
CATEGORIES_INTRO = '''
四个主题分类标识符及含义：
- compression：模型压缩（量化、低比特、剪切、蒸馏、稀疏化等）
- weight_edit：权重操控（知识编辑、模型编辑、权重编辑、模型合并等）
- edge：端侧与消费级推理（on-device 推理、消费级 GPU、llama.cpp、GGUF、本地推理等）
- low_resource：低资源训练（MeZO、零阶优化、显存高效微调、低资源训练、单卡训练等）
'''

RESOURCE_SIGNAL_INTRO = '''
resource_signal 允许取值：
- a100_80g：A100 80G 级别算力
- consumer_24g：消费级 24G 显卡（如 RTX 3090/4090）
- consumer_16g：消费级 16G 显卡（如 RTX 4070 Ti/3060 等）
- cpu：仅 CPU 可运行
- colab：Colab 等免费云笔记本
- single_gpu：单消费级卡（未明确显存）
- unspecified：未提及或无法判断
'''

AIDAILY_ENRICH_SYS_PMT = '''
你是一位“AI 平权日报”的编辑，编辑视角是“算力与模型能力下沉、普惠”，关注普通开发者与小机构能否用消费级硬件落地模型。

你的任务是阅读一条采集来的文章/论文，判断其与“AI 平权”主题的相关度，打上主题分类，提炼一句话要点，并给出算力资源信号。

{categories}

{resource_signal}

输出要求：
1. 只输出 JSON，不要包含任何其他说明文字。
2. JSON 必须用三个反引号包裹（```json ... ```）。
3. relevance 取 0~1 之间的浮点数；与算法下沉/普惠无关的内容（如纯量子物理、融资裁员、公司八卦） relevance 应为 0。
4. categories 为命中的主题分类标识符列表，可多个，无匹配则为空数组 []。
5. one_liner 用一句中文概括文章要点。
6. resource_signal 必须取上述允许值之一，文中未提及则用 unspecified。
'''.strip().replace('{categories}', CATEGORIES_INTRO).replace('{resource_signal}', RESOURCE_SIGNAL_INTRO)

AIDAILY_ENRICH_USER_PMT = '''
## 待分析条目

- 信源：{source}
- 标题：{title}
- URL：{url}
- 摘要：{summary}

{hint}

## 任务

请按 system prompt 中的要求，输出结构化 JSON（三个反引号包裹）。
'''

# 提示词版本，作为 LLM 缓存键的一部分；提示词任意改动都会导致版本变更
prompt_version = md5(AIDAILY_ENRICH_SYS_PMT + AIDAILY_ENRICH_USER_PMT)
