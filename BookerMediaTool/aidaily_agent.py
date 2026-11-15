# -*- coding: utf-8 -*-
"""aidaily 子命令的智能体：LLM 精筛单条条目。"""
import json
import logging

from pydantic import parse_obj_as

from .util import render_prompt, ext_code_block
from .openai import OpenAIClient
from .aidaily_models import Article, Enrichment
from .aidaily_pmt import AIDAILY_ENRICH_SYS_PMT, AIDAILY_ENRICH_USER_PMT

logger = logging.getLogger(__name__)

# 按信源分支给 LLM 的提示词 hint，与任务书 §3.3 约定一致
SOURCE_HINT = {
    'arxiv': '侧重判断是否涉及可在消费级硬件上运行/训练的模型（算力要求、显存、量化、单卡可行性等）。',
    'hn': '侧重开发者部署经验（能否在本地/消费级显卡上跑、编译安装、推理速度等）。',
    'tds': '侧重教程的可复现性与硬件门槛（Colab、显卡需求、低成本方案等）。',
    'machine_heart': '侧重开源工具与消费级硬件适配（是否开源、可在什么设备上运行等）。',
    'quantumbit': '侧重开源工具与消费级硬件适配（是否开源、可在什么设备上运行等）。',
    'xinzhiyuan': '侧重开源工具与消费级硬件适配（是否开源、可在什么设备上运行等）。',
}


def _hint_for(source: str) -> str:
    return SOURCE_HINT.get(source, '侧重是否与消费级/低资源 AI 能力下沉相关。')


class AidailyAgent:
    """封装 aidaily 的大模型调用：对单条条目做精筛（相关度/分类/资源信号/一句话说点）。"""

    def __init__(self, args):
        self.args = args
        self.client = OpenAIClient.from_args(args)

    def enrich_one(self, article: Article) -> Enrichment:
        """对单条文章做 LLM 精筛，返回 Enrichment；解析失败抛异常以供重试。"""
        user = render_prompt(
            AIDAILY_ENRICH_USER_PMT,
            source=article.source,
            title=article.title,
            summary=article.summary,
            url=article.url,
            hint=_hint_for(article.source),
        )
        msgs = [
            {'role': 'system', 'content': AIDAILY_ENRICH_SYS_PMT},
            {'role': 'user', 'content': user},
        ]
        logger.debug('enrich one: source=%s title=%s', article.source, article.title)
        return self.client.call_llm_retry(
            msgs,
            self.args.model,
            retry=getattr(self.args, 'retry', 10),
            temp=getattr(self.args, 'temp', 1),
            max_tokens=getattr(self.args, 'llm_max_tokens', 512),
            parse_output=self._parse_enrichment,
        )

    def _parse_enrichment(self, text: str) -> Enrichment:
        """解析大模型输出：三反引号代码块 → JSON → Enrichment。失败抛异常触发重试。"""
        try:
            block = ext_code_block(text)
        except Exception:
            block = text
        data = json.loads(block)
        return parse_obj_as(Enrichment, data)
