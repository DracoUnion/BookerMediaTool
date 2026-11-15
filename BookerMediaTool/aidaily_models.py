# -*- coding: utf-8 -*-
"""aidaily 子命令的 Pydantic 数据模型。"""
from typing import List, Literal
from pydantic import BaseModel, Field

Category = Literal["compression", "weight_edit", "edge", "low_resource"]
ResourceSignal = Literal[
    "a100_80g",
    "consumer_24g",
    "consumer_16g",
    "cpu",
    "colab",
    "single_gpu",
    "unspecified",
]

CATEGORY_CN = {
    "compression": "模型压缩",
    "weight_edit": "权重操控",
    "edge": "端侧与消费级推理",
    "low_resource": "低资源训练",
}


class Article(BaseModel):
    """采集到的原始信源条目。"""

    id: str = Field(..., description='条目唯一 ID，md5(source+url+title)[:16]')
    source: str = Field(..., description='信源标识符：arxiv/hn/tds/machine_heart/quantumbit/xinzhiyuan')
    url: str = Field(..., description='原文链接')
    title: str = Field(..., description='标题')
    summary: str = Field('', description='摘要或正文片段')
    published_at: str = Field(..., description='发布时间，保留信源原始字符串')
    lang: str = Field(..., description='语言标识：zh/en')
    run_date: str = Field(..., description='采集所属日期，YYYY-MM-DD')


class DedupedArticle(Article):
    """去重后的主条目，低优先级重复条目挂在 evidence 下。"""

    is_primary: bool = Field(True, description='是否为去重组的主条目')
    evidence_for: str = Field('', description='若非主条目，指向主条目的 id')
    evidence: List[dict] = Field(default_factory=list, description='与主条目重复的低优先级条目列表')


class Enrichment(BaseModel):
    """大模型精筛后的结构化判断。"""

    relevance: float = Field(..., ge=0, le=1, description='与 AI 平权主题的相关度，0~1')
    categories: List[Category] = Field(..., description='命中的主题分类标识符列表')
    one_liner: str = Field(..., description='一句话要点')
    resource_signal: ResourceSignal = Field(..., description='算力资源信号')


class EnrichedArticle(DedupedArticle):
    """带大模型精筛结果的条目。"""

    enrichment: Enrichment = Field(..., description='大模型精筛结果')


class RunStatus(BaseModel):
    """单次运行状态，写入 run.json。"""

    run_date: str = Field(..., description='运行日期，YYYY-MM-DD')
    started_at: str = Field(..., description='开始时间，ISO 格式')
    finished_at: str = Field('', description='结束时间，ISO 格式')
    status: Literal["running", "success", "degraded", "failed"] = Field(
        "running", description='运行状态：running/success/degraded/failed'
    )
    exit_code: int = Field(0, description='退出码')
    source_status: dict = Field(default_factory=dict, description='各信源采集状态')
    counts: dict = Field(default_factory=dict, description='各阶段条目数统计')
    cli_args: dict = Field(default_factory=dict, description='本次运行的关键 CLI 参数')
