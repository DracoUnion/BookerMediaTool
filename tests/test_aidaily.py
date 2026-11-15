# -*- coding: utf-8 -*-
"""aidaily 子命令测试（离线，mock LLM，覆盖任务书 §10 B–G）。"""
import argparse
import datetime as dt
import json
import os
import shutil
from os import path
from typing import List

from BookerMediaTool import aidaily
from BookerMediaTool.aidaily_models import Article, Enrichment
from BookerMediaTool.aidaily import AidailyOrchestrator, FileLock, _ArgError
from BookerMediaTool.util import md5

FIXTURE = path.join(path.dirname(__file__), 'fixtures', 'aidaily')
LOCAL_TZ = dt.datetime.now().astimezone().tzinfo


def _make_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('-m', '--model', help='模型名称（公共参数，由 __main__ 根解析器提供）')
    sub = parser.add_subparsers()
    aidaily.reg_subparser(sub)
    return parser


def _args(tmp_path, **kw):
    parser = _make_parser()
    ns = parser.parse_args(['aidaily', '--out-dir', str(tmp_path)])
    ns.date = kw.get('date', '2026-11-15')
    ns.run_date = kw.get('run_date', '2026-11-15')
    # 公共 LLM 参数由 __main__ 根解析器提供；此处补齐默认值供编排器使用
    ns.model = kw.get('model', 'gpt-4o-mini')
    ns.key = kw.get('key', 'sk-test')
    ns.host = kw.get('host', None)
    ns.temp = kw.get('temp', 1)
    ns.retry = kw.get('retry', 1_000_000)
    ns.emb = kw.get('emb', None)
    for k, v in kw.items():
        setattr(ns, k, v)
    return ns


def test_help_defaults():
    parser = _make_parser()
    ns = parser.parse_args(['aidaily'])
    assert ns.stage == 'all'
    assert ns.out_dir == './ai_daily_out'
    assert ns.simhash_max_distance == 3
    assert ns.semantic_threshold == 0.85
    assert ns.hot_window_boost == 0.80
    assert ns.hot_window_count == 200
    assert ns.llm_max_concurrency == 4
    assert ns.llm_max_tokens == 512
    assert 'xinzhiyuan' not in ns.sources


def test_model_before_subcommand(tmp_path):
    # -m 在子命令之前：走 __main__ 根解析器（公共参数由根提供）
    ns = _make_parser().parse_args(['-m', 'm1', 'aidaily', '--out-dir', str(tmp_path)])
    assert ns.model == 'm1'


def test_validation_errors(tmp_path):
    parser = _make_parser()

    def parse(extra):
        return parser.parse_args(['aidaily', '--out-dir', str(tmp_path)] + extra)

    for extra in [
        ['--simhash-max-distance', '99'],
        ['--simhash-max-distance', '-1'],
        ['--semantic-threshold', '1.5'],
        ['--semantic-threshold', '0'],
        ['--sources', 'arxiv,foo'],
        ['--window-hours', '0'],
        ['--stage', 'bogus'],
    ]:
        ns = parse(extra)
        try:
            AidailyOrchestrator.validate_args(parser, ns)
        except SystemExit as e:
            assert e.code == 2, extra
        except _ArgError:
            pass
        else:
            raise AssertionError(f'期望退出码 2：{extra}')

    # --since/--until 必须成对
    ns = parse(['--since', '2026-01-01'])
    try:
        AidailyOrchestrator.validate_args(parser, ns)
    except SystemExit as e:
        assert e.code == 2
    except _ArgError:
        pass
    else:
        raise AssertionError('期望退出码 2：--since 单独出现')


def test_stats_empty(tmp_path):
    args = _args(tmp_path, stage='stats')
    code = AidailyOrchestrator(args).run()
    assert code == 0
    assert path.isfile(path.join(tmp_path, '2026-11-15', 'stats.txt'))


def test_dry_run(tmp_path):
    args = _args(tmp_path, stage='all', dry_run=True)
    code = AidailyOrchestrator(args).run()
    assert code == 0
    assert not path.isfile(path.join(tmp_path, '2026-11-15', 'run.json'))


def test_filter_rules(tmp_path):
    args = _args(tmp_path)
    out = path.join(tmp_path, args.run_date)
    os.makedirs(out, exist_ok=True)
    os.makedirs(out, exist_ok=True)
    shutil.copy(path.join(FIXTURE, 'raw_sample.yaml'), path.join(out, 'raw.yaml'))
    assert path.isfile(path.join(out, 'raw.yaml')), out
    orch = AidailyOrchestrator(args)
    filtered = orch.step_filter()
    orch = AidailyOrchestrator(args)
    filtered = orch.step_filter()
    titles = [a.title for a in filtered]
    assert 'INT4 Quantization for LLMs on a Single 24G GPU' in titles
    assert 'Hands-on Fine-tuning Llama on Colab' in titles   # tds 额外关键词 colab/hands-on
    assert 'llama.cpp 1.0 released' in titles                  # hn github.com 直保
    assert 'Quantum Error Correction for Surface Codes' not in titles  # 无主题词
    assert 'My weekend project' not in titles                  # hn 非兜底域名且无关键词
    assert '某 AI 公司 CEO 裁员并新一轮融资' not in titles       # 黑名单
    assert '消费级显卡本地运行大模型指南' not in titles          # 无主题词匹配
    assert orch.counts['filtered'] == 4   # 含重复 arXiv 条目，去重在 dedup 阶段


def test_dedup_simhash(tmp_path):
    args = _args(tmp_path)
    out = path.join(tmp_path, args.run_date)
    os.makedirs(out, exist_ok=True)
    os.makedirs(out, exist_ok=True)
    shutil.copy(path.join(FIXTURE, 'raw_sample.yaml'), path.join(out, 'raw.yaml'))
    orch = AidailyOrchestrator(args)
    filtered = orch.step_filter()
    deduped = orch.step_dedup()
    ids = [d.id for d in deduped]
    assert len(deduped) == 3, ids
    primary = [d for d in deduped if d.id == 'a0000001'][0]
    assert len(primary.evidence) == 1
    assert primary.evidence[0]['source'] == 'arxiv'
    assert path.isfile(path.join(tmp_path, '.index', 'articles.json'))
    assert path.isfile(path.join(tmp_path, '.index', 'simhash.json'))


def test_lock_conflict(tmp_path):
    lock = path.join(tmp_path, '.index', '.lock')
    l1 = FileLock(lock)
    assert l1.acquire() is True
    l2 = FileLock(lock)
    assert l2.acquire() is False
    l1.release()
    l3 = FileLock(lock)
    assert l3.acquire() is True
    l3.release()


def _fake_enrichment(relevance, categories, one_liner, signal):
    return Enrichment(
        relevance=relevance, categories=categories,
        one_liner=one_liner, resource_signal=signal,
    )


def _bind(agent, func):
    return func.__get__(agent, type(agent))


def test_enrich_and_cache(tmp_path):
    args = _args(tmp_path, stage='enrich')
    out = path.join(tmp_path, args.run_date)
    os.makedirs(out, exist_ok=True)
    shutil.copy(path.join(FIXTURE, 'deduped_sample.yaml'), path.join(out, 'deduped.yaml'))
    orch = AidailyOrchestrator(args)

    calls = []

    def fake_enrich_one(self, article):
        calls.append(article.id)
        by_title = {
            'INT4 Quantization for LLMs on a Single 24G GPU': _fake_enrichment(
                0.9, ['compression'], '4位量化单卡24G', 'consumer_24g'),
            'Transformer Math on CPU': _fake_enrichment(
                0.2, ['edge'], 'CPU 推理', 'cpu'),
            'Hands-on Fine-tuning Llama on Colab': _fake_enrichment(
                0.95, ['low_resource'], 'Colab 微调', 'consumer_16g'),
            'llama.cpp 1.0 released': _fake_enrichment(
                0.8, ['edge'], '本地推理', 'consumer_24g'),
        }
        return by_title[article.title]

    orch.agent.enrich_one = _bind(orch.agent, fake_enrich_one)
    enriched = orch.step_enrich()
    ids = [e.id for e in enriched]
    assert 'Unrelated GitHub repo' not in ids  # relevance 0 被剔除（本批次无此条目）
    assert len(enriched) == 3
    assert orch.counts['enrich_called'] == 3
    assert orch.counts['enrich_failed'] == 0

    cache_dir = path.join(tmp_path, '.cache', 'enrich')
    cache_files = [f for f in os.listdir(cache_dir) if path.isfile(path.join(cache_dir, f))]
    assert len(cache_files) == 3, cache_files

    # 第二次运行：命中缓存，enrich_one 不再被调用
    calls2 = []

    def fake_enrich_one2(self, article):
        calls2.append(article.id)
        return _fake_enrichment(0.9, ['compression'], 'x', 'unspecified')

    orch2 = AidailyOrchestrator(args)
    orch2.agent.enrich_one = _bind(orch2.agent, fake_enrich_one2)
    enriched2 = orch2.step_enrich()
    assert calls2 == []
    assert [e.id for e in enriched2] == ids


def test_enrich_failure_exit_code(tmp_path):
    args = _args(tmp_path, stage='enrich')
    out = path.join(tmp_path, args.run_date)
    os.makedirs(out, exist_ok=True)
    shutil.copy(path.join(FIXTURE, 'deduped_sample.yaml'), path.join(out, 'deduped.yaml'))
    orch = AidailyOrchestrator(args)

    def fake_raise(self, article):
        raise RuntimeError('LLM down')

    orch.agent.enrich_one = _bind(orch.agent, fake_raise)
    code = orch.run()
    assert code == 4


def test_render_output(tmp_path):
    args = _args(tmp_path, stage='render')
    out = path.join(tmp_path, args.run_date)
    os.makedirs(out, exist_ok=True)
    os.makedirs(out, exist_ok=True)
    shutil.copy(path.join(FIXTURE, 'raw_sample.yaml'), path.join(out, 'raw.yaml'))
    shutil.copy(path.join(FIXTURE, 'deduped_sample.yaml'), path.join(out, 'filtered.yaml'))
    shutil.copy(path.join(FIXTURE, 'deduped_sample.yaml'), path.join(out, 'deduped.yaml'))
    shutil.copy(path.join(FIXTURE, 'enriched_sample.yaml'), path.join(out, 'enriched.yaml'))
    orch = AidailyOrchestrator(args)
    code = orch.run()
    assert code == 0

    def _read(name):
        return open(path.join(out, name), encoding='utf8').read()

    md = _read('digest.md')
    assert '# AI平权日报 · 2026-11-15' in md
    assert '## 模型压缩' in md
    assert '## 权重操控' in md
    assert '## 端侧与消费级推理' in md
    assert '## 低资源训练' in md
    assert '## 观察区（相关度 < 0.3）' in md
    assert '统计：采集' in md
    assert '🟢' in md  # consumer_24g / consumer_16g 条目
    assert ('[INT4 Quantization for LLMs on a Single 24G GPU]'
            '(https://arxiv.org/abs/2401.00001)') in md

    html = _read('digest.html')
    assert html.startswith('<!DOCTYPE html>')
    assert '🟢' in html

    data = json.loads(_read('digest.json'))
    assert data['date'] == '2026-11-15'
    assert set(data) >= {'stats', 'categories', 'observations'}
    assert data['stats']['formal'] == 2
    assert len(data['observations']) == 1


def test_render_idempotent_overwrite(tmp_path):
    args = _args(tmp_path, stage='render')
    out = path.join(tmp_path, args.run_date)
    os.makedirs(out, exist_ok=True)
    shutil.copy(path.join(FIXTURE, 'raw_sample.yaml'), path.join(out, 'raw.yaml'))
    shutil.copy(path.join(FIXTURE, 'deduped_sample.yaml'), path.join(out, 'filtered.yaml'))
    shutil.copy(path.join(FIXTURE, 'deduped_sample.yaml'), path.join(out, 'deduped.yaml'))
    shutil.copy(path.join(FIXTURE, 'enriched_sample.yaml'), path.join(out, 'enriched.yaml'))

    def _read(name):
        return open(path.join(out, name), encoding='utf8').read()

    AidailyOrchestrator(args).run()
    md1 = _read('digest.md')
    AidailyOrchestrator(args).run()
    md2 = _read('digest.md')
    assert md1 == md2
    assert md1.count('# AI平权日报') == 1


def test_all_idempotent_when_success(tmp_path):
    args = _args(tmp_path, stage='all')
    out = path.join(tmp_path, args.run_date)
    os.makedirs(out, exist_ok=True)
    os.makedirs(out, exist_ok=True)
    open(path.join(out, 'run.json'), 'w', encoding='utf8').write(
        json.dumps({'run_date': '2026-11-15', 'status': 'success'}, ensure_ascii=False)
    )
    code = AidailyOrchestrator(args).run()
    assert code == 0


def test_collect_xinzhiyuan_skipped(tmp_path):
    """xinzhiyuan 无 feed 时应标记 skipped、不联网、不算失败。"""
    args = _args(tmp_path, sources='xinzhiyuan')
    orch = AidailyOrchestrator(args)
    now = dt.datetime.now(LOCAL_TZ)
    arts = orch._collect_source(
        'xinzhiyuan',
        now - dt.timedelta(hours=24),
        now,
    )
    assert arts == []
    # step_collect 会把 source_status 置为 skipped
    orch.step_collect()
    assert orch.source_status.get('xinzhiyuan') == 'skipped'
    assert orch.counts.get('raw') == 0


def test_make_article_window():
    since = dt.datetime(2026, 11, 14, tzinfo=LOCAL_TZ)
    until = dt.datetime(2026, 11, 15, tzinfo=LOCAL_TZ)
    entry = {'title': 'T', 'url': 'https://example.com/1', 'summary': 'S',
             'published_at': '2026-11-14 10:00:00'}
    a = aidaily._make_article('arxiv', entry, since, until, '2026-11-15')
    assert a is not None
    assert a.id == md5('arxivhttps://example.com/1T')[:16]
    assert a.lang == 'en'
    entry_zh = dict(entry, title='中文标题 T', published_at='2026-11-14 10:00:00')
    a = aidaily._make_article('arxiv', entry_zh, since, until, '2026-11-15')
    assert a.lang == 'zh'
    entry_out = dict(entry, published_at='2026-11-13 10:00:00')
    assert aidaily._make_article('arxiv', entry_out, since, until, '2026-11-15') is None
