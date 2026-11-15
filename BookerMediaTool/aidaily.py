# -*- coding: utf-8 -*-
"""aidaily 子命令：AI平权日报——从固定信源采集、粗筛、去重、LLM 精筛、渲染简报。"""
import argparse
import json
import logging
import os
import re
import sys
import time
import xml.etree.ElementTree as ET
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timedelta, timezone
from os import path
from typing import Dict, List, Optional

import feedparser
import numpy as np
import requests
from pydantic import parse_obj_as

from .util import (
    md5, render_prompt, read_text,
    read_yaml_model, write_yaml_model, write_text,
)
from .openai import logger as oai_logger
from .aidaily_agent import AidailyAgent
from .aidaily_models import (
    Article, DedupedArticle, Enrichment, EnrichedArticle, RunStatus,
    CATEGORY_CN,
)
from .aidaily_pmt import prompt_version

logging.basicConfig(
    level=logging.INFO,
    format='[%(asctime)s][%(name)s][%(levelname)s] %(message)s',
)
logger = logging.getLogger(__name__)

LOCAL_TZ = datetime.now().astimezone().tzinfo

KNOWN_SOURCES = ('arxiv', 'hn', 'tds', 'machine_heart', 'quantumbit', 'xinzhiyuan')
SOURCE_PRIORITY = {'arxiv': 0, 'tds': 1, 'hn': 2, 'machine_heart': 3, 'quantumbit': 4, 'xinzhiyuan': 5}
ALL_STAGES = ('collect', 'filter', 'dedup', 'enrich', 'render')

GREEN_SIGNALS = {'consumer_24g', 'consumer_16g', 'colab', 'cpu'}
HN_KEEP_DOMAINS = ('github.com', 'arxiv.org', 'huggingface.co')
SIGNAL_CN = {
    'a100_80g': 'A100 80G', 'consumer_24g': '消费级 24G', 'consumer_16g': '消费级 16G',
    'cpu': 'CPU 仅', 'colab': 'Colab', 'single_gpu': '单卡', 'unspecified': '未标注',
}
CATEGORY_ORDER = ('compression', 'weight_edit', 'edge', 'low_resource')

DEFAULT_SOURCES = 'arxiv,hn,tds,machine_heart,quantumbit'
DEFAULT_OUT_DIR = './ai_daily_out'
DATE_DIR_NAME = '%Y-%m-%d'

TOPIC_KEYWORD_ARGS = (
    'compression_keywords', 'weight_edit_keywords', 'edge_keywords', 'low_resource_keywords',
)


class _ArgError(Exception):
    """参数校验失败；由 aidaily_hdl 转为 parser.error（退出码 2）。"""


# ── 跨平台文件锁（fcntl.flock 在 Linux，msvcrt 在 Windows） ───────────────

try:
    import msvcrt  # type: ignore
    _HAVE_MSVCRT = True
except ImportError:  # pragma: no cover - Linux 环境
    _HAVE_MSVCRT = False


class FileLock:
    """非阻塞互斥锁，锁文件为 {out_dir}/.index/.lock；冲突返回 False。"""

    def __init__(self, lock_path: str):
        self.lock_path = lock_path
        self.fd: Optional[int] = None

    def acquire(self) -> bool:
        os.makedirs(path.dirname(self.lock_path), exist_ok=True)
        self.fd = os.open(self.lock_path, os.O_RDWR | os.O_CREAT, 0o644)
        try:
            if _HAVE_MSVCRT:
                msvcrt.locking(self.fd, msvcrt.LK_NBLCK, 1)
            else:
                import fcntl  # type: ignore
                fcntl.flock(self.fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError:
            try:
                os.close(self.fd)
            except OSError:
                pass
            self.fd = None
            return False
        return True

    def release(self) -> None:
        if self.fd is None:
            return
        try:
            if _HAVE_MSVCRT:
                msvcrt.locking(self.fd, msvcrt.LK_UNLCK, 1)
            else:
                import fcntl  # type: ignore
                fcntl.flock(self.fd, fcntl.LOCK_UN)
        finally:
            try:
                os.close(self.fd)
            except OSError:
                pass
            self.fd = None


# ── 时间与文本工具 ──────────────────────────────────────────────────────

def _parse_window(text: str, end_of_day: bool = False) -> Optional[datetime]:
    """解析 'YYYY-MM-DD' 或 ISO 8601 时间，带本地时区；失败返回 None。"""
    if not text:
        return None
    text = text.strip()
    try:
        dt = datetime.strptime(text, DATE_DIR_NAME)
        if end_of_day:
            dt = dt.replace(hour=23, minute=59, second=59)
        return dt.replace(tzinfo=LOCAL_TZ)
    except ValueError:
        pass
    try:
        dt = datetime.fromisoformat(text)
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=LOCAL_TZ)
        return dt
    except ValueError:
        logger.warning('无法解析时间 "%s"，忽略', text)
        return None


def _domain(url: str) -> str:
    m = re.search(r'https?://([^/]+)/', url)
    return m.group(1).lower() if m else ''


def _merged_topic_keywords(args) -> List[str]:
    kw: List[str] = []
    for name in TOPIC_KEYWORD_ARGS:
        for k in (getattr(args, name, '') or '').split(','):
            k = k.strip().lower()
            if k and k not in kw:
                kw.append(k)
    return kw


# ── SimHash（纯 Python 实现，无第三方依赖） ─────────────────────────────

def _norm_tokens(text: str) -> List[str]:
    """归一化：小写、去标点；中文按字符 3-gram、英文按空格分词。"""
    text = text.lower()
    text = re.sub(r'[^\w一-鿿\s]', ' ', text)
    out: List[str] = []
    for t in text.split():
        if not t:
            continue
        if re.search(r'[一-鿿]', t):
            if len(t) < 3:
                out.append(t)
            else:
                out.extend(t[i:i + 3] for i in range(len(t) - 2))
        else:
            out.append(t)
    return out


def _simhash(tokens: List[str]) -> int:
    v = [0] * 64
    mask = (1 << 64) - 1
    for t in tokens:
        h = int(md5(t), 16) & mask
        for i in range(64):
            if h & (1 << i):
                v[i] += 1
            else:
                v[i] -= 1
    return sum(1 << i for i in range(64) if v[i] > 0)


def _hamming(a: int, b: int) -> int:
    return bin(a ^ b).count('1')


# ── 信源采集 ────────────────────────────────────────────────────────────

def _feed_entries(url: str) -> List[dict]:
    """抓取 RSS/Atom feed，返回条目列表；feedparser 不可用时回退到 ElementTree RSS 解析。"""
    try:
        r = requests.get(url, timeout=30, headers={'User-Agent': 'BookerMediaTool/1.0'})
        r.raise_for_status()
        text = r.text
    except Exception as ex:
        logger.warning('feed 抓取失败 %s：%s', url, ex)
        return []
    try:
        d = feedparser.parse(text)
        entries: List[dict] = []
        for e in d.entries:
            published = ''
            if getattr(e, 'published_parsed', None):
                published = datetime(*e.published_parsed[:6], tzinfo=timezone.utc).astimezone(LOCAL_TZ).strftime('%Y-%m-%d %H:%M:%S')
            elif getattr(e, 'updated_parsed', None):
                published = datetime(*e.updated_parsed[:6], tzinfo=timezone.utc).astimezone(LOCAL_TZ).strftime('%Y-%m-%d %H:%M:%S')
            entries.append({
                'title': getattr(e, 'title', '') or '',
                'url': getattr(e, 'link', '') or getattr(e, 'id', '') or '',
                'summary': getattr(e, 'summary', '') or getattr(e, 'description', '') or '',
                'published_at': published,
            })
        return entries
    except Exception:
        try:
            root = ET.fromstring(text)
            ns = {'atom': 'http://www.w3.org/2005/Atom'}
            items = root.findall('.//item')
            if not items:
                items = root.findall('atom:entry', ns)
            entries = []
            for it in items:
                link = it.find('link')
                url = link.get('href') if link is not None else it.findtext('link') or ''
                entries.append({
                    'title': it.findtext('title') or '',
                    'url': url,
                    'summary': it.findtext('description') or it.findtext('atom:summary', ns) or '',
                    'published_at': it.findtext('pubDate') or it.findtext('atom:published', ns) or '',
                })
            return entries
        except Exception as ex:
            logger.warning('feed 解析失败 %s：%s', url, ex)
            return []


def _make_article(source: str, entry: dict, since: datetime, until: datetime,
                  run_date: str) -> Optional[Article]:
    title = (entry.get('title') or '').strip()
    url = (entry.get('url') or '').strip()
    if not title or not url:
        return None
    summary = (entry.get('summary') or '').strip()
    published_at = (entry.get('published_at') or '').strip()
    dt = _parse_window(published_at)
    if dt is not None and not (since <= dt < until):
        return None
    aid = md5(source + url + title)[:16]
    lang = 'zh' if re.search(r'[一-鿿]', title + summary) else 'en'
    return Article(
        id=aid, source=source, url=url, title=title, summary=summary[:2000],
        published_at=published_at, lang=lang, run_date=run_date,
    )


class AidailyOrchestrator:
    """编排器：采集 → 粗筛 → 去重 → LLM 精筛 → 渲染，支持 --stage 单步运行。"""

    def __init__(self, args):
        self.args = args
        self.run_date = args.date or datetime.now(LOCAL_TZ).strftime(DATE_DIR_NAME)
        self.out_dir = args.out_dir
        self.date_dir = path.join(self.out_dir, self.run_date)
        self.index_dir = path.join(self.out_dir, '.index')
        self.cache_dir = path.join(self.out_dir, '.cache', 'enrich')
        self.lock = FileLock(path.join(self.index_dir, '.lock'))
        self.agent = AidailyAgent(args)
        self.source_status: Dict[str, str] = {}
        self.counts: Dict[str, int] = {}
        self._started_at: Optional[str] = None

    # ── 路径与读写 ────────────────────────────────────────────────────

    def _fname(self, name: str) -> str:
        return path.join(self.date_dir, name)

    def _read_yaml(self, fname: str, model):
        fpath = self._fname(fname)
        if not path.isfile(fpath):
            return None
        return read_yaml_model(fpath, model)

    def _write_yaml(self, fname: str, obj) -> None:
        write_yaml_model(self._fname(fname), obj)

    # ── 参数校验 ────────────────────────────────────────────────────

    @staticmethod
    def validate_args(parser, args) -> None:
        def fail(msg: str) -> None:
            if parser is not None:
                parser.error(msg)
            raise _ArgError(msg)

        if bool(args.since) != bool(args.until):
            fail('--since 与 --until 必须成对出现')
        if args.window_hours <= 0:
            fail('--window-hours 必须大于 0')
        if not (0 <= args.simhash_max_distance <= 64):
            fail('--simhash-max-distance 必须在 [0, 64] 之间')
        if not (0 < args.semantic_threshold <= 1):
            fail('--semantic-threshold 必须在 (0, 1] 之间')
        srcs = [s.strip() for s in (args.sources or '').split(',') if s.strip()]
        unknown = [s for s in srcs if s not in KNOWN_SOURCES]
        if unknown:
            fail(f'未知信源：{", ".join(unknown)}（可用：{", ".join(KNOWN_SOURCES)}）')
        if args.stage not in ALL_STAGES + ('stats', 'all'):
            fail(f'--stage 必须是 {{"all", "collect", "filter", "dedup", "enrich", "render", "stats"}}')

    # ── 运行状态 ────────────────────────────────────────────────────

    def _init_run_status(self) -> None:
        self._started_at = datetime.now(LOCAL_TZ).isoformat()

    def _final_counts(self) -> Dict[str, int]:
        c = dict(self.counts)
        for name, fname, model in (
            ('raw', 'raw.yaml', List[Article]),
            ('filtered', 'filtered.yaml', List[Article]),
            ('deduped', 'deduped.yaml', List[DedupedArticle]),
            ('enriched', 'enriched.yaml', List[EnrichedArticle]),
        ):
            if name not in c:
                data = self._read_yaml(fname, model)
                c[name] = len(data) if data else 0
        return c

    def _save_run_status(self, code: int, status: str) -> None:
        rs = RunStatus(
            run_date=self.run_date,
            started_at=self._started_at or '',
            finished_at=datetime.now(LOCAL_TZ).isoformat(),
            status=status, exit_code=code,
            source_status=self.source_status,
            counts=self._final_counts(),
            cli_args={k: v for k, v in vars(self.args).items()
                      if k not in ('func', 'key')},
        )
        write_text(self._fname('run.json'),
                   json.dumps(rs.dict(), indent=2, ensure_ascii=False))

    # ── 锁 ────────────────────────────────────────────────────────────

    def _acquire_lock(self) -> bool:
        return self.lock.acquire()

    def _release_lock(self) -> None:
        self.lock.release()

    # ── 各阶段 ────────────────────────────────────────────────────────

    def _collect_source(self, source: str, since: datetime, until: datetime) -> List[Article]:
        args = self.args
        arts: List[Article] = []
        if source == 'arxiv':
            cats = [c.strip() for c in (args.arxiv_categories or '').split(',') if c.strip()]
            for cat in cats:
                for entry in _feed_entries(f'https://rss.arxiv.org/rss/{cat}'):
                    a = _make_article('arxiv', entry, since, until, self.run_date)
                    if a:
                        arts.append(a)
            # arXiv API 按主题词补充（同一 id 内部去重）
            for kw in _merged_topic_keywords(args)[:8]:
                try:
                    r = requests.get(
                        'https://export.arxiv.org/api/query',
                        params={'search_query': f'all:{kw}', 'start': 0, 'max_results': 5},
                        timeout=30,
                        headers={'User-Agent': 'BookerMediaTool/1.0'},
                    )
                    r.raise_for_status()
                    for entry in _feed_entries(r.text):
                        a = _make_article('arxiv', entry, since, until, self.run_date)
                        if a:
                            arts.append(a)
                except Exception as ex:
                    logger.debug('arXiv API 补充失败 keyword=%s：%s', kw, ex)
                time.sleep(0.5)
        elif source == 'hn':
            ts = int(since.timestamp())
            for kw in _merged_topic_keywords(args):
                try:
                    url = (
                        'https://hn.algolia.com/api/v1/search?'
                        f'query={kw}&numericFilters=created_at_i>{ts},points>{args.hn_min_points}'
                        '&tags=story&hitsPerPage=50'
                    )
                    r = requests.get(url, timeout=30, headers={'User-Agent': 'BookerMediaTool/1.0'})
                    r.raise_for_status()
                    for hit in (r.json().get('hits') or []):
                        title = hit.get('title') or ''
                        url = hit.get('url') or f"https://news.ycombinator.com/item?id={hit.get('objectID')}"
                        if not title or not url:
                            continue
                        entry = {
                            'title': title,
                            'url': url,
                            'summary': (hit.get('story_title') or '') + ' | ' + str(hit.get('points') or 0) + ' points',
                            'published_at': datetime.fromtimestamp(hit.get('created_at_i') or 0, tz=LOCAL_TZ).strftime('%Y-%m-%d %H:%M:%S'),
                        }
                        a = _make_article('hn', entry, since, until, self.run_date)
                        if a:
                            arts.append(a)
                except Exception as ex:
                    logger.debug('hn 查询失败 keyword=%s：%s', kw, ex)
                time.sleep(0.2)
        else:
            feed = getattr(args, f'{source}_feed', '') or ''
            if not feed:
                logger.warning('[collect] 信源 %s 未配置 feed，跳过', source)
                return []
            for entry in _feed_entries(feed):
                a = _make_article(source, entry, since, until, self.run_date)
                if a:
                    arts.append(a)
        return arts

    def step_collect(self) -> List[Article]:
        args = self.args
        until = _parse_window(args.until, end_of_day=True) or datetime.now(LOCAL_TZ)
        since = _parse_window(args.since) or (until - timedelta(hours=args.window_hours))
        logger.info('[collect] 窗口 [%s, %s)，信源：%s', since, until, args.sources)
        all_arts: List[Article] = []
        self.source_status = {}
        for source in [s.strip() for s in (args.sources or '').split(',') if s.strip()]:
            try:
                arts = self._collect_source(source, since, until)
            except Exception as ex:
                logger.exception('[collect] 信源 %s 失败：%s', source, ex)
                arts = []
                self.source_status[source] = 'failed'
            else:
                self.source_status[source] = 'skipped' if not arts and source == 'xinzhiyuan' else 'ok'
            logger.info('[collect] 信源 %s：%d 条（状态 %s）', source, len(arts), self.source_status[source])
            all_arts.extend(arts)
        uniq: Dict[str, Article] = {}
        for a in all_arts:
            uniq.setdefault(a.id, a)
        articles = list(uniq.values())
        self.counts['raw'] = len(articles)
        self._write_yaml('raw.yaml', articles)
        self._append_index(articles)
        logger.info('[collect] 共 %d 条（去重后 %d 条）', len(all_arts), len(articles))
        return articles

    def _append_index(self, articles: List[Article]) -> None:
        fpath = path.join(self.index_dir, 'articles.json')
        index: Dict[str, dict] = {}
        if path.isfile(fpath):
            try:
                index = json.loads(read_text(fpath) or '{}')
            except json.JSONDecodeError:
                index = {}
        for a in articles:
            index[a.id] = {
                'source': a.source, 'title': a.title,
                'published_at': a.published_at, 'run_date': a.run_date,
            }
        os.makedirs(self.index_dir, exist_ok=True)
        write_text(fpath, json.dumps(index, ensure_ascii=False, indent=2))

    def step_filter(self) -> List[Article]:
        args = self.args
        raw = self._read_yaml('raw.yaml', List[Article])
        if raw is None:
            raise FileNotFoundError('raw.yaml 不存在，请先运行 collect 阶段')
        blacklist = [b.strip().lower() for b in (args.blacklist or '').split(',') if b.strip()]
        topic_kw = _merged_topic_keywords(args)
        tds_extra = [k.strip().lower() for k in (args.tds_extra_keywords or '').split(',') if k.strip()]
        out: List[Article] = []
        for a in raw:
            text = (a.title + ' ' + a.summary).lower()
            if any(b and b in text for b in blacklist):
                continue
            hit = any(k in text for k in topic_kw)
            if a.source == 'tds':
                hit = hit or any(k in text for k in tds_extra)
            if not hit and a.source == 'hn' and _domain(a.url) in HN_KEEP_DOMAINS:
                hit = True
            if not hit:
                continue
            out.append(a)
        self.counts['filtered'] = len(out)
        self._write_yaml('filtered.yaml', out)
        logger.info('[filter] 粗筛后 %d / %d 条', len(out), len(raw))
        return out

    def _load_embedding_model(self):
        model_path = getattr(self.args, 'emb', None) or os.environ.get('EMB_MODEL_PATH', '')
        if not model_path:
            logger.warning('--emb 未提供且 EMB_MODEL_PATH 未设置，跳过语义去重，仅做 SimHash')
            return None
        try:
            from sentence_transformers import SentenceTransformer  # type: ignore
        except Exception:
            logger.warning('sentence_transformers 未安装，跳过语义去重，仅做 SimHash')
            return None
        try:
            return SentenceTransformer(model_path)
        except Exception as ex:
            logger.warning('语义去重模型加载失败（%s），跳过语义去重，仅做 SimHash', ex)
            return None

    def step_dedup(self) -> List[DedupedArticle]:
        args = self.args
        filtered = self._read_yaml('filtered.yaml', List[Article])
        if filtered is None:
            raise FileNotFoundError('filtered.yaml 不存在，请先运行 filter 阶段')

        dist = args.simhash_max_distance
        lookback = args.lookback_days
        cutoff = datetime.now(LOCAL_TZ) - timedelta(days=lookback)

        # 历史 SimHash 索引
        sim_fpath = path.join(self.index_dir, 'simhash.json')
        sim_index: Dict[str, int] = {}
        if path.isfile(sim_fpath):
            try:
                sim_index = {k: int(v) for k, v in json.loads(read_text(sim_fpath) or '{}').items()}
            except json.JSONDecodeError:
                sim_index = {}
        # 历史条目元数据（用于 lookback 过滤与 evidence）
        index_fpath = path.join(self.index_dir, 'articles.json')
        index_meta: Dict[str, dict] = {}
        if path.isfile(index_fpath):
            try:
                index_meta = json.loads(read_text(index_fpath) or '{}')
            except json.JSONDecodeError:
                index_meta = {}

        model = self._load_embedding_model()
        emb_enabled = model is not None
        # 语义索引向量（已 L2 归一化）
        emb_ids: List[str] = []
        emb_vecs = np.zeros((0, 0), dtype='float32')
        emb_fpath = path.join(self.index_dir, 'embeddings.npz')
        if emb_enabled and path.isfile(emb_fpath):
            try:
                npz = np.load(emb_fpath)
                emb_ids = [str(x) for x in npz['ids'].tolist()]
                emb_vecs = np.asarray(npz['vectors'], dtype='float32')
            except Exception as ex:
                logger.warning('读取语义索引失败：%s', ex)
                emb_vecs = np.zeros((0, 0), dtype='float32')

        def in_lookback(meta: Optional[dict]) -> bool:
            if not meta or not meta.get('published_at'):
                return True
            dt = _parse_window(meta['published_at'])
            return dt is None or dt >= cutoff

        def text_of(a: Article) -> str:
            return (a.title + ' ' + a.summary)[:512]

        def simhash_of(a: Article) -> int:
            return _simhash(_norm_tokens(text_of(a)))

        def hist_simhash_dup(h: int) -> bool:
            """与历史索引中回看窗口内的条目比对 SimHash。"""
            for aid, m in index_meta.items():
                if aid not in sim_index or not in_lookback(m):
                    continue
                if _hamming(h, sim_index[aid]) <= dist:
                    return True
            return False

        primaries: List[DedupedArticle] = []
        ordered = sorted(
            filtered,
            key=lambda a: (SOURCE_PRIORITY.get(a.source, 99), a.published_at),
        )
        hot = len(filtered) > args.hot_window_count
        threshold = args.hot_window_boost if hot else args.semantic_threshold

        for a in ordered:
            # 同一条目已在历史索引中（完全重复采集），直接跳过
            if a.id in index_meta:
                continue
            h = simhash_of(a)
            dup: Optional[DedupedArticle] = None
            for p in primaries:
                if _hamming(h, simhash_of(p)) <= dist:
                    dup = p
                    break
            if dup is None and hist_simhash_dup(h):
                # 历史去重命中：该条目在回看窗口内已入库，不再入选
                logger.debug('[dedup] %s 历史 SimHash 重复，跳过', a.id)
                continue
            if dup is None and emb_enabled:
                v = model.encode([text_of(a)])[0].astype('float32')
                v = v / (np.linalg.norm(v) + 1e-9)
                best = 0.0
                for pos, aid in enumerate(emb_ids):
                    if not in_lookback(index_meta.get(aid, {})):
                        continue
                    s = float(np.dot(emb_vecs[pos], v))
                    if s > best:
                        best = s
                if best >= threshold:
                    logger.debug('[dedup] %s 历史语义重复（%.2f），跳过', a.id, best)
                    continue
            if dup is not None:
                dup.evidence.append({
                    'source': a.source, 'url': a.url, 'title': a.title,
                    'published_at': a.published_at,
                })
                logger.debug('[dedup] %s 并入 %s（%s）', a.id, dup.id, a.source)
            else:
                primaries.append(DedupedArticle(
                    id=a.id, source=a.source, url=a.url, title=a.title, summary=a.summary,
                    published_at=a.published_at, lang=a.lang, run_date=a.run_date,
                    is_primary=True, evidence_for='', evidence=[],
                ))
                sim_index[a.id] = h
                if emb_enabled:
                    vec = model.encode([text_of(a)])[0].astype('float32')
                    vec = vec / (np.linalg.norm(vec) + 1e-9)
                    emb_ids.append(a.id)
                    emb_vecs = np.vstack([emb_vecs, vec.reshape(1, -1)])

        self.counts['deduped'] = len(primaries)
        self._write_yaml('deduped.yaml', primaries)

        # 更新跨日索引
        index_meta.update({
            a.id: {'source': a.source, 'title': a.title,
                   'published_at': a.published_at, 'run_date': a.run_date}
            for a in primaries
        })
        keep: Dict[str, dict] = {}
        for aid, m in index_meta.items():
            if in_lookback(m):
                keep[aid] = m
        os.makedirs(self.index_dir, exist_ok=True)
        write_text(index_fpath, json.dumps(keep, ensure_ascii=False, indent=2))
        write_text(sim_fpath, json.dumps({aid: int(h) for aid, h in sim_index.items()
                                          if aid in keep}, ensure_ascii=False, indent=2))
        if emb_enabled:
            rows = [(aid, vec) for aid, vec in zip(emb_ids, emb_vecs.tolist()) if aid in keep]
            if rows:
                ids_arr = np.array([r[0] for r in rows])
                vecs_arr = np.array([r[1] for r in rows], dtype='float32')
                np.savez(emb_fpath, ids=ids_arr, vectors=vecs_arr)
            elif path.isfile(emb_fpath):
                os.remove(emb_fpath)
        logger.info('[dedup] 去重后 %d 条（主条目）', len(primaries))
        return primaries

    def step_enrich(self) -> List[EnrichedArticle]:
        args = self.args
        deduped = self._read_yaml('deduped.yaml', List[DedupedArticle])
        if deduped is None:
            raise FileNotFoundError('deduped.yaml 不存在，请先运行 dedup 阶段')
        os.makedirs(self.cache_dir, exist_ok=True)
        no_cache = bool(getattr(args, 'llm_no_cache', False))

        results: List[Optional[EnrichedArticle]] = [None] * len(deduped)
        called = cached = failed = 0

        def job(i: int, a: DedupedArticle):
            key = md5(a.id + prompt_version + args.model + (args.host or ''))
            fpath = path.join(self.cache_dir, key + '.yaml')
            if not no_cache:
                hit = read_yaml_model(fpath, Enrichment)
                if hit is not None:
                    return i, hit, True
            env = self.agent.enrich_one(a)
            if not no_cache:
                write_yaml_model(fpath, env)
            return i, env, False

        with ThreadPoolExecutor(max_workers=max(1, getattr(args, 'llm_max_concurrency', 4))) as pool:
            futs = {pool.submit(job, i, a): i for i, a in enumerate(deduped)}
            for fut in as_completed(futs):
                try:
                    i, env, is_cached = fut.result()
                except KeyboardInterrupt:
                    raise
                except Exception as ex:
                    logger.exception('[enrich] 条目 %s 失败：%s', deduped[futs[fut]].id, ex)
                    failed += 1
                    continue
                if is_cached:
                    cached += 1
                else:
                    called += 1
                results[i] = EnrichedArticle(**deduped[i].dict(), enrichment=env)

        enriched: List[EnrichedArticle] = [r for r in results if r is not None]
        formal = sum(1 for r in enriched if r.enrichment.relevance >= 0.7)
        observations = sum(1 for r in enriched if 0 < r.enrichment.relevance < 0.7)
        # relevance == 0 的条目不入 enriched.yaml
        enriched = [r for r in enriched if r.enrichment.relevance > 0]
        self.counts['enriched'] = len(enriched)
        self.counts['enrich_called'] = called
        self.counts['enrich_cached'] = cached
        self.counts['enrich_failed'] = failed
        self.counts['enrich_formal'] = formal
        self.counts['enrich_observations'] = observations
        self._write_yaml('enriched.yaml', enriched)
        logger.info('[enrich] 入选 %d 条（正式 %d，观察 %d，API 调用 %d，缓存命中 %d，失败 %d）',
                    len(enriched), formal, observations, called, cached, failed)
        return enriched

    def _relevance_label(self, r: float) -> str:
        if r >= 0.7:
            return '高'
        if r >= 0.3:
            return '中'
        return '低'

    def step_render(self) -> None:
        args = self.args
        enriched = self._read_yaml('enriched.yaml', List[EnrichedArticle])
        if enriched is None:
            raise FileNotFoundError('enriched.yaml 不存在，请先运行 enrich 阶段')
        raw = self._read_yaml('raw.yaml', List[Article]) or []
        filtered = self._read_yaml('filtered.yaml', List[Article]) or []
        deduped = self._read_yaml('deduped.yaml', List[DedupedArticle]) or []

        formal = [r for r in enriched if r.enrichment.relevance >= 0.7]
        observations = [r for r in enriched if 0 < r.enrichment.relevance < 0.7]

        self.counts['raw'] = len(raw)
        self.counts['filtered'] = len(filtered)
        self.counts['deduped'] = len(deduped)
        self.counts['enriched'] = len(enriched)
        self.counts['enrich_formal'] = len(formal)
        self.counts['enrich_observations'] = len(observations)

        def badge(r: EnrichedArticle) -> str:
            return '🟢' if r.enrichment.resource_signal in GREEN_SIGNALS else ''

        def signal_cn(r: EnrichedArticle) -> str:
            return SIGNAL_CN.get(r.enrichment.resource_signal, r.enrichment.resource_signal)

        def evidence_sources(r: EnrichedArticle) -> str:
            seen: List[str] = []
            for e in r.evidence:
                s = e.get('source', '')
                if s and s not in seen:
                    seen.append(s)
            return '、'.join(seen) or r.source

        # Markdown
        lines: List[str] = [
            f'# AI平权日报 · {self.run_date}',
            '',
            f'> 统计：采集 {len(raw)} 条 → 粗筛 {len(filtered)} 条 → 去重 {len(deduped)} 条 → 入选 {len(formal)} 条',
            '',
        ]
        html_lines: List[str] = [
            '<!DOCTYPE html><html><head><meta charset="utf-8">',
            '<style>body{font-family:-apple-system,BlinkMacSystemFont,"Segoe UI",Roboto,"Helvetica Neue",Arial,sans-serif;max-width:900px;margin:24px auto;padding:0 16px;line-height:1.7;color:#222}',
            'h1{border-bottom:2px solid #eee;padding-bottom:8px}h2{border-bottom:1px solid #eee;padding-bottom:4px}',
            'ul{padding-left:20px}li{margin:8px 0}.meta{color:#666;font-size:0.95em}',
            '.green::before{content:"🟢 "}a{color:#0969da}</style>',
            '</head><body>',
            f'<h1>AI平权日报 · {self.run_date}</h1>',
            f'<p class="meta">统计：采集 {len(raw)} 条 → 粗筛 {len(filtered)} 条 → 去重 {len(deduped)} 条 → 入选 {len(formal)} 条</p>',
        ]
        cat_stats: Dict[str, List[dict]] = {c: [] for c in CATEGORY_ORDER}
        obs_items: List[dict] = []
        for r in formal:
            label = self._relevance_label(r.enrichment.relevance)
            item = {
                'title': r.title, 'source': r.source, 'url': r.url,
                'one_liner': r.enrichment.one_liner, 'relevance': r.enrichment.relevance,
                'label': label, 'signal': r.enrichment.resource_signal,
                'signal_cn': signal_cn(r), 'badge': badge(r),
                'evidence_sources': evidence_sources(r),
            }
            for cat in r.enrichment.categories:
                cat_stats.setdefault(cat, []).append(item)
        for r in observations:
            item = {
                'title': r.title, 'source': r.source, 'url': r.url,
                'one_liner': r.enrichment.one_liner, 'relevance': r.enrichment.relevance,
                'label': self._relevance_label(r.enrichment.relevance),
                'signal': r.enrichment.resource_signal, 'signal_cn': signal_cn(r),
                'badge': badge(r), 'evidence_sources': evidence_sources(r),
            }
            obs_items.append(item)

        for cat in CATEGORY_ORDER:
            items = cat_stats.get(cat, [])
            lines.append(f'## {CATEGORY_CN[cat]}')
            lines.append('')
            if items:
                for it in items:
                    lines += [
                        f"- **{it['title']}{it['badge']}**（{it['source']} · 相关度 {it['label']}）",
                        f'  {it["one_liner"]}',
                        f'  资源信号：{it["signal_cn"]} ｜ 佐证：{it["evidence_sources"]}',
                        f'  [{it["title"]}]({it["url"]})',
                        '',
                    ]
            else:
                lines += ['(暂无)', '']
            html_lines.append(f'<h2>{CATEGORY_CN[cat]}</h2><ul>')
            for it in items:
                html_lines.append(
                    f'<li><strong>{it["title"]}<span class="green">{it["badge"]}</span></strong>'
                    f'（{it["source"]} · 相关度 {it["label"]}）<br>'
                    f'{it["one_liner"]}<br>'
                    f'<span class="meta">资源信号：{it["signal_cn"]} ｜ 佐证：{it["evidence_sources"]}</span><br>'
                    f'<a href="{it["url"]}">{it["title"]}</a></li>'
                )
            html_lines.append('</ul>')

        lines.append('## 观察区（相关度 < 0.3）')
        lines.append('')
        html_lines.append('<h2>观察区（相关度 < 0.3）</h2><ul>')
        for it in obs_items:
            if it['relevance'] >= 0.3:
                continue
            lines += [
                f"- **{it['title']}{it['badge']}**（{it['source']} · 相关度 {it['label']}）",
                f'  {it["one_liner"]}',
                f'  [{it["title"]}]({it["url"]})',
                '',
            ]
            html_lines.append(
                f'<li><strong>{it["title"]}<span class="green">{it["badge"]}</span></strong>'
                f'（{it["source"]} · 相关度 {it["label"]}）<br>{it["one_liner"]}<br>'
                f'<a href="{it["url"]}">{it["title"]}</a></li>'
            )
        html_lines.append('</ul></body></html>')

        write_text(self._fname('digest.md'), '\n'.join(lines))
        write_text(self._fname('digest.html'), '\n'.join(html_lines))

        digest_json = {
            'date': self.run_date,
            'stats': {
                'raw': len(raw), 'filtered': len(filtered), 'deduped': len(deduped),
                'enriched': len(enriched), 'formal': len(formal),
                'observations': len([x for x in obs_items if x['relevance'] < 0.3]),
            },
            'categories': {
                cat: [{k: it[k] for k in ('title', 'source', 'url', 'relevance', 'one_liner', 'signal')}
                      for it in cat_stats.get(cat, [])] for cat in CATEGORY_ORDER
            },
            'observations': [{k: it[k] for k in ('title', 'source', 'url', 'relevance', 'one_liner', 'signal')}
                             for it in obs_items if it['relevance'] < 0.3],
        }
        write_text(self._fname('digest.json'), json.dumps(digest_json, ensure_ascii=False, indent=2))
        logger.info('[render] 已写入 digest.md / digest.html / digest.json')

    # ── stats ─────────────────────────────────────────────────────────

    def print_stats(self) -> None:
        run_path = self._fname('run.json')
        run = {}
        if path.isfile(run_path):
            try:
                run = json.loads(read_text(run_path) or '{}')
            except json.JSONDecodeError:
                run = {}
        counts = run.get('counts', {}) or {}
        source_status = run.get('source_status', self.source_status) or {}
        enriched = counts.get('enriched')
        if enriched is None:
            data = self._read_yaml('enriched.yaml', None)
            enriched = len(data) if data else 0
        called = counts.get('enrich_called', 0)
        cached = counts.get('enrich_cached', 0)
        raw = counts.get('raw')
        filtered = counts.get('filtered')
        if raw is None:
            data = self._read_yaml('raw.yaml', None)
            raw = len(data) if data else 0
        if filtered is None:
            data = self._read_yaml('filtered.yaml', None)
            filtered = len(data) if data else 0
        ratio = (filtered / raw) if raw else 0.0
        lines: List[str] = [
            f'AI平权日报 统计 · {self.run_date}',
            f'  阶段条目数：raw={raw} filtered={filtered} deduped={counts.get("deduped", "-")} '
            f'enriched={counts.get("enriched", "-")}',
            f'  正式={counts.get("enrich_formal", "-")} 观察={counts.get("enrich_observations", "-")} '
            f'enrich_failed={counts.get("enrich_failed", "-")}',
            f'  信源状态：{json.dumps(source_status, ensure_ascii=False)}',
            f'  LLM 实际调用={called} 缓存命中={cached} enriched条目={enriched}',
            f'  filter/raw 比例={ratio:.0%}（调参目标 10%–20%）',
            f'  状态文件：{run_path}',
        ]
        text = '\n'.join(lines)
        print(text)
        write_text(self._fname('stats.txt'), text)
        logger.info('[stats] 已写入 stats.txt')

    # ── 主流程 ────────────────────────────────────────────────────────

    def _status_map(self, code: int) -> str:
        if code == 0:
            return 'success'
        if code == 3:
            return 'degraded'
        return 'failed'

    def run(self) -> int:
        args = self.args
        if args.verbose:
            logger.setLevel(logging.DEBUG)
            oai_logger.setLevel(logging.DEBUG)

        if args.dry_run:
            print('[aidaily] 干运行：不联网、不调 LLM、不写文件')
            print(f'  stage={args.stage} date={self.run_date} out_dir={self.out_dir}')
            print(f'  sources={args.sources} window={args.window_hours}h lookback={args.lookback_days}d')
            print(f'  LLM：model={getattr(args, "model", None)} host={getattr(args, "host", None)} '
                  f'concurrency={getattr(args, "llm_max_concurrency", 4)} '
                  f'cache={"off" if getattr(args, "llm_no_cache", False) else "on"}')
            return 0

        if not self._acquire_lock():
            print(f'[aidaily] 另一运行正在占用 {self.out_dir}（锁冲突），退出码 5')
            return 5
        try:
            return self._run()
        finally:
            self._release_lock()

    def _run(self) -> int:
        args = self.args
        self._init_run_status()

        if args.stage == 'stats':
            self.print_stats()
            self._save_run_status(0, 'success')
            return 0

        # 幂等：当日已成功且未强制，则不重复生成
        if (args.stage == 'all' and not args.force and path.isfile(self._fname('run.json'))):
            try:
                prev = json.loads(read_text(self._fname('run.json')) or '{}')
            except json.JSONDecodeError:
                prev = {}
            if prev.get('status') == 'success':
                logger.info('[run] %s 已完成（status=success），跳过；使用 --force 强制重跑', self.run_date)
                self._save_run_status(0, 'success')
                return 0

        stages = ALL_STAGES if args.stage == 'all' else (args.stage,)
        code = 0
        stage_ok = True
        for stage in stages:
            try:
                getattr(self, f'step_{stage}')()
            except Exception as ex:
                logger.exception('[run] 阶段 %s 失败：%s', stage, ex)
                stage_ok = False
                code = max(code, 1)

        digest_exists = path.isfile(self._fname('digest.md'))
        source_failed = any(v == 'failed' for v in self.source_status.values())
        enriched_cnt = self.counts.get('enriched')
        if enriched_cnt is None:
            data = self._read_yaml('enriched.yaml', None)
            enriched_cnt = len(data) if data else 0
        deduped_cnt = self.counts.get('deduped')
        if deduped_cnt is None:
            data = self._read_yaml('deduped.yaml', None)
            deduped_cnt = len(data) if data else 0
        enrich_failed = self.counts.get('enrich_failed', 0) > 0

        if digest_exists:
            code = 0
        elif stage_ok:
            code = 0
        else:
            code = 1
        if source_failed:
            code = max(code, 3)
        if enrich_failed and enriched_cnt < deduped_cnt:
            code = max(code, 4)

        status = self._status_map(code)
        self._save_run_status(code, status)
        logger.info('[run] 完成，run_date=%s status=%s exit_code=%d', self.run_date, status, code)
        return code


def aidaily_hdl(args):  # pragma: no cover - 由 __main__ 调用，进程由 sys.exit 托管
    """入口函数：校验参数、实例化编排器并运行，统一由 sys.exit 处理退出码。"""
    parser = getattr(aidaily_hdl, 'parser', None)
    try:
        AidailyOrchestrator.validate_args(parser, args)
    except _ArgError:
        sys.exit(2)
    # enrich/all 阶段需要 API Key（dry-run 除外）
    if args.stage in ('enrich', 'all') and not args.dry_run and not args.key:
        logger.error('未提供 API Key（-k 或环境变量 OPENAI_API_KEY），无法运行 enrich 阶段')
        sys.exit(2)
    try:
        code = AidailyOrchestrator(args).run()
    except SystemExit:
        raise
    except Exception:
        logger.exception('aidaily 异常终止')
        code = 1
    sys.exit(code)


def reg_subparser(subparsers) -> None:
    """注册 aidaily 子命令。

    项目未使用 parents 机制，故公共 LLM 参数（-m/-k/-H/-tm/-r/--emb）在此处按与
    __main__ 根解析器一致的定义附加，以兼容 `md-tool aidaily -m xxx` 与
    `md-tool -m xxx aidaily` 两种写法；aidaily 不新增除 --llm-* 以外的 LLM 参数。
    """
    p = subparsers.add_parser(
        'aidaily',
        help='AI平权日报：从固定信源采集、粗筛、去重、LLM 精筛、渲染简报',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # 运行控制
    p.add_argument('--stage', default='all',
                   help='运行阶段：all/collect/filter/dedup/enrich/render/stats，默认 all')
    p.add_argument('--date', default='',
                   help='目标日期 YYYY-MM-DD，默认本地今天')
    p.add_argument('--since', default='',
                   help='自定义采集窗口起点（与 --until 成对）')
    p.add_argument('--until', default='',
                   help='自定义采集窗口终点（与 --since 成对）')
    p.add_argument('--out-dir', default=DEFAULT_OUT_DIR,
                   help='产物与索引根目录，默认 ./ai_daily_out')
    p.add_argument('--window-hours', type=int, default=24,
                   help='采集窗口小时数，默认 24')
    p.add_argument('--lookback-days', type=int, default=7,
                   help='跨日去重回看天数，默认 7')
    p.add_argument('--force', action='store_true',
                   help='忽略“当日已完成”标记，强制重跑')
    p.add_argument('--dry-run', action='store_true',
                   help='只打印计划，不联网、不调 LLM、不写文件')
    p.add_argument('--resume', action='store_true',
                   help='enrich 阶段跳过已缓存条目')
    p.add_argument('-v', '--verbose', action='store_true',
                   help='DEBUG 级别日志')

    # 信源
    p.add_argument('--sources', default=DEFAULT_SOURCES,
                   help='采集信源，逗号分隔，默认 arxiv,hn,tds,machine_heart,quantumbit')
    p.add_argument('--arxiv-categories', default='cs.LG,cs.CL,cs.AI',
                   help='arXiv 分类，逗号分隔')
    p.add_argument('--hn-min-points', type=int, default=5,
                   help='HN 最小点数')
    p.add_argument('--tds-feed', default='https://towardsdatascience.com/feed',
                   help='TDS RSS 链接')
    p.add_argument('--machine-heart-feed', default='https://www.jiqizhixin.com/feed',
                   help='新机器人 RSS 链接（实现者提供）')
    p.add_argument('--quantumbit-feed', default='https://www.qbitai.com/feed',
                   help='量子位 RSS 链接（实现者提供）')
    p.add_argument('--xinzhiyuan-feed', default='',
                   help='新智元 RSS 链接，空则跳过')

    # 主题词
    p.add_argument('--compression-keywords',
                   default='quantization,low-bit,int4,gptq,awq,pruning,sparsity,distillation',
                   help='模型压缩主题词，逗号分隔')
    p.add_argument('--weight-edit-keywords',
                   default='knowledge editing,model editing,weight editing,model merging',
                   help='权重操控主题词，逗号分隔')
    p.add_argument('--edge-keywords',
                   default='on-device,edge inference,consumer gpu,llama.cpp,local inference,gguf',
                   help='端侧与消费级推理主题词，逗号分隔')
    p.add_argument('--low-resource-keywords',
                   default='mezo,zeroth-order,memory-efficient fine-tuning,low-resource training,single gpu',
                   help='低资源训练主题词，逗号分隔')
    p.add_argument('--tds-extra-keywords',
                   default='tutorial,hands-on,guide,colab,low-cost,budget,peft,lora',
                   help='TDS 额外主题词，命中任一即通过粗筛')
    p.add_argument('--blacklist',
                   default='funding,acquisition,ceo,layoff,融资,裁员',
                   help='标题或摘要命中即丢弃，逗号分隔')

    # 去重
    p.add_argument('--simhash-max-distance', type=int, default=3,
                   help='SimHash 最大汉明距离，范围 [0,64]')
    p.add_argument('--semantic-threshold', type=float, default=0.85,
                   help='语义去重相似度阈值，范围 (0,1]')
    p.add_argument('--hot-window-boost', type=float, default=0.80,
                   help='热门窗口（当日条目数超过 --hot-window-count）时的语义阈值')
    p.add_argument('--hot-window-count', type=int, default=200,
                   help='热门窗口条目数阈值')

    # LLM 新增参数（不与公共参数冲突）
    p.add_argument('--llm-max-concurrency', type=int, default=4,
                   help='enrich 阶段线程池大小')
    p.add_argument('--llm-no-cache', action='store_true',
                   help='关闭 LLM 缓存')
    p.add_argument('--llm-max-tokens', type=int, default=512,
                   help='传给 call_llm_retry 的 max_tokens')

    # -m/-k/-H/-tm/-r/--emb 等公共 LLM 参数由 __main__.py 根解析器统一提供，
    # aidaily 不重复定义；使用时请放在子命令之前：
    #   md-tool -m gpt-4o-mini -k "$OPENAI_API_KEY" aidaily ...
    p.set_defaults(func=aidaily_hdl)
    aidaily_hdl.parser = p
