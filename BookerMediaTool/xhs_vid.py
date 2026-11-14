"""xhs-vid — 发布视频到小红书。"""

from __future__ import annotations

import logging
import os
from os import path

from .xhs_api import publish_video_note, NoCookieError, XhsApiError

logger = logging.getLogger(__name__)


def xhs_vid_handle(args):
    video = args.video
    if not os.path.isfile(video):
        print(f"视频文件不存在: {video}")
        return

    # 标题：默认取文件名（无扩展名），可被 -t 覆盖
    default_title = path.splitext(path.basename(video))[0]
    title = args.title or default_title
    desc = args.desc or ""

    cover = getattr(args, "cover", None)
    logger.info("发布视频笔记: title=%r, video=%s", title, video)

    try:
        result = publish_video_note(
            title=title, desc=desc, vid_path=video, cover_path=cover,
        )
        print(f"✓ 发布成功: {result}")
    except NoCookieError as exc:
        print(f"✗ {exc}")
    except XhsApiError as exc:
        print(f"✗ 发布失败: {exc}")


def reg_subparser(subparsers):
    p = subparsers.add_parser("xhs-vid", help="发布视频到小红书")
    p.add_argument("video", help="视频文件路径")
    p.add_argument("-tt", "--title", help="笔记标题（默认取视频文件名）")
    p.add_argument("-d", "--desc", default="", help="笔记正文（默认空）")
    p.add_argument("-c", "--cover", help="封面图路径（可选）")
    p.set_defaults(func=xhs_vid_handle)
    return p
