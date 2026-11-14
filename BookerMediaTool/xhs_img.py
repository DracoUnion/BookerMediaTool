"""xhs-img — 发布图片系列到小红书。"""

from __future__ import annotations

import logging
import os
from os import path

from .xhs_api import publish_image_note, NoCookieError, XhsApiError

logger = logging.getLogger(__name__)

IMG_EXTS = {".png", ".jpg", ".jpeg", ".webp", ".gif", ".bmp"}


def _validate_images(paths: list[str]) -> list[str]:
    valid = []
    for p in paths:
        if not os.path.isfile(p):
            raise FileNotFoundError(f"图片文件不存在: {p}")
        if os.path.splitext(p)[1].lower() not in IMG_EXTS:
            logger.warning("未知图片格式: %s", p)
        valid.append(p)
    return valid


def xhs_img_handle(args):
    images = _validate_images(list(args.images))
    if not images:
        print("未提供任何图片")
        return

    # 标题：默认取第一张图片文件名（无扩展名），可被 -t 覆盖
    default_title = path.splitext(path.basename(images[0]))[0]
    title = args.title or default_title
    desc = args.desc or ""

    logger.info("发布图片笔记: title=%r, %d 张图片", title, len(images))

    try:
        result = publish_image_note(title=title, desc=desc, paths=images)
        print(f"✓ 发布成功: {result}")
    except NoCookieError as exc:
        print(f"✗ {exc}")
    except XhsApiError as exc:
        print(f"✗ 发布失败: {exc}")


def reg_subparser(subparsers):
    p = subparsers.add_parser("xhs-img", help="发布图片系列到小红书")
    p.add_argument("images", nargs="+", help="图片文件路径（按顺序）")
    p.add_argument("-tt", "--title", help="笔记标题（默认取第一张图片文件名）")
    p.add_argument("-d", "--desc", default="", help="笔记正文（默认空）")
    p.set_defaults(func=xhs_img_handle)
    return p
