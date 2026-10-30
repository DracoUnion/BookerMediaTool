# BookerMediaTool

BookerMediaTool 是一个基于 OpenAI 兼容 API 的命令行媒体内容生成工具集，源自 [BookerGptTool](https://github.com/DracoUnion/BookerGptTool) 的媒体管线子命令，包含：小红书信息图、文章配图、网页 PPT、小红书轮播图、即刻文案、播客、公众号出稿、人脸融合等。

## 安装

推荐使用 Python 3.8 或更高版本（项目声明支持 Python 3.6+）：

```bash
pip install git+https://github.com/DracoUnion/BookerMediaTool.git
```

也可以在源码目录中安装：

```bash
pip install .
```

安装后提供两个等价的命令：`media-tool` 和 `BookerMediaTool`。

## 配置 API

工具通过 OpenAI 兼容接口访问大模型。至少需要设置 API Key：

```bash
export OPENAI_API_KEY="sk-..."
export OPENAI_BASE_URL="https://api.openai.com/v1"
export OPENAI_CHAT_MODEL="gpt-4o-mini"
```

Windows PowerShell：

```powershell
$env:OPENAI_API_KEY = "sk-..."
$env:OPENAI_BASE_URL = "https://api.openai.com/v1"
$env:OPENAI_CHAT_MODEL = "gpt-4o-mini"
```

`OPENAI_BASE_URL` 可替换为其他 OpenAI 兼容服务的地址。视觉任务可额外配置：

```bash
export OPENAI_VIS_MODEL="视觉模型名称"
export OPENAI_TTI_MODEL="文生图模型名称"
```

不要把 API Key 提交到 Git 仓库或写入公开脚本中。

## 基本用法

查看帮助和版本：

```bash
media-tool -h
media-tool --version
media-tool <子命令> -h
```

通用参数应放在子命令之前，例如：

```bash
media-tool --model gpt-4o-mini --retry 3 xhs-art article.md -o out
```

常用全局参数：

| 参数 | 说明 | 默认值 |
| --- | --- | --- |
| `-m, --model` | 聊天模型名称 | `OPENAI_CHAT_MODEL`，未设置时为 `gpt-3.5-turbo` |
| `-k, --key` | OpenAI API Key | `OPENAI_API_KEY` |
| `-H, --host` | API 地址 | `OPENAI_BASE_URL` |
| `-vm, --vmodel` | 视觉模型名称 | `OPENAI_VIS_MODEL` |
| `-im, --tti-model` | 文生图模型名称 | `OPENAI_TTI_MODEL` |
| `-r, --retry` | 重试次数 | `1000000` |
| `-tm, --temp` | temperature | `1` |
| `-tp, --top-p` | top-p | 无 |
| `-fp, --frequency-penalty` | 频率惩罚 | 无 |
| `-pp, --presence-penalty` | 存在惩罚 | 无 |
| `-mt, --max-tokens` | 最大输出 token 数 | 不限制 |
| `-ST, --no-stream` | 关闭流式输出 | 关闭 |
| `-eb, --extra-body` | 额外请求体（JSON） | 无 |
| `-ct, --conn-timeout` | 连接超时秒数 | `60` |
| `-rt, --read-timeout` | 读取超时秒数 | `120` |
| `-rr, --repetition-regex` | 重复检测正则 | 空 |
| `-TN, --no-think` | 关闭思考模式 | 关闭 |
| `-ua, --user-agent` | HTTP User-Agent 请求头 | `claude-cli/2.1.41` |

## 子命令

### 小红书信息图卡片系列（xhs-img）

12 种视觉风格 · 8 种布局 · 3 种配色 · 25+ 预设。分析内容 → 生成大纲 → 组装 Prompt → 调用 TTI 生成图片。

```bash
# 分析文章并输出大纲与加工后的图片 Prompt
media-tool xhs-img outline.md

# 指定预设、图片数量与输出目录
media-tool xhs-img outline.md --preset <预设名称> --count 5 -o output

# 从标准输入读取
cat outline.md | media-tool xhs-img - --count 4
```

可用选项：`--style`、`--layout`、`--palette`、`--preset`、`--count`、`--output-dir`、`--strategy`（`a` 故事驱动 / `b` 信息密集 / `c` 视觉优先）。

### 文章配图（art-img）

```bash
# 文章 → 01fish 风格配图 HTML（浏览器打开可逐张下载 / 全量打包 ZIP）
media-tool art-img article.md

# 可选：用 Playwright 导出 PNG（需要 pip install playwright）
media-tool art-img article.md --png

# 指定输出路径 / 调试模式
media-tool art-img article.md -o out.html -D
```

### 网页 PPT（ppt）

素材（md/txt）→ 单文件 HTML 横向翻页 PPT（电子杂志风 / 瑞士国际主义风）：

```bash
media-tool ppt article.md
media-tool ppt article.md -a "行业内部" -t 30   # 指定受众与分享时长
```

输出写到 `<素材路径>_ppt/index.html`；规划与 Deck 有缓存，同素材重复运行不会重复消耗 token。

### 小红书轮播图 + 发布文案（xhs-art）

```bash
# 文章 → 8-10 张轮播卡片 HTML + 小红书发布文案 txt
media-tool xhs-art article.md -o out
media-tool xhs-art article.md --base 自定义文件名
```

### 即刻发布文案（jike-art）

```bash
# 文章 → 即刻文案 txt（正文 + 话题圈）
media-tool jike-art article.md -o out
```

### 播客（podcast）

```bash
# 文章 → 播客脚本 + 封面 + 小宇宙文案（音频依赖本地 IndexTTS2）
media-tool podcast article.md

# 跳过 TTS 音频与封面
media-tool podcast article.md --no-tts --no-cover
```

`podcast` 的音频生成依赖本地 IndexTTS2（`INDEXTTS_DIR` / `VOICE_REF` 环境变量），未配置时自动跳过音频，仅输出脚本与封面。

### 公众号出稿（gzh-art）

```bash
# 素材目录（其中的 .md 文件）→ 公众号文章 Markdown（01fish 写作风格）
media-tool gzh-art drafts -o article.md
media-tool gzh-art drafts --owner "作者名"
```

`gzh-art` 的输入是**素材目录**，读取其中所有 `.md` 文件作为素材列表。输出默认写在输入目录下，文件名为目录名 + `.md`。风格指南内置在 `gzh_art_assets/references/writing-style.md`。

### 人脸融合（zhanshimian）

```bash
# 人脸照片 + 多张场景照片 → 融合图（OpenAI gpt-image-edit）
media-tool zhanshimian face.jpg scenes_dir -o output_dir

# 也支持逗号分隔的文件列表或 glob 模式
media-tool zhanshimian face.jpg "a.jpg,b.jpg,c.jpg"
media-tool zhanshimian face.jpg "*.png" --size 1024x1024 --model gpt-image-1
```

## 文件与输出说明

- 需要文件路径的命令通常在输入文件旁边生成结果，具体输出位置以命令帮助或运行日志为准。
- 所有 LLM 调用都经过 `openai.py` 的 OpenAIClient（带重试、流式、思考模式开关等）。
- 通用参数需放在子命令名之前；子命令自身的参数放在子命令名之后。

## 开发

```bash
git clone https://github.com/DracoUnion/BookerMediaTool.git
cd BookerMediaTool
pip install -e .
```

项目主页：<https://github.com/DracoUnion/BookerMediaTool>