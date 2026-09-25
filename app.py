"""EnDictation 后端（NAS 迁移版）。

主要变更见 docs/NAS_MIGRATION.md：
- /upload 与 /generate-tts 受理后台任务并立即返回 202 与 status_url，不在 HTTP 请求内等待外部 API。
- 会话隔离：签名 cookie 内保存服务端生成的随机会话 ID，任务按 owner 校验；
  写接口要求有效会话并通过同源检查。
- 输入上限（请求体大小、图片格式/像素、句子/重点词数量与长度、合成总量、执行预算）
  在请求入口校验，均为写入代码常量的初始设计值，不提供环境变量配置。
- Azure TTS 采用文本转语音 REST 接口（步骤 1 验证结论见 NAS_MIGRATION §3.5），
  不再依赖 Azure Speech SDK；所有外部调用显式配置有限超时，不自动重试。
"""

import argparse
import html
import io
import json
import logging
import math
import os
import re
import uuid
from datetime import timedelta
from pathlib import Path

import requests
from flask import (Flask, abort, jsonify, render_template, request,
                   send_from_directory, session)
from PIL import Image
from google import genai
from google.genai import types as genai_types

from tasks import CapacityError, TaskFailure, TaskManager

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

logging.getLogger('werkzeug').setLevel(logging.WARNING)

try:
    from gtts import gTTS
    TTS_GTTS_AVAILABLE = True
except ImportError:
    logger.warning("gtts库未安装，gTTS引擎不可用")
    TTS_GTTS_AVAILABLE = False

try:
    from google.cloud import texttospeech
    from google.oauth2 import service_account
    TTS_GOOGLE_CLOUD_AVAILABLE = True
except ImportError:
    logger.warning("google-cloud-texttospeech库未安装，Google Cloud TTS引擎不可用")
    TTS_GOOGLE_CLOUD_AVAILABLE = False

# ---- 初始设计值常量（NAS_MIGRATION §3.5/§3.6；不提供环境变量配置） ----

DEFAULT_DATA_DIR = ".local-data"
MAX_REQUEST_BYTES = 10 * 1024 * 1024          # 单次请求体上限，Flask MAX_CONTENT_LENGTH
MAX_IMAGE_PIXELS = 20_000_000                 # 图片像素数上限
IMAGE_FORMATS = {                              # Pillow 实际格式 → (扩展名, MIME)
    "JPEG": (".jpg", "image/jpeg"),
    "PNG": (".png", "image/png"),
    "WEBP": (".webp", "image/webp"),
}
MAX_NON_TITLE_SENTENCES = 50                  # 单次 TTS 非标题句子数上限
MAX_SENTENCE_CHARS = 500                      # 单句长度上限
MAX_BOLD_WORDS_PER_SENTENCE = 20              # 每句重点词数量上限
MAX_BOLD_WORD_CHARS = 100                     # 单个重点词长度上限
MAX_TOTAL_SYNTH_CHARS = 10000                 # 单次合成文本总量上限（句+词，含重复部分）
TASK_BUDGET_SECONDS = 600                     # 单任务执行预算（从实际开始执行起计，不含排队）
SPEED_MIN, SPEED_MAX = -50, 50                # 语速百分比范围
SESSION_LIFETIME_DAYS = 7                     # 会话 cookie 生存期

OCR_MODEL = "gemini-3-flash-preview"
# genai HttpOptions.timeout 的单位是毫秒（SDK 内部会除以 1000 换算成秒传给 httpx），
# 实测误传秒值 120 会在 ~1 秒内触发 read timeout；这里表示 120 秒。
OCR_TIMEOUT_MS = 120_000
OCR_PROMPT = "请你将图片处理成markdown文本，根据句号、句点、数字标号将文本分割为句子并换行。如果句子中有被圈出、粗体、放大、与众不同的字体或颜色的文本，则把它们也用粗体标记。请仅输出markdown代码即可。"

# Azure REST（步骤 1 验证结论：requests、timeout=(5, 30)、不自动重试）
AZURE_REST_CONNECT_TIMEOUT = 5.0
AZURE_REST_READ_TIMEOUT = 30.0
AZURE_OUTPUT_FORMAT = "audio-24khz-48kbitrate-mono-mp3"
# gTTS 2.5.4 构造器默认无限等待，必须显式传入
GTTS_TIMEOUT_SECONDS = 30.0
# Google Cloud TTS 按调用传入超时，并显式关闭 GAPIC 默认重试
GOOGLE_CLOUD_TTS_TIMEOUT_SECONDS = 30.0

AUDIO_FILENAME_RE = re.compile(r"^(sentence|word)_[0-9]+(_[0-9]+)?\.mp3$")


def get_env_first(*names, default=None):
    """按顺序读取第一个非空环境变量"""
    for name in names:
        value = os.environ.get(name)
        if value:
            return value
    return default


GEMINI_API_KEY = get_env_first("GOOGLE_API_KEY", "GEMINI_API_KEY")
AZURE_SPEECH_KEY = get_env_first("AZURE_API_KEY", "AZURE_SPEECH_KEY", "SPEECH_KEY")
AZURE_SPEECH_REGION = get_env_first("AZURE_SPEECH_REGION", "AZURE_SERVICE_REGION", default="southeastasia")

# TTS服务配置字典
tts_models = {
    "SG-man": {
        "label": "新加坡英语-男声 (Microsoft)",
        "type": "ms-tts",
        "speech_key": AZURE_SPEECH_KEY,
        "service_region": AZURE_SPEECH_REGION,
        "voice_name": "en-SG-WayneNeural",
        "speed": "-10%",
    },
    "SG-woman": {
        "label": "新加坡英语-女声 (Microsoft)",
        "type": "ms-tts",
        "speech_key": AZURE_SPEECH_KEY,
        "service_region": AZURE_SPEECH_REGION,
        "voice_name": "en-SG-LunaNeural",
        "speed": "-15%",
    },
    "UK-man": {
        "label": "英式英语/中文-男声 (Microsoft)",
        "type": "ms-tts",
        "speech_key": AZURE_SPEECH_KEY,
        "service_region": AZURE_SPEECH_REGION,
        "voice_name": "en-GB-OllieMultilingualNeural",
        "speed": "-10%",
    },
    "UK-woman": {
        "label": "英式英语/中文-女声 (Microsoft)",
        "type": "ms-tts",
        "speech_key": AZURE_SPEECH_KEY,
        "service_region": AZURE_SPEECH_REGION,
        "voice_name": "en-GB-LibbyNeural",
        "speed": "-10%",
    },
    "US-Azure-man": {
        "label": "美式英语-男声 (Microsoft)",
        "type": "ms-tts",
        "speech_key": AZURE_SPEECH_KEY,
        "service_region": AZURE_SPEECH_REGION,
        "voice_name": "en-US-GuyNeural",
        "speed": "-10%",
    },
    "US-Azure-woman": {
        "label": "美式英语-女声 (Microsoft)",
        "type": "ms-tts",
        "speech_key": AZURE_SPEECH_KEY,
        "service_region": AZURE_SPEECH_REGION,
        "voice_name": "en-US-JennyNeural",
        "speed": "-10%",
    },
    "CH-man": {
        "label": "中文-男声 (Microsoft)",
        "type": "ms-tts",
        "speech_key": AZURE_SPEECH_KEY,
        "service_region": AZURE_SPEECH_REGION,
        "voice_name": "zh-CN-YunyangNeural",
        "speed": "-20%",
    },
    "CH-woman": {
        "label": "中文-女声 (Microsoft)",
        "type": "ms-tts",
        "speech_key": AZURE_SPEECH_KEY,
        "service_region": AZURE_SPEECH_REGION,
        "voice_name": "zh-CN-XiaoxiaoNeural",
        "speed": "-20%",
    },
    "French-Azure-man": {
        "label": "法语-男声 (Microsoft)",
        "type": "ms-tts",
        "speech_key": AZURE_SPEECH_KEY,
        "service_region": AZURE_SPEECH_REGION,
        "voice_name": "fr-FR-HenriNeural",
        "speed": "-10%",
    },
    "French-Azure-woman": {
        "label": "法语-女声 (Microsoft)",
        "type": "ms-tts",
        "speech_key": AZURE_SPEECH_KEY,
        "service_region": AZURE_SPEECH_REGION,
        "voice_name": "fr-FR-DeniseNeural",
        "speed": "-10%",
    },
    "UK-Google": {
        "label": "英式英语 (Google)",
        "type": "gtts",
        "lang": "en",
        "tld": "co.uk",
    },
    "US-Google": {
        "label": "美式英语 (Google)",
        "type": "gtts",
        "lang": "en",
        "tld": "com",
    },
    "French-Google": {
        "label": "法语 (Google)",
        "type": "gtts",
        "lang": "fr",
        "tld": "fr",
    },
    "Chinese-Google": {
        "label": "中文 (Google)",
        "type": "gtts",
        "lang": "zh",
        "tld": "com",
    },
    "English-Chirp": {
        "label": "英语 (Google Cloud Chirp 3 HD)",
        "type": "google-cloud-tts",
        "language_code": "en-GB",
        "voice_name": "en-GB-Chirp3-HD-Leda",
        "speed": "-10%",
    },
    "UK-Chirp-man": {
        "label": "英式英语-男声 (Google Cloud Chirp 3 HD)",
        "type": "google-cloud-tts",
        "language_code": "en-GB",
        "voice_name": "en-GB-Chirp3-HD-Charon",
        "speed": "-10%",
    },
    "UK-Chirp-woman": {
        "label": "英式英语-女声 (Google Cloud Chirp 3 HD)",
        "type": "google-cloud-tts",
        "language_code": "en-GB",
        "voice_name": "en-GB-Chirp3-HD-Leda",
        "speed": "-10%",
    },
    "US-Chirp-man": {
        "label": "美式英语-男声 (Google Cloud Chirp 3 HD)",
        "type": "google-cloud-tts",
        "language_code": "en-US",
        "voice_name": "en-US-Chirp3-HD-Charon",
        "speed": "-10%",
    },
    "US-Chirp-woman": {
        "label": "美式英语-女声 (Google Cloud Chirp 3 HD)",
        "type": "google-cloud-tts",
        "language_code": "en-US",
        "voice_name": "en-US-Chirp3-HD-Leda",
        "speed": "-10%",
    },
    "Chinese-Chirp": {
        "label": "中文 (Google Cloud Chirp 3 HD)",
        "type": "google-cloud-tts",
        "language_code": "cmn-CN",
        "voice_name": "cmn-CN-Chirp3-HD-Leda",
        "speed": "-10%",
    },
    "Chinese-Chirp-man": {
        "label": "中文普通话-男声 (Google Cloud Chirp 3 HD)",
        "type": "google-cloud-tts",
        "language_code": "cmn-CN",
        "voice_name": "cmn-CN-Chirp3-HD-Charon",
        "speed": "-10%",
    },
    "Chinese-Chirp-woman": {
        "label": "中文普通话-女声 (Google Cloud Chirp 3 HD)",
        "type": "google-cloud-tts",
        "language_code": "cmn-CN",
        "voice_name": "cmn-CN-Chirp3-HD-Leda",
        "speed": "-10%",
    },
    "French-Chirp": {
        "label": "法语 (Google Cloud Chirp 3 HD)",
        "type": "google-cloud-tts",
        "language_code": "fr-FR",
        "voice_name": "fr-FR-Chirp3-HD-Leda",
        "speed": "-10%",
    },
    "French-Chirp-man": {
        "label": "法语-男声 (Google Cloud Chirp 3 HD)",
        "type": "google-cloud-tts",
        "language_code": "fr-FR",
        "voice_name": "fr-FR-Chirp3-HD-Charon",
        "speed": "-10%",
    },
    "French-Chirp-woman": {
        "label": "法语-女声 (Google Cloud Chirp 3 HD)",
        "type": "google-cloud-tts",
        "language_code": "fr-FR",
        "voice_name": "fr-FR-Chirp3-HD-Leda",
        "speed": "-10%",
    },
}

tts_engines = [
    {"id": "azure", "label": "Azure"},
    {"id": "google", "label": "Google"},
    {"id": "gtts", "label": "gTTS"},
]

tts_languages = [
    {"id": "uk-en", "label": "英式英语"},
    {"id": "us-en", "label": "美式英语"},
    {"id": "sg-en", "label": "新加坡英语"},
    {"id": "cmn-cn", "label": "中文"},
    {"id": "fr-fr", "label": "法语"},
]

voice_genders = [
    {"id": "male", "label": "男声"},
    {"id": "female", "label": "女声"},
]

tts_voice_matrix = {
    "azure": {
        "uk-en": {"male": "UK-man", "female": "UK-woman"},
        "us-en": {"male": "US-Azure-man", "female": "US-Azure-woman"},
        "sg-en": {"male": "SG-man", "female": "SG-woman"},
        "cmn-cn": {"male": "CH-man", "female": "CH-woman"},
        "fr-fr": {"male": "French-Azure-man", "female": "French-Azure-woman"},
    },
    "google": {
        "uk-en": {"male": "UK-Chirp-man", "female": "UK-Chirp-woman"},
        "us-en": {"male": "US-Chirp-man", "female": "US-Chirp-woman"},
        "cmn-cn": {"male": "Chinese-Chirp-man", "female": "Chinese-Chirp-woman"},
        "fr-fr": {"male": "French-Chirp-man", "female": "French-Chirp-woman"},
    },
    "gtts": {
        "uk-en": {"male": "UK-Google", "female": "UK-Google"},
        "us-en": {"male": "US-Google", "female": "US-Google"},
        "cmn-cn": {"male": "Chinese-Google", "female": "Chinese-Google"},
        "fr-fr": {"male": "French-Google", "female": "French-Google"},
    },
}


def tts_supports_speed(tts_model):
    """判断TTS模型是否支持自定义语速"""
    return tts_model.get("type") in ("ms-tts", "google-cloud-tts")


def parse_speed_percent(speed_value, default=0):
    """将语速配置解析为百分比整数"""
    if speed_value is None:
        return default

    if isinstance(speed_value, str):
        speed_value = speed_value.strip().rstrip('%')

    try:
        speed_percent = int(float(speed_value))
    except (TypeError, ValueError):
        return default

    return max(-50, min(50, speed_percent))


def format_speed_percent(speed_percent):
    """转成Azure SSML rate格式"""
    speed_percent = parse_speed_percent(speed_percent)
    return f"{speed_percent:+d}%"


def format_google_cloud_speaking_rate(speed_percent):
    """转成Google Cloud TTS speakingRate倍率"""
    speed_percent = parse_speed_percent(speed_percent)
    return 1 + speed_percent / 100


def get_tts_options():
    """返回前端需要的TTS选项元数据"""
    return [
        {
            "id": key,
            "label": value.get("label", key),
            "supports_speed": tts_supports_speed(value),
            "default_speed": parse_speed_percent(value.get("speed")),
        }
        for key, value in tts_models.items()
    ]


# ---- 请求校验 ----

class RequestValidationError(Exception):
    """请求入口校验失败；携带 HTTP 状态码与稳定错误类型。"""

    def __init__(self, status, code, message):
        super().__init__(message)
        self.status = status
        self.code = code
        self.message = message


def validate_speed_value(value, tts_model):
    """校验 speed：缺省取模型默认；非有限数值或越界返回 400；不支持速度的引擎忽略该值。"""
    if value is None:
        if tts_supports_speed(tts_model):
            return parse_speed_percent(tts_model.get("speed"))
        return None
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise RequestValidationError(400, "invalid_input", "speed 必须是数字")
    if not math.isfinite(value):
        raise RequestValidationError(400, "invalid_input", "speed 必须是有限数值")
    if value < SPEED_MIN or value > SPEED_MAX:
        raise RequestValidationError(400, "invalid_input",
                                     f"speed 超出 {SPEED_MIN} 到 {SPEED_MAX} 的范围")
    if not tts_supports_speed(tts_model):
        return None
    return int(value)


def _validate_sentence_row(row, position):
    """校验单个句子行，返回规范化 dict。"""
    if not isinstance(row, dict):
        raise RequestValidationError(400, "invalid_input", f"第 {position} 项不是有效对象")
    text = row.get("text")
    if not isinstance(text, str) or not text.strip():
        raise RequestValidationError(400, "invalid_input", f"第 {position} 句缺少有效文本")
    if len(text) > MAX_SENTENCE_CHARS:
        raise RequestValidationError(400, "invalid_input",
                                     f"第 {position} 句超过 {MAX_SENTENCE_CHARS} 字符")
    is_title = row.get("is_title", False)
    if not isinstance(is_title, bool):
        raise RequestValidationError(400, "invalid_input", f"第 {position} 项 is_title 必须是布尔值")
    original_text = row.get("original_text")
    if original_text is not None and not isinstance(original_text, str):
        raise RequestValidationError(400, "invalid_input", f"第 {position} 项 original_text 必须是字符串")
    title = row.get("title")
    if title is not None and not isinstance(title, str):
        raise RequestValidationError(400, "invalid_input", f"第 {position} 项 title 必须是字符串")

    bold_words = row.get("bold_words") or []
    if not isinstance(bold_words, list):
        raise RequestValidationError(400, "invalid_input", f"第 {position} 句 bold_words 必须是数组")
    if len(bold_words) > MAX_BOLD_WORDS_PER_SENTENCE:
        raise RequestValidationError(
            400, "invalid_input", f"第 {position} 句重点词数量超过 {MAX_BOLD_WORDS_PER_SENTENCE}")
    words = []
    for word_item in bold_words:
        if not isinstance(word_item, dict) or not isinstance(word_item.get("word"), str) \
                or not word_item["word"].strip():
            raise RequestValidationError(400, "invalid_input",
                                         f"第 {position} 句存在无效的重点词条目")
        if len(word_item["word"]) > MAX_BOLD_WORD_CHARS:
            raise RequestValidationError(
                400, "invalid_input", f"第 {position} 句重点词超过 {MAX_BOLD_WORD_CHARS} 字符")
        words.append(word_item["word"])

    return {
        "text": text,
        "original_text": original_text if original_text is not None else text,
        "is_title": is_title,
        "title": title or "",
        "bold_words": words,
    }


def validate_tts_request(data):
    """校验 /generate-tts 请求；返回 (规范化句子列表, tts_model, speed_percent)。"""
    if not isinstance(data, dict):
        raise RequestValidationError(400, "invalid_input", "请求体必须是 JSON 对象")

    sentences = data.get("sentences")
    if not isinstance(sentences, list) or not sentences:
        raise RequestValidationError(400, "invalid_input", "没有可生成音频的OCR文本")

    clean_rows = []
    non_title_count = 0
    total_chars = 0
    for position, row in enumerate(sentences, start=1):
        clean = _validate_sentence_row(row, position)
        if not clean["is_title"]:
            non_title_count += 1
            total_chars += len(clean["text"]) + sum(len(word) for word in clean["bold_words"])
        clean_rows.append(clean)

    if non_title_count == 0:
        raise RequestValidationError(400, "invalid_input", "没有可生成音频的句子")
    if non_title_count > MAX_NON_TITLE_SENTENCES:
        raise RequestValidationError(
            400, "invalid_input", f"非标题句子数超过 {MAX_NON_TITLE_SENTENCES} 条")
    if total_chars > MAX_TOTAL_SYNTH_CHARS:
        raise RequestValidationError(
            400, "invalid_input", f"单次合成文本总量超过 {MAX_TOTAL_SYNTH_CHARS} 字符")

    engine = data.get("tts_engine")
    language = data.get("tts_language")
    gender = data.get("voice_gender")
    if not all(isinstance(field, str) and field for field in (engine, language, gender)):
        raise RequestValidationError(400, "invalid_input", "必须提供 tts_engine、tts_language 与 voice_gender")
    engine_options = tts_voice_matrix.get(engine)
    language_options = (engine_options or {}).get(language)
    tts_key = (language_options or {}).get(gender)
    if tts_key is None:
        raise RequestValidationError(400, "invalid_tts_combo",
                                     "所选引擎、语言与音色组合不受支持，不回退到默认音色")
    tts_model = tts_models[tts_key]

    speed_percent = validate_speed_value(data.get("speed"), tts_model)
    return clean_rows, tts_model, speed_percent


def validate_and_save_image(file_storage, task_dir):
    """校验上传图片的实际内容并写入任务目录，返回 (输入文件名, MIME 类型)。

    上传文件名只作为显示信息，不参与目录定位；扩展名由 Pillow 实际格式决定。
    """
    raw = file_storage.read()
    if not raw:
        raise RequestValidationError(400, "invalid_input", "上传的文件为空")
    try:
        with Image.open(io.BytesIO(raw)) as img:
            img.verify()
        with Image.open(io.BytesIO(raw)) as img:
            image_format = img.format
            width, height = img.size
    except Exception:
        raise RequestValidationError(415, "invalid_input", "文件不是有效图片或已损坏")

    if image_format not in IMAGE_FORMATS:
        raise RequestValidationError(
            415, "invalid_input", f"图片格式不支持，仅支持 JPEG、PNG、WebP（实际为 {image_format}）")
    if width * height > MAX_IMAGE_PIXELS:
        raise RequestValidationError(
            413, "too_large", f"图片像素数超过上限 {MAX_IMAGE_PIXELS}（实际 {width}x{height}）")

    extension, mime_type = IMAGE_FORMATS[image_format]
    input_name = f"input{extension}"
    (task_dir / input_name).write_bytes(raw)
    return input_name, mime_type


# ---- OCR ----

_gemini_client = None


def get_gemini_client():
    """惰性创建 genai 客户端：显式 HTTP 超时、单次尝试不自动重试。"""
    global _gemini_client
    if _gemini_client is None:
        client_options = {"timeout": OCR_TIMEOUT_MS, "retry_options": {"attempts": 1}}
        if GEMINI_API_KEY:
            _gemini_client = genai.Client(
                api_key=GEMINI_API_KEY,
                http_options=genai_types.HttpOptions(**client_options),
            )
        else:
            _gemini_client = genai.Client(
                http_options=genai_types.HttpOptions(**client_options),
            )
    return _gemini_client


def merge_standalone_number_labels(lines):
    """将独立一行的编号合并到下一行句子"""
    merged_lines = []
    line_index = 0
    number_label_pattern = re.compile(r'^\s*\d+\s*[\.\)、\):：]?\s*\*?\s*$')

    while line_index < len(lines):
        line = lines[line_index].strip()
        line_index += 1

        if not line:
            continue

        if number_label_pattern.match(line):
            while line_index < len(lines) and not lines[line_index].strip():
                line_index += 1

            if line_index < len(lines):
                merged_lines.append(f"{line} {lines[line_index].strip()}")
                line_index += 1
            else:
                merged_lines.append(line)
        else:
            merged_lines.append(line)

    return merged_lines


def parse_ocr_response(text):
    """解析 OCR 返回的文本，输出 §3.4 契约的结构化句子数组。

    bold_words 统一为对象数组（每项含 word），标题行不带 bold_words；
    首个非空行为标题，所有句子都携带该标题。
    """
    text = re.sub(r'^```markdown\s*|\s*```$', '', text, flags=re.MULTILINE)
    lines = merge_standalone_number_labels(text.split('\n'))
    first_nonempty = next((i for i, line in enumerate(lines) if line.strip()), None)
    title = lines[first_nonempty].strip() if first_nonempty is not None else ''

    sentences = []
    for i, line in enumerate(lines):
        if not line.strip():
            continue
        clean_sentence = re.sub(r'\*', '', line)
        if i == first_nonempty:
            sentences.append({
                'text': clean_sentence.strip(),
                'original_text': line.strip(),
                'is_title': True,
                'title': title,
            })
            continue
        bold_words = [{'word': word.strip()} for word in re.findall(r'\*\*(.*?)\*\*', line) if word.strip()]
        sentences.append({
            'text': clean_sentence.strip(),
            'original_text': line.strip(),
            'is_title': False,
            'title': title,
            'bold_words': bold_words,
        })
    return sentences


def extract_text_cloud(image_path, mime_type):
    """使用 Gemini API 进行 OCR 识别"""
    logger.info("使用Gemini OCR服务处理图片")
    client = get_gemini_client()

    with open(image_path, 'rb') as f:
        image_bytes = f.read()

    response = client.models.generate_content(
        model=OCR_MODEL,
        contents=[
            {"inline_data": {"mime_type": mime_type, "data": image_bytes}},
            OCR_PROMPT,
        ],
    )

    text = response.text
    return parse_ocr_response(text)


def run_ocr_task(context, mime_type):
    """OCR 任务执行函数：一次外部调用，成功或失败都在任务状态中明确表达。"""
    context.set_progress(0, 0, "正在识别图片文本")
    try:
        sentences = extract_text_cloud(context.input_path, mime_type)
    except Exception as exc:
        raise TaskFailure("ocr_failed", f"OCR 识别失败：{exc}") from exc
    if not sentences:
        raise TaskFailure("ocr_failed", "OCR 未能从图片中提取到文本")
    logger.info("OCR识别成功，提取了%d个句子", len(sentences))
    return sentences, None


# ---- TTS 合成 ----

def build_ssml(text, voice_name, speed):
    """与既有 SSML 结构保持一致：文本转义，prosody 控制语速。"""
    ssml_body = html.escape(text, quote=False)
    return f"""
<speak version='1.0' xmlns='http://www.w3.org/2001/10/synthesis' xml:lang='en-US'>
    <voice name='{voice_name}'>
        <prosody rate='{speed}'>
            {ssml_body}
        </prosody>
    </voice>
</speak>
"""


def synthesize_azure_rest(text, tts_model, speed_percent, output_path):
    """Azure 文本转语音 REST 接口（NAS_MIGRATION §3.5 步骤 1 选定实现）。

    超时 (connect, read) = (5, 30) 秒，requests 默认不自动重试；
    失败直接抛出异常，不写空文件。
    """
    speech_key = tts_model.get("speech_key")
    if not speech_key:
        raise ValueError("Azure Speech 密钥未配置（AZURE_API_KEY）")
    speed = format_speed_percent(speed_percent) if speed_percent is not None else tts_model["speed"]
    ssml = build_ssml(text, tts_model["voice_name"], speed)
    url = f"https://{tts_model['service_region']}.tts.speech.microsoft.com/cognitiveservices/v1"
    headers = {
        "Ocp-Apim-Subscription-Key": speech_key,
        "Content-Type": "application/ssml+xml",
        "X-Microsoft-OutputFormat": AZURE_OUTPUT_FORMAT,
    }
    response = requests.post(
        url,
        data=ssml.encode("utf-8"),
        headers=headers,
        timeout=(AZURE_REST_CONNECT_TIMEOUT, AZURE_REST_READ_TIMEOUT),
    )
    response.raise_for_status()
    if not response.content:
        raise ValueError("Azure TTS 返回空音频")
    output_path.write_bytes(response.content)


def synthesize_gtts(text, tts_model, output_path):
    """gTTS：构造器显式传入有限超时（2.5.4 默认无限等待）。"""
    if not TTS_GTTS_AVAILABLE:
        raise ValueError("gtts 库未安装，gTTS 引擎不可用")
    tts = gTTS(text=text, lang=tts_model["lang"], tld=tts_model["tld"],
               slow=False, timeout=GTTS_TIMEOUT_SECONDS)
    try:
        tts.save(str(output_path))
    except Exception:
        output_path.unlink(missing_ok=True)
        raise
    if not output_path.is_file() or output_path.stat().st_size == 0:
        output_path.unlink(missing_ok=True)
        raise ValueError("gTTS 未产生有效音频")


_google_cloud_tts_client = None


def get_google_cloud_tts_client():
    """惰性创建 Google Cloud TTS 客户端（进程内复用）。"""
    global _google_cloud_tts_client
    if _google_cloud_tts_client is None:
        credentials_json = get_env_first(
            "GOOGLE_CLOUD_TTS_CREDENTIALS_JSON",
            "GOOGLE_CREDENTIALS_JSON",
            "GOOGLE_SERVICE_ACCOUNT_JSON",
            "GCP_SERVICE_ACCOUNT_JSON",
        )
        if credentials_json:
            credentials = service_account.Credentials.from_service_account_info(json.loads(credentials_json))
            _google_cloud_tts_client = texttospeech.TextToSpeechClient(credentials=credentials)
        else:
            _google_cloud_tts_client = texttospeech.TextToSpeechClient()
    return _google_cloud_tts_client


def synthesize_google_cloud(text, tts_model, speed_percent, output_path):
    """Google Cloud TTS：按调用传入超时，并显式关闭默认重试。"""
    if not TTS_GOOGLE_CLOUD_AVAILABLE:
        raise ValueError("google-cloud-texttospeech 库未安装，Google Cloud TTS 引擎不可用")
    client = get_google_cloud_tts_client()
    input_text = texttospeech.SynthesisInput(text=text)
    voice = texttospeech.VoiceSelectionParams(
        language_code=tts_model["language_code"],
        name=tts_model["voice_name"],
    )
    if speed_percent is not None:
        speaking_rate = format_google_cloud_speaking_rate(speed_percent)
    else:
        speaking_rate = format_google_cloud_speaking_rate(tts_model.get("speed"))
    audio_config = texttospeech.AudioConfig(
        audio_encoding=texttospeech.AudioEncoding.MP3,
        speaking_rate=speaking_rate,
    )
    response = client.synthesize_speech(
        input=input_text,
        voice=voice,
        audio_config=audio_config,
        retry=None,
        timeout=GOOGLE_CLOUD_TTS_TIMEOUT_SECONDS,
    )
    if not response.audio_content:
        raise ValueError("Google Cloud TTS 返回空音频")
    output_path.write_bytes(response.audio_content)


def synthesize_item(text, tts_model, speed_percent, output_path):
    """合成单个条目并写入任务目录；失败抛异常，不生成空文件。"""
    if not text or not text.strip():
        raise ValueError("输入文本为空，无法生成音频")
    text = text.strip()
    tts_type = tts_model["type"]
    if tts_type == "ms-tts":
        synthesize_azure_rest(text, tts_model, speed_percent, output_path)
    elif tts_type == "gtts":
        # gTTS 不支持语速，非默认速度在校验层已被忽略
        synthesize_gtts(text, tts_model, output_path)
    elif tts_type == "google-cloud-tts":
        synthesize_google_cloud(text, tts_model, speed_percent, output_path)
    else:
        raise ValueError(f"未知TTS类型: {tts_type}")


def run_tts_task(context, sentences, tts_model, speed_percent):
    """TTS 任务执行函数：串行合成，进度按尝试次数计数，预算在连续调用之间检查。"""
    speak_rows = [(idx, row) for idx, row in enumerate(sentences) if not row["is_title"]]
    total = len(speak_rows) + sum(len(row["bold_words"]) for _, row in speak_rows)
    context.set_progress(0, total, "开始生成音频")

    current = 0
    warnings = []
    any_audio = False
    built = {}

    for idx, row in speak_rows:
        output_row = {
            "text": row["text"],
            "original_text": row["original_text"],
            "is_title": False,
            "title": row["title"],
            "bold_words": [],
        }

        sentence_out = f"sentence_{idx}.mp3"
        if context.elapsed_seconds() >= TASK_BUDGET_SECONDS:
            warnings.append(f"句子 {idx + 1} 未尝试：任务执行预算已耗尽")
        else:
            try:
                synthesize_item(row["text"], tts_model, speed_percent,
                                context.task_dir / sentence_out)
                output_row["audio_path"] = f"/audio/{context.task_id}/{sentence_out}"
                any_audio = True
            except Exception as exc:
                warnings.append(f"句子 {idx + 1} 音频生成失败：{exc}")
            current += 1
            context.set_progress(current, total, f"正在生成音频 {current}/{total}")

        for widx, word in enumerate(row["bold_words"]):
            word_out = f"word_{idx}_{widx}.mp3"
            word_entry = {"word": word}
            if context.elapsed_seconds() >= TASK_BUDGET_SECONDS:
                warnings.append(f"重点词 {word} 未尝试：任务执行预算已耗尽")
            else:
                try:
                    synthesize_item(word, tts_model, speed_percent,
                                    context.task_dir / word_out)
                    word_entry["audio_path"] = f"/audio/{context.task_id}/{word_out}"
                    any_audio = True
                except Exception as exc:
                    warnings.append(f"重点词 {word} 音频生成失败：{exc}")
                current += 1
                context.set_progress(current, total, f"正在生成音频 {current}/{total}")
            output_row["bold_words"].append(word_entry)

        built[idx] = output_row

    # 按原顺序组装结果，标题行原样保留
    result = []
    for idx, row in enumerate(sentences):
        if row["is_title"]:
            result.append({
                "text": row["text"],
                "original_text": row["original_text"],
                "is_title": True,
                "title": row["title"],
            })
        else:
            result.append(built[idx])

    if not any_audio:
        if current == 0 and context.elapsed_seconds() >= TASK_BUDGET_SECONDS:
            raise TaskFailure("tts_failed", "任务执行预算已耗尽，未能生成任何音频")
        reason = warnings[0] if warnings else "未知原因"
        raise TaskFailure("tts_failed", f"全部音频生成失败：{reason}")

    return result, warnings or None


# ---- 应用与路由 ----

def _error_response(status, code, message):
    return jsonify({"error": {"code": code, "message": message}}), status


def _require_session_id():
    """写接口与状态查询都要求有效会话；缺少时返回 403（状态查询处另行处理）。"""
    sid = session.get("sid")
    if not sid:
        abort(403)
    return sid


def _require_same_origin():
    """写接口同源检查：拒绝不匹配的 Origin；无 Origin 时按 Referer 校验；都没有则拒绝。"""
    header = request.headers.get("Origin") or request.headers.get("Referer")
    if not header:
        abort(403)
    match = re.match(r"^[a-zA-Z][a-zA-Z0-9+.-]*://([^/]+)", header)
    if not match or match.group(1) != request.host:
        abort(403)


def create_app(task_manager, *, max_content_length=MAX_REQUEST_BYTES):
    """构建 Flask 应用；测试可注入独立 TaskManager 与更小的请求体上限。"""
    flask_app = Flask(__name__)
    flask_app.config.update(
        SECRET_KEY=os.environ["SECRET_KEY"],
        MAX_CONTENT_LENGTH=max_content_length,
        SESSION_COOKIE_NAME="endictation_session",
        SESSION_COOKIE_HTTPONLY=True,
        SESSION_COOKIE_SAMESITE="Lax",
        # 本阶段为本机/受信 LAN HTTP 验证，不启用仅 HTTPS 可发送的 Secure cookie
        SESSION_COOKIE_SECURE=False,
        PERMANENT_SESSION_LIFETIME=timedelta(days=SESSION_LIFETIME_DAYS),
    )

    @flask_app.errorhandler(413)
    def request_too_large(_error):
        return _error_response(413, "too_large", "请求体超过大小限制，未被受理")

    @flask_app.errorhandler(403)
    def forbidden(_error):
        return _error_response(403, "forbidden", "缺少有效会话或同源检查未通过")

    @flask_app.get("/")
    def index():
        if "sid" not in session:
            session["sid"] = uuid.uuid4().hex
            session.permanent = True
        return render_template(
            "index.html",
            tts_options=get_tts_options(),
            tts_engines=tts_engines,
            tts_languages=tts_languages,
            voice_genders=voice_genders,
            tts_voice_matrix=tts_voice_matrix,
        )

    @flask_app.get("/health")
    def health():
        """只证明进程可响应，不调用收费 API，也不宣称凭据有效。"""
        return jsonify({"status": "ok"})

    @flask_app.post("/upload")
    def upload_file():
        """受理 OCR 任务：校验后立即返回 202，OCR 在后台执行。"""
        _require_same_origin()
        sid = _require_session_id()

        if "file" not in request.files:
            return _error_response(400, "invalid_input", "没有文件上传")
        file = request.files["file"]
        if not file.filename:
            return _error_response(400, "invalid_input", "未选择文件")

        holder = {}

        def input_saver(task_dir):
            input_name, mime_type = validate_and_save_image(file, task_dir)
            holder["mime_type"] = mime_type
            return str(task_dir / input_name)

        def execute(context):
            return run_ocr_task(context, holder["mime_type"])

        try:
            snapshot = task_manager.submit(
                stage="ocr", owner=sid, execute=execute, input_saver=input_saver)
        except RequestValidationError as exc:
            return _error_response(exc.status, exc.code, exc.message)
        except CapacityError as exc:  # noqa: F821 - 来自 tasks 模块
            return _error_response(429, "capacity", str(exc))
        return jsonify({
            "task_id": snapshot["task_id"],
            "status_url": f"/tasks/{snapshot['task_id']}",
        }), 202

    @flask_app.post("/generate-tts")
    def generate_tts():
        """受理 TTS 任务：校验后立即返回 202，合成在后台执行。"""
        _require_same_origin()
        sid = _require_session_id()

        data = request.get_json(silent=True)
        if data is None:
            return _error_response(400, "invalid_input", "请求体不是有效 JSON")
        try:
            sentences, tts_model, speed_percent = validate_tts_request(data)
        except RequestValidationError as exc:
            return _error_response(exc.status, exc.code, exc.message)

        def execute(context):
            return run_tts_task(context, sentences, tts_model, speed_percent)

        try:
            snapshot = task_manager.submit(stage="tts", owner=sid, execute=execute)
        except CapacityError as exc:  # noqa: F821
            return _error_response(429, "capacity", str(exc))
        return jsonify({
            "task_id": snapshot["task_id"],
            "status_url": f"/tasks/{snapshot['task_id']}",
        }), 202

    @flask_app.get("/tasks/<task_id>")
    def get_task(task_id):
        """任务状态查询；轮询返回 200 不代表任务成功，由 status 字段表达结果。"""
        sid = session.get("sid")
        snapshot = task_manager.get(task_id, sid) if sid else None
        if snapshot is None:
            # 统一提示，不区分不存在、过期与不属于该会话
            return _error_response(404, "task_not_found", "任务不存在或已过期")
        return jsonify(snapshot)

    @flask_app.get("/audio/<task_id>/<filename>")
    def serve_audio(task_id, filename):
        """按任务目录返回音频；文件名受限，owner 校验失败或清理竞争均返回 404。"""
        sid = session.get("sid")
        if not sid or not AUDIO_FILENAME_RE.match(filename):
            return _error_response(404, "not_found", "音频不存在或已过期")
        task_dir = task_manager.task_dir(task_id, sid)
        if task_dir is None or not (task_dir / filename).is_file():
            return _error_response(404, "not_found", "音频不存在或已过期")
        return send_from_directory(task_dir, filename, mimetype="audio/mpeg")

    return flask_app


# ---- 启动引导 ----

def bootstrap_runtime(data_dir_env=None):
    """进程启动入口检查与初始化；Gunicorn worker 导入与 python app.py 共用。

    - 必须设置 SECRET_KEY，无任何回退默认值；
    - 任务目录必须可写，权限错误直接启动失败并给出路径与所需权限。
    """
    if not os.environ.get("SECRET_KEY"):
        raise RuntimeError(
            "启动失败：必须设置环境变量 SECRET_KEY（用于会话签名），不得使用回退默认值。")
    data_dir = Path(data_dir_env or os.environ.get("DATA_DIR") or DEFAULT_DATA_DIR)
    tasks_root = data_dir / "tasks"
    try:
        tasks_root.mkdir(parents=True, exist_ok=True)
        probe = tasks_root / ".write-probe"
        probe.write_text("")
        probe.unlink()
    except OSError as exc:
        raise RuntimeError(
            f"启动失败：任务目录 {tasks_root} 不可写（需要写入权限）：{exc}") from exc
    task_manager = TaskManager(tasks_root)
    # 进程重启后旧任务一律失效：只清理本应用专属任务目录
    task_manager.purge_all_task_dirs()
    task_manager.start_cleanup_loop()
    logger.info("运行数据目录：%s", tasks_root)
    return task_manager


manager = bootstrap_runtime()
app = create_app(manager)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--port', type=int, default=int(os.environ.get('PORT', 5001)),
                        help='Port to run the server on')
    args = parser.parse_args()
    app.run(host='0.0.0.0', debug=False, port=args.port)
