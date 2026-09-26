"""运行时配置：可提交 Git 的默认文件与本机私有配置。

配置内容（管理员经页面「配置」按钮导入/导出）：
- secrets：Gemini / Azure / Google Cloud TTS 凭据
- ocr_model：OCR 模型名
- tts_engines / tts_languages / tts_models / tts_voice_matrix：引擎、语言、
  音色定义与组合矩阵（管理员可增删语言与音色，例如为 gTTS 增加语言）

明确不进入配置文件（终端用户在页面选择）：速度、男女声。
输入限制、超时、保留期等保护参数仍为代码常量（NAS_MIGRATION §3.5/§3.6）。

凭据不读环境变量（2026-09-25 用户决策）：配置文件是唯一凭据来源；
首次运行无私有配置文件时使用 config.default.json（密钥为空，调用时明确报错，
经页面「配置」导入后即可用）。
实际配置文件含密钥，权限应为 0600，不入 Git、不进镜像；默认文件不得含密钥。
"""

from __future__ import annotations

import copy
import json
import os
import re
import tempfile
import threading
from pathlib import Path

DATA_DIR = Path(__file__).resolve().parent / ".local-data"

MODEL_TYPES = ("ms-tts", "gtts", "google-cloud-tts")
VOICE_GENDERS = ("male", "female")

# 各类型模型必填字段；ms-tts 可选 service_region 覆盖全局区域
REQUIRED_MODEL_FIELDS = {
    "ms-tts": ("voice_name",),
    "gtts": ("lang", "tld"),
    "google-cloud-tts": ("language_code", "voice_name"),
}

DEFAULT_CONFIG_PATH = Path(__file__).resolve().with_name("config.default.json")

SECRET_KEYS = (
    "google_api_key",
    "azure_speech_key",
    "azure_speech_region",
    "google_cloud_tts_credentials_json",
)


class ConfigError(ValueError):
    """配置结构不合法；message 面向管理员，可安全返回。"""


def build_default_config():
    """读取可提交 Git 的无凭据默认配置；真实凭据只保存在运行目录。"""
    data = json.loads(DEFAULT_CONFIG_PATH.read_text(encoding="utf-8"))
    secrets = data.get("secrets", {})
    if any(secrets.get(key) for key in SECRET_KEYS if key != "azure_speech_region"):
        raise ConfigError("config.default.json 不允许包含密钥或服务账号")
    return data


def _require(condition, message):
    if not condition:
        raise ConfigError(message)


def _check_id_list(value, name):
    _require(isinstance(value, list) and value, f"{name} 必须是非空数组")
    seen = set()
    for item in value:
        _require(isinstance(item, dict), f"{name} 的每一项必须是对象")
        _require(set(item) == {"id", "label"}, f"{name} 的每一项只能包含 id 和 label")
        item_id = item.get("id")
        label = item.get("label")
        _require(isinstance(item_id, str) and item_id.strip(), f"{name} 存在缺少 id 的条目")
        _require(isinstance(label, str) and label.strip(), f"{name} 的 {item_id} 缺少 label")
        _require(item_id not in seen, f"{name} 的 id 重复：{item_id}")
        seen.add(item_id)


def validate_config(data):
    """校验并规范化配置，返回深拷贝的合法配置 dict；不合法抛 ConfigError。"""
    _require(isinstance(data, dict), "配置必须是 JSON 对象")
    _require(set(data) - {"preferred_tts"} == {"version", "secrets", "ocr_model", "tts_engines",
                           "tts_languages", "tts_models", "tts_voice_matrix"},
             "配置字段不完整或包含未知字段；不接受速度、性别等用户偏好")
    version = data.get("version")
    _require(type(version) is int and version == 1, "不支持的配置版本")

    secrets = data.get("secrets")
    _require(isinstance(secrets, dict), "secrets 必须是对象")
    _require(not set(secrets) - set(SECRET_KEYS), "secrets 包含未知字段")
    clean_secrets = {}
    for key in SECRET_KEYS:
        value = secrets.get(key)
        if key == "google_cloud_tts_credentials_json" and value is not None:
            _require(isinstance(value, dict), "Google Cloud 凭据必须是服务账号 JSON 对象")
            _require(value.get("type") == "service_account", "Google Cloud 凭据必须是 service_account")
            _require(all(isinstance(value.get(k), str) and value[k].strip()
                         for k in ("client_email", "private_key", "token_uri")), "Google Cloud 服务账号缺少必填字段")
            _require(value["token_uri"] == "https://oauth2.googleapis.com/token", "Google Cloud token_uri 必须使用官方地址")
            try:
                from google.oauth2 import service_account
                service_account.Credentials.from_service_account_info(value)
            except (ValueError, TypeError) as exc:
                raise ConfigError("Google Cloud 服务账号私钥无效") from exc
            clean_secrets[key] = copy.deepcopy(value)
        elif value is not None:
            _require(isinstance(value, str), f"secrets.{key} 必须是字符串")
            clean_secrets[key] = value
        else:
            clean_secrets[key] = None
    region = clean_secrets.get("azure_speech_region")
    _require(not region or re.fullmatch(r"[a-z0-9-]+", region), "Azure 区域格式无效")

    ocr_model = data.get("ocr_model")
    _require(isinstance(ocr_model, str) and ocr_model.strip(), "ocr_model 必须是非空字符串")

    engines = data.get("tts_engines")
    languages = data.get("tts_languages")
    models = data.get("tts_models")
    matrix = data.get("tts_voice_matrix")
    _check_id_list(engines, "tts_engines")
    _check_id_list(languages, "tts_languages")
    engine_ids = {item["id"] for item in engines}
    language_ids = {item["id"] for item in languages}

    _require(isinstance(models, dict) and models, "tts_models 必须是非空对象")
    clean_models = {}
    for key, model in models.items():
        _require(isinstance(key, str) and key.strip(), "音色 ID 必须是非空字符串")
        _require(isinstance(model, dict), f"音色 {key} 必须是对象")
        model_type = model.get("type")
        _require(isinstance(model_type, str), f"音色 {key} 的 type 必须是字符串")
        allowed = {"label", "type", *REQUIRED_MODEL_FIELDS.get(model_type, ())}
        if model_type == "ms-tts":
            allowed.add("service_region")
        _require(not set(model) - allowed,
                 f"音色 {key} 包含未知字段，不接受默认速度或性别偏好")
        _require(model_type in MODEL_TYPES,
                 f"音色 {key} 的 type 必须是 {MODEL_TYPES} 之一（实际 {model_type}）")
        _require(isinstance(model.get("label"), str) and model["label"].strip(),
                 f"音色 {key} 缺少 label")
        for field in REQUIRED_MODEL_FIELDS[model_type]:
            _require(isinstance(model.get(field), str) and model[field].strip(),
                     f"音色 {key}（{model_type}）缺少必填字段 {field}")
        clean_model = {"label": model["label"], "type": model_type}
        if "service_region" in model:
            _require(isinstance(model["service_region"], str) and model["service_region"].strip(),
                     f"音色 {key} 的 service_region 必须是非空字符串")
        for field in ("voice_name", "lang", "tld", "language_code", "service_region"):
            if isinstance(model.get(field), str) and model[field].strip():
                clean_model[field] = model[field]
        clean_models[key] = clean_model
        region = clean_model.get("service_region")
        _require(not region or re.fullmatch(r"[a-z0-9-]+", region), f"音色 {key} 的区域格式无效")

    _require(isinstance(matrix, dict) and matrix, "tts_voice_matrix 必须是非空对象")
    _require(set(matrix) == engine_ids, "每个引擎必须配置至少一种语言")
    clean_matrix = {}
    for engine_id, language_map in matrix.items():
        _require(engine_id in engine_ids, f"矩阵引用了未定义的引擎：{engine_id}")
        _require(isinstance(language_map, dict) and language_map,
                 f"矩阵中引擎 {engine_id} 必须是非空对象")
        clean_language_map = {}
        for language_id, gender_map in language_map.items():
            _require(language_id in language_ids, f"矩阵引用了未定义的语言：{language_id}")
            _require(isinstance(gender_map, dict) and gender_map,
                     f"矩阵中 {engine_id}/{language_id} 必须是非空对象")
            clean_gender_map = {}
            for gender, model_key in gender_map.items():
                _require(gender in VOICE_GENDERS,
                         f"矩阵中 {engine_id}/{language_id} 出现无效性别 {gender}（仅支持 male/female）")
                _require(isinstance(model_key, str), f"矩阵中 {engine_id}/{language_id}/{gender} 必须是音色键")
                _require(model_key in clean_models,
                         f"矩阵引用了未定义的音色：{model_key}（{engine_id}/{language_id}/{gender}）")
                clean_gender_map[gender] = model_key
            clean_language_map[language_id] = clean_gender_map
        clean_matrix[engine_id] = clean_language_map

    if "preferred_tts" in data:
        preferred = data["preferred_tts"]
        _require(isinstance(preferred, dict) and set(preferred) == {"engine", "language"},
                 "preferred_tts 必须仅包含 engine 和 language")
        _require(all(isinstance(preferred[key], str) for key in ("engine", "language")),
                 "首选引擎和语言必须是字符串")
        _require(preferred["language"] in clean_matrix.get(preferred["engine"], {}),
                 "首选引擎和语言组合不存在，请重新选择首选项")

    return {
        "version": 1,
        "secrets": clean_secrets,
        "ocr_model": ocr_model,
        "tts_engines": copy.deepcopy(engines),
        "tts_languages": copy.deepcopy(languages),
        "tts_models": clean_models,
        "tts_voice_matrix": clean_matrix,
        **({"preferred_tts": copy.deepcopy(data["preferred_tts"])} if "preferred_tts" in data else {}),
    }


class RuntimeConfig:
    """已校验的运行配置。"""

    def __init__(self, data):
        self._data = validate_config(data)

    @property
    def raw(self):
        return copy.deepcopy(self._data)

    @property
    def secrets(self):
        return copy.deepcopy(self._data["secrets"])

    @property
    def ocr_model(self):
        return self._data["ocr_model"]

    @property
    def tts_engines(self):
        return copy.deepcopy(self._data["tts_engines"])

    @property
    def tts_languages(self):
        return copy.deepcopy(self._data["tts_languages"])

    @property
    def tts_models(self):
        return copy.deepcopy(self._data["tts_models"])

    @property
    def tts_voice_matrix(self):
        return copy.deepcopy(self._data["tts_voice_matrix"])

    @property
    def preferred_tts(self):
        if "preferred_tts" in self._data:
            return copy.deepcopy(self._data["preferred_tts"])
        # 旧配置保持原有初选行为，读取时不迁移、不写盘。
        matrix = self._data["tts_voice_matrix"]
        engine = "azure" if "azure" in matrix else self._data["tts_engines"][0]["id"]
        language = "sg-en" if "sg-en" in matrix[engine] else next(iter(matrix[engine]))
        return {"engine": engine, "language": language}


class ConfigStore:
    """单进程内串行保存，落盘成功后才替换当前配置。"""

    def __init__(self, path):
        self.path = Path(path)
        self._lock = threading.Lock()
        self.current = RuntimeConfig(json.loads(self.path.read_text(encoding="utf-8"))
                                     if self.path.exists() else build_default_config())

    def save(self, data):
        candidate = RuntimeConfig(data)
        with self._lock:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            name = None
            try:
                with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", dir=self.path.parent,
                                                 prefix=".config-", delete=False) as output:
                    name = output.name
                    json.dump(candidate.raw, output, ensure_ascii=False, indent=2)
                    output.flush()
                    os.fsync(output.fileno())
                os.replace(name, self.path)
                self.current = candidate
            finally:
                if name and os.path.exists(name):
                    os.unlink(name)


def session_key(data_dir):
    """签名密钥由本机生成并持久化，不从服务配置导出或环境变量读取。"""
    path = Path(data_dir) / ".session-key"
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    except FileExistsError:
        key = path.read_bytes()
        if len(key) != 32:
            raise ConfigError("会话签名密钥文件损坏")
        return key
    with os.fdopen(fd, "wb") as output:
        key = os.urandom(32)
        output.write(key)
    return key
