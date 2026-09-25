# 多语言听写练习应用

基于 Web 的听写练习工具。上传听写列表图片，自动 OCR 识别句子和重点单词，再按需生成对应语言和口音的音频。

## 功能特点

- 图片上传：支持拖放或选择文件上传听写列表图片
- OCR 识别：自动识别图片中的文本内容和加粗/圈出的重点单词
- TTS 生成：OCR 后手动选择引擎、语言、音色和速度，可重复生成音频
- 多语言支持：新加坡英语、英式英语、美式英语、中文普通话、法语
- 音频播放：点击播放整句或重点单词的发音
- 后台任务：OCR 与 TTS 在后台串行执行，提交接口立即返回，任务按会话互相隔离

## 技术实现

- **前端**: HTML + CSS + JavaScript + Bootstrap 5（DOM API 渲染，不注入服务端 HTML）
- **后端**: Flask (Python 3.11) + Gunicorn（`1 worker + gthread 4 线程`，不使用 preload）
- **任务模型**: 进程内串行 FIFO 队列（`tasks.py`），全局最多一个生成任务在执行，其余排队
- **OCR**: Google Gemini 多模态模型（`gemini-3-flash-preview`）
- **TTS**: Microsoft Azure 文本转语音 REST 接口 + Google TTS (gTTS) + Google Cloud Text-to-Speech

## 环境变量配置

| 变量 | 必需 | 说明 |
|------|------|------|
| `SECRET_KEY` | 是 | Flask 会话签名密钥；缺失时应用启动直接失败，没有回退默认值 |
| `GOOGLE_API_KEY` | OCR 必需 | Gemini API 密钥，兼容别名 `GEMINI_API_KEY` |
| `AZURE_API_KEY` | 使用 Azure 音色必需 | Azure Speech 密钥，兼容别名 `AZURE_SPEECH_KEY`、`SPEECH_KEY` |
| `AZURE_SPEECH_REGION` | 否 | Azure 区域，默认 `southeastasia`，兼容别名 `AZURE_SERVICE_REGION` |
| `DATA_DIR` | 否 | 运行数据根目录，默认 `.local-data`，任务文件在 `DATA_DIR/tasks/<task_id>/` |
| `PORT` | 否 | 服务端口，默认 5001 |
| `GOOGLE_CLOUD_TTS_CREDENTIALS_JSON` | 否 | Google Cloud TTS 服务账号 JSON 单行字符串（容器推荐） |
| `GOOGLE_APPLICATION_CREDENTIALS` | 否 | Google Cloud TTS 服务账号文件路径（本地可用） |

Google Cloud TTS JSON 兼容别名 `GOOGLE_CREDENTIALS_JSON`、`GOOGLE_SERVICE_ACCOUNT_JSON`、`GCP_SERVICE_ACCOUNT_JSON`；文件路径兼容 `GOOGLE_CLOUD_TTS_CREDENTIALS_FILE`、`GOOGLE_SERVICE_ACCOUNT_FILE`、`GCP_SERVICE_ACCOUNT_FILE`。

## 本地运行

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
export SECRET_KEY="某个随机长字符串"      # 必需
export GOOGLE_API_KEY="您的Gemini API密钥"  # OCR 需要
python app.py                            # 开发入口，默认 http://localhost:5001
# 或生产方式启动（带配置与目录权限检查）：
./startup.sh
```

## 容器运行（NAS 迁移版）

```bash
export SECRET_KEY="某个随机长字符串"
docker compose -f compose.nas.yml config --quiet   # 缺 SECRET_KEY 时直接失败
docker compose -f compose.nas.yml up -d
docker compose -f compose.nas.yml ps               # 仅应显示 127.0.0.1:15901->5001
```

- 镜像为 `linux/amd64`，单服务 + 独立数据卷；健康检查用 Python 标准库请求 `/health`。
- 端口只绑定回环地址 `127.0.0.1:15901`；NAS 上的实际端口与 LAN 绑定在部署阶段另行确认。
- 任务状态与音频只在内存和容器卷中保留：进程重启后旧任务一律失效（返回 404，提示重新提交），启动时会清理遗留任务目录。
- 容器验证详情与后续 NAS 部署边界见 [NAS 迁移计划](docs/NAS_MIGRATION.md)。

## 使用方法

1. 访问首页建立会话（签名 cookie，7 天生存期）
2. 上传听写列表图片（不做本地格式/像素/大小校验，内容由提供方判定；iPhone 相册原图可直接上传）
3. OCR 完成后选择引擎、语言、速度和音色，点击「生成TTS」
4. 排队中显示「排队中」；TTS 生成时显示进度；完成后播放句子和单词音频
5. 同一 OCR 结果可重复选择不同音色生成；任务过多时会明确提示未受理

## 使用限制

- 上传图片不做本地校验（格式/像素/大小不限，恢复直接上传；无效内容由提供方判定并明确报错）
- 单次 TTS 最多 50 条非标题句子；单句 500 字符；每句最多 20 个重点词、每词 100 字符；合成总量 10000 字符
- 单任务从实际开始执行起预算 10 分钟（不含排队）；超时不再尝试剩余项，已生成部分可用并逐条提示
- 任务记录最多 100 条（含排队/执行/完成）；完成任务保留 24 小时后自动清理
- 免费接口可能出现使用频率限制，图片处理失败请等待 3 分钟后重试
- 不同 TTS 引擎可用的语言不同，页面会自动隐藏当前引擎不支持的语言；无效组合会被拒绝而不是回退默认音色

## 测试

```bash
pip install -r requirements.txt -r requirements-dev.txt
python -m pytest tests/ -q
```

测试默认用假实现替换所有外部 API，不消耗额度。

## Azure App Service

NAS 迁移进行中，项目改造计划见 [NAS 迁移计划](docs/NAS_MIGRATION.md)。迁移期间 GitHub Actions 仅执行构建/测试，不再部署 Azure；现有 Azure 网页保持已部署版本，不随 `main` 推送更新，停用时另行确认。

## 目录结构

```text
app.py              # Flask 应用：路由、校验、OCR/TTS 执行
tasks.py            # 进程内任务模型：串行 FIFO、容量、owner 校验、过期清理
templates/
  index.html        # 前端单页面（轮询任务状态、DOM 渲染）
tests/              # 任务模型与接口测试（外部 API 为假实现）
requirements.txt    # 运行依赖
requirements-dev.txt# 测试依赖（pytest）
Dockerfile          # NAS 容器镜像（python:3.11-slim，非 root）
compose.nas.yml     # 单服务编排（amd64、回环端口绑定、标准库 healthcheck）
startup.sh          # 带检查的 Gunicorn 启动脚本
```
