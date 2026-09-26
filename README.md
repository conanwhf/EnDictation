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

## 配置文件

首页「配置」按钮以表单编辑各服务密钥和 OCR 模型，按引擎增删语言、设置音色，并支持导入和导出完整 JSON 文件。Google Cloud 服务账号可单独导入；微软区域不在页面展示，原有区域值保留供 API 调用。首次启动不需要密钥；缺少凭据时对应任务明确失败，不读取环境变量或 Google 默认凭据。

- 导入文件后点击「保存并应用」才会生效；导出的是编辑器当前内容，包括尚未保存的修改。
- Google Cloud 的「导入服务账号」接收原始服务账号 JSON，并将其嵌入完整配置，不依赖外部文件路径。
- 配置保存在项目根目录 `.local-data/config.json`，文件权限为 `0600`，不进入 Git 或镜像。保存后即时生效，重启后保留。
- `config.default.json` 是可提交 Git、可进入镜像的默认配置，密钥和服务账号均为空。没有本机配置时才加载它；页面保存不会修改默认文件。
- 速度和男/女声选择仅由终端用户在页面设置，不进入配置文件。音色定义中的男/女映射用于提供可选项，不代表用户选择或默认偏好。
- 每种语言旁的「设为首选」单选框可指定全局首选引擎及语言，保存后即时应用，并用于之后打开的页面；初始为 Azure · 新加坡英语。
- 已受理任务沿用提交时的配置，新任务使用新配置；保存不刷新页面、不清空 OCR 结果。
- 会话签名密钥自动生成到 `.local-data/.session-key`，不导入导出；保护参数（超时、任务容量、输入上限）仍为代码常量。

**配置仅允许本机、家庭 LAN `192.168.0.0/24` 或 Tailscale `100.64.0.0/10` 直连访问。** 外部页面隐藏配置入口，配置读取、导入和导出均受服务端保护；Cloudflare 或带转发头的请求一律拒绝。不鉴别内网用户身份，公网部署仍需单独验收代理拓扑。配置格式和新增 gTTS 语言示例见 [配置说明](docs/CONFIGURATION.md)。

## 本地运行

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python app.py                            # 仅监听 http://127.0.0.1:5001，端口用 --port 指定
# 或生产方式启动（带配置与目录权限检查）：
./startup.sh                             # 可传端口：./startup.sh 5002
```

## 容器运行（NAS 迁移版）

NAS 手动部署使用 [NAS 部署与更新文档](docs/NAS_DEPLOYMENT.md) 和 [compose.qnap.yml](compose.qnap.yml)：在 Container Station 拉取 `conanwhf1984/endictation-nas:latest`，创建 external volume，再粘贴 YAML。无需在 NAS 手动拉代码或构建。

以下 `compose.nas.yml` 仅供开发端本机构建与回环端口验证，不要直接粘贴到 QNAP：

```bash
docker compose -f compose.nas.yml config --quiet
docker compose -f compose.nas.yml up -d
docker compose -f compose.nas.yml ps               # 仅应显示 127.0.0.1:15901->5001
```

- 镜像为 `linux/amd64`，单服务 + 独立数据卷；健康检查用 Python 标准库请求 `/health`。
- 数据卷挂载到 `/app/.local-data`。从旧 `/data` 挂载更新时沿用同一个 named volume 即可；旧环境变量不会自动迁入配置文件，需在页面重新配置。
- 首次安装含新启动程序的镜像后，日常代码更新只需在 Container Station 点击 Restart：启动时自动拉取 GitHub `main`，不需要在 NAS 上手动执行 Git 命令，不添加网页更新按钮。代码须先在开发端测试并推送；本地开发启动不自动拉取。
- 拉取失败会明确记录日志并使用兼容的缓存代码或镜像内代码。依赖、Dockerfile 或容器启动程序变化时，手动运行 GitHub Actions 的 `Publish NAS image`，成功后在 Container Station 执行 Pull + Update Application；Restart 不更新镜像。
- 项目更新不会覆盖实际配置：保留原 Compose 项目和数据卷，不使用 `down -v`。新版默认配置只用于未保存配置的实例，具体流程与失败处理见 [更新与配置保留](docs/CONFIGURATION.md#更新与配置保留)。
- 端口只绑定回环地址 `127.0.0.1:15901`；NAS 上的实际端口与 LAN 绑定在部署阶段另行确认。
- 任务状态与音频只在内存和容器卷中保留：进程重启后旧任务一律失效（返回 404，提示重新提交），启动时会清理遗留任务目录。
- 容器验证详情与后续 NAS 部署边界见 [NAS 迁移计划](docs/NAS_MIGRATION.md)。

## 使用方法

1. 访问首页建立会话（签名 cookie，7 天生存期）
2. 上传听写列表图片（不做本地格式/像素/大小校验，内容由提供方判定；iPhone 相册原图可直接上传）

> 图片字节**不做任何本地处理**（不缩放、不转码、不重压缩、不解码），原样交给 OCR。
> 这是有意为之：听写识别依赖原图的准确细节——文字本身以及被**加粗/圈出**的重点词标记，
> 任何有损转换都可能损失这些细节。
3. OCR 完成后选择引擎、语言、速度和音色，点击「生成TTS」
4. 排队中显示「排队中」；TTS 生成时显示进度；完成后播放句子和单词音频
5. 同一 OCR 结果可重复选择不同音色生成；任务过多时会明确提示未受理

## 使用限制

- 上传图片不做本地校验，也不做任何图像处理（原样转发；OCR 需要原图准确细节，无效内容由提供方判定并明确报错）
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

测试默认用假实现替换所有外部 API，不消耗额度。前端轮询竞态测试通过 Node.js 执行，未安装 Node.js 时该项会明确跳过；也可单独运行 `node --test tests/frontend.test.cjs`。

## Azure App Service

NAS 迁移进行中，项目改造历史见 [NAS 迁移计划](docs/NAS_MIGRATION.md)，当前部署操作见 [NAS 部署与更新](docs/NAS_DEPLOYMENT.md)。普通 `main` 推送只运行构建/测试，镜像工作流单独手动触发，不再部署 Azure；现有 Azure 网页保持已部署版本，停用时另行确认。

## 目录结构

```text
app.py              # Flask 应用：路由、校验、OCR/TTS 执行
config.py           # 默认服务定义、配置校验与原子保存
config.default.json # 无密钥默认配置，可提交 Git
tasks.py            # 进程内任务模型：串行 FIFO、容量、owner 校验、过期清理
templates/
  index.html        # 前端单页面（轮询任务状态、DOM 渲染）
static/
  config-editor.js  # 独立配置表单，不阻断上传事件绑定
tests/              # 任务模型与接口测试（外部 API 为假实现）
requirements.txt    # 运行依赖
requirements-dev.txt# 测试依赖（pytest）
Dockerfile          # NAS 容器镜像（python:3.11-slim，非 root）
compose.nas.yml     # 单服务编排（amd64、回环端口绑定、标准库 healthcheck）
compose.qnap.yml    # QNAP GUI 部署（Docker Hub 镜像、LAN 绑定、external volume）
startup.sh          # 带检查的 Gunicorn 启动脚本
docker_start.py     # 容器启动时拉取代码，再启动 Gunicorn
```
