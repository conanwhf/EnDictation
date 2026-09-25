# NAS 迁移计划：项目改造阶段

> 状态：待实施。本文记录计划，不表示代码已修改或 NAS 已部署。
> 本阶段仅准备项目代码、测试和容器运行方式；不配置公网入口，不操作 NAS 生产服务。

## 1. 目标与范围

将现有 Flask 应用改造成可在 QNAP Container Station 运行的独立容器，同时解决长期运行、任务隔离和长请求问题。保留现有上传 OCR、选择引擎和音色、重复生成 TTS、播放句子和重点词的使用流程。

本阶段完成标准：

- 使用 Python 3.11、Linux amd64 容器和生产 WSGI 服务启动应用。
- OCR 和 TTS 在后台执行，HTTP 提交请求不等待外部 API 完成。
- 不同浏览器会话之间的进度、图片和音频互不覆盖。
- 请求量、输入大小和临时文件保留时间有明确上限。
- 常见错误可见、可复现，不用空音频或默认音色掩盖失败。
- 完成自动化测试、本地容器验证和文档同步。

本阶段不做：

- 域名、DNS、Tunnel、Access、Google 登录或任何公网配置。
- NAS 应用创建、端口开放、镜像推送及现有服务变更。
- 新账号、新云服务、本地语音模型、数据库、Redis 或 Celery。
- EstateAnalysis 的 Git 自动拉取、数据更新和 Runtime 发布机制。
- 历史练习库、跨设备同步、任务断点续跑和应用内账号系统。

### 已确认的迁移取舍

- 迁移彻底完成前，Azure 网页继续运行当前已部署版本，不更新、不维护新功能。NAS 项目改造不再要求兼容 Azure 的新版本部署；不修改线上 Azure 配置，不停止或删除现有网站。
- 实施时先移除仓库工作流中的 Azure 部署 job 和对应部署权限，保留构建/测试；必须在首次推送迁移代码时一并生效，避免 main 推送覆盖旧站。实际迁移完成和 Azure 停用须另行确认，本阶段不执行。
- 生成任务串行执行，同时到达的有效请求按受理顺序排队，自动开始下一项。正常并发不返回繁忙错误，也不要求用户稍后重试。
- Azure 网页托管与 Azure Speech 是不同资源；继续复用现有 TTS 服务和凭据。

## 2. 现状与依据

代码依据：`app.py`、`templates/index.html`、`requirements.txt`、`startup.sh`、`.github/workflows/main_endictaion.yml`。现有功能说明见 [README](../README.md) 和 [架构文档](ARCHITECTURE.md)。

NAS 环境依据为用户提供的「EstateAnalysis NAS 部署留档」「EstateAnalysis 公网部署留档」「NAS 基本情况」。这些文档包含私人信息，不复制凭据到项目中。文档记录的设备为 QNAP TS-451D、Intel J4025、12 GB RAM；当前可用内存和端口仍需在实际部署阶段核实。

| 当前实现 | 具体影响 | 本阶段处理 |
| --- | --- | --- |
| `app.run()` 启动，缺少容器文件 | 尚无可验证的生产容器入口 | 增加 Gunicorn、Dockerfile、Compose |
| `/upload`、`/generate-tts` 同步等待结果 | 慢 API 和长列表使 HTTP 请求持续挂起 | 返回任务 ID，轮询任务结果 |
| 全局 `processing_status` | 不同请求覆盖进度；多个 worker 各有一份状态 | 单 worker，按任务存储状态 |
| `clean_audio_folder()` 删除全部 MP3 | 后一次操作删除前一次仍在播放的音频 | 每个任务独立目录，按时间清理 |
| 上传以原文件名保存 | 同名图片覆盖，长期运行累积图片 | 服务端生成任务目录，OCR 结束删除原图 |
| TTS 异常写空文件或跳过条目 | 部分失败无法清楚反馈 | 显式错误和部分成功提示 |
| 缺少业务输入上限 | 单次请求可消耗过多内存、磁盘和 API 额度 | 在请求入口校验 |
| 默认端口 `5001` | NAS 的 QTS HTTPS 已使用该宿主端口 | 容器内部保留，宿主端口独立映射 |

现有架构文档中「重启后文件丢失」的表述需要在实施时修正：内存状态随进程重启丢失，磁盘文件是否保留取决于容器重建和卷挂载，不能混为一谈。

## 3. 实施方案

### 3.1 进程与任务模型

采用 Gunicorn `1 worker + 4 threads`，另用标准库 `ThreadPoolExecutor(max_workers=1)` 执行 OCR 或 TTS。上述数量为本阶段初始设计值，不是性能测试结论。

- 全局最多一个生成任务在执行；其余有效任务进入进程内 FIFO 队列，提交后立即返回 `202`。
- 使用锁原子地完成容量检查、任务登记和执行器提交，保证受理顺序就是执行顺序；提交失败时撤销登记并清理上传文件。
- 一个任务结束或失败后自动执行下一项；排队不占用 HTTP 请求线程，不引入外部队列服务。
- 状态查询、首页和音频读取不占生成名额，生成期间仍可访问。
- 任务字典按 `task_id` 存储；状态更新和清理使用同一锁保护，返回状态快照。
- 执行器在 worker 进程中初始化，不开启 Gunicorn preload，不用多个 worker 共享内存状态。
- 不添加取消、自动重复生成或跨进程恢复功能。

这使两个人可以分别保留并播放自己的结果，也可以同时提交；后台依次生成，前端等待结果即可。后续只有出现实际并行执行需求时才扩展。

### 3.2 会话与文件隔离

使用 Flask 签名 session cookie 保存服务端生成的随机会话 ID；首次打开首页时建立会话。任务登记 owner，状态和音频读取都校验 owner。会话仅用于隔离，不是登录认证，也不能据此开放给不可信网络。

- NAS/容器运行要求环境变量 `SECRET_KEY`；不得把真实值写入镜像、Compose 或 Git。检查在应用进程启动入口执行，Gunicorn worker 与本地 `python app.py` 开发入口共用同一套检查，无任何回退默认值；生产环境静默回退到开发密钥是被禁止的行为，测试中使用固定假密钥不受此限。
- Cookie 使用 `HttpOnly`、`SameSite=Lax`。本阶段是本机或受信 LAN HTTP 验证，不启用仅 HTTPS 可发送的 Secure cookie；未来 HTTPS 阶段另行配置。
- 会话 cookie 设置明确生存期（初始设计 7 天），避免浏览器会话级 cookie 在关闭浏览器后丢失会话、导致已受理任务无法查询。排队与执行时长没有严格上限，7 天不构成任务时限承诺；页面重开后的任务恢复不在本阶段范围。
- 写接口检查同源请求，不开放跨域访问。拒绝不匹配的 Origin；没有 Origin 时按 Referer 校验，二者都没有则拒绝。测试客户端应明确发送同源头。
- 使用完整随机 UUID 作为任务 ID；客户端不再发送旧 `request_id`，后端也不把它作为授权或目录标识。
- 同一浏览器的多个标签页共享会话，允许读取该会话任务；独立浏览器或无痕窗口用于验证隔离。

容器中的目录安排：

```text
/app/                         镜像内代码，只读使用
/data/tasks/<task_id>/         当前任务文件
  input.<实际图片扩展名>       OCR 临时输入
  sentence_<index>.mp3         TTS 整句音频
  word_<index>_<index>.mp3     TTS 重点词音频
```

所有路径由服务端构造。上传文件名只作为显示信息，不参与目录定位。音频统一返回 `/audio/<task_id>/<filename>`，使用受限文件名和 Flask 安全目录发送方法，拒绝路径穿越。

### 3.3 保留时间与重启行为

初始设计为完成任务保留 24 小时、内存最多保留 100 个任务，包含排队中、执行中和已完成任务。沿用此总量上限限制队列，不另加队列容量配置。达到数量上限且无过期项可清理时返回 `429` 和错误类型 `capacity`，明确说明未受理；这属于容量异常，不是日常并发时的繁忙处理。不提前删除其他用户仍有效的结果。这些是项目保护参数，不代表云服务免费额度。

- OCR 原图在成功或失败后的 `finally` 中删除，不保留到 24 小时。
- 完成时间决定任务过期时间；清理时同时移除任务记录和音频目录。
- 启动时和每分钟检查一次清理；清理线程只在 worker 内启动。
- 排队中和正在执行的任务不参与过期清理；排队 OCR 的输入文件必须保留到任务执行结束。文件读取与清理竞争时明确返回 `404`，不得返回空 MP3。
- 任务状态、队列和 owner 只在内存中保存。进程重启后，包括排队项在内的旧任务失效，状态和音频接口返回 `404`，提示重新提交。
- 因为旧任务不能恢复，worker 启动时删除 `/data/tasks` 下遗留的任务目录；只清理应用专属目录，不遍历或删除其他挂载内容。
- 一个数据卷只允许一个应用实例使用，不允许新旧实例同时读写。

容器卷用于明确写入位置和控制权限，不承诺重启后恢复任务。页面刷新后的历史恢复也不在本阶段范围内。

### 3.4 API 变更

前后端作为同一版本一起更新，不保留旧同步响应的兼容分支。以下均为计划接口：

| 接口 | 输入 | 成功响应 | 失败响应 |
| --- | --- | --- | --- |
| `POST /upload` | multipart `file` | `202`，`task_id`、`status_url` | `400` 参数错误；`413` 过大；`415` 类型不支持；`429` 容量已满 |
| `POST /generate-tts` | `sentences`、`tts_engine`、`tts_language`、`voice_gender`、`speed` | `202`，`task_id`、`status_url` | `400` 参数或组合错误；`413` 过大；`429` 容量已满 |
| `GET /tasks/<task_id>` | 当前会话 cookie | `200`，状态、进度、结果或错误 | `404` 不存在、过期或不属于该会话 |
| `GET /audio/<task_id>/<filename>` | 当前会话 cookie | `200`，MP3 | `404` 无权限、过期或文件不存在 |
| `GET /health` | 无 | `200`，`{"status":"ok"}` | 应用不能服务时连接失败或非 200 |

缺少有效会话或同源检查失败的写请求返回 `403`。缺少必需的部署配置时启动失败，不等到首次用户请求才发现。

`POST /generate-tts` 只接收 `sentences`、`tts_engine`、`tts_language`、`voice_gender`、`speed`。引擎、语言、音色三元组必须命中现有 `tts_voice_matrix` 组合，否则返回 `400` 与错误类型 `invalid_tts_combo`，不回退。`speed` 可缺省并按所选模型默认处理；出现非有限数值或越界值时返回 `400`；不支持速度的引擎收到非默认速度时忽略该值。`resolve_tts_key()` 的 `tts_select` 优先返回、逐级回退和 `tts_profile` 旧映射一并删除，不保留任何静默回退路径。

任务响应包含 `task_id`、`stage`（`ocr` 或 `tts`）、`status`（`queued`、`running`、`succeeded`、`failed`）、`current`、`total`、`progress`、`message`，部分成功时另有 `warnings`；`progress` 为 `current` 占 `total` 的百分比，`total` 为 `0` 时固定为 `0`。任务先登记为 `queued`，后台开始时改为 `running`。排队阶段不显示生成进度，仅显示「排队中」，继续自动轮询，不提供「稍后重试」提示。OCR 阶段 `current`、`total`、`progress` 固定为 `0`，前端只显示识别中、不渲染进度条；TTS 阶段 `total` 为计划尝试的合成项数：非标题句子数与这些句子的重点词数之和；每次合成尝试结束——无论成功或失败——`current` 加一，`current` 与 `total` 必须同源计数；任务提前终止时保持实际进度，不伪装为全部完成。成功时 `result` 为句子数组，元素只含结构化数据：`text`、`original_text`、`is_title`、`title`、`bold_words` 与整句 `audio_path`；不再包含服务端拼接的 `html_text`。`bold_words` 统一为对象数组，每项含 `word`；`audio_path` 仅在该项音频生成成功时存在，OCR 结果与合成失败项省略该字段，整句 `audio_path` 同理。OCR 解析直接输出该对象数组，不再保留字符串数组与对象数组并存的兼容结构。前端用 DOM API 渲染：标题以 `textContent` 显示；正文按普通文本节点与 `<strong>` 节点拼装，各节点内容以 `textContent` 设置，`**...**` 加粗由前端格式化保留，不出现裸露星号；播放按钮按「固定模板 + 逐词转义」构建（现有兜底分支即参考实现），仅对含有效 `audio_path` 的句子和重点词创建，无音频的重点词保留加粗文字。失败时 `error` 为稳定错误类型（`invalid_input`、`invalid_tts_combo`、`too_large`、`capacity`、`ocr_failed`、`tts_failed`）和可读说明；`404` 响应统一提示任务不存在或已过期，不区分原因，不向任何请求方泄露任务是否曾经存在。轮询成功不代表任务成功：任务失败仍返回 HTTP `200`，由 `status=failed` 表达。

`result` 结构示例。OCR 成功（尚无任何音频，省略所有 `audio_path` 字段）：

```json
[
  {"text": "Unit 1", "original_text": "Unit 1", "is_title": true, "title": "Unit 1"},
  {"text": "She has a beautiful garden.", "original_text": "She has a **beautiful** garden.", "is_title": false, "title": "Unit 1", "bold_words": [{"word": "beautiful"}]}
]
```

TTS 部分成功（第二句整句与重点词都有音频；第三句整句失败，重点词仅保留文字）：

```json
[
  {"text": "Unit 1", "original_text": "Unit 1", "is_title": true, "title": "Unit 1"},
  {"text": "She has a beautiful garden.", "original_text": "She has a **beautiful** garden.", "is_title": false, "title": "Unit 1", "audio_path": "/audio/<task_id>/sentence_1.mp3", "bold_words": [{"word": "beautiful", "audio_path": "/audio/<task_id>/word_1_0.mp3"}]},
  {"text": "He runs quickly.", "original_text": "He runs **quickly**.", "is_title": false, "title": "Unit 1", "bold_words": [{"word": "quickly"}]}
]
```

`warnings` 为字符串数组，逐条给出失败句子或重点词的可读原因，例如 `["句子 3 音频生成失败：Azure TTS取消", "重点词 quickly 音频生成失败"]`。

TTS 的预期提供方错误与单条调用超时使用同一处理规则：底层调用已退出后，该项记为失败，`current` 加一，记录原因并继续剩余项，不自动重试。部分音频成功时返回 `succeeded`、可播放结果和 `warnings`；全部音频失败时返回 `failed` 与 `tts_failed`。不生成空文件，不返回失败项的 `audio_path`。

达到任务级 10 分钟预算时，不再发起后续合成调用，保留已完成结果和实际进度，未尝试项不增加 `current`。已有有效音频则返回 `succeeded + warnings`，逐条说明失败项和因预算耗尽而未尝试的句子或重点词；没有有效音频则返回 `failed` 与 `tts_failed`，说明预算耗尽。预算检查不强制中断正在进行的调用。OCR 只有一次外部调用，调用失败或超时即返回任务 `failed` 与 `ocr_failed`。

移除旧 `/status` 与 `/audio/<filename>` 单段路由，前端不再读取全局状态。

`/health` 只证明进程可以响应，不调用收费 API，也不宣称凭据有效。外部服务连通性通过单独的人工用例验证。

### 3.5 输入与外部调用

以下数值是待实施的初始设计值，写入代码常量并用测试覆盖，不增加一组可配置环境变量：

| 项目 | 初始限制 |
| --- | --- |
| 单次请求体 | 10 MiB，使用 Flask `MAX_CONTENT_LENGTH` |
| 图片格式 | JPEG、PNG、WebP；用 Pillow 验证实际内容，并传递实际 MIME 类型 |
| 图片像素数 | 2000 万像素；超限直接拒绝 |
| TTS 非标题句子数 | 最多 50 条 |
| 单句长度 | 最多 500 字符 |
| 每句重点词数量 | 最多 20 个，每个最多 100 字符 |
| 单次合成文本总量 | 句子和重点词累计最多 10000 字符，包含重复合成部分 |
| 单任务执行预算 | 从实际开始执行起计 10 分钟，不计排队时间；在连续合成调用之间检查 |

引擎、语言、音色和速度以现有能力矩阵为准。拒绝无效组合、非有限数值和越界速度，不回退到另一种付费引擎。标题、正文和重点词按实际输出位置转义；进度、错误和提示用 DOM `textContent` 显示。

保留现有 `GOOGLE_API_KEY`、`AZURE_API_KEY`、区域变量及 Google Cloud TTS 凭据别名。凭据只通过进程环境或只读文件挂载传入，日志不打印密钥、图片字节或凭据 JSON。

外部调用必须显式配置有限超时。本阶段接受连接/读取超时或提供方对应的调用超时，不将其描述为所有故障情况下的严格总时限。已核对的接口事实：google-genai 在 `Client` 的 HTTP 选项中设置超时；Google Cloud TTS 按调用传入超时；gTTS 2.5.4 构造器带 `timeout` 参数且默认无限等待，必须显式传入；Azure Speech SDK 的 `ResultFuture.get()` 本身没有超时参数，不能仅依靠该方法限制等待。

串行执行放大挂起后果：若底层调用持续不返回且无法停止，其后所有已受理任务一并停留在 `queued`；`/health` 仍可能正常响应，现有健康检查与重启策略不能恢复生成能力，需要人工重启。因此 Azure 合成路径必须先验证、后实施（步骤 1），由实测结论确定唯一实现：

- 候选一：文本转语音 REST 接口——与现有 Speech 资源同区域、同密钥，请求体即现有 SSML，`X-Microsoft-OutputFormat` 支持 MP3 输出。HTTP 客户端的连接与读取超时覆盖「无响应挂起」的典型场景，但不构成整个请求的严格总时限（服务器持续缓慢发送字节时仍可超过）。官方文档将 REST 定位为 SDK 不适用时的替代，采用即为有意识取舍；若采用，可从依赖移除 Azure Speech SDK，镜像不再安装其系统库。
- REST 满足步骤 1 的功能与受控故障测试要求即可选定，不再验证 SDK 候选。
- 仅当 REST 不满足要求时考虑保留 SDK，实测 `stop_speaking_async()` 能否终止所模拟的挂起合成并让 `get()` 返回；不得把调用了停止方法等同于底层已经停止。
- 两个候选均不能通过步骤 1 的验证时，停止迁移并报告，不增加看门狗或进程管理框架，也不在底层仍执行时提前宣告任务结束：迟到的调用仍可能写文件、更新状态，与已判定失败的任务冲突。缺少凭据或测试条件时记为未验证，不当作候选技术失败或自行绕过验证。

#### 步骤 1 验证结论（2026-09-24，本机 macOS arm64，Python 3.11.15）

- 选定实现：Azure 文本转语音 REST 接口 `POST https://{region}.tts.speech.microsoft.com/cognitiveservices/v1`，请求体沿用现有 SSML，请求头 `Ocp-Apim-Subscription-Key`、`Content-Type: application/ssml+xml`、`X-Microsoft-OutputFormat: audio-24khz-48kbitrate-mono-mp3`（格式名与官方 `SpeechSynthesisOutputFormat` 枚举 `Audio24Khz48KBitRateMonoMp3` 核对一致）。
- 客户端与超时参数：`requests`（实测 2.34.2），`timeout=(5, 30)`（连接 5 秒、读取 30 秒），未配置自动重试（requests 默认 `max_retries=0`）。连接超时取 5 秒的原因：macOS 上 TCP SYN 重试约 7.8 秒即放弃，10 秒的套接字超时永远不会被触发（实测 10 秒配置下 `ConnectTimeout` 提前至 7.83 秒），5 秒可确定性由套接字超时生效。
- 受控故障测试（`tools/verify_azure_rest.py`，假凭据仅发往 127.0.0.1 本地故障服务，6/6 通过）：连接拒绝 0.00s 明确 `ConnectionError`；塞满 backlog 的本地监听器触发 `ConnectTimeout` 实测 5.00s；本地挂起服务（接受连接、读完整请求、永不响应）触发 `ReadTimeout` 实测 30.00s，且服务端在等待窗口内观察到客户端 FIN，证明超时后连接释放；超时发生后同进程内下一次调用正常完成（串行恢复）；本地 401 应答触发 `requests.HTTPError` 明确传播，失败时不写音频文件。
- 故障注入方式与排查记录：挂起服务必须按 `Content-Length` 读完整个请求再挂起——HTTP 请求头部与包体常分两个 TCP 段到达，把迟到包体当作客户端后续动作会使服务端提前关闭连接、客户端收到 `RemoteDisconnected`，该缺陷已在脚本中修复。
- 未验证项：真实端点与密钥的合成、现有音色可合成性、默认/调整语速对比与 MP3 实际播放（当前环境缺少 `AZURE_API_KEY`，按计划记为未验证，不当作候选技术失败）。按用户指示采用 REST 路径继续实施，步骤 3 将从依赖中移除 Azure Speech SDK；凭据可用后执行 `python tools/verify_azure_rest.py --real` 补齐真实合成验证，若实测不通过再回到本节修订决策。
- 已接受的限制：读取超时不能限制「服务器持续缓慢发送字节」的总时长，本验证不构成严格总时限证明。SDK 候选（`stop_speaking_async()`）无需验证。

不能把 `future.result(timeout=...)` 当作底层调用已经停止；旧调用未结束时不得释放名额并启动下一次调用。调用超时与任务级 10 分钟预算是互补的两层，都不保证强制终止原生调用。连接/读取超时的剩余限制须随验证结果记录，不能因一次正常合成成功而宣称不存在挂起风险。

仅在任务执行边界兜底捕获未预期异常，将任务明确标为失败并记录异常栈；提供方失败不再通过 `create_empty_audio()`、宽泛捕获后继续或静默默认值隐藏。已生成的有效音频可作为部分结果保留。

### 3.6 其他初始设计值

以下数值与 §3.5 表同性质：写入代码常量并用测试覆盖，不增加可配置环境变量。

| 项目 | 初始设计值 |
| --- | --- |
| 状态轮询间隔 | 1 秒 |
| 轮询网络错误重试 | 首次失败后最多再重试 3 次，间隔依次 1s/2s/4s；仍失败则停止并提供对当前任务的重新查询入口 |
| 测试框架与命令 | `requirements-dev.txt` 仅含 `pytest` 及其依赖；运行 `python -m pytest tests/ -q` |
| `SECRET_KEY` 传入 Compose | 插值 `${SECRET_KEY:?SECRET_KEY required}`，变量缺失时 `compose config` 直接失败，不在文件中落任何默认值 |
| 图片扩展名映射 | Pillow 实际格式 `JPEG→.jpg`、`PNG→.png`、`WEBP→.webp`；其余一律返回 `415` |

## 4. 文件改动清单

| 文件 | 计划改动 |
| --- | --- |
| `app.py` | 路由改异步提交；会话、输入校验；传入任务输出目录和进度回调；清除全局状态、全目录删除、空音频降级和服务端 HTML 拼接；增加健康检查 |
| `tasks.py`（新增） | 集中管理任务字典、锁、串行 FIFO 执行、过期清理和生命周期；不抽象通用任务框架 |
| `templates/index.html` | 按任务 ID 轮询；从任务结果用 DOM API 渲染结构化数据（不用 `html_text`）；播放完整音频 URL；显示排队状态、部分失败和异常错误 |
| `requirements.txt` | 增加经 Python 3.11/amd64 构建验证的 Gunicorn、Pillow 版本；不顺手升级无关依赖 |
| `requirements-dev.txt`（新增） | 测试依赖，与运行依赖分开 |
| `Dockerfile`（新增） | Debian 系 Python 3.11 slim 镜像，安装 Azure Speech 需要的系统库（若按 §3.5 决策移除 SDK 则取消）、Python 依赖和应用，非 root 启动 |
| `.dockerignore`（新增） | 排除 Git、虚拟环境、运行数据、本地临时文件；凭据按具体文件名模式排除，不用 `*.json` 一刀切 |
| `compose.nas.yml`（新增） | 单服务、`linux/amd64`、独立数据卷、环境变量、端口映射和健康检查，不含公网组件 |
| `startup.sh` | 改为带失败检查的启动脚本，检查配置和目录权限，用 `exec gunicorn` 启动；保留 `PORT` 环境变量支持 |
| `.gitignore` | 忽略新的本地运行数据与秘密配置文件，仅增加必要规则 |
| `tests/`（新增） | 任务、接口、文件隔离、错误和输入限制测试；默认替换外部 API，不消耗额度 |
| `README.md`、`docs/ARCHITECTURE.md` | 实现完成后同步启动、接口、存储和限制；删除已失效的同步流程描述 |
| `.github/workflows/main_endictaion.yml` | 移除 Azure 部署 job 及其部署权限，保留并调整构建/测试；不新增 NAS 发布动作，不修改现有 Azure 网站 |

## 5. 容器细节

容器工作目录固定为 `/app`，运行数据根目录由 `DATA_DIR` 指定，Compose 使用 `/data`。开发环境默认使用项目内被忽略的运行目录。保留容器内部 `PORT=5001`。

本地验证只绑定回环地址：Compose 端口映射写为 `127.0.0.1:15901:5001`，不使用会绑定全部网卡的 `"15901:5001"` 简写；使用前先检查宿主端口是否空闲，验证时用 `docker compose ps` 确认 published 地址不含 `0.0.0.0`。未来 NAS 的实际端口及 LAN 绑定留到部署阶段确认，不把建议值写成已占用或已批准的事实。

- Gunicorn 使用 `--workers 1 --worker-class gthread --threads 4`，日志输出到 stdout/stderr。
- 镜像构建时创建非 root 用户和可写 `/data/tasks`；空 named volume 首次挂载和已有卷分别验证权限。
- 权限错误直接启动失败，输出路径和所需权限；不通过 `chmod 777` 或临时 root 运行掩盖。
- Azure Speech 系统依赖在选定基础镜像中实测导入和真实合成，不只检查 `pip install` 成功；若 §3.5 决策改用 REST 并移除 SDK，本条改为实测 REST 合成与 MP3 输出。
- 健康检查用 Python 标准库请求 `/health`，不额外安装 curl。Docker unhealthy 本身不会自动重启容器，不能把它描述成自动恢复。
- Compose 使用 `restart: unless-stopped`；进程退出后重启意味着未完成任务丢失，不表示任务续跑。
- 本阶段不添加 CPU/内存硬限制；先记录容器实测占用，再决定 NAS 部署值。

## 6. 实施顺序

### 步骤 0：停止后续 Azure 发布

先将现有工作流改为仅构建/测试，移除 Azure 登录、部署步骤和对应权限。迁移期间的第一次 main 推送必须同时包含此工作流修改与迁移代码，不单独逐步提交；在此之前不向 main 推送任何其他变更，包括仅修改文档的提交——旧工作流会因任何推送执行全量构建并部署重启 Azure 站，与「不更新」承诺冲突。推送前确认不存在仍会部署的旧 workflow run 或其他已配置的自动部署入口；若存在，先停止该发布执行，不停止网站。现有 Azure 网页保持已部署版本；不为新版本给旧站补 `SECRET_KEY`，也不承诺旧站具备新任务功能。

### 步骤 1：Azure 合成路径验证

在生产代码接入 Azure 新调用路径之前完成本验证，允许编写隔离验证脚本。先验证 REST；通过后直接选定，只有功能或故障测试不满足要求时才验证 SDK。真实调用消耗少量现有额度，不注册新账号，凭据仅通过进程环境注入。

1. 真实 API 测试：确认同区域端点、现有密钥和现有 SSML 可用；使用短文本检查项目现有 Azure 音色可合成，至少对默认音色比较正常与调整后的语速，并实际播放 MP3。记录所测音色、语速和未覆盖项，不能只以 HTTP 200 判为通过。
2. REST 受控故障测试：使用本地测试服务或故障代理模拟连接失败、连接建立超时及服务端迟迟不返回数据。使用拟采用的 HTTP 客户端及超时参数，确认调用返回明确错误、连接与文件资源释放，并能串行执行下一次调用。这部分使用假凭据，不把真实 Azure key 发给测试服务；仅替换高层函数直接抛异常不能证明网络超时生效。
3. 若需要验证 SDK：记录如何模拟连接或合成停滞；确认停止请求后原 `get()` 确实退出、文件写入停止，再验证后续调用可执行。若只能让包装线程超时而原调用仍未结束，判为未通过。
4. 记录采用的客户端/SDK 版本、超时参数的值与单位、自动重试设置、故障注入方式、实际耗时和资源释放结果。受控超时用例按选定超时值加测试容差判断；连接拒绝只证明错误传播，不替代连接建立超时测试。持续缓慢传输可能超过读取超时值的总时长，作为已接受的限制记录，不把本测试当作严格总时限证明。

将已验证路径的结论写回 §3.5，确定唯一实现及其超时参数，再进入步骤 2。REST 通过时把 SDK 候选记为无需验证；两条路径均未通过或必要验证无法完成时，停止并报告具体原因。不得在看门狗或进程管理方向上自行发挥。此步骤只验证调用层能恢复，任务队列的自动推进在步骤 2 和最终验收中另测。

### 步骤 2：任务模型与针对性测试

先建立任务创建、FIFO 排队、容量竞争、owner 校验、过期及失败后继续执行的测试，再实现 `tasks.py`。外部操作使用可控制的假实现。确认两个并发提交都被受理，但最多一个执行；前项结束或失败后，下一项自动开始。

### 步骤 3：后端接入与文件处理

改造上传、OCR、TTS 和音频服务；保留文本解析与音色矩阵，除新边界校验和错误传播外不重构无关逻辑。所有生成函数接收任务输出位置；移除因此成为孤儿的旧代码：`resolve_tts_key()` 全部回退与 `tts_select`/`tts_profile` 兼容、`sanitize_html()`、`create_empty_audio()`、`clean_audio_folder()`、`get_bold_word_texts()` 一类的双结构兼容辅助、`update_processing_status()` 一族、`UPLOAD_FOLDER`/`AUDIO_FOLDER` 常量及模块级目录创建；`safe_filename()` 按新代码实际引用决定去留。

### 步骤 4：前端接入

将 `uploadFile()` 和 `generateTts()` 改为提交后轮询 `status_url`；仅当任务成功时调用 `displayResults()`。提交请求只发送 §3.4 契约字段，不再发送 `request_id` 和 `tts_select`。`displayResults()` 改为从结构化数据用 DOM API 渲染：标题与正文节点内容均以 `textContent` 设置，正文按普通文本节点与 `<strong>` 节点拼装；重点词按钮按固定模板加逐词转义构建，且仅对含有效 `audio_path` 的项创建，其余保留加粗文字；`**加粗**` 前端格式化保留。移除 `html_text` 注入与 onclick 字符串改写路径。保持 `currentOcrData` 作为原始 OCR 数据，重复 TTS 不触发新 OCR。

轮询采用单条可停止的定时链，不叠加并发定时器。`queued` 与 `running` 都持续轮询，受理后禁用重复提交按钮直到完成。切换任务时忽略旧响应；临时网络错误有限重试，连续失败后停止并提供对当前任务重新查询的入口，不自动重复提交生成。`404` 提示任务已失效；仅容量已满时的 `429` 显示明确异常并保留 OCR 内容和选择，不能伪装为已经入队。正常排队不提示用户重试。刷新页面不承诺恢复已有任务。

### 步骤 5：镜像与本地验证

新增容器文件并修改启动脚本。以下命令在文件实现后执行，不表示现在已有这些文件：

```bash
docker compose -f compose.nas.yml config --quiet
docker compose -f compose.nas.yml build
docker compose -f compose.nas.yml up -d
docker compose -f compose.nas.yml ps
docker compose -f compose.nas.yml logs --tail=100
```

配置检查前只在当前进程设置凭据；不输出展开后的 Compose，以免泄露环境变量。若本机为 Apple Silicon，明确记录 amd64 模拟构建，不能把本机模拟运行当成 NAS 原生验收。

### 步骤 6：文档同步与集中验证

实现期间按修改模块运行针对性测试；完成后运行一次完整自动化测试和 `git diff --check`。使用 Chrome 验证时在沙盒外执行。只有实际完成的检查才能标为通过。

文档、代码和测试一致后，再按用户要求形成一个完整提交；不自动推送。首次推送前必须确认同一版本已移除 Azure 部署 job，防止新代码覆盖保留的旧站。本阶段不下线 Azure 服务，迁移彻底完成后另行确认停用。

## 7. 验收用例

| 用例 | 预期结果 |
| --- | --- |
| 提供方假实现阻塞，提交 OCR/TTS | 提供方未完成时已返回 `202`，首页与状态查询仍能响应 |
| 两个会话同时提交 | 两项均返回 `202`，最多一项执行，另一项为 `queued`；按受理顺序自动执行，无繁忙重试提示 |
| 首项失败、后项排队 | 首项明确失败，底层调用结束后后项自动开始，无须重新提交 |
| 排队期间执行清理或等待超过执行预算 | 输入文件不被清理；执行预算从实际开始时计算 |
| 两会话依次生成、使用同名图片 | B 不覆盖 A 文件；A 仍能播放自己的音频 |
| B 查询 A 的任务或音频 URL | 返回 `404`，不返回任务文字、进度或音频 |
| 同一 OCR 多次选择音色生成 | 不再次调用 OCR；新旧音频在保留期内独立存在 |
| 图片伪扩展名、损坏图片、像素超限、请求超限 | 返回对应 4xx，不调用 OCR API |
| 错误引擎组合、异常 JSON、超长句子和重点词 | 返回 4xx，不静默选择默认引擎 |
| OCR 失败、TTS 全失败、TTS 部分失败 | 状态分别正确；部分失败可见，无空 MP3 播放按钮 |
| OCR 调用失败或超时 | 底层调用退出后任务为 `failed`、错误为 `ocr_failed`，队列自动执行下一任务 |
| 单条 TTS 调用失败或超时 | 该项失败且 `current` 加一，记录原因并继续剩余项；全部失败则任务 `failed`，部分成功则 `succeeded + warnings` |
| TTS 执行预算耗尽 | 不再调用未尝试项，保持实际进度；有有效音频则部分成功并列出未完成项，无有效音频则 `failed` |
| 任务过期及满 100 条记录 | 只删除过期项；无可清理空间时明确拒绝新任务 |
| 正在生成且有排队任务时重启容器 | 执行中及排队中的旧任务均变为不存在，页面停止等待；不承诺恢复 |
| 首次推送迁移代码 | 工作流只构建/测试，不执行 Azure 登录或部署；旧站不被新代码覆盖 |
| 重启后卷有遗留文件 | 仅清理专属任务目录，新任务可正常写入 |
| 服务无 API 连通性或凭据失效 | `/health` 不消耗额度；真实任务明确报错 |
| 状态查询暂时断网、浏览器切换任务 | 有限重试，不无限转圈，不让旧结果覆盖新任务 |
| 新卷和已有卷、缺配置或目录不可写 | 有效配置可启动；无效配置立即失败且日志可定位 |
| 本地容器端口绑定 | `docker compose ps` 仅显示 `127.0.0.1:15901->5001`，不出现 `0.0.0.0` |
| OCR 结果渲染（尚无音频） | 重点词为加粗文字、无播放按钮；正文无裸露 `**` 星号 |
| TTS 成功后渲染 | 有音频的句子和重点词带播放按钮；正文无裸露 `**` 星号，无服务端 HTML 注入 |
| TTS 部分失败渲染 | 成功项带播放按钮，失败项保留加粗文字；`warnings` 逐条可见 |
| 受控故障触发所选调用路径超时 | 按步骤 1 确定的超时参数与测试容差返回错误并释放资源；任务按 OCR/TTS 规则处理，结束后队列继续下一项，不把包装线程超时当作底层调用退出 |

真实 API 检查另行记录：用现有账号测试一张图片，以及 Azure、gTTS、Google Cloud TTS 各一条短文本。缺少凭据、额度或网络时明确标为未验证，不为测试注册账号。自动化测试全部通过不能替代真实语音播放和 amd64 系统库验证。

## 8. 交付状态与后续边界

### 交付进度

- 2026-09-24 步骤 0 完成（本地提交，未推送）：`.github/workflows/main_endictaion.yml` 移除 `deploy` job、Azure OIDC 登录、`id-token: write` 权限及部署用 zip/artifact 步骤，仅保留构建/语法检查。README 中「push 到 main 自动部署 Azure」的描述已同步删除。现有 Azure 网页保持已部署版本；首次推送将包含本工作流修改与全部迁移代码。
- 2026-09-24 步骤 1 完成（本地提交，未推送）：Azure 合成路径验证。受控故障测试 6/6 通过（详见 §3.5 步骤 1 验证结论），选定 REST 实现（requests、`timeout=(5, 30)`、`audio-24khz-48kbitrate-mono-mp3`）。真实 Azure 合成因无凭据记为未验证；真实图片 OCR 同样待 `GOOGLE_API_KEY`。验证脚本保留在 `tools/verify_azure_rest.py`（`--real` 补做真实合成）。
- 2026-09-24 步骤 2 完成（本地提交，未推送）：先写针对性测试再实现 `tasks.py`（`tests/test_tasks.py` 16 项全部通过）。覆盖任务创建、FIFO 受理序执行、容量拒绝与过期恢复、owner 校验、过期清理（记录+目录）、排队/执行中任务不过期、输入文件排队保留/结束后删除、进度与 warnings、快照隔离、提交失败撤销、启动清目录、后台清理线程。CI 工作流加入 pytest 步骤并修正虚拟环境 PATH 传递。外部操作全部为可控制假实现，不触网。
- 2026-09-24 步骤 3 完成（本地提交，未推送）：`app.py` 后端接入。`/upload`、`/generate-tts` 改为 202 + `status_url`；会话（签名 cookie + sid、HttpOnly/Lax/7 天）与同源校验（Origin→Referer→拒绝）就位；输入上限（请求体 10 MiB、Pillow 实测格式与 2000 万像素、句子/重点词/总量、语速）全部在入口校验；任务目录按 `task_id` 隔离，OCR 原图 finally 删除；Azure 改用 REST（`timeout=(5,30)`、无自动重试），gTTS 显式 `timeout=30`，Google Cloud TTS `timeout=30, retry=None`，genai `timeout=120, attempts=1`；TTS 进度按尝试计数、预算在调用间检查、部分成功 `succeeded+warnings`；`resolve_tts_key` 回退、`sanitize_html`、`create_empty_audio`、`clean_audio_folder`、全局状态族、`UPLOAD_FOLDER`/`AUDIO_FOLDER`、`safe_filename` 及 Azure Speech SDK 依赖全部移除。`requirements.txt` 增加 `requests`/`Pillow`，移除 `azure-cognitiveservices-speech`。`tests/test_app.py` 新增 29 项接口测试，全套 45 项通过（外部 API 均为假实现）。
- 2026-09-24 步骤 4 完成（本地提交，未推送）：`templates/index.html` 前端接入。`uploadFile()`/`generateTts()` 改为提交 `202` 响应中的 `task_id` 并轮询 `status_url`，仅任务成功时渲染结果；提交只发送契约字段（`sentences`/`tts_engine`/`tts_language`/`voice_gender`/`speed`），不再发送 `request_id` 与 `tts_select`。轮询为单条可停止定时链（1 秒间隔），`queued` 仅显示「排队中」继续轮询、OCR 阶段不渲染进度条、TTS 按尝试进度渲染；网络错误按 1s/2s/4s 有限重试，仍失败停止并提供「重新查询当前任务」入口；`404` 提示任务已过期。渲染全部使用 DOM API：标题/正文 `textContent`，正文按文本节点与 `<strong>` 拼装，`**加粗**` 前端格式化保留；播放按钮按固定模板+逐词转义构建，仅含有效 `audio_path` 的句子/重点词创建，其余保留加粗文字；`warnings` 逐条可见；容量 `429` 明确提示未受理并保留 OCR 内容与选择。移除 `html_text` 注入、onclick 字符串改写、旧 `/status` 轮询与 `request_id`。`currentOcrData` 保留为原始 OCR 数据，重复 TTS 不触发新 OCR。模板渲染检查通过；浏览器端完整验证留待集中验证阶段。
- 2026-09-24 步骤 5 完成（本地提交，未推送）：容器文件与本地验证。新增 `Dockerfile`（`python:3.11-slim`、非 root 用户、预创建可写 `/data/tasks`、Gunicorn `1 worker + gthread 4` 且不 preload，REST 决策后无需 Azure Speech 系统库）、`.dockerignore`（凭据按具体文件名模式排除）、`compose.nas.yml`（`linux/amd64`、独立数据卷、`127.0.0.1:15901:5001` 回环绑定、标准库 healthcheck、`restart: unless-stopped`、`SECRET_KEY` 用 `${VAR:?}` 插值）、`startup.sh` 改为带配置与目录权限检查的 `exec gunicorn` 启动（保留 `PORT`）；`requirements.txt` 增加经构建验证的 `gunicorn==23.0.0`。本机验证记录（Apple Silicon，amd64 为模拟构建/运行，不作为 NAS 原生验收）：`compose config --quiet` 通过、缺 `SECRET_KEY` 时插值直接失败；镜像构建成功（79.5MB）；`compose ps` 仅显示 `127.0.0.1:15901->5001`，无 `0.0.0.0`；容器 healthy，`/health` 200、首页 200 且种 `HttpOnly; SameSite=Lax` 会话 cookie、无会话写请求 403 JSON；只读 `/data` 卷启动失败并输出路径与所需权限（退出码 3）；已有卷 `down`→`up` 后仍健康；空转占用约 99MiB 内存 / 0.12% CPU（供 NAS 部署值参考）。验证后已 `down -v` 清理。
- 2026-09-25 步骤 6 完成（本地提交，未推送）：文档同步与集中验证。README 与 docs/ARCHITECTURE.md 已按实现重写（任务模型、会话与隔离、API 契约、输入限制、外部调用与超时、容器运行、已知限制；并修正「重启后文件丢失」的混用表述：内存状态随进程重启丢失，磁盘文件取决于容器重建与卷挂载）。
  - 集中验证由 glm-5.3-flash 子代理独立执行：`python -m pytest tests/ -q` 45/45 通过；`py_compile` 三个源文件无语法错误；`git diff --check` 无空白问题。
  - 浏览器端到端验证（Chrome headless + agent-browser，假 OCR/假 TTS 拷贝真实 MP3，不消耗额度）：首轮发现真实回归——OCR 成功后 `currentOcrData` 未赋值，「生成TTS」必然提前返回；已修复（`handleTaskUpdate` 在 `stage==='ocr'` 成功时保存 `task.result`）。复验通过：上传→OCR 渲染（加粗、无裸露 `**`、无播放按钮）→生成TTS（实际发出 POST、202 后轮询）→整句播放按钮与重点词胶囊→真实播放（currentTime 前进、onended 移除元素、无错误提示）→无 warnings。
- **交付结论：项目改造完成，可进入 NAS 部署验证；不能写「NAS 已迁移完成」。** 未验证项：①真实 Azure 合成与语速对比（无 `AZURE_API_KEY`，`tools/verify_azure_rest.py --real` 待补）；②真实图片 OCR 与 Gemini 额度（无 `GOOGLE_API_KEY`）；③gTTS / Google Cloud TTS 真实短文本合成；④真实凭据下的浏览器播放（本轮播放用的是仓库历史 MP3）；⑤amd64 系统库与运行验证为本机 Apple Silicon 模拟，非 NAS 原生（Intel J4025）验收；⑥NAS 实机端口、卷权限与资源余量。首次推送前需确认：本版本已移除 Azure 部署 job（推送将一并携带），且不存在仍会部署的旧 workflow run；本阶段全程未推送。

### 2026-09-25 真实 API 验证（补充，本地提交）

**第二轮（补 Gemini 真实 OCR，用户提供测试密钥）：**
- 修复真实测试发现的单位缺陷：`genai HttpOptions.timeout` 单位为毫秒（SDK 源码 `get_timeout_in_seconds` 明确除以 1000），原 `timeout=120` 实际只有 0.12 秒，真实调用约 1 秒即 `read timeout`；改为 `OCR_TIMEOUT_MS = 120_000`（120 秒）。修复后 45 项自动化测试仍全部通过。
- 真实 Gemini OCR（`gemini-3-flash-preview`，用户图片转换副本 5099×3824）：`succeeded`，耗时 27.2s，识别 28 行（标题 "Ex(1) - Words with ea, ai, ay, ou, ow"，27 条非标题句子 + 32 个重点词，日期行与编号句解析正常）。
- 真实全链路（真实图片 → 真实 OCR → 真实 Azure TTS → 音频下载）：TTS `59/59` 项 100% `succeeded`、无 warnings，耗时 13.1s；整句与重点词 MP3 落盘 `tools/real_api_verification/real_chain_sentence_1.mp3`、`real_chain_word_1_0.mp3`（重点词 "29th September"），**待人工播放确认**。
- 至此原未验证项仅剩：Google Cloud TTS（无服务账号/ADC 凭据）；amd64 NAS 原生验收；NAS 实机端口/卷权限/资源余量；全部 MP3 的人工听感确认。

**第一轮（仅 Azure 凭据）：**

用真实凭据与用户提供图片 `IMG_9888.jpeg` 补做真实测试；凭据取自用户环境变量（仅 `AZURE_SPEECH_KEY` 与 `AZURE_SPEECH_REGION`，无 Gemini / Google Cloud 凭据；`GOOGLE_PLACES_API_KEY` 属 Places 服务，与 Gemini 无关，未挪用）。

**已验证：**
- Azure REST 真实合成（`tools/verify_azure_rest.py --real`，7/7 通过）：默认语速 -15% 与调整语速 0% 各一条，24192 / 22752 字节，耗时 0.80s / 0.30s；同轮复验受控故障 6 项仍全部通过。MP3 已落盘 `tools/azure_rest_verify_output/`，**待人工播放确认**。
- 应用链路真实 Azure TTS 端到端（`python app.py` + curl，真实密钥）：会话 → `/generate-tts` 202 → 排队/执行 → `4/4` 项 100% `succeeded`、无 warnings → 经 `/audio/<task_id>/` 下载整句与重点词 MP3，帧头 `FF F3` 有效。文件在 `tools/real_api_verification/`（`app_e2e_sentence_1.mp3`、`app_e2e_word_1_0.mp3`）。
- gTTS 真实合成：30720 字节，`gtts_sample.mp3`，**待人工播放确认**。
- 真实图片入口校验：iPhone 原图（MPO 容器、5712×4284=2447 万像素）按设计被拒——`415 实际为 MPO`（白名单外）；转基线 JPEG 并缩至 5099×3824（1949 万像素、2.5MB）后 `202` 受理。
- 无凭据失败路径：受理后的 OCR 任务明确 `failed` + `ocr_failed`（"No API key was provided..."），符合「真实任务明确报错」验收项。

**仍未验证：** 真实 Gemini OCR 与额度（无密钥）；Google Cloud TTS（无服务账号/ADC）；真实听写内容的 OCR→TTS 全链路（依赖前者）；MP3 实际听感（文件已生成，待人工播放）；amd64 NAS 原生验收；NAS 实机端口/卷权限/资源余量。

**产品观察（待用户决策）：** iPhone 相册原图常见为 MPO 多图容器且超过 2000 万像素，当前设计会直接拒绝（格式白名单与像素上限均为计划既定值）。本轮通过「转基线 JPEG + 缩放」完成测试；是否在服务端自动接受 MPO/超限缩图，属于新的产品决策，未擅自改动。

项目改造交付时应报告：实际修改文件、自动化结果、本地容器结果、真实 API/浏览器结果、未验证项。若只完成本地验证，应写「可进入 NAS 部署验证」，不能写「NAS 已迁移完成」。

后续 NAS 部署需单独确认端口、卷权限、资源余量和镜像获取方式；公网工作另起阶段。OCR 模型可用性、云服务免费额度和账号账单配置也需使用现有账号核实。本阶段不承诺无限免费，不更换模型来掩盖调用失败。
