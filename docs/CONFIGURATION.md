# 配置文件

## 操作步骤

1. 从本机、家庭 LAN 或 Tailscale 直连应用，打开首页「配置」。外部来源不显示配置入口。
2. 按服务填写 Gemini API 密钥、OCR 模型和 Azure API 密钥；密钥可临时显示或隐藏。未使用的服务可不填密钥。微软区域不在页面展示，保存时原样保留配置文件中的区域参数供服务调用使用。
3. 使用「导入服务账号」选择 Google Cloud 原始授权 JSON。服务账号内容将嵌入同一个配置文件，不记录原文件路径。
4. 在「语言与音色」选择引擎，编辑语言名称及对应参数：gTTS 使用语言代码和口音域名，Azure 使用男女声音色名称，Google Cloud 使用语言代码和音色名称。可添加、移除语言，不需要编辑 JSON；也可使用「导入配置」载入完整配置。
5. 在目标语言旁勾选「设为首选」单选框，全局仅保留一组首选引擎和语言，例如「Azure · 新加坡英语」。点击「保存并应用」后当前页面和以后打开的页面均采用该首选项；速度、男女声仍由使用者选择，不写入首选配置。校验或写盘失败时显示错误，原配置保持不变。
6. 用「导出配置」下载编辑器当前内容（包括未保存修改），文件名为 `endictation-config.json`。导出文件包含所有密钥，应按凭据保管。

导入文件只填充编辑器，不自动保存。关闭窗口不保存修改，再次打开时读取服务器上的配置。

## 字段

| 字段 | 类型 | 含义 |
|---|---|---|
| `version` | 整数 | 当前为 `1` |
| `secrets.google_api_key` | 字符串或 null | Gemini OCR 密钥 |
| `secrets.azure_speech_key` | 字符串或 null | Azure Speech 密钥 |
| `secrets.azure_speech_region` | 字符串或 null | Azure 区域，例如 `southeastasia` |
| `secrets.google_cloud_tts_credentials_json` | 对象或 null | 原始 `service_account` JSON 对象，不能是路径或 JSON 字符串 |
| `ocr_model` | 字符串 | Gemini 模型名 |
| `tts_engines` | 对象数组 | 引擎的 `id`、`label` |
| `tts_languages` | 对象数组 | 语言的 `id`、`label` |
| `tts_models` | 对象 | 音色 ID 到提供方参数的映射 |
| `tts_voice_matrix` | 对象 | 引擎 ID → 语言 ID → `male` / `female` → 音色 ID |
| `preferred_tts` | 对象，可选 | `engine` 和 `language`，指定全局首选引擎、语言组合；不能包含速度或性别 |

删除首选语言前须先将另一种语言设为首选。旧配置缺少 `preferred_tts` 时，延续原有初选行为：优先 Azure、新加坡英语，缺少时使用首个可用引擎及语言；读取不会改写旧文件。显式填写的首选组合无效时拒绝保存，不静默替换。

每个引擎必须至少配置一种语言；矩阵只能引用已定义的语言、音色和 `male` / `female`。默认配置由仓库根目录 `config.default.json` 提供，可以直接编辑并提交 Git。该文件所有密钥和服务账号必须为空；区域、模型、语言和音色可保留。页面导出的是实际配置，可能包含密钥，不能当作默认文件直接提交。

音色字段：

- 所有音色：`label`、`type`。
- Azure：`type: "ms-tts"`，必填 `voice_name`；可选 `service_region` 覆盖全局区域。
- gTTS：`type: "gtts"`，必填 `lang`、`tld`。
- Google Cloud：`type: "google-cloud-tts"`，必填 `language_code`、`voice_name`。

速度、已选性别等用户偏好不属于配置。男/女映射只是候选音色定义，gTTS 的两个选项可指向同一模型，不代表它支持切换男女声。新增语言或音色不会改变提供方的实际能力、权限或额度；保存只验证结构、引用和服务账号私钥格式，不调用云服务验证可用性。

## 示例：增加 gTTS 西班牙语

在页面选择 gTTS，点击「添加语言」，填写名称「西班牙语」、语言代码 `es`、口音域名 `com`，保存即可。若编辑导出文件，则完成以下三处修改，保留其他字段：

1. 在 `tts_languages` 数组追加：

```json
{"id": "es", "label": "西班牙语"}
```

2. 在 `tts_models` 对象新增：

```json
"Spanish-gTTS": {"label": "西班牙语", "type": "gtts", "lang": "es", "tld": "com"}
```

3. 在 `tts_voice_matrix.gtts` 新增：

```json
"es": {"male": "Spanish-gTTS", "female": "Spanish-gTTS"}
```

保存后，该引擎的语言选项即时更新；其他引擎不会自动获得这项语言。

## 存储与接口

- 配置优先级：存在 `.local-data/config.json` 时整体使用该文件；不存在时读取 `config.default.json`，不合并、不自动覆盖已有实际配置。页面保存只写运行目录。更新默认文件后，如需应用到已有实例，可在内网导入并补回密钥后保存。
- 不读取原有密钥环境变量、别名、Google ADC、`DATA_DIR`、`PORT` 或 `SECRET_KEY`。开发端口使用 `python app.py --port 5002`；Gunicorn 使用 `--bind` 或启动脚本的端口参数。
- `GET /config` 读取完整配置，`POST /config` 校验并保存配置。两者首先校验连接来源，非受信直连返回 `403 config_forbidden`。GET 要求先访问首页建立会话；POST 另做同源检查。首页与配置响应都禁止缓存，避免不同访问来源复用配置入口。
- 保存使用权限 `0600` 的临时文件并原子替换，同一进程的保存操作串行执行。写盘失败不替换内存配置。
- OCR 和 TTS 均捕获提交时的配置，排队或执行中修改配置不影响已受理任务。撤销密钥不会取消旧任务。
- `.local-data/.session-key` 由系统生成并持久化，不属于导出配置。任务状态仍只在内存中，重启后失效。
- 单 Gunicorn worker 的约束不变。配置文件不支持跨 worker 热同步。

## 更新与配置保留

项目更新只替换代码、依赖和 `config.default.json`，不复制默认文件到 `.local-data/config.json`，不自动合并或重置实际配置。只要运行目录或数据卷保留，已有密钥、OCR 模型、语言与音色定义、会话签名密钥均继续使用。

### 本地 Git 更新

在同一工作目录更新代码后重启服务即可，保留 `.local-data/`。不要用 `git clean -fdx` 清理该目录，也不要使用带 `--delete` 的全目录同步覆盖运行目录。更换代码目录或机器时，需单独迁移 `.local-data/`；新的空目录不会自动找到旧目录中的配置。

### Docker / NAS 更新

1. 更新前在内网导出已保存配置并妥善保管。等待正在执行或排队的任务结束；重启不能续跑任务。
2. 保持原 Compose 项目名、服务名和 `endictation-data` 卷引用不变。Compose 的实际卷名带项目名前缀，更换目录、`-p` 参数或 Container Station Application 名称可能连接到另一个空卷，看起来像配置丢失。
3. 在已部署的同一项目目录更新代码后执行：

```bash
docker compose -f compose.nas.yml up -d --build endictation
docker compose -f compose.nas.yml ps
```

容器被替换，但原卷继续挂载到 `/app/.local-data`。不需要先 `down`，禁止为普通更新使用 `down -v`、删除卷或重新初始化运行目录。若采用预构建镜像，则先拉取目标版本，再在同一 Application 中重建服务并保留原卷。

4. 更新后从受信直连入口确认配置仍在，再验证服务功能；`/health` 正常不能证明密钥有效。

默认文件的新增语言或模型只影响首次安装，不自动改写已有实例。需要采用新版选项时，在现有配置上手动增补对应字段；不要为更新语言而直接导入空密钥默认文件覆盖实际配置。

若将来新代码不再支持已有配置格式，启动应明确失败并保留原文件，不允许用默认配置覆盖后继续运行。需要格式迁移时另行提供有备份的迁移步骤；回退镜像时同样保留原卷并确认格式兼容。

## 安全边界

沿用 EstateAnalysis 的家庭 LAN / Tailscale 直连边界，并将保护扩展到配置读取和导出（本项目返回完整密钥，而房产项目的公开读取不返回密钥）：

- 允许连接对端：`192.168.0.0/24`、`100.64.0.0/10`；另保留本机直连 `127.0.0.1`、`::1`。IPv4-mapped IPv6 按对应 IPv4 判定。
- 只读取 socket 对端 `request.remote_addr`，不使用 Host 或转发头扩大权限，不安装 `ProxyFix`。
- 带 `CF-Connecting-IP`、`Cf-Access-Jwt-Assertion`、`Forwarded`、`X-Forwarded-For`、`X-Real-IP`、`X-Forwarded-Host` 或 `X-Forwarded-Proto` 的请求均拒绝配置访问，包括空值；其他功能不因此受限。
- 不把全部私网或 Docker bridge 加入白名单。Docker Desktop / 代理若把连接改写成其他地址，配置访问会被拒绝；应使用可保留真实来源的直连入口，不能用信任转发头绕过。
- 尚未做内网用户身份鉴别：受信直连用户均可读取和修改密钥。若未来代理去掉上述标记并把所有请求转换为受信对端，应用将无法区分来源；公网部署前必须验证该边界，不可只依据按钮隐藏判断安全。

`config.default.json` 不含凭据，可提交 Git 并进入镜像；实际 `.local-data/config.json`、签名密钥与导出文件保持忽略。导出文件包含明文凭据，任意改名后仍须妥善保管，不得提交 Git。

本轮仅本地项目改造，不变更 Azure 已部署网页，不配置公网或 NAS 入口。
