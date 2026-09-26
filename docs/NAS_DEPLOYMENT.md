# EnDictation NAS 部署与更新

本文用于 QNAP Container Station 手动部署。范围为家庭 LAN，不配置域名、Cloudflare Tunnel、反向代理或路由器公网端口转发。现有 Azure 网页保持原版本，本次不停止、不更新它。

部署文件以仓库根目录 [compose.qnap.yml](../compose.qnap.yml) 为准。本文解释字段与 GUI 操作，不维护第二份 Compose。旧 [NAS 迁移计划](NAS_MIGRATION.md) 保留实施历史，不作为当前部署参数来源。

## 1. 部署配置

| 项目 | 配置 |
| --- | --- |
| NAS | QNAP TS-451D，Intel J4025，`linux/amd64` |
| NAS LAN 地址 | `192.168.0.128`，部署前确认仍是该地址 |
| Container Station Application | `endictation` |
| Container / Compose service | `endictation` |
| 镜像 | `docker.io/conanwhf1984/endictation-nas:latest` |
| 镜像架构 | `linux/amd64` |
| 宿主绑定 | `192.168.0.128:15901`，不绑定所有网卡 |
| 容器端口 | `5001/tcp` |
| LAN 网址 | `http://192.168.0.128:15901` |
| 健康检查 | `http://192.168.0.128:15901/health` |
| external named volume | `endictation-data`，首次部署前创建 |
| 容器内挂载路径 | `/app/.local-data`，读写 |
| 容器用户 | 镜像默认非 root 用户，UID `1000`；GUI 不覆盖 User |
| 启动命令 | 镜像默认 `python /opt/endictation/docker_start.py`；GUI 不覆盖 Command / Entrypoint |
| 自动重启策略 | `unless-stopped` |
| CPU / 内存限制 | 本阶段不设置额外硬限制，不复制房产项目的资源参数 |
| 日志 | `json-file`，每个文件最多 `10m`，保留 `3` 个文件 |
| Git 仓库与分支 | `https://github.com/conanwhf/EnDictation.git`，`main` |
| 更新时机 | 每次容器启动时拉取；运行中不定时更新 |
| 环境变量 | 不需额外填写；密钥、模型、语言在网页配置，不设置 `SECRET_KEY` / `DATA_DIR` / `PORT` |

`15901` 是本项目部署选用端口。部署前在 Container Station 与 NAS 服务配置中确认无占用；不能把一次连接失败当作端口空闲的证明。若 NAS IP 或端口不同，先修改 `compose.qnap.yml` 中的宿主绑定，再使用相应网址。

此 Compose 不绑定 NAS 的 Tailscale 网卡，不承诺 `http://hy-nas...:15901` 可直达；先完成 LAN 验收。应用允许 Tailscale 来源不等于宿主机已经开放对应监听。以后增加入口时另行验证，不通过信任转发头放宽配置权限。

## 2. 账号与密钥

复用现有 GitHub 与 Docker Hub 账号，不注册新账号。

| 内容 | 填写位置 | 是否进入 Git / 镜像 |
| --- | --- | --- |
| Docker Hub 发布令牌 | EnDictation 仓库的 Actions secret `DOCKERHUB` | 否，仅发布工作流使用 |
| Git 拉取凭据 | 不填写，EnDictation 是公开仓库 | 不需要 |
| NAS 拉取镜像凭据 | 公开镜像不需要；遇 Docker Hub 限流时可在 Container Station Registry 使用现有账号登录 | 不放入应用 Compose |
| Gemini API 密钥 | 网页「配置」 | 否，保存在数据卷 |
| Azure Speech API 密钥 | 网页「配置」 | 否，保存在数据卷 |
| Google Cloud 服务账号 JSON | 网页「配置 → 导入服务账号」 | 否，嵌入数据卷中的配置文件 |

`config.default.json` 是完整的无密钥模板，含 OCR 模型、引擎、语言、音色及首选项。首次启动可打开网页，但未配置的云服务会明确报错。微软区域不在表单展示，底层默认 `southeastasia`；沿用该区域资源，或导入含正确区域的配置。不要把区域参数当作不再需要。

本地实际配置不会随 Git 推送或镜像发布到 NAS。迁移已有配置：在当前本地网页「配置」中导出，再在 NAS 网页导入并保存。导出文件含明文密钥，应私下保管，不放 Git、共享下载目录或文档附件。

## 3. 首次镜像发布

此节在 GitHub 操作，不需要 NAS 终端。镜像发布与 NAS 部署是两个动作，发布镜像不会自动修改 NAS。

1. 确认目标代码已推送到 `main`，`Build and test EnDictation` 成功；不运行旧 Azure 部署工作流。
2. 在 GitHub 仓库 **Settings → Secrets and variables → Actions** 确认存在 `DOCKERHUB`，使用现有 Docker Hub 发布令牌。不要把令牌写入 YAML 或本仓库文档。
3. 打开 **Actions → Publish NAS image → Run workflow**，选择 `main`。该工作流只接受 `main`，先运行测试，再构建并推送 `linux/amd64` 镜像。
4. 等待整个 workflow 成功，核对目标提交与 `conanwhf1984/endictation-nas:latest` 的发布结果。失败时不继续首次部署。
5. 在 Docker Hub 确认仓库是 Public；匿名可拉取后，再执行下一节。

工作流为 [.github/workflows/nas-image.yml](../.github/workflows/nas-image.yml)，仅手动触发。普通 `main` 推送只运行 CI，不重复发布镜像；Azure 部署步骤已经移除。发布过程中不要同时推送待验证代码，因为容器启动时会读取届时的 `main`。

## 4. Container Station 首次部署

### 4.1 部署前确认

- NAS 地址仍为 `192.168.0.128`，TCP `15901` 没有被其他服务占用。
- NAS 能访问 GitHub、Docker Hub 及所需的 Gemini / Azure / Google 语音服务。
- NAS 防火墙允许家庭 LAN 访问 TCP `15901`，路由器未转发此端口，不创建公网入口。
- 镜像发布已经成功；已在本机导出需要迁移的配置，或准备好现有服务密钥。
- 这次新建的是 EnDictation，不修改房产项目的 Application、volume、端口或镜像。

### 4.2 创建数据卷

1. 打开 **Container Station → Volumes → Create**。
2. 名称填写 `endictation-data`，使用默认 local volume。
3. 确认卷已存在。若此名称已有数据，先辨认归属，不删除、不清空、不换成一个新空卷。

空 Docker volume 首次挂载时会复制镜像目标目录的已有内容；镜像已把该目录交给 UID `1000`。使用原生 named volume，不改成 NAS 共享目录 bind mount，不启用 `nocopy`。已有卷不可写时查看权限错误并保留原数据，不用删除卷或改为 root 运行来掩盖问题。

### 4.3 拉取镜像

1. 打开 **Images → Pull**，镜像填写 `docker.io/conanwhf1984/endictation-nas:latest`。
2. 等待下载完成，确认架构为 `amd64`。
3. 记录镜像 ID 或 digest，供重建后核对。标签仍叫 `latest` 不代表已更新本地镜像。

### 4.4 创建 Application

1. 打开 **Applications → Create**，名称填写 `endictation`。
2. 从本次已推送的仓库打开 [compose.qnap.yml](../compose.qnap.yml)，将完整内容粘贴到 YAML 编辑器。
3. 不添加 `build`、额外环境变量、特权模式、Docker socket、外部服务或自定义启动命令。
4. 点击 **Validate**，确认镜像、端口和 external volume 均符合第 1 节。
5. 创建 Application。预览应复用 `endictation-data`；如果提示创建其他数据卷，先停止并核对 YAML。

不同 Container Station 版本可能显示 Create Application / 创建应用程序等名称。操作目标始终是 Application 的 Compose，不是单独用容器向导创建一套不同配置。

### 4.5 查看首次启动

在容器 **Logs** 中查看：

```text
Source ready: <当前 Git 提交>
Starting gunicorn
Listening at: http://0.0.0.0:5001
Booting worker
```

Gunicorn 在容器内监听 `0.0.0.0` 是正常的；宿主机发布地址仍必须是 `192.168.0.128:15901`。应用为单 worker、4 个 HTTP 线程，OCR / TTS 仍串行排队，不自行增大 worker 数。

拉取失败时可能出现 `Source update skipped` 与 `Starting code bundled in the image` / `Starting cached source`。这表示回退启动，不是更新成功。每条 Git 命令最多等待 60 秒，健康检查启动宽限为 90 秒；这不是云服务任务超时。Docker 的 unhealthy 状态本身不会触发自动重启。

## 5. 首次配置与验收

1. 从家庭 LAN 的浏览器打开 `http://192.168.0.128:15901`，确认显示「配置」按钮。
2. 导入本机导出的完整配置，或填写 Gemini / Azure 密钥并导入 Google Cloud 服务账号文件。检查 OCR 模型、语言、音色与「设为首选」，点击「保存并应用」。速度和男女声仍由使用者选择，不写入首选项。
3. 上传一张真实听写图片，确认 OCR 完成、重点词显示正确；不要只凭 `/health` 成功判断云服务可用。
4. 用日常使用的语音引擎生成少量音频并播放，确认整句、重点词与语速符合预期。云服务是否收费由原账号额度与账单规则决定，本项目不承诺无限免费。
5. 等待任务结束，在内网再次导出已保存配置并保管。执行一次 Restart，确认服务恢复、设置仍在。旧页面中的任务和音频失效属于预期，重新上传图片验收。

| 验收项 | 通过条件 |
| --- | --- |
| 镜像 | 架构为 amd64，容器使用本次拉取的镜像 |
| 端口 | 宿主只发布 `192.168.0.128:15901`，没有公网转发 |
| 数据 | `endictation-data` 挂载 `/app/.local-data`，容器可写 |
| 来源检查 | LAN 直连显示配置按钮且可保存；不通过代理绕过 |
| 进程 | `/health` 返回 `{"status":"ok"}` |
| 代码 | Logs 中 `Source ready` 为预期提交；回退日志单独记录 |
| 云服务 | 真实 OCR、生成、播放通过，不仅是 HTTP 200 |
| 持久化 | Restart 后密钥、模型、语言和首选项保持不变 |

配置仅信任连接对端：本机、`192.168.0.0/24`、Tailscale `100.64.0.0/10`；带 Cloudflare 或转发头的请求拒绝。尚未鉴别内网用户身份，受信直连用户可以读取完整密钥，因此仅供受信家庭网络使用。HTTP 无传输加密，不在公共 Wi-Fi 配置密钥。

若首页可用但配置按钮缺失或 `/config` 返回 `403`，先确认浏览器直连 LAN 地址。QNAP 网络若把来源改写为 Docker bridge 地址，本阶段应停止配置验收并检查实际拓扑；不要加入全部 Docker 网段、信任 `X-Forwarded-For` 或取消服务端检查。

## 6. 日常更新

| 改动 | 操作 |
| --- | --- |
| 普通 Python、页面、默认语言模板 | 开发端测试并推送 `main`，CI 成功后在 NAS 点击 Restart |
| `requirements.txt`、`Dockerfile`、`docker_start.py` | 发布新镜像，然后 Images Pull，再 Update / Recreate Application，保留原卷 |
| Compose、NAS IP、端口或卷引用 | 更新 `compose.qnap.yml`，再 Update Application |
| 密钥、OCR 模型、语言、首选项 | 内网页面保存即可，不需重启 |

**普通代码更新：** 先等正在执行和排队的任务结束，再 Restart；检查 `Source ready` 和功能。宿主机重启、进程退出后的自动重启也会检查代码，因此 `main` 只放已经验证的版本。没有定时后台更新或网页更新按钮。

**镜像更新：** 等 `Publish NAS image` 成功，Images 中 Pull `latest`，然后在 Application 中选择 Update / Recreate，使用当前 `compose.qnap.yml`。确认仍挂载原 `endictation-data`，取消任何删除关联卷的选项，再确认容器使用新的 image ID。只点击 Restart 不会换用后来拉取的新镜像。

首次使用启动拉取功能必须安装新镜像，旧镜像无法通过 Restart 自行获得新启动程序。更新默认模板不会覆盖已有私有配置，新增语言需在设置里手动增补。具体实现与失败行为见 [更新与配置保留](CONFIGURATION.md#更新与配置保留)。

## 7. 数据保留与恢复

| 数据卷内路径 | 内容 |
| --- | --- |
| `config.json` | 实际密钥、模型、语言、音色与首选项；保存权限为 `0600` |
| `.session-key` | 会话签名密钥，不在配置导出中 |
| `source/` | Git 代码缓存；其 `.local-data` 是指向父数据目录的链接 |
| `tasks/` | 临时图片和音频；进程启动时清理，任务记录不跨重启保存 |

恢复应用时先停止原容器并保留日志，在卷列表确认实际卷名，再用相同 external volume 重建。不要删除 `endictation-data`、执行 `down -v`、使用空卷替代或同时启动两个实例共享该卷。

日常备份优先使用页面导出配置。整卷备份应在应用停止后使用 NAS 支持的卷备份或备份工具，必须保留符号链接、不跟随 `source/.local-data` 递归遍历。导出配置不包括签名密钥；整卷备份才包含它。若数据卷确实丢失，新建卷后导入备份配置可以恢复服务设置，旧任务和浏览器会话不能恢复。

普通错误版本的恢复在开发端通过 `git revert` 生成正常的新提交，测试推送后再 Restart。不要在 NAS 强制 reset 代码，也不要只换旧镜像就宣称回滚完成：启动程序仍会拉取 `main`，缓存代码和配置还可能比镜像新。

## 8. 故障处理

| 现象 / 日志 | 检查与处理 |
| --- | --- |
| Pull 提示不存在或无权限 | 确认 workflow 发布成功、镜像名和 Public 状态；限流时用原 Docker Hub 账号登录 Registry |
| `external volume ... not found` | 首次部署先创建 `endictation-data`；恢复时核对实际卷名，不创建空卷冒充原卷 |
| 端口绑定失败 | 核对 NAS IP 和 `15901` 占用；修改 Compose 后 Update Application |
| 页面打不开 | 检查 Application 状态、端口、防火墙与 Logs；不用公网域名作为 LAN 验收入口 |
| 任务目录或配置不可写 | 保留原卷，确认是否误用了共享目录、只读挂载或不同 UID；不要删除数据解决权限问题 |
| `Source update skipped` | 看紧接着的 Git 错误；检查 GitHub DNS / HTTPS、网络和是否有本地修改 |
| `Image update required` | 远端依赖或启动文件与镜像不一致，完成镜像发布、Pull、Update Application |
| `Starting cached source` / `Starting code bundled in the image` | 服务可启动，但未应用远端更新；记录原因，不把 health 成功当作更新成功 |
| Git 分叉 / 本地修改 | 保存日志与实际缓存状态，不自动覆盖；按开发端修复流程处理 |
| 配置 `403` 或按钮不显示 | 检查来源网络和转发头，按第 5 节处理，不放宽信任范围 |
| OCR / TTS 失败 | 检查所用服务密钥、OCR 模型、区域、额度和出口网络；单次外部失败不等于 Docker 损坏 |
| Restart 后任务 404 | 预期行为，任务不跨重启恢复，重新上传生成 |
| 反复退出 / 启动异常 | Stop Application 并保留 Logs，检查配置格式与代码；不要连续重启或清卷 |

## 9. NAS 验收记录

本地 amd64 容器验证不等于 QNAP 实机验收。本节由部署者在完成操作后记录；部署前保持「未部署」，不把镜像发布成功写成 NAS 已上线。

| 项目 | 实际结果 |
| --- | --- |
| NAS 部署状态 | 未部署，由 Conan 手动执行 |
| 部署时间 / Container Station 版本 | 待实机填写 |
| 镜像 ID / 启动日志 Git 提交 | 待实机填写 |
| 实际 LAN 网址 / volume | 待实机填写 |
| health / 配置入口与保存 | 待实机验证 |
| 真实 OCR / TTS / 播放 | 待实机验证 |
| Restart 后配置保留 | 待实机验证 |

GUI 操作参考 [QNAP Container Station 3 官方说明](https://www.qnap.com/en/how-to/tutorial/article/how-to-use-container-station-3)；数据卷行为参考 [Docker Volumes](https://docs.docker.com/engine/storage/volumes/)。具体应用参数以本仓库 Compose 和代码为准。
