# 多语言听写练习应用

基于 Web 的听写练习工具。上传听写列表图片，自动 OCR 识别句子和重点单词，再按需生成对应语言和口音的音频。

## 功能特点

- 图片上传：支持拖放或选择文件上传听写列表图片
- OCR 识别：自动识别图片中的文本内容和加粗/圈出的重点单词
- TTS 生成：OCR 后手动生成音频，可重复按不同引擎和速度生成
- 多语言支持：英语、中文、法语等多种语言和口音
- 音频播放：点击播放整句或重点单词的发音

## 技术实现

- **前端**: HTML + CSS + JavaScript + Bootstrap 5
- **后端**: Flask (Python 3.11)
- **OCR**: Google Gemini 3 Flash Preview 多模态模型
- **TTS**: Microsoft Azure Cognitive Services + Google TTS (gTTS)

## 依赖

```bash
pip install -r requirements.txt
```

核心依赖：Flask、google-genai、gtts、azure-cognitiveservices-speech

## 使用方法

1. 启动应用：`python app.py`
2. 浏览器访问：`http://localhost:5001`
3. 上传听写列表图片（建议分辨率不低于 1920x1080）
4. OCR 完成后选择 TTS 引擎和速度，点击“生成TTS”
5. 生成完成后播放句子和单词音频

### 使用限制

- Gemini API 有调用频率限制，失败时请稍后重试
- Microsoft TTS 如卡顿，可刷新页面切换到 Google TTS
- 音频生成单次请求限制 10 分钟以内内容

## 环境变量配置

```bash
export GOOGLE_API_KEY="您的Gemini API密钥"    # 必需，用于OCR
export AZURE_API_KEY="您的Azure API密钥"     # 使用Microsoft TTS时必需
```

## Azure App Service 部署

本应用已适配 Azure App Service，push 到 `main` 分支通过 GitHub Actions 自动部署。

### Azure 应用设置

在 Azure 门户的应用设置中配置：
- `GOOGLE_API_KEY`: Gemini API 密钥
- `AZURE_API_KEY`: Azure 语音服务密钥

### 手动部署

```bash
az login
az webapp create --resource-group myResourceGroup --plan myAppServicePlan \
  --name your-app-name --runtime "PYTHON:3.11" --deployment-local-git
git remote add azure <deployment-url>
git push azure main
```

## 目录结构

```text
app.py              # Flask 主应用
templates/
  index.html        # 前端单页面
uploads/            # 上传图片临时目录
audio/              # 生成的音频文件
requirements.txt    # Python 依赖
startup.sh          # Azure 启动脚本
web.config          # IIS 配置
```
