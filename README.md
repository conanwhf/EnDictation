# 英文听写练习应用

这是一个基于Web的英文听写练习应用，允许用户上传英文听写列表的图片，自动识别其中带有标号的句子和加粗单词，并为每个句子和重点单词生成英式发音的音频。

## 功能特点

- 图片上传：支持拖放或选择文件上传英文听写列表图片
- OCR识别：自动识别图片中的文本内容和加粗单词
- 英式发音：为每个句子和重点单词生成自然清晰的英式发音
- 音频播放：提供简洁的播放界面，可选择性播放句子或单词

## 技术实现

- 前端：HTML, CSS, JavaScript, Bootstrap 5
- 后端：Flask (Python)
- OCR和TTS：通过Qwen-omni-turbo API实现

## 依赖库清单

本应用依赖以下Python库：

### 核心依赖
- Flask: Web应用框架
- requests: HTTP请求库
- openai: 通过API调用Qwen模型
- numpy: 数据处理

### 音频处理依赖
- soundfile: 音频文件处理

### 本地OCR依赖（可选）
- pytesseract: OCR文本识别
- Pillow (PIL): 图像处理

### 本地TTS依赖（可选）
- pyttsx3: 本地文本转语音
- gTTS: Google文本转语音

## 安装说明

1. 克隆或下载本项目到本地

2. 安装核心依赖库：
   ```
   pip install flask requests openai numpy soundfile
   ```

3. 安装可选的本地OCR依赖（如果不需要本地OCR功能可跳过）：
   ```
   pip install pytesseract pillow
   ```
   注意：pytesseract还需要安装Tesseract OCR引擎：
   - Windows: 下载并安装[Tesseract OCR](https://github.com/UB-Mannheim/tesseract/wiki)
   - macOS: `brew install tesseract`
   - Linux: `sudo apt install tesseract-ocr`

4. 安装可选的本地TTS依赖（如果不需要对应的TTS功能可跳过）：
   ```
   pip install pyttsx3 gtts
   pip install azure-cognitiveservices-speech
   ```

## 使用方法

1. 启动应用：`python app.py`
2. 在浏览器中访问：`http://localhost:5000`
3. 上传英文听写列表图片
4. 等待处理完成后，可以播放句子和单词的音频

## API配置

应用使用Qwen和Gemini模型API进行OCR识别和TTS语音生成。API密钥可以通过环境变量配置，也可以使用默认值。

### 环境变量配置

您可以通过设置以下环境变量来配置API密钥和其他参数：

```bash
# 阿里云Qwen API配置
export QWEN_API_KEY="您的阿里云API密钥"
# Google Gemini API配置
export GOOGLE_API_KEY="您的Gemini API密钥"
# Azure API配置
export AZURE_API_KEY="您的Azure API密钥"

```
如果不设置环境变量，应用将使用默认值。

## Azure App Service 部署说明

本应用已适配Azure App Service，可以按照以下步骤进行部署：

### 前提条件

1. 安装Azure CLI工具
2. 拥有Azure账号和订阅

### 部署步骤

1. **登录Azure**

   ```bash
   az login
   ```

2. **创建资源组**（如果尚未创建）

   ```bash
   az group create --name myResourceGroup --location eastasia
   ```

3. **创建App Service计划**

   ```bash
   az appservice plan create --name myAppServicePlan --resource-group myResourceGroup --sku B1 --is-linux
   ```

4. **创建Web应用**

   ```bash
   az webapp create --resource-group myResourceGroup --plan myAppServicePlan --name your-app-name --runtime "PYTHON|3.9" --deployment-local-git
   ```

5. **配置环境变量**

   在Azure门户中，导航到您的Web应用 -> 配置 -> 应用程序设置，添加以下环境变量：
   - `QWEN_API_KEY`: 您的阿里云API密钥
   - `GOOGLE_API_KEY`: 您的Google API密钥

6. **部署应用**

   ```bash
   git init
   git add .
   git commit -m "Initial commit"
   git remote add azure <deployment-url-from-previous-step>
   git push azure master
   ```

   或者，您也可以使用Azure门户中的部署中心，通过GitHub、BitBucket或Azure Repos进行持续部署。

7. **访问应用**

   部署完成后，您可以通过以下URL访问应用：
   ```
   https://your-app-name.azurewebsites.net
   ```

### 故障排除

- 如果遇到部署问题，请检查Azure门户中的应用日志
- 确保所有必要的依赖项都已在requirements.txt中列出
- 验证环境变量是否正确配置

## 目录结构

- `app.py`: 主应用程序
- `templates/index.html`: 前端界面
- `uploads/`: 上传的图片存储目录
- `audio/`: 生成的音频文件存储目录
- `demo.jpg`: 示例图片