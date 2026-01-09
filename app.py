from flask import Flask, render_template, request, jsonify, send_file
import os
import re
import requests
import json
import base64
import argparse
import numpy as np
import soundfile as sf
import logging
from openai import OpenAI

# 配置日志记录
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# 设置Werkzeug日志级别为WARNING以隐藏访问日志
logging.getLogger('werkzeug').setLevel(logging.WARNING)

# 导入阿里云TTS相关库
try:
    from dashscope.audio.tts_v2 import SpeechSynthesizer
    import dashscope
    TTS_QWEN_AVAILABLE = True
except ImportError:
    logger.warning("dashscope库未安装，云端TTS功能不可用，请使用pip install dashscope安装")
    TTS_QWEN_AVAILABLE = False

# 导入Azure语音SDK
try:
    import azure.cognitiveservices.speech as speechsdk
    TTS_AZURE_AVAILABLE = True
except ImportError:
    logger.warning("azure-cognitiveservices-speech库未安装，Azure TTS功能不可用，请使用pip install azure-cognitiveservices-speech安装")
    TTS_AZURE_AVAILABLE = False

# 导入本地TTS相关库（调用Google TTS实现）
try:
    from gtts import gTTS
    TTS_GTTS_AVAILABLE = True
except ImportError:
    logger.warning("gtts库未安装，本地TTS功能不可用")
    TTS_GTTS_AVAILABLE = False

app = Flask(__name__)

# AI服务配置字典
ocr_ai_models = {
    "qwen-ocr": {
        "key": os.environ.get("QWEN_API_KEY", "demo"),
        "url": "https://dashscope.aliyuncs.com/compatible-mode/v1",
        "name": "qwen-vl-max-latest",
    },
    "gemini-ocr": {
        "key": os.environ.get("GOOGLE_API_KEY", "demo"),
        "url": "https://generativelanguage.googleapis.com/v1beta/openai/",
        "name": "gemini-2.5-flash",
    },
}

# TTS服务配置字典
tts_models = {
    "Qwen-man": {
        "type": "qwen-tts",
        "key": os.environ.get("QWEN_API_KEY", "demo"),
        "model": "cosyvoice-v1",
        "voice": "longxiang",
        "speed": "0.8",
    },
    "Qwen-woman": {
        "type": "qwen-tts",
        "key": os.environ.get("QWEN_API_KEY", "demo"),
        "model": "cosyvoice-v1",
        "voice": "longjing",
        "speed": "0.8",
    },
    "SG-man": {
        "type": "ms-tts",
        "speech_key": os.environ.get("AZURE_API_KEY", "demo"),
        "service_region": "southeastasia",
        "voice_name": "en-SG-WayneNeural",
        "speed": "-10%",
    },
    "SG-woman": {
        "type": "ms-tts",
        "speech_key": os.environ.get("AZURE_API_KEY", "demo"),
        "service_region": "southeastasia",
        "voice_name": "en-SG-LunaNeural",
        "speed": "-10%",
    },
    "UK-man": {
        "type": "ms-tts",
        "speech_key": os.environ.get("AZURE_API_KEY", "demo"),
        "service_region": "southeastasia",
        "voice_name": "en-GB-OllieMultilingualNeural",
        "speed": "-10%",
    },
    "UK-woman": {
        "type": "ms-tts",
        "speech_key": os.environ.get("AZURE_API_KEY", "demo"),
        "service_region": "southeastasia",
        "voice_name": "en-GB-LibbyNeural",
        "speed": "-10%",
    },
    "CH-man": {
        "type": "ms-tts",
        "speech_key": os.environ.get("AZURE_API_KEY", "demo"),
        "service_region": "southeastasia",
        "voice_name": "zh-CN-YunyangNeural",
        "speed": "-20%",
    },
    "UK-Google": {
        "type": "gtts",
        "lang": "en",
        "tld": "co.uk",
    },
    "US-Google": {
        "type": "gtts",
        "lang": "en",
        "tld": "com",
    },
    "French-Google": {
        "type": "gtts",
        "lang": "fr",
        "tld": "fr",
    },
    "Chinese-Google": {
        "type": "gtts",
        "lang": "zh",
        "tld": "com",
    },
}

# 模型配置
def get_selected_model(request, model_type):
    if model_type == 'ocr':
        selected = request.form.get('ocr-select', 'gemini-ocr')
        return ocr_ai_models[selected]
    elif model_type == 'tts':
        selected = request.form.get('tts-select', 'UK-Google')
        return tts_models[selected]

@app.route('/')
def index():
    return render_template('index.html', 
        ocr_options=ocr_ai_models.keys(), 
        tts_options=tts_models.keys()
    )

# OCR提示词
OCR_PROMPT = "请你将图片处理成markdown文本，根据句号、句点、数字标号将文本分割为句子并换行。如果句子中有被圈出、粗体、放大、与众不同的字体或颜色的文本，则把它们也用粗体标记。请仅输出markdown代码即可。"

# 确保上传和音频文件夹存在
UPLOAD_FOLDER = 'uploads'
AUDIO_FOLDER = 'audio'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(AUDIO_FOLDER, exist_ok=True)

def encode_image_to_base64(image_path):
    """将图片转换为base64编码"""
    try:
        with open(image_path, 'rb') as image_file:
            return base64.b64encode(image_file.read()).decode('utf-8')
    except Exception as e:
        logger.error(f"图片编码失败: {str(e)}")
        raise ValueError(f"无法读取或编码图片: {str(e)}")

def init_ocr_client(ocr_model):
    """初始化OCR客户端，处理代理设置"""
    # 临时保存当前环境变量中可能存在的代理设置
    http_proxy = os.environ.pop('HTTP_PROXY', None)
    https_proxy = os.environ.pop('HTTPS_PROXY', None)
    no_proxy = os.environ.pop('NO_PROXY', None)
    
    try:
        # 不使用代理初始化客户端
        client = OpenAI(
            api_key=ocr_model["key"],
            base_url=ocr_model["url"]
        )
        logger.info("成功初始化OCR客户端")
        return client
    except Exception as e:
        logger.error(f"OCR客户端初始化错误: {e}")
        raise
    finally:
        # 恢复环境变量
        if http_proxy:
            os.environ['HTTP_PROXY'] = http_proxy
        if https_proxy:
            os.environ['HTTPS_PROXY'] = https_proxy
        if no_proxy:
            os.environ['NO_PROXY'] = no_proxy

def parse_ocr_response(text):
    """解析OCR返回的文本，提取句子和加粗单词"""
    # 移除markdown代码块标记
    text = re.sub(r'^```markdown\s*|\s*```$', '', text, flags=re.MULTILINE)
    
    # 分割文本行
    lines = text.split('\n')
    title = lines[0].strip() if lines and lines[0].strip() else ''
    
    # 解析文本，提取句子和加粗单词
    sentences = []
    for i, line in enumerate(lines):
        if not line.strip():
            continue
            
        # 提取加粗单词和清理句子
        bold_words = re.findall(r'\*\*(.*?)\*\*', line)
        clean_sentence = re.sub(r'\*\*', '', line)
        
        # 构建句子数据
        sentences.append({
            'text': clean_sentence.strip(),
            'bold_words': bold_words,
            'original_text': line.strip(),
            'is_title': i == 0,
            'title': title if i == 0 else ''
        })
    
    return sentences

def extract_text_cloud(image_path, ocr_model):
    """使用云API进行OCR识别"""
    logger.info("使用云端OCR服务处理图片")
    
    try:
        # 编码图片
        image_base64 = encode_image_to_base64(image_path)
        
        # 初始化客户端
        client = init_ocr_client(ocr_model)
        
        # 调用OCR API
        response = client.chat.completions.create(
            model=ocr_model["name"],
            messages=[{
                "role": "user",
                "content": [
                    {"type": "text", "text": OCR_PROMPT},
                    {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{image_base64}"}}
                ]
            }]
        )
        
        # 解析OCR结果
        text = response.choices[0].message.content
        sentences = parse_ocr_response(text)
        
        logger.info(f"OCR识别成功，提取了{len(sentences)}个句子")
        return sentences
        
    except Exception as e:
        logger.error(f"OCR识别失败: {str(e)}")
        raise

def create_empty_audio(filename):
    """创建空音频文件作为后备方案"""
    audio_path = os.path.join(AUDIO_FOLDER, filename)
    with open(audio_path, 'wb') as audio_file:
        audio_file.write(b'')
    return audio_path

def validate_text(text):
    """验证文本是否有效"""
    if not text or not text.strip():
        raise ValueError("输入文本为空，无法生成音频")
    return text.strip()

def generate_audio(text, filename, tts_model=None):
    """根据选择的TTS模型生成音频"""
    try:
        # 验证文本
        text = validate_text(text)
        
        # 根据TTS类型选择相应的生成函数
        if tts_model["type"] == "gtts" and TTS_GTTS_AVAILABLE:
            logger.info(f"使用GTTS服务生成音频: {filename}")
            return generate_audio_gtts(text, filename, tts_model)
        elif tts_model["type"] == "qwen-tts" and TTS_QWEN_AVAILABLE:
            logger.info(f"使用QWEN TTS服务生成音频: {filename}")
            return generate_audio_qwen(text, filename, tts_model)
        elif tts_model["type"] == "ms-tts" and TTS_AZURE_AVAILABLE:
            logger.info(f"使用Azure TTS服务生成音频: {filename}")
            return generate_audio_azure(text, filename, tts_model)
        else:
            logger.warning(f"警告: 所选TTS服务不可用或未启用，无法生成音频")
            return create_empty_audio(filename)
    except Exception as e:
        logger.error(f"音频生成失败: {str(e)}")
        return create_empty_audio(filename)

def generate_audio_gtts(text, filename, tts_model=None):
    """使用Google TTS库生成音频"""
    audio_path = os.path.join(AUDIO_FOLDER, filename)
    if tts_model is None:
        tts_model = tts_models["UK"]

    try:
        # 使用gTTS生成音频
        tts = gTTS(text=text, lang=tts_model["lang"], tld=tts_model["tld"], slow=False)
        tts.save(audio_path)
        logger.info(f"GTTS音频生成成功: {filename}")
        return audio_path
    except Exception as e:
        logger.error(f"GTTS音频生成失败: {str(e)}")
        return create_empty_audio(filename)

def generate_audio_qwen(text, filename, tts_model=None):
    """使用阿里云TTS服务生成音频"""
    audio_path = os.path.join(AUDIO_FOLDER, filename)
    if tts_model is None:
        tts_model = tts_models["Chinese"]
        
    try:
        # 配置阿里云TTS
        dashscope.api_key = tts_model["key"]
        synthesizer = SpeechSynthesizer(model=tts_model["model"], voice=tts_model["voice"], speech_rate=tts_model["speed"])
        
        # 调用API生成音频
        audio = synthesizer.call(text)
        
        # 验证返回的音频数据
        if audio is None or not audio:
            raise ValueError("TTS API返回了空数据")
            
        # 记录请求指标
        logger.info(f"[TTS指标] 请求ID: {synthesizer.get_last_request_id()}, 首包延迟: {synthesizer.get_first_package_delay()}ms")
        
        # 保存音频文件
        with open(audio_path, 'wb') as f:
            f.write(audio)
            
        logger.info(f"阿里云TTS生成成功: {filename}")
        return audio_path
    except Exception as e:
        logger.error(f"阿里云TTS生成失败: {str(e)}")
        return create_empty_audio(filename)

def generate_audio_azure(text, filename, tts_model=None):
    """使用Azure语音服务生成音频"""
    audio_path = os.path.join(AUDIO_FOLDER, filename)
    if tts_model is None:
        tts_model = tts_models["SG-man"]
        
    try:
        # 配置Azure语音服务
        speech_config = speechsdk.SpeechConfig(
            subscription=tts_model["speech_key"], 
            region=tts_model["service_region"]
        )
        speech_config.speech_synthesis_voice_name = tts_model["voice_name"]
        
        # 创建音频输出配置
        audio_config = speechsdk.audio.AudioOutputConfig(filename=audio_path)
        speech_synthesizer = speechsdk.SpeechSynthesizer(speech_config=speech_config, audio_config=audio_config)
        
        # 使用SSML格式设置语音
        ssml_text = f"""
<speak version='1.0' xmlns='http://www.w3.org/2001/10/synthesis' xml:lang='en-US'>
    <voice name='{tts_model["voice_name"]}'>
        <prosody rate='{tts_model["speed"]}'>
            {text}
        </prosody>
    </voice>
</speak>
"""
        
        # 合成音频
        result = speech_synthesizer.speak_ssml_async(ssml_text).get()
        
        # 检查结果
        if result.reason == speechsdk.ResultReason.SynthesizingAudioCompleted:
            with open(audio_path, 'wb') as f:
                f.write(result.audio_data)
            logger.info(f"Azure TTS生成成功: {filename}")
            return audio_path
        else:
            # 处理错误情况
            if result.reason == speechsdk.ResultReason.Canceled:
                details = result.cancellation_details
                error_msg = f"Azure TTS取消: {details.reason}"
                if details.reason == speechsdk.CancellationReason.Error:
                    error_msg += f", 错误详情: {details.error_details}"
                raise ValueError(error_msg)
            else:
                raise ValueError(f"Azure TTS失败，未知原因: {result.reason}")
    except Exception as e:
        logger.error(f"Azure TTS生成失败: {str(e)}")
        return create_empty_audio(filename)



def clean_audio_folder():
    """清理音频文件夹中的所有MP3文件"""
    try:
        count = 0
        for audio_file in os.listdir(AUDIO_FOLDER):
            if audio_file.endswith('.mp3'):
                audio_path = os.path.join(AUDIO_FOLDER, audio_file)
                os.remove(audio_path)
                count += 1
        logger.info(f"已清理{count}个音频文件")
        return True
    except Exception as e:
        logger.error(f"清理音频文件时出错: {e}")
        return False

def process_bold_words(sentence, idx, tts_model):
    """处理句子中的加粗单词，生成音频和HTML"""
    word_audios = []
    html_text = sentence['text']
    
    if not sentence['bold_words'] or len(sentence['bold_words']) == 0:
        return [], html_text, False
    
    for widx, word in enumerate(sentence['bold_words']):
        try:
            # 生成单词音频
            word_audio = generate_audio(
                word,
                f'word_{idx}_{widx}.mp3',
                tts_model
            )
            
            # 添加到结果列表
            word_audios.append({
                'word': word,
                'audio_path': f'/audio/word_{idx}_{widx}.mp3'
            })
            
            # 创建带播放按钮的HTML
            button_html = f'<span class="word-item bold" onclick="playAudio(\'word_{idx}_{widx}.mp3\')"><i class="bi bi-play-circle-fill"></i> {word}</span>'
            
            # 替换HTML中的单词
            pattern = r'\b' + re.escape(word) + r'\b'
            html_text = re.sub(pattern, button_html, html_text, count=1)
            
        except Exception as e:
            logger.error(f"单词音频生成错误: {e}")
            continue
    
    return word_audios, html_text, len(word_audios) > 0

def process_sentence(sentence, idx, tts_model, processed_count, total_sentences):
    """处理单个句子，生成音频和数据结构"""
    # 初始化基本信息
    sentence_data = {
        'text': sentence['text'],
        'has_bold_words': False,
        'html_text': sentence['text']
    }
    
    # 更新处理状态
    global processing_status
    processing_status = {
        'status': 'processing',
        'message': f'正在处理第 {processed_count}/{total_sentences} 个句子',
        'current': processed_count,
        'total': total_sentences,
        'progress': int((processed_count / total_sentences) * 100)
    }
    
    # 生成整句音频 (不再有Word Wall的跳过逻辑)
    sentence_audio = generate_audio(
        sentence['text'],
        f'sentence_{idx}.mp3',
        tts_model
    )
    sentence_data['audio_path'] = f'sentence_{idx}.mp3'
    
    # 处理加粗单词
    if sentence['bold_words'] and len(sentence['bold_words']) > 0:
        processing_status['message'] = f'正在处理第 {processed_count}/{total_sentences} 个句子的加粗单词'
        word_audios, html_text, has_bold_words = process_bold_words(sentence, idx, tts_model)
        
        sentence_data['bold_words'] = word_audios
        sentence_data['has_bold_words'] = has_bold_words
        sentence_data['html_text'] = html_text
        
    return sentence_data

@app.route('/upload', methods=['POST'])
def upload_file():
    """处理上传的图片文件，执行OCR和TTS"""
    # 初始化处理状态
    processing_status = {
        'status': 'processing',
        'message': '开始处理上传的图片'
    }
    
    try:
        # 验证上传文件
        if 'file' not in request.files:
            return jsonify({'error': '没有文件上传'}), 400
        
        file = request.files['file']
        if file.filename == '':
            return jsonify({'error': '未选择文件'}), 400
        
        # 保存上传的图片
        processing_status['message'] = '保存上传的图片'
        image_path = os.path.join(UPLOAD_FOLDER, file.filename)
        file.save(image_path)
        logger.info(f"已保存图片: {image_path}")
        
        # 清理之前的音频文件
        processing_status['message'] = '清理之前的音频文件'
        clean_audio_folder()
        
        # OCR处理
        try:
            processing_status['message'] = '正在进行OCR识别'
            ocr_model = get_selected_model(request, 'ocr')
            sentences = extract_text_cloud(image_path, ocr_model)
            
            if not sentences:
                return jsonify({'error': 'OCR识别失败，未能提取文本'}), 500
                
        except Exception as e:
            logger.error(f"OCR处理错误: {e}")
            return jsonify({'error': f'OCR处理错误: {str(e)}'}), 500
        
        # 音频生成
        processing_status['message'] = '正在生成音频'
        result = []
        
        # 计算需要处理的句子总数 (现在是所有句子)
        total_sentences = len(sentences)
        processed_count = 0
        tts_model = get_selected_model(request, 'tts')
        
        # 更新处理状态中的总句子数
        processing_status['total'] = total_sentences
        processing_status['current'] = 0
        processing_status['progress'] = 0
        
        # 处理每个句子
        for idx, sentence in enumerate(sentences):
            try:
                processed_count += 1 
                
                # 更新处理状态中的当前进度
                processing_status['current'] = processed_count
                processing_status['progress'] = int((processed_count / total_sentences) * 100)
                
                # 处理句子 (移除 word_wall_index 参数)
                sentence_data = process_sentence(
                    sentence, idx, 
                    tts_model, processed_count, total_sentences
                )
                
                result.append(sentence_data)
            except Exception as e:
                logger.error(f"句子处理错误: {e}")
                continue
        
        # 验证结果
        if not result:
            return jsonify({'error': 'OCR识别失败，未能提取任何文本'}), 500
        
        # 更新状态为完成
        processing_status['status'] = 'done'
        processing_status['message'] = '处理完成'
        processing_status['current'] = total_sentences
        processing_status['total'] = total_sentences
        processing_status['progress'] = 100
        logger.info(f"处理完成，共生成{len(result)}个句子数据")
            
        return jsonify(result)
    except Exception as e:
        logger.error(f"上传处理过程中发生错误: {e}")
        return jsonify({'error': f'处理失败: {str(e)}'}), 500

@app.route('/audio/<filename>')
def serve_audio(filename):
    return send_file(
        os.path.join(AUDIO_FOLDER, filename),
        mimetype='audio/mpeg',
        as_attachment=False
    )

# 初始化处理状态
def init_processing_status():
    return {
        'status': 'idle',  # idle, processing, done
        'message': '准备就绪',
        'current': 0,
        'total': 0,
        'progress': 0
    }

# 创建处理状态实例
processing_status = init_processing_status()

@app.route('/status')
def get_status():
    return jsonify(processing_status)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--port', type=int, default=int(os.environ.get('PORT', 5001)), help='Port to run the server on')
    args = parser.parse_args()
    app.run(host='0.0.0.0', debug=False, port=args.port)