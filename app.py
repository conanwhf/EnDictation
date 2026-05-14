from flask import Flask, render_template, request, jsonify, send_file
import html
import os
import re
import argparse
import logging
import unicodedata
import uuid
from google import genai

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

logging.getLogger('werkzeug').setLevel(logging.WARNING)

try:
    import azure.cognitiveservices.speech as speechsdk
    TTS_AZURE_AVAILABLE = True
except ImportError:
    logger.warning("azure-cognitiveservices-speech库未安装，Azure TTS功能不可用")
    TTS_AZURE_AVAILABLE = False

try:
    from gtts import gTTS
    TTS_GTTS_AVAILABLE = True
except ImportError:
    logger.warning("gtts库未安装，Google TTS功能不可用")
    TTS_GTTS_AVAILABLE = False

app = Flask(__name__)

OCR_MODEL = "gemini-3-flash-preview"

# TTS服务配置字典
tts_models = {
    "SG-man": {
        "label": "新加坡英语-男声 (Microsoft)",
        "type": "ms-tts",
        "speech_key": os.environ.get("AZURE_API_KEY"),
        "service_region": "southeastasia",
        "voice_name": "en-SG-WayneNeural",
        "speed": "-10%",
    },
    "SG-woman": {
        "label": "新加坡英语-女声 (Microsoft)",
        "type": "ms-tts",
        "speech_key": os.environ.get("AZURE_API_KEY"),
        "service_region": "southeastasia",
        "voice_name": "en-SG-LunaNeural",
        "speed": "-10%",
    },
    "UK-man": {
        "label": "英式英语/中文-男声 (Microsoft)",
        "type": "ms-tts",
        "speech_key": os.environ.get("AZURE_API_KEY"),
        "service_region": "southeastasia",
        "voice_name": "en-GB-OllieMultilingualNeural",
        "speed": "-10%",
    },
    "UK-woman": {
        "label": "英式英语/中文-女声 (Microsoft)",
        "type": "ms-tts",
        "speech_key": os.environ.get("AZURE_API_KEY"),
        "service_region": "southeastasia",
        "voice_name": "en-GB-LibbyNeural",
        "speed": "-10%",
    },
    "CH-man": {
        "label": "中文-男声 (Microsoft)",
        "type": "ms-tts",
        "speech_key": os.environ.get("AZURE_API_KEY"),
        "service_region": "southeastasia",
        "voice_name": "zh-CN-YunyangNeural",
        "speed": "-20%",
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
}

def tts_supports_speed(tts_model):
    """判断TTS模型是否支持自定义语速"""
    return tts_model.get("type") == "ms-tts"

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

@app.route('/')
def index():
    return render_template('index.html',
        tts_options=get_tts_options()
    )

# OCR提示词
OCR_PROMPT = "请你将图片处理成markdown文本，根据句号、句点、数字标号将文本分割为句子并换行。如果句子中有被圈出、粗体、放大、与众不同的字体或颜色的文本，则把它们也用粗体标记。请仅输出markdown代码即可。"

# 确保上传和音频文件夹存在
UPLOAD_FOLDER = 'uploads'
AUDIO_FOLDER = 'audio'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(AUDIO_FOLDER, exist_ok=True)

def safe_filename(filename):
    """清理文件名，防止路径遍历攻击"""
    filename = os.path.basename(filename)
    filename = unicodedata.normalize('NFKD', filename)
    filename = re.sub(r'[^\w\s.-]', '', filename).strip()
    if not filename:
        filename = 'upload.jpg'
    return filename

def sanitize_html(text):
    """清理OCR返回文本中的潜在危险HTML"""
    text = text.replace('<', '&lt;').replace('>', '&gt;')
    text = re.sub(r'\*\*(.*?)\*\*', r'<strong>\1</strong>', text)
    return text

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
    """解析OCR返回的文本，提取句子和加粗单词"""
    # 移除markdown代码块标记
    text = re.sub(r'^```markdown\s*|\s*```$', '', text, flags=re.MULTILINE)
    
    # 分割文本行
    lines = merge_standalone_number_labels(text.split('\n'))
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

def extract_text_cloud(image_path):
    """使用Gemini API进行OCR识别"""
    logger.info("使用Gemini OCR服务处理图片")

    try:
        client = genai.Client()

        with open(image_path, 'rb') as f:
            image_bytes = f.read()

        response = client.models.generate_content(
            model=OCR_MODEL,
            contents=[
                {"inline_data": {"mime_type": "image/jpeg", "data": image_bytes}},
                OCR_PROMPT,
            ],
        )

        text = response.text
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

def generate_audio(text, filename, tts_model=None, speed_percent=None):
    """根据选择的TTS模型生成音频"""
    try:
        # 验证文本
        text = validate_text(text)
        
        # 根据TTS类型选择相应的生成函数
        if tts_model["type"] == "gtts" and TTS_GTTS_AVAILABLE:
            logger.info(f"使用GTTS服务生成音频: {filename}")
            return generate_audio_gtts(text, filename, tts_model)
        elif tts_model["type"] == "ms-tts" and TTS_AZURE_AVAILABLE:
            logger.info(f"使用Azure TTS服务生成音频: {filename}")
            return generate_audio_azure(text, filename, tts_model, speed_percent)
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
        tts_model = tts_models["UK-Google"]

    try:
        # 使用gTTS生成音频
        tts = gTTS(text=text, lang=tts_model["lang"], tld=tts_model["tld"], slow=False)
        tts.save(audio_path)
        logger.info(f"GTTS音频生成成功: {filename}")
        return audio_path
    except Exception as e:
        logger.error(f"GTTS音频生成失败: {str(e)}")
        return create_empty_audio(filename)

def generate_audio_azure(text, filename, tts_model=None, speed_percent=None):
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
        speed = format_speed_percent(speed_percent) if speed_percent is not None else tts_model["speed"]
        ssml_body = html.escape(text, quote=False)
        ssml_text = f"""
<speak version='1.0' xmlns='http://www.w3.org/2001/10/synthesis' xml:lang='en-US'>
    <voice name='{tts_model["voice_name"]}'>
        <prosody rate='{speed}'>
            {ssml_body}
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

def update_processing_status(**kwargs):
    """安全更新全局处理状态"""
    global processing_status
    processing_status.update(kwargs)

def get_bold_word_texts(sentence):
    """兼容OCR原始结果和TTS生成后的重点词结构"""
    words = []
    for word in sentence.get('bold_words') or []:
        if isinstance(word, dict):
            word = word.get('word', '')
        if str(word).strip():
            words.append(str(word).strip())
    return words

def process_bold_words(sentence, idx, tts_model, run_id, speed_percent=None):
    """处理句子中的加粗单词，生成音频和HTML"""
    word_audios = []
    html_text = sanitize_html(sentence.get('original_text') or sentence['text'])
    bold_words = get_bold_word_texts(sentence)
    
    if not bold_words:
        return [], html_text, False
    
    for widx, word in enumerate(bold_words):
        try:
            audio_filename = f'{run_id}_word_{idx}_{widx}.mp3'

            # 生成单词音频
            generate_audio(
                word,
                audio_filename,
                tts_model,
                speed_percent
            )
            
            # 添加到结果列表
            word_audios.append({
                'word': word,
                'audio_path': f'/audio/{audio_filename}'
            })
            
            # 创建带播放按钮的HTML
            safe_word = sanitize_html(word)
            button_html = f'<span class="word-item bold" onclick="playAudio(\'{audio_filename}\')"><i class="bi bi-play-circle-fill"></i> {safe_word}</span>'
            
            # 替换HTML中的单词
            pattern = r'\b' + re.escape(word) + r'\b'
            html_text = re.sub(pattern, button_html, html_text, count=1)
            
        except Exception as e:
            logger.error(f"单词音频生成错误: {e}")
            continue
    
    return word_audios, html_text, len(word_audios) > 0

def process_sentence(sentence, idx, tts_model, processed_count, total_sentences, run_id, speed_percent=None):
    """处理单个句子，生成音频和数据结构"""
    # 初始化基本信息
    sentence_data = {
        'text': sentence['text'],
        'original_text': sentence.get('original_text', sentence['text']),
        'is_title': sentence.get('is_title', False),
        'title': sentence.get('title', ''),
        'has_bold_words': False,
        'html_text': sanitize_html(sentence.get('original_text') or sentence['text'])
    }
    
    # 更新处理状态
    update_processing_status(
        status='processing',
        message=f'正在处理第 {processed_count}/{total_sentences} 个句子',
        current=processed_count,
        total=total_sentences,
        progress=int((processed_count / total_sentences) * 100)
    )
    
    # 生成整句音频 (不再有Word Wall的跳过逻辑)
    audio_filename = f'{run_id}_sentence_{idx}.mp3'
    generate_audio(
        sentence['text'],
        audio_filename,
        tts_model,
        speed_percent
    )
    sentence_data['audio_path'] = audio_filename
    
    # 处理加粗单词
    if get_bold_word_texts(sentence):
        update_processing_status(message=f'正在处理第 {processed_count}/{total_sentences} 个句子的加粗单词')
        word_audios, html_text, has_bold_words = process_bold_words(sentence, idx, tts_model, run_id, speed_percent)
        
        sentence_data['bold_words'] = word_audios
        sentence_data['has_bold_words'] = has_bold_words
        sentence_data['html_text'] = html_text
        
    return sentence_data

@app.route('/upload', methods=['POST'])
def upload_file():
    """处理上传的图片文件，只执行OCR"""
    update_processing_status(
        status='processing',
        message='开始处理上传的图片',
        current=0,
        total=0,
        progress=0
    )

    try:
        if 'file' not in request.files:
            return jsonify({'error': '没有文件上传'}), 400

        file = request.files['file']
        if file.filename == '':
            return jsonify({'error': '未选择文件'}), 400

        update_processing_status(message='保存上传的图片')
        filename = safe_filename(file.filename)
        image_path = os.path.join(UPLOAD_FOLDER, filename)
        file.save(image_path)
        logger.info(f"已保存图片: {image_path}")

        update_processing_status(message='清理之前的音频文件')
        clean_audio_folder()

        try:
            update_processing_status(message='正在进行OCR识别')
            sentences = extract_text_cloud(image_path)

            if not sentences:
                return jsonify({'error': 'OCR识别失败，未能提取文本'}), 500

        except Exception as e:
            logger.error(f"OCR处理错误: {e}")
            return jsonify({'error': f'OCR处理错误: {str(e)}'}), 500

        total_sentences = len(sentences)
        update_processing_status(
            status='done',
            message='OCR识别完成',
            current=total_sentences,
            total=total_sentences,
            progress=100
        )
        logger.info(f"OCR处理完成，共提取{len(sentences)}个句子数据")

        return jsonify(sentences)
    except Exception as e:
        logger.error(f"上传处理过程中发生错误: {e}")
        return jsonify({'error': f'处理失败: {str(e)}'}), 500

@app.route('/generate-tts', methods=['POST'])
def generate_tts():
    """根据OCR结果单独生成TTS音频"""
    update_processing_status(
        status='processing',
        message='开始生成音频',
        current=0,
        total=0,
        progress=0
    )

    try:
        data = request.get_json(silent=True) or {}
        sentences = data.get('sentences') or []
        if not isinstance(sentences, list) or not sentences:
            return jsonify({'error': '没有可生成音频的OCR文本'}), 400

        tts_key = data.get('tts_select') or data.get('tts-select') or 'UK-Google'
        tts_model = tts_models.get(tts_key, tts_models['UK-Google'])
        speed_percent = None
        if tts_supports_speed(tts_model):
            default_speed = parse_speed_percent(tts_model.get('speed'))
            speed_percent = parse_speed_percent(data.get('speed'), default_speed)

        update_processing_status(message='清理之前的音频文件')
        clean_audio_folder()

        items_to_speak = [sentence for sentence in sentences if sentence.get('text') and not sentence.get('is_title')]
        if not items_to_speak:
            return jsonify({'error': '没有可生成音频的句子'}), 400

        total_sentences = len(items_to_speak)
        processed_count = 0
        run_id = uuid.uuid4().hex[:8]
        result = []

        for idx, sentence in enumerate(sentences):
            if sentence.get('is_title'):
                result.append({
                    'text': sentence.get('text', ''),
                    'bold_words': get_bold_word_texts(sentence),
                    'original_text': sentence.get('original_text', sentence.get('text', '')),
                    'is_title': True,
                    'title': sentence.get('title', sentence.get('text', '')),
                    'has_bold_words': False
                })
                continue

            try:
                processed_count += 1
                sentence_data = process_sentence(
                    sentence, idx, tts_model,
                    processed_count, total_sentences,
                    run_id, speed_percent
                )
                result.append(sentence_data)
            except Exception as e:
                logger.error(f"句子TTS处理错误: {e}")
                continue

        if not result:
            return jsonify({'error': '音频生成失败，未能处理任何文本'}), 500

        update_processing_status(
            status='done',
            message='音频生成完成',
            current=total_sentences,
            total=total_sentences,
            progress=100
        )
        logger.info(f"音频生成完成，共生成{processed_count}个句子数据")

        return jsonify(result)
    except Exception as e:
        logger.error(f"TTS生成过程中发生错误: {e}")
        return jsonify({'error': f'TTS生成失败: {str(e)}'}), 500

@app.route('/audio/<filename>')
def serve_audio(filename):
    filename = safe_filename(filename)
    audio_path = os.path.join(AUDIO_FOLDER, filename)
    if not os.path.isfile(audio_path):
        return jsonify({'error': '文件不存在'}), 404
    return send_file(audio_path, mimetype='audio/mpeg', as_attachment=False)

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
