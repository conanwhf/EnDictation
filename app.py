from flask import Flask, render_template, request, jsonify, send_file
import os
import re
import requests
import json
import base64
import numpy as np
import soundfile as sf
from openai import OpenAI

# 导入阿里云TTS相关库
try:
    from dashscope.audio.tts_v2 import SpeechSynthesizer
    import dashscope
    TTS_QWEN_AVAILABLE = True
except ImportError:
    print("警告: dashscope库未安装，云端TTS功能不可用，请使用pip install dashscope安装")
    TTS_QWEN_AVAILABLE = False

# 导入Azure语音SDK
try:
    import azure.cognitiveservices.speech as speechsdk
    TTS_AZURE_AVAILABLE = True
except ImportError:
    print("警告: azure-cognitiveservices-speech库未安装，Azure TTS功能不可用，请使用pip install azure-cognitiveservices-speech安装")
    TTS_AZURE_AVAILABLE = False

# 导入本地TTS相关库（调用Google TTS实现）
try:
    from gtts import gTTS
    TTS_GTTS_AVAILABLE = True
except ImportError:
    print("警告: gtts库未安装，本地TTS功能不可用")
    TTS_GTTS_AVAILABLE = False

app = Flask(__name__)

# AI服务配置字典
ocr_ai_models = {
    "qwen-ocr": {
        "key": os.environ.get("QWEN_API_KEY", "demo"),
        "url": "https://dashscope.aliyuncs.com/compatible-mode/v1",
        "name": "qvq-max-latest",
    },
    "gemini-ocr": {
        "key": os.environ.get("GOOGLE_API_KEY", "demo"),
        "url": "https://generativelanguage.googleapis.com/v1beta/openai/",
        "name": "gemini-2.0-flash",
    },
}

# TTS服务配置字典
tts_models = {
    "qwen-tts": {
        "key": os.environ.get("QWEN_API_KEY", "demo"),
        "model": "cosyvoice-v1",
        "voice": "longxiaochun",
    },
    "gtts": {
        "lang": "en", 
        "tld": "co.uk", 
    },
    "ms-tts": {
        "speech_key": os.environ.get("AZURE_API_KEY", "demo"),
        "service_region": "southeastasia",
        "voice_name": "en-SG-LunaNeural",
        #"voice_name": "en-SG-WayneNeural",
        "speed": "-15%",
    },
}
# 模型配置
OCR_MODEL = ocr_ai_models["gemini-ocr"]
TTS_MODEL = tts_models["gtts"]
# OCR提示词
OCR_PROMT = "请你将图片处理成文本，使用markdown输出。对于每个句子：如果句子中有被圈出的部分，仅把圈出来的部分用粗体标记；如果没有被圈出来的部分，则将识别到的粗体单词用粗体标记"
# 打印当前使用的OCR模型和API密钥
print("QWEN key: ", ocr_ai_models["qwen-ocr"]["key"])
print("GOOGLE key: ", ocr_ai_models["gemini-ocr"]["key"])

# 确保上传和音频文件夹存在
UPLOAD_FOLDER = 'uploads'
AUDIO_FOLDER = 'audio'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(AUDIO_FOLDER, exist_ok=True)


def extract_text_cloud(image_path):
    """使用云API进行OCR识别"""
    print("使用云端OCR服务处理图片")
    # 将图片转换为base64编码
    with open(image_path, 'rb') as image_file:
        import base64
        image_base64 = base64.b64encode(image_file.read()).decode('utf-8')
    
    # 根据AI demo.py示例构建正确的OCR请求格式
    # 修改客户端初始化部分
    # 最简化的客户端初始化 - 避免使用proxies参数
    import os
    
    # 临时保存当前环境变量中可能存在的代理设置
    http_proxy = os.environ.pop('HTTP_PROXY', None)
    https_proxy = os.environ.pop('HTTPS_PROXY', None)
    no_proxy = os.environ.pop('NO_PROXY', None)
    
    try:
        # 不使用代理初始化客户端
        client = OpenAI(
            api_key=OCR_MODEL["key"],
            base_url=OCR_MODEL["url"]
        )
        print("成功初始化OpenAI客户端")
    except Exception as e:
        print(f"OpenAI客户端初始化错误: {e}")
        raise
    finally:
        # 恢复环境变量
        if http_proxy:
            os.environ['HTTP_PROXY'] = http_proxy
        if https_proxy:
            os.environ['HTTPS_PROXY'] = https_proxy
        if no_proxy:
            os.environ['NO_PROXY'] = no_proxy
    
    # 使用与AI demo.py一致的API调用方式
    response = client.chat.completions.create(
        model=OCR_MODEL["name"],
        messages=[
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": OCR_PROMT,
                    },
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/jpeg;base64,{image_base64}"
                        },
                    },
                ],
            }
        ],
    )
    
    # 获取OCR结果
    text = response.choices[0].message.content
    
    # 移除markdown代码块标记
    text = re.sub(r'^```markdown\s*|\s*```$', '', text, flags=re.MULTILINE)
    
    # 获取第一行作为标题
    lines = text.split('\n')
    title = ''
    if lines and lines[0].strip():
        title = lines[0].strip()
    
    # 解析文本，提取句子和加粗单词
    sentences = []
    for i, line in enumerate(lines):
        if line.strip():
            # 使用正则表达式查找加粗单词（假设加粗单词被**包围）
            bold_words = re.findall(r'\*\*(.*?)\*\*', line)
            # 移除标记符号得到原始句子
            clean_sentence = re.sub(r'\*\*', '', line)
            
            # 检查是否包含数字标号
            has_number = bool(re.search(r'^\d+\.\s', clean_sentence.strip()))
            
            # 添加所有句子，包括一个标志表示是否有数字标号
            sentences.append({
                'text': clean_sentence.strip(),
                'bold_words': bold_words,
                'original_text': line.strip(),  # 保存原始带标记的文本
                'has_number': has_number,
                'is_title': i == 0,  # 标记第一行为标题
                'title': title if i == 0 else ''  # 添加标题字段
            })
    return sentences

def generate_audio(text, filename):
    # 优先检查是否可以使用Google TTS
    if (TTS_MODEL == tts_models["gtts"]) and TTS_GTTS_AVAILABLE:
        print(f"使用GTTS服务生成音频: {filename}")
        return generate_audio_gtts(text, filename)
    # 其次检查是否可以使用Qwen TTS
    if (TTS_MODEL == tts_models["qwen-tts"]) and TTS_QWEN_AVAILABLE:
        print(f"使用QWEN TTS服务生成音频: {filename}")
        return generate_audio_qwen(text, filename)
    # 再次检查是否可以使用Azure TTS
    if (TTS_MODEL == tts_models["ms-tts"]) and TTS_AZURE_AVAILABLE:
        print(f"使用Azure TTS服务生成音频: {filename}")
        return generate_audio_azure(text, filename)
    
    # 如果以上方式都不可用，创建空文件作为后备方案
    print(f"警告: TTS服务不可用或未启用，无法生成音频")
    audio_path = os.path.join(AUDIO_FOLDER, filename)
    with open(audio_path, 'wb') as audio_file:
        audio_file.write(b'')
    return audio_path

def generate_audio_gtts(text, filename):
    """使用本地TTS库生成音频"""
    audio_path = os.path.join(AUDIO_FOLDER, filename)
    
    # 使用gTTS (需要网络连接，但质量较好)
    tts = gTTS(text=text, lang=TTS_MODEL["lang"], tld=TTS_MODEL["tld"], slow=False)
    tts.save(audio_path)
    return audio_path

def generate_audio_qwen(text, filename):
    """使用阿里云TTS服务生成音频"""
    audio_path = os.path.join(AUDIO_FOLDER, filename)
    
    try:
        # 检查输入文本是否为空
        if not text or not text.strip():
            raise ValueError("输入文本为空，无法生成音频")
            
        # 配置阿里云TTS模型和声音
        dashscope.api_key = TTS_MODEL["key"]
        model = TTS_MODEL["model"]  # 使用与API demo相同的模型
        voice = TTS_MODEL["voice"]  # 使用与API demo相同的声音
        
        # 初始化语音合成器
        synthesizer = SpeechSynthesizer(model=model, voice=voice)
        
        # 调用API生成音频
        audio = synthesizer.call(text)
        
        # 检查返回的音频数据是否为None或空
        if audio is None:
            raise ValueError("TTS API返回了空数据(None)")
        elif not audio:  # 检查是否为空字节
            raise ValueError("TTS API返回了空字节")
            
        # 输出请求指标信息
        print(f"[TTS指标] 请求ID: {synthesizer.get_last_request_id()}, 首包延迟: {synthesizer.get_first_package_delay()}ms")
        
        # 保存音频文件
        with open(audio_path, 'wb') as f:
            f.write(audio)
            
        print(f"云端TTS生成成功: {filename}")
        return audio_path
    except Exception as e:
        print(f"云端TTS生成失败: {str(e)}")
       
    # 失败后创建空文件作为后备方案
    with open(audio_path, 'wb') as audio_file:
        audio_file.write(b'')
    return audio_path

def generate_audio_azure(text, filename):
    """使用Azure语音服务生成音频"""
    audio_path = os.path.join(AUDIO_FOLDER, filename)
    
    try:
        # 检查输入文本是否为空
        if not text or not text.strip():
            raise ValueError("输入文本为空，无法生成音频")
        
        # 配置Azure语音服务
        speech_key = TTS_MODEL["speech_key"]
        service_region = TTS_MODEL["service_region"]
        voice_name = TTS_MODEL["voice_name"]
        speed_rate = TTS_MODEL["speed"]
        
        # 创建语音配置
        speech_config = speechsdk.SpeechConfig(subscription=speech_key, region=service_region)
        speech_config.speech_synthesis_voice_name = voice_name
        
        # 创建语音合成器
        speech_synthesizer = speechsdk.SpeechSynthesizer(speech_config=speech_config)
        
        # 使用SSML格式设置慢速语音
        ssml_text = f"""
<speak version='1.0' xmlns='http://www.w3.org/2001/10/synthesis' xml:lang='en-US'>
    <voice name='{voice_name}'>
        <prosody rate='{speed_rate}'>
            {text}
        </prosody>
    </voice>
</speak>
"""
        
        # 合成音频
        result = speech_synthesizer.speak_ssml_async(ssml_text).get()
        
        # 检查结果
        if result.reason == speechsdk.ResultReason.SynthesizingAudioCompleted:
            # 获取音频数据
            audio_data = result.audio_data
            
            # 保存音频文件
            with open(audio_path, 'wb') as f:
                f.write(audio_data)
                
            print(f"Azure TTS生成成功: {filename}")
            return audio_path
        elif result.reason == speechsdk.ResultReason.Canceled:
            cancellation_details = result.cancellation_details
            print(f"Azure TTS合成取消: {cancellation_details.reason}")
            if cancellation_details.reason == speechsdk.CancellationReason.Error:
                print(f"错误详情: {cancellation_details.error_details}")
            raise ValueError(f"Azure TTS合成失败: {cancellation_details.reason}")
        else:
            raise ValueError(f"Azure TTS合成失败，未知原因: {result.reason}")
    except Exception as e:
        print(f"Azure TTS生成失败: {str(e)}")
    
    # 失败后创建空文件作为后备方案
    with open(audio_path, 'wb') as audio_file:
        audio_file.write(b'')
    return audio_path

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_file():
    global processing_status
    try:
        # 更新状态为处理中
        processing_status['status'] = 'processing'
        processing_status['message'] = '开始处理上传的图片'
        
        if 'file' not in request.files:
            processing_status['status'] = 'idle'
            processing_status['message'] = '准备就绪'
            return jsonify({'error': '没有文件上传'}), 400
        
        file = request.files['file']
        if file.filename == '':
            processing_status['status'] = 'idle'
            processing_status['message'] = '准备就绪'
            return jsonify({'error': '未选择文件'}), 400
        
        # 保存上传的图片
        processing_status['message'] = '保存上传的图片'
        image_path = os.path.join(UPLOAD_FOLDER, file.filename)
        file.save(image_path)
        
        # 清理之前生成的音频文件
        processing_status['message'] = '清理之前的音频文件'
        try:
            for audio_file in os.listdir(AUDIO_FOLDER):
                if audio_file.endswith('.mp3'):
                    audio_path = os.path.join(AUDIO_FOLDER, audio_file)
                    os.remove(audio_path)
                    print(f"已删除音频文件: {audio_path}")
        except Exception as e:
            print(f"清理音频文件时出错: {e}")
            # 继续处理，不中断上传流程
        
        # 处理图片并提取文本
        try:
            processing_status['message'] = '正在进行OCR识别'
            sentences = extract_text_cloud(image_path)
            if not sentences:
                processing_status['status'] = 'idle'
                processing_status['message'] = '准备就绪'
                return jsonify({'error': 'OCR识别失败，未能提取文本'}), 500
        except Exception as e:
            print(f"OCR处理错误: {e}")
            processing_status['status'] = 'idle'
            processing_status['message'] = '准备就绪'
            return jsonify({'error': f'OCR处理错误: {str(e)}'}), 500
        
        # 为每个句子和加粗单词生成音频
        processing_status['message'] = '正在生成音频'
        result = []
        
        # 检查是否有"My Word Wall"部分
        word_wall_index = -1
        for i, s in enumerate(sentences):
            if s['text'].strip().startswith("My Word Wall"):
                word_wall_index = i
                break
                
        # 计算需要处理的句子总数（排除My Word Wall后面的部分）
        total_sentences = len(sentences) if word_wall_index == -1 else word_wall_index
        processed_count = 0
        
        for idx, sentence in enumerate(sentences):
            try:
                # 初始化基本信息
                sentence_data = {
                    'text': sentence['text'],
                    'has_number': sentence.get('has_number', False),
                    'has_bold_words': False,
                    'html_text': sentence['text']  # 初始化为原始文本
                }
                
                # 为所有句子生成音频，但排除My Word Wall后面的部分
                if word_wall_index == -1 or idx < word_wall_index:
                    processed_count += 1
                    processing_status['message'] = f'正在处理第 {processed_count}/{total_sentences} 个句子'
                
                    # 生成整句音频
                    sentence_audio = generate_audio(
                        sentence['text'],
                        f'sentence_{idx}.mp3'
                    )
                    sentence_data['audio_path'] = f'sentence_{idx}.mp3'
                    
                    # 生成加粗单词的音频
                    word_audios = []
                    html_text = sentence['text']  # 初始化为原始文本
                    
                    if sentence['bold_words'] and len(sentence['bold_words']) > 0:
                        processing_status['message'] = f'正在处理第 {processed_count}/{total_sentences} 个句子的加粗单词'
                        
                        for widx, word in enumerate(sentence['bold_words']):
                            try:
                                word_audio = generate_audio(
                                    word,
                                    f'word_{idx}_{widx}.mp3'
                                )
                                word_audios.append({
                                    'word': word,
                                    'audio_path': f'/audio/word_{idx}_{widx}.mp3'
                                })
                                
                                # 直接在句子中嵌入单词按钮
                                # 创建一个带有播放按钮的HTML片段，替换原始单词
                                # 使用正则表达式进行精确替换，避免替换句子中所有相同的单词
                                button_html = f'<span class="word-item bold" onclick="playAudio(\'word_{idx}_{widx}.mp3\')"><i class="bi bi-play-circle-fill"></i> {word}</span>'
                                # 使用正则表达式确保只替换完整的单词，而不是单词的一部分
                                pattern = r'\b' + re.escape(word) + r'\b'
                                # 只替换第一次出现的实例
                                html_text = re.sub(pattern, button_html, html_text, count=1)
                            except Exception as e:
                                print(f"单词音频生成错误: {e}")
                                # 继续处理其他单词
                                continue
                        
                        sentence_data['bold_words'] = word_audios
                        sentence_data['has_bold_words'] = len(word_audios) > 0
                        sentence_data['html_text'] = html_text
                else:
                    # 非数字标号句子不生成音频，但仍然添加到结果中
                    sentence_data['has_bold_words'] = False
                    sentence_data['bold_words'] = []
                
                result.append(sentence_data)
            except Exception as e:
                print(f"句子处理错误: {e}")
                # 继续处理其他句子
                continue
        
        # 检查是否有任何结果
        if not result:
            processing_status['status'] = 'idle'
            processing_status['message'] = '准备就绪'
            return jsonify({'error': 'OCR识别失败，未能提取任何文本'}), 500
            
        # 检查是否有带数字标号的句子
        if not any(item.get('has_number', False) for item in result):
            print("警告: 未找到带数字标号的句子，但仍返回所有识别出的文本")
            # 继续处理，返回所有识别出的句子
        
        # 更新状态为完成
        processing_status['status'] = 'done'
        processing_status['message'] = '处理完成'
            
        return jsonify(result)
    except Exception as e:
        print(f"上传处理过程中发生错误: {e}")
        return jsonify({'error': f'处理失败: {str(e)}'}), 500

@app.route('/audio/<filename>')
def serve_audio(filename):
    return send_file(
        os.path.join(AUDIO_FOLDER, filename),
        mimetype='audio/mpeg',
        as_attachment=False
    )

# 添加全局变量用于跟踪处理状态
processing_status = {
    'status': 'idle',  # idle, processing, done
    'message': '准备就绪'
}

@app.route('/status')
def get_status():
    return jsonify(processing_status)

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--port', type=int, default=int(os.environ.get('PORT', 5001)), help='Port to run the server on')
    args = parser.parse_args()
    app.run(host='0.0.0.0', debug=False, port=args.port)