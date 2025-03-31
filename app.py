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
        "key": os.environ.get("QWEN_API_KEY", "sk-demo"),
        "url": "https://dashscope.aliyuncs.com/compatible-mode/v1",
        "name": "qwen-vl-max-latest",
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
        "key": os.environ.get("QWEN_API_KEY", "sk-demo"),
        "model": "cosyvoice-v1",
        "voice": "longxiaochun",
    },
    "gtts": {
        "lang": "en", 
        "tld": "co.uk", 
    },
}
# 模型配置
OCR_MODEL = ocr_ai_models["gemini-ocr"]
#TTS_MODEL = tts_models["qwen-tts"]
TTS_MODEL = tts_models["gtts"]

# 打印当前使用的OCR模型和API密钥
print("--- All Environment Variables ---")
# 使用 json.dumps 让字典输出更易读，特别是值包含特殊字符时
print(json.dumps(dict(os.environ), indent=2))
print("--- End Environment Variables ---")
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
    client = OpenAI(
        api_key=OCR_MODEL["key"],
        base_url=OCR_MODEL["url"],
    )
    
    # 使用与AI demo.py一致的API调用方式
    response = client.chat.completions.create(
        model=OCR_MODEL["name"],
        messages=[
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": "请你将图片处理成文本，使用markdown输出，加粗的单词用粗体标记",
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
    # 优先检查是否可以使用云端TTS
    if (TTS_MODEL == tts_models["gtts"]) and TTS_GTTS_AVAILABLE:
        print(f"使用GTTS服务生成音频: {filename}")
        return generate_audio_gtts(text, filename)
    # 其次检查是否可以使用本地TTS
    if (TTS_MODEL == tts_models["qwen-tts"]) and TTS_QWEN_AVAILABLE:
        print(f"使用QWEN TTS服务生成音频: {filename}")
        return generate_audio_qwen(text, filename)
    
    # 如果以上两种方式都不可用，创建空文件作为后备方案
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
        total_sentences = len([s for s in sentences if s.get('has_number', False)])
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
                
                # 只为带数字标号的句子生成音频
                if sentence.get('has_number', False):
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
                                button_html = f'<span class="word-item bold" onclick="playAudio(\'word_{idx}_{widx}.mp3\')"><i class="bi bi-play-circle-fill"></i> {word}</span>'
                                html_text = html_text.replace(word, button_html)
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
    parser.add_argument('--port', type=int, default=int(os.environ.get('PORT', 5000)), help='Port to run the server on')
    args = parser.parse_args()
    app.run(host='0.0.0.0', debug=False, port=args.port)