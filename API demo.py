# OCR demo from Google Gemini Demo
from pydub import AudioSegment
import pyttsx3
import azure.cognitiveservices.speech as speechsdk
from dashscope.audio.tts_v2 import *
import dashscope
import base64
from openai import OpenAI

client = OpenAI(
    api_key="GEMINI_API_KEY",
    base_url="https://generativelanguage.googleapis.com/v1beta/openai/"
)
# Function to encode the image
def encode_image(image_path):
  with open(image_path, "rb") as image_file:
    return base64.b64encode(image_file.read()).decode('utf-8')
# Getting the base64 string
base64_image = encode_image("Path/to/agi/image.jpeg")
response = client.chat.completions.create(
    model="gemini-2.0-flash",
    messages=[
        {
          "role": "user",
          "content": [
              {
                  "type": "text",
                  "text": "What is in this image?",
              },
              {
                  "type": "image_url",
                  "image_url": {
                      "url":  f"data:image/jpeg;base64,{base64_image}"
                  },
              },
          ],
        }
    ],
)
print(response.choices[0])


# Ali TTS demo
model = "cosyvoice-v1"
voice = "longxiaochun"

synthesizer = SpeechSynthesizer(model=model, voice=voice)
audio = synthesizer.call("今天天气怎么样？")
print('[Metric] requestId: {}, first package delay ms: {}'.format(
    synthesizer.get_last_request_id(),
    synthesizer.get_first_package_delay()))

with open('output.mp3', 'wb') as f:
    f.write(audio)




'''
  For more samples please visit https://github.com/Azure-Samples/cognitive-services-speech-sdk 
'''
# Creates an instance of a speech config with specified subscription key and service region.
speech_key = "undefined"
service_region = "undefined"

speech_config = speechsdk.SpeechConfig(
    subscription=speech_key, region=service_region)
# Note: the voice setting will not overwrite the voice element in input SSML.
speech_config.speech_synthesis_voice_name = "en-SG-LunaNeural"

text = "Hi, this is Luna"

# use the default speaker as audio output.
speech_synthesizer = speechsdk.SpeechSynthesizer(speech_config=speech_config)

text_to_synthesize = """
<speak version='1.0' xmlns='http://www.w3.org/2001/10/synthesis' xml:lang='zh-CN'>
    <prosody rate='-20%'>
        这段文字将会以比正常速度慢 20% 的速度播放。
    </prosody>
    <prosody rate='0.8'>
        这段文字将会以 80% 的速度播放，也是慢速。
    </prosody>
    <prosody rate='slow'>
        这段文字将会以预定义的慢速播放。
    </prosody>
</speak>
"""
result = speech_synthesizer.speak_ssml_async(text_to_synthesize).get()

# Check result
if result.reason == speechsdk.ResultReason.SynthesizingAudioCompleted:
    print("Speech synthesized for text [{}]".format(text))
elif result.reason == speechsdk.ResultReason.Canceled:
    cancellation_details = result.cancellation_details
    print("Speech synthesis canceled: {}".format(cancellation_details.reason))
    if cancellation_details.reason == speechsdk.CancellationReason.Error:
        print("Error details: {}".format(cancellation_details.error_details))


#pyttsx3
def generate_audio_pyttsx3(text: str, filename: str):
    """
    使用 pyttsx3 库根据全局设置生成音频文件。

    Args:
        text (str): 要转换为语音的文本。
        filename (str): 要保存的音频文件名 (建议使用 .wav 或 .mp3 后缀, 但实际格式取决于引擎)。

    Returns:
        str: 生成的音频文件的完整路径，如果失败则返回 None。
    """
    audio_path = os.path.join(AUDIO_FOLDER, filename)

    try:
        engine = pyttsx3.init()
    except ImportError:
        print('错误：无法导入 TTS 引擎驱动。请确保 pyttsx3 及其依赖已正确安装。')
        return None
    except RuntimeError as e:
        print(f'错误：TTS 引擎初始化失败: {e}。请检查系统 TTS 是否正常工作。')
        return None
    except Exception as e:
        print(f'未知错误在初始化时发生: {e}')
        return None

    try:
        # 1. 设置语速
        rate = PYTTSX3_SETTINGS.get('rate')
        if rate:
            try:
                engine.setProperty('rate', int(rate))
                print(f"设置语速为: {rate}")
            except Exception as e:
                print(f"警告：设置语速失败: {e}")

        # 2. 设置声音 (语言)
        voice_id = PYTTSX3_SETTINGS.get('voice_id')
        if voice_id:
            try:
                engine.setProperty('voice', voice_id)
                # 验证一下当前声音（可选）
                current_voice = engine.getProperty('voice')
                print(f"尝试设置声音 ID: {voice_id}")
                # 注意：getProperty('voice') 返回的可能不是设置的ID，而是内部标识
                # print(f"当前实际使用的声音标识: {current_voice}")
            except Exception as e:
                print(f"警告: 无法设置声音 ID '{voice_id}'。将使用默认声音。错误: {e}")
        else:
            # 获取并打印默认声音信息（可选）
            try:
                 default_voice_id = engine.getProperty('voice')
                 voices = engine.getProperty('voices')
                 default_voice_name = "Unknown"
                 for v in voices:
                     if v.id == default_voice_id:
                         default_voice_name = v.name
                         break
                 print(
                   f"信息: 未指定 voice_id，使用系统默认声音 (ID: {default_voice_id}, Name: {default_voice_name})")
            except Exception as e:
                 print(f"信息: 未指定 voice_id，使用系统默认声音 (获取详情失败: {e})")

        # 3. 生成并保存音频文件
        print(f"准备生成音频: '{text}' -> {audio_path}")
        engine.save_to_file(text, audio_path)

        # 4. 执行保存操作 (阻塞直到完成)
        engine.runAndWait()

        # 5. 停止引擎 (在 runAndWait 后通常是好的做法，尤其是在循环或长时间运行的应用中)
        engine.stop()

        print(f"音频文件已成功生成: {audio_path}")
        return audio_path

    except Exception as e:
        print(f"在生成音频过程中发生错误: {e}")
        # 尝试停止引擎以防万一
        try:
            engine.stop()
        except:
            pass  # 忽略停止时的错误
        return None
