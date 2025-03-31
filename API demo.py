# OCR demo from Google Gemini Demo
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
