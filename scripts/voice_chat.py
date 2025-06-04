import argparse
import subprocess
import requests
import json
import speech_recognition as sr


def recognize_speech(recognizer, microphone):
    with microphone as source:
        audio = recognizer.listen(source)
    try:
        return recognizer.recognize_google(audio, language='ko-KR')
    except sr.UnknownValueError:
        return ""


def generate_text(prompt):
    cmd = [
        'python', 'run_inference.py',
        '--prompt', prompt,
        '--n-predict', '128'
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    return result.stdout


def tts(text):
    payload = {'text': text}
    resp = requests.post('http://localhost:8000/tts', data=json.dumps(payload))
    if resp.ok:
        with open('output.wav', 'wb') as f:
            f.write(resp.content)


def main():
    parser = argparse.ArgumentParser(description='Voice conversation using BitNet and Parakeet')
    parser.add_argument('--mic', type=int, default=0, help='Microphone device index')
    args = parser.parse_args()

    recognizer = sr.Recognizer()
    microphone = sr.Microphone(device_index=args.mic)

    while True:
        print('Say something...')
        text = recognize_speech(recognizer, microphone)
        if not text:
            print('Could not understand')
            continue
        print('User:', text)
        response = generate_text(text)
        print('Bot:', response)
        tts(response)


if __name__ == '__main__':
    main()
