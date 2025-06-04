import argparse
import subprocess
import requests
import json

try:
    import speech_recognition as sr
    HAS_SR = True
except Exception:  # covers missing PyAudio or speech_recognition
    sr = None
    HAS_SR = False


def recognize_speech(recognizer=None, microphone=None):
    if HAS_SR and recognizer and microphone:
        with microphone as source:
            audio = recognizer.listen(source)
        try:
            return recognizer.recognize_google(audio, language='ko-KR')
        except sr.UnknownValueError:
            return ""
    else:
        # Fallback to text input when speech recognition is unavailable
        try:
            return input()
        except EOFError:
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
    try:
        resp = requests.post('http://localhost:8000/tts', data=json.dumps(payload))
        if resp.ok:
            with open('output.wav', 'wb') as f:
                f.write(resp.content)
    except requests.RequestException:
        print('TTS service unavailable')


def main():
    parser = argparse.ArgumentParser(description='Voice conversation using BitNet and Parakeet')
    parser.add_argument('--mic', type=int, default=0, help='Microphone device index')
    args = parser.parse_args()

    recognizer = sr.Recognizer() if HAS_SR and args.mic >= 0 else None
    microphone = sr.Microphone(device_index=args.mic) if HAS_SR and args.mic >= 0 else None
    use_sr = recognizer is not None and microphone is not None

    while True:
        print('Say something...')
        text = recognize_speech(recognizer, microphone)
        if not text:
            print('Could not understand')
            if not use_sr:
                break
            continue
        print('User:', text)
        response = generate_text(text)
        print('Bot:', response)
        tts(response)
        if not use_sr:
            break


if __name__ == '__main__':
    main()
