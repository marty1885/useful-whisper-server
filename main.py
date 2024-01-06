from bottle import route, run, request, post
from useful_transformers import WhisperModel, decode_pcm
import useful_transformers
import io
import librosa
import soundfile as sf

print("Loading model...")
model = WhisperModel()
print("Model loaded!")

@post('/transcribe')
def transcribe():
    body = request.body.read()
    audio, sr = sf.read(io.BytesIO(body))
    if sr != 16000:
        audio = librosa.resample(audio, orig_sr=sr, target_sr=16000)
    text = decode_pcm(audio, model)
    print(text)

    return text

run(host='0.0.0.0', port=4997)
