from fastapi import FastAPI, Request
from fastapi.responses import PlainTextResponse
import uvicorn
from useful_transformers import WhisperModel, decode_pcm
import useful_transformers
import io
import librosa
import soundfile as sf

print("Loading model...")
model = WhisperModel()
print("Model loaded!")

app = FastAPI()

@app.post("/transcribe")
async def transcribe(request: Request):
    body = await request.body()
    audio, sr = sf.read(io.BytesIO(body))
    if sr != 16000:
        audio = librosa.resample(audio, orig_sr=sr, target_sr=16000)
    text = decode_pcm(audio, model)
    print(text)

    return PlainTextResponse(text)

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=4997)
