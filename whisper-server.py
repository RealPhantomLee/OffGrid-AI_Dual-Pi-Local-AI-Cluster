import uvicorn
import tempfile
import os
import whisper
from fastapi import FastAPI, UploadFile, File

app = FastAPI(title="Whisper Server")
model = whisper.load_model("base")


@app.post("/transcribe")
async def transcribe(audio: UploadFile = File(...)):
    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as f:
        f.write(await audio.read())
        tmp_path = f.name
    try:
        result = model.transcribe(tmp_path)
        return {"text": result["text"], "language": result.get("language", "en")}
    finally:
        os.unlink(tmp_path)


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=9000)
