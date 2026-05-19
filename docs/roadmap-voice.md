# Roadmap: Voice Pipeline

> **Status: STT is operational. TTS and wake-word are not yet implemented.**
> Whisper STT is running on port 9000 and accepting audio uploads. This document covers completing the full hands-free voice interface.

## Current State

| Component | Status | Notes |
|---|---|---|
| Speech-to-Text (Whisper) | ✅ Running on port 9000 | `whisper-server.service`, openai-whisper base model |
| Text-to-Speech (Piper) | ❌ Not installed | Fast ONNX TTS, excellent ARM support |
| Wake Word Detection | ❌ Not installed | OpenWakeWord, CPU-only, always-on |
| Voice orchestration | ❌ Not wired | `voice` agent stub exists in `agent-hub/main.py` |

## Target Architecture

```
Microphone (USB)
    │  always listening (low-power wake word loop)
    ▼
OpenWakeWord — "hey aipi" / "hey jarvis"
    │  trigger on detection (confidence > 0.5)
    ▼
Audio capture (5 seconds WAV)
    │
    ▼
Whisper STT (:9000/transcribe)
    │  returns transcript text
    ▼
Agent Hub — voice agent (:8000/chat with voice=true flag)
    │  routes to phi3.5 (fast) or Mistral (complex)
    ▼
LLM response text
    │
    ▼
Piper TTS → WAV audio
    │
    ▼
Speaker (USB / 3.5mm)
```

## Hardware Required

| Item | Recommendation | Cost |
|---|---|---|
| USB Microphone | Fifine K053, TONOR TC30, or any USB mic | $15–25 |
| USB Speaker | Anker Soundcore Mini, or 3.5mm audio output | $15–30 |
| Combo option | ReSpeaker 2-Mic HAT (RPi HAT with mic array) | $20 |

> **Note:** The ReSpeaker HAT uses the 40-pin GPIO header — confirm it doesn't conflict with the Hailo-8 HAT if using both.

## Testing Whisper STT (Already Running)

```bash
# Record a test clip
arecord -f cd -t wav -d 5 test.wav

# Transcribe
curl -X POST http://localhost:9000/transcribe \
  -F "audio=@test.wav" | python3 -m json.tool
# {"text": "hello this is a test", "language": "en"}
```

## Installing Piper TTS

```bash
source ~/venv/bin/activate
pip install piper-tts

# Download English voice model
mkdir -p ~/models/piper
wget "https://huggingface.co/rhasspy/piper-voices/resolve/main/en/en_US/jenny/medium/en_US-jenny-medium.onnx" \
  -O ~/models/piper/en_US-jenny-medium.onnx
wget "https://huggingface.co/rhasspy/piper-voices/resolve/main/en/en_US/jenny/medium/en_US-jenny-medium.onnx.json" \
  -O ~/models/piper/en_US-jenny-medium.onnx.json

# Test synthesis
echo "Hello, I am your local AI assistant." | \
  ~/venv/bin/piper \
    --model ~/models/piper/en_US-jenny-medium.onnx \
    --output_file /tmp/test_tts.wav && \
  aplay /tmp/test_tts.wav
```

Other available voices: `en_US-lessac-medium` (male), `en_GB-alan-medium` (British), and many others at `rhasspy/piper-voices` on HuggingFace.

## Installing OpenWakeWord

```bash
source ~/venv/bin/activate
pip install openwakeword pyaudio

# Test — listens for "hey jarvis"
python3 -c "
import openwakeword, numpy as np
from openwakeword.model import Model
model = Model(wakeword_models=['hey_jarvis'])
print('Wake word model loaded. Say: hey jarvis')
"
```

Custom wake words ("hey aipi") can be trained with OpenWakeWord's training pipeline using ~30 minutes of audio samples.

## Voice Orchestration Script

```python
# /home/merry/voice-pipeline/voice.py
import asyncio, io, wave, numpy as np
import pyaudio, httpx, subprocess
from openwakeword.model import Model
from piper.voice import PiperVoice

WHISPER_URL = "http://localhost:9000/transcribe"
AGENT_URL   = "http://localhost:8000/chat"
PIPER_MODEL = "/home/merry/models/piper/en_US-jenny-medium.onnx"

RATE, CHUNK, RECORD_SECS = 16000, 1280, 5

wakeword_model = Model(wakeword_models=["hey_jarvis"])
tts_voice = PiperVoice.load(PIPER_MODEL)

async def transcribe(wav_bytes: bytes) -> str:
    async with httpx.AsyncClient(timeout=30) as c:
        r = await c.post(WHISPER_URL,
                         files={"audio": ("audio.wav", wav_bytes, "audio/wav")})
    return r.json().get("text", "")

async def get_response(text: str) -> str:
    async with httpx.AsyncClient(timeout=120) as c:
        r = await c.post(AGENT_URL, json={"message": text})
    return r.json().get("response", "Sorry, I didn't catch that.")

def speak(text: str):
    buf = io.BytesIO()
    with wave.open(buf, "wb") as wf:
        wf.setnchannels(1); wf.setsampwidth(2); wf.setframerate(22050)
        for audio in tts_voice.synthesize_stream_raw(text):
            wf.writeframes(audio)
    buf.seek(0)
    subprocess.run(["aplay", "-q", "-"], input=buf.read())

def capture_audio(stream) -> bytes:
    frames = [stream.read(CHUNK) for _ in range(int(RATE / CHUNK * RECORD_SECS))]
    buf = io.BytesIO()
    pa = pyaudio.PyAudio()
    with wave.open(buf, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(pa.get_sample_size(pyaudio.paInt16))
        wf.setframerate(RATE)
        wf.writeframes(b"".join(frames))
    return buf.getvalue()

async def main():
    pa = pyaudio.PyAudio()
    stream = pa.open(rate=RATE, channels=1, format=pyaudio.paInt16,
                     input=True, frames_per_buffer=CHUNK)
    print("Listening for wake word...")
    while True:
        chunk = np.frombuffer(stream.read(CHUNK), dtype=np.int16)
        pred = wakeword_model.predict_chunk(chunk)
        if pred.get("hey_jarvis", 0) > 0.5:
            print("Wake word detected!")
            wav = capture_audio(stream)
            transcript = await transcribe(wav)
            if transcript.strip():
                print(f"You: {transcript}")
                response = await get_response(transcript)
                print(f"AIPI: {response}")
                speak(response)

asyncio.run(main())
```

## Implementation Steps

1. Acquire USB microphone and speaker
2. Install Piper TTS: `pip install piper-tts` + download voice model
3. Install OpenWakeWord: `pip install openwakeword pyaudio`
4. Test each component individually (STT, TTS, wake word detection)
5. Wire `voice.py` orchestration script, test full loop
6. Create `voice-pipeline.service` systemd unit (see below)
7. Enable at boot: `sudo systemctl enable --now voice-pipeline`
8. (Optional) Train custom "hey aipi" wake word with OpenWakeWord

## Planned Systemd Service

```ini
[Unit]
Description=OffGridAI Voice Pipeline
After=whisper-server.service agent-hub.service
Requires=whisper-server.service

[Service]
Type=simple
User=merry
WorkingDirectory=/home/merry/voice-pipeline
Environment="VIRTUAL_ENV=/home/merry/venv"
Environment="PATH=/home/merry/venv/bin:/usr/local/bin:/usr/bin:/bin"
ExecStart=/home/merry/venv/bin/python3 voice.py
Restart=on-failure
RestartSec=5
StandardOutput=journal
StandardError=journal

[Install]
WantedBy=multi-user.target
```
