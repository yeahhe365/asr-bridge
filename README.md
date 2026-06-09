# Fun-ASR OpenAI Proxy

Local OpenAI-compatible transcription proxy for Alibaba Cloud Bailian Fun-ASR.

It is meant for apps such as Spokenly that let you configure an OpenAI-compatible
speech-to-text endpoint.

## Spokenly Settings

In Spokenly's **OpenAI compatible API** dialog, use:

```text
API key: local
Model: fun-asr
URL: http://localhost:8000
```

The API key in Spokenly is only used to satisfy its UI. The proxy uses
`DASHSCOPE_API_KEY` from your local environment to call Bailian.

## Run

Install and start the proxy:

```bash
cd fun-asr-openai-proxy
uv sync
export DASHSCOPE_API_KEY="sk-your-new-bailian-api-key"
uv run python -m fun_asr_openai_proxy
```

The server listens on `http://127.0.0.1:8000`.

## Docker Compose

Create `.env` with your Bailian API key:

```bash
DASHSCOPE_API_KEY=sk-your-new-bailian-api-key
```

Build and start the container:

```bash
cd fun-asr-openai-proxy
docker compose up -d --build
```

Check the service:

```bash
curl http://localhost:8000/health
```

Stop it:

```bash
docker compose down
```

For Spokenly, keep using:

```text
API key: local
Model: fun-asr
URL: http://localhost:8000
```

## API

The proxy exposes:

```text
GET  /health
GET  /v1/models
GET  /models
POST /v1/audio/transcriptions
POST /audio/transcriptions
```

`POST /v1/audio/transcriptions` accepts OpenAI-style multipart form data:

```bash
curl http://localhost:8000/v1/audio/transcriptions \
  -H "Authorization: Bearer local" \
  -F model=fun-asr \
  -F language=zh \
  -F file=@/path/to/audio.wav
```

Default response:

```json
{"text":"识别结果"}
```

Plain text response:

```bash
curl http://localhost:8000/v1/audio/transcriptions \
  -F model=fun-asr \
  -F response_format=text \
  -F file=@/path/to/audio.wav
```

## Notes

- The proxy converts uploaded audio into a `data:<mime>;base64,...` URI and sends
  it to Bailian's asynchronous `fun-asr` transcription API.
- If a client sends `whisper-1` or another Whisper-style model name, the proxy
  maps it to `fun-asr` before calling Bailian.
- Do not put your Bailian API key into this repository or Spokenly. Keep it in
  the local `DASHSCOPE_API_KEY` environment variable.

## License

MIT
