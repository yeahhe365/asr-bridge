# Fun-ASR OpenAI Proxy

[![linux.do](https://shorturl.at/ggSqS)](https://linux.do)

一个将阿里云百炼 Fun-ASR 封装为 OpenAI Whisper 兼容接口的轻量代理服务。

它适合接入 Spokenly 这类支持「OpenAI 兼容 STT / Whisper API」的客户端：客户端仍然按 `/v1/audio/transcriptions` 上传音频，代理内部将请求转换为百炼 Fun-ASR 的异步识别任务，并返回 OpenAI 风格的转写结果。

## ✨ 功能特性

- 🔌 **OpenAI 兼容接口**：支持 `POST /v1/audio/transcriptions`
- 🎙 **Spokenly 友好**：可直接在 Spokenly 的 OpenAI 兼容 API 配置中使用
- ☁️ **百炼 Fun-ASR 后端**：音频会转换为 `data:<mime>;base64,...` 后提交给百炼异步识别接口
- 🐋 **Docker Compose 部署**：内置 `Dockerfile` 和 `compose.yaml`
- 🧩 **模型名兼容**：客户端传入 `whisper-1` 等模型名时，会自动映射到 `fun-asr`
- 🕳 **空语音兼容**：百炼返回 `ASR_RESPONSE_HAVE_NO_WORDS` 时，会返回空文本，避免客户端连接测试直接报错
- 🔐 **密钥本地保存**：百炼 API Key 只从本地环境变量或 `.env` 读取，不需要填到 Spokenly

## 🚀 快速开始

### 方式 1：直接运行

1. 克隆仓库：

   ```bash
   git clone https://github.com/yeahhe365/fun-asr-openai-proxy.git
   cd fun-asr-openai-proxy
   ```

2. 安装依赖：

   ```bash
   uv sync
   ```

3. 配置百炼 API Key：

   ```bash
   export DASHSCOPE_API_KEY="sk-your-bailian-api-key"
   ```

4. 启动服务：

   ```bash
   uv run python -m fun_asr_openai_proxy
   ```

   默认服务地址为 `http://127.0.0.1:8000`。

### 方式 2：Docker Compose 部署

1. 复制环境变量文件：

   ```bash
   cp .env.example .env
   ```

2. 编辑 `.env`，填入百炼 API Key：

   ```bash
   DASHSCOPE_API_KEY=sk-your-bailian-api-key
   ```

3. 构建并启动容器：

   ```bash
   docker compose up -d --build
   ```

4. 检查服务状态：

   ```bash
   curl http://localhost:8000/health
   ```

   正常返回：

   ```json
   {"status":"ok"}
   ```

5. 停止服务：

   ```bash
   docker compose down
   ```

## 🎙 Spokenly 配置

在 Spokenly 的 **OpenAI 兼容 API** 配置中填写：

```text
API 密钥: local
模型: fun-asr
URL: http://localhost:8000
```

其中 `API 密钥` 只用于满足 Spokenly 的输入要求，代理不会使用它调用百炼。真正的百炼 API Key 由服务端读取 `DASHSCOPE_API_KEY`。

## 📡 API 说明

### 健康检查

```text
GET /health
```

### 模型列表

```text
GET /v1/models
GET /models
```

### 音频转写

```text
POST /v1/audio/transcriptions
POST /audio/transcriptions
```

请求示例：

```bash
curl http://localhost:8000/v1/audio/transcriptions \
  -H "Authorization: Bearer local" \
  -F model=fun-asr \
  -F language=zh \
  -F file=@/path/to/audio.wav
```

默认返回：

```json
{"text":"识别结果"}
```

如果需要纯文本返回：

```bash
curl http://localhost:8000/v1/audio/transcriptions \
  -F model=fun-asr \
  -F response_format=text \
  -F file=@/path/to/audio.wav
```

## 🧪 测试与校验

运行完整测试：

```bash
uv run pytest -q
```

本地 smoke test：

```bash
scripts/smoke_local.sh
```

> `scripts/smoke_local.sh` 会向 `http://127.0.0.1:8000/v1/audio/transcriptions` 上传一段测试音频，因此需要先启动服务。

## ⚠️ 注意事项

- 请不要把真实的百炼 API Key 提交到仓库。
- `.env` 已被 `.gitignore` 和 `.dockerignore` 排除。
- Fun-ASR 的计费由阿里云百炼侧决定，本项目只做协议转换。
- 当前项目主要面向 Spokenly 和 OpenAI Whisper 兼容客户端，不提供完整的 OpenAI API 网关能力。
- 如果客户端连接测试发送的是静音或极短音频，百炼可能返回 `ASR_RESPONSE_HAVE_NO_WORDS`，本项目会将其转换为空转写结果。

## 📄 开源协议

本项目基于 [MIT License](LICENSE) 开源。

## 🙏 致谢

本项目已在 [LINUX DO](https://linux.do) 社区分享，感谢社区的支持与反馈。
