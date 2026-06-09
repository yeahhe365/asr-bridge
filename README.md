# ASR Bridge

[![linux.do](https://shorturl.at/ggSqS)](https://linux.do)

一个将阿里云百炼 Fun-ASR、豆包语音识别等 ASR 服务封装为 OpenAI Whisper 兼容接口的轻量代理服务。

它适合接入 Spokenly 这类支持「OpenAI 兼容 STT / Whisper API」的客户端：客户端仍然按 `/v1/audio/transcriptions` 上传音频，代理内部将请求转换为百炼 Fun-ASR 或豆包语音识别任务，并返回 OpenAI 风格的转写结果。

## ✨ 功能特性

- 🔌 **OpenAI 兼容接口**：支持 `POST /v1/audio/transcriptions`
- 🎙 **Spokenly 友好**：可直接在 Spokenly 的 OpenAI 兼容 API 配置中使用
- ☁️ **百炼 Fun-ASR 后端**：音频会转换为 `data:<mime>;base64,...` 后提交给百炼异步识别接口
- 🌋 **豆包语音识别后端**：支持豆包录音文件识别标准版 2.0，音频会转为 Base64 后走 `submit` / `query`
- 🐋 **Docker Compose 部署**：内置 `Dockerfile` 和 `compose.yaml`
- 🖥 **Web 管理界面**：可在浏览器里配置语言提示、热词表，并上传音频或录音试转写
- 🧩 **模型名兼容**：客户端传入 `whisper-1` 等模型名时，会自动映射到 `fun-asr`
- 🌐 **语言提示**：支持 OpenAI `language` 字段，也支持百炼原生 `language_hints`
- 📝 **自定义热词**：支持创建百炼热词表，并在转写时使用 `vocabulary_id`
- 🕳 **空语音兼容**：百炼返回 `ASR_RESPONSE_HAVE_NO_WORDS` 时，会返回空文本，避免客户端连接测试直接报错
- 🔐 **密钥本地保存**：百炼和豆包密钥只从本地环境变量或 `.env` 读取，不需要填到 Spokenly

## 🚀 快速开始

### 方式 1：直接运行

1. 克隆仓库：

   ```bash
   git clone https://github.com/yeahhe365/asr-bridge.git
   cd asr-bridge
   ```

2. 安装依赖：

   ```bash
   uv sync
   ```

3. 配置模型服务密钥：

   ```bash
   export DASHSCOPE_API_KEY="sk-your-bailian-api-key"
   export DASHSCOPE_LANGUAGE_HINTS=""
   export DASHSCOPE_VOCABULARY_ID=""
   export DOUBAO_API_KEY=""
   export DOUBAO_APP_ID=""
   export DOUBAO_ACCESS_TOKEN=""
   export DOUBAO_RESOURCE_ID="volc.seedasr.auc"
   ```

   `DASHSCOPE_LANGUAGE_HINTS` 留空表示自动检测语言。豆包新版控制台可以只配置 `DOUBAO_API_KEY`；旧版控制台使用 `DOUBAO_APP_ID` + `DOUBAO_ACCESS_TOKEN`。

4. 启动服务：

   ```bash
   uv run python -m asr_bridge
   ```

   默认服务地址为 `http://127.0.0.1:8000`。

### 方式 2：Docker Compose 部署

1. 复制环境变量文件：

   ```bash
   cp .env.example .env
   ```

2. 编辑 `.env`，填入模型服务密钥：

   ```bash
   DASHSCOPE_API_KEY=sk-your-bailian-api-key
   DASHSCOPE_LANGUAGE_HINTS=
   DASHSCOPE_VOCABULARY_ID=
   DOUBAO_API_KEY=
   DOUBAO_APP_ID=
   DOUBAO_ACCESS_TOKEN=
   DOUBAO_RESOURCE_ID=volc.seedasr.auc
   ```

   `DASHSCOPE_LANGUAGE_HINTS` 留空表示自动检测语言。只使用豆包模型时，可以不填 `DASHSCOPE_API_KEY`；只使用百炼 Fun-ASR 时，可以不填豆包密钥。

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
模型: fun-asr 或 doubao-asr
URL: http://localhost:8000
```

其中 `API 密钥` 只用于满足 Spokenly 的输入要求，代理不会使用它调用模型服务。真正的服务端密钥由 `.env` 读取。

## 🖥 Web 管理界面

服务启动后，打开：

```text
http://localhost:8000
```

可以在页面里完成：

- 查看服务状态
- 临时修改运行配置：`language_hints`、`vocabulary_id`
- 创建百炼热词表
- 上传音频文件试转写
- 使用浏览器麦克风录音试转写
- 实时查看服务日志和转写状态

Web 页面不会显示或要求输入百炼 API Key。API Key 仍然只从服务端环境变量读取。

> Web 管理界面里的运行配置只在当前服务进程内生效。实时日志保存在服务进程内存里，重启容器后会清空。浏览器录音需要麦克风权限，并会在前端编码为 WAV 后提交给转写接口。

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

当前内置模型名：

```text
fun-asr
doubao-asr
```

### 运行配置

```text
GET /api/settings
PUT /api/settings
```

请求示例：

```bash
curl -X PUT http://localhost:8000/api/settings \
  -H "Content-Type: application/json" \
  -d '{"language_hints":["zh"],"vocabulary_id":"vocab-stock-123"}'
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

使用豆包录音文件识别标准版 2.0：

```bash
curl http://localhost:8000/v1/audio/transcriptions \
  -H "Authorization: Bearer local" \
  -F model=doubao-asr \
  -F file=@/path/to/audio.wav
```

也可以直接使用百炼原生的语言提示参数：

```bash
curl http://localhost:8000/v1/audio/transcriptions \
  -H "Authorization: Bearer local" \
  -F model=fun-asr \
  -F language_hints=zh \
  -F file=@/path/to/audio.wav
```

`language_hints` 支持逗号分隔或 JSON 数组字符串，例如 `zh,en`、`["zh"]`。优先级为：请求中的 `language_hints` > 请求中的 `language` > `.env` 中的 `DASHSCOPE_LANGUAGE_HINTS`。

豆包后端会使用请求中的 `language_hints` 第一项或 OpenAI `language` 字段作为语言提示，并将常见值如 `zh`、`en` 映射为豆包接口需要的语言代码。留空则交给模型自动判断。

如果已经创建了热词表，可以传入 `vocabulary_id`：

```bash
curl http://localhost:8000/v1/audio/transcriptions \
  -H "Authorization: Bearer local" \
  -F model=fun-asr \
  -F vocabulary_id=vocab-stock-123 \
  -F file=@/path/to/audio.wav
```

`vocabulary_id` 的优先级为：请求中的 `vocabulary_id` > `.env` 中的 `DASHSCOPE_VOCABULARY_ID`。

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

### 创建热词表

```text
POST /v1/audio/vocabularies
POST /audio/vocabularies
```

请求示例：

```bash
curl http://localhost:8000/v1/audio/vocabularies \
  -H "Content-Type: application/json" \
  -d '{
    "prefix": "stock",
    "target_model": "fun-asr",
    "vocabulary": [
      {"text": "英伟达", "weight": 4, "lang": "zh"},
      {"text": "纳斯达克", "weight": 4, "lang": "zh"},
      {"text": "美联储", "weight": 4, "lang": "zh"}
    ]
  }'
```

返回示例：

```json
{"vocabulary_id":"vocab-stock-123","status":"PENDING"}
```

拿到 `vocabulary_id` 后，可以先等百炼侧热词表状态变为可用，再将它填入转写请求或 `.env` 的 `DASHSCOPE_VOCABULARY_ID`。

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
- Web 管理界面没有额外鉴权，请不要直接暴露到公网。
- 浏览器录音需要在 `localhost` 或 HTTPS 环境下使用。
- 当前项目主要面向 Spokenly 和 OpenAI Whisper 兼容客户端，不提供完整的 OpenAI API 网关能力。
- 如果客户端连接测试发送的是静音或极短音频，百炼可能返回 `ASR_RESPONSE_HAVE_NO_WORDS`，本项目会将其转换为空转写结果。

## 📄 开源协议

本项目基于 [MIT License](LICENSE) 开源。

## 🙏 致谢

本项目已在 [LINUX DO](https://linux.do) 社区分享，感谢社区的支持与反馈。
