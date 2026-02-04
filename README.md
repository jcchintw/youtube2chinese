# YouTube to Chinese Dubbing Pipeline 🎬🇹🇼

自動將 YouTube 影片轉換為中文配音版本，包含中文字幕。

[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

## ✨ 功能特色

- **一鍵處理**：從 YouTube URL 直接產出中文配音影片
- **高品質轉錄**：使用 whisper.cpp（本地運行，無需 API）
- **智慧翻譯**：支援 OpenAI GPT-4o / Google Gemini（適應性翻譯，符合語速）
- **自然語音**：Edge-TTS 台灣中文語音（免費、無需 API Key）
- **時間對齊**：自動調整語速，確保音畫同步
- **多版本輸出**：
  - 中文配音 + 中文字幕
  - 原音 + 中文字幕
  - 原音 + 英文字幕

## 📋 系統需求

- Python 3.9+
- ffmpeg
- yt-dlp
- whisper.cpp（自動下載模型）

## 🚀 快速開始

### 1. 安裝依賴

```bash
# Clone 專案
git clone https://github.com/jcchintw/youtube2chinese.git
cd youtube2chinese

# 安裝 Python 依賴
pip install -r requirements.txt

# 安裝系統工具（macOS）
brew install ffmpeg yt-dlp

# 安裝 whisper.cpp（如尚未安裝）
git clone https://github.com/ggerganov/whisper.cpp.git ~/.whisper.cpp
cd ~/.whisper.cpp && make
```

### 2. 設定翻譯來源（三選一）

**選項 A：OpenClaw（推薦，如已安裝）**
```bash
# 無需設定 API Key，直接使用 OpenClaw 配置的 LLM
python y2c.py video.mp4 -o ./output --translator openclaw
```

**選項 B：OpenAI**
```bash
export OPENAI_API_KEY="your-api-key"
python y2c.py video.mp4 -o ./output --translator openai
```

**選項 C：Google Gemini**
```bash
export GOOGLE_API_KEY="your-api-key"
python y2c.py video.mp4 -o ./output --translator gemini
```

### 3. 執行

```bash
# 基本用法（YouTube URL）
python y2c.py "https://www.youtube.com/watch?v=VIDEO_ID" -o ./output

# 使用本地影片
python y2c.py video.mp4 -o ./output

# 使用 Gemini 翻譯
python y2c.py video.mp4 -o ./output --translator gemini

# 自訂選項
python y2c.py video.mp4 -o ./output \
    --whisper-model large \
    --tts-voice zh-CN-XiaoxiaoNeural \
    --translator openai \
    --translation-model gpt-4o
```

## 📁 輸出結構

```
output/
├── downloads/          # 下載的原始影片
├── subtitles/
│   ├── english.srt     # 英文字幕
│   └── chinese.srt     # 中文字幕
├── tts/                # TTS 音檔
├── translated.json     # 翻譯資料
├── aligned_chinese.wav # 對齊後的中文音軌
└── output/
    ├── chinese_dubbed.mp4         # 中文配音版
    ├── chinese_subtitles_only.mp4 # 原音+中文字幕
    └── english_subtitles.mp4      # 原音+英文字幕
```

## ⚙️ 參數說明

| 參數 | 預設值 | 說明 |
|------|--------|------|
| `input` | (必填) | YouTube URL 或本地影片路徑 |
| `-o, --output-dir` | `./y2c_output` | 輸出目錄 |
| `--api-key` | 環境變數 | 翻譯 API Key |
| `--translator` | `openai` | 翻譯提供者 (openai/gemini/openclaw) |
| `--translation-model` | 自動 | 翻譯模型 |
| `--whisper-model` | `medium` | Whisper 模型 (tiny/base/small/medium/large) |
| `--whisper-cpp-path` | `~/.whisper.cpp` | whisper.cpp 路徑 |
| `--tts-voice` | `zh-TW-YunJheNeural` | Edge-TTS 語音 |

## 🎙️ 可用 TTS 語音

### 台灣中文
- `zh-TW-YunJheNeural` (男聲，預設)
- `zh-TW-HsiaoChenNeural` (女聲)

### 中國中文
- `zh-CN-XiaoxiaoNeural` (女聲)
- `zh-CN-YunxiNeural` (男聲)

查看所有可用語音：
```bash
edge-tts --list-voices | grep zh
```

## 🔧 Pipeline 流程

```
YouTube URL
    │
    ▼
┌─────────────────┐
│  1. yt-dlp      │ ─── 下載影片
└─────────────────┘
    │
    ▼
┌─────────────────┐
│  2. whisper.cpp │ ─── 語音轉文字 (STT)
└─────────────────┘
    │
    ▼
┌─────────────────┐
│  3. LLM API     │ ─── 適應性翻譯
└─────────────────┘
    │
    ▼
┌─────────────────┐
│  4. Edge-TTS    │ ─── 中文語音合成
└─────────────────┘
    │
    ▼
┌─────────────────┐
│  5. ffmpeg      │ ─── 音軌對齊
└─────────────────┘
    │
    ▼
┌─────────────────┐
│  6. ffmpeg      │ ─── 影片合成
└─────────────────┘
    │
    ▼
Chinese Dubbed Video 🎉
```

## 📝 注意事項

1. **翻譯品質**：使用「適應性翻譯」，會根據時間長度調整譯文長度，確保語速自然
2. **語速調整**：TTS 音檔最多加速 1.25 倍，超過則保留原速
3. **Whisper 模型**：
   - `medium`：平衡速度與品質（推薦）
   - `large`：最高品質，但速度較慢
4. **Edge-TTS**：免費服務，無需 API Key，但依賴網路連線

## 🤝 貢獻

歡迎提交 Issue 和 Pull Request！

## 📄 License

MIT License

## 🙏 致謝

- [whisper.cpp](https://github.com/ggerganov/whisper.cpp) - 高效的本地語音辨識
- [yt-dlp](https://github.com/yt-dlp/yt-dlp) - YouTube 下載工具
- [Edge-TTS](https://github.com/rany2/edge-tts) - Microsoft Edge 語音合成
- [ffmpeg](https://ffmpeg.org/) - 影音處理

---

Made with ❤️ by JCBOT
