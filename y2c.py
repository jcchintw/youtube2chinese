#!/usr/bin/env python3
"""
YouTube to Chinese Dubbing Pipeline v1.0

A complete pipeline to download YouTube videos, transcribe, translate to Chinese,
generate TTS audio, and produce dubbed videos with Chinese subtitles.
"""

import argparse
import json
import os
import re
import subprocess
import sys
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed

# ============================================================================
# Configuration
# ============================================================================

DEFAULT_WHISPER_MODEL = "medium"
DEFAULT_TTS_VOICE = "zh-TW-YunJheNeural"
DEFAULT_TRANSLATOR = "openai"
MAX_TTS_SPEEDUP = 1.25
SILENCE_SAMPLE_RATE = 24000

# ============================================================================
# Utility Functions
# ============================================================================

def run_cmd(cmd, check=True, capture=False):
    """Run a shell command."""
    if capture:
        result = subprocess.run(cmd, shell=True, capture_output=True, text=True, check=check)
        return result.stdout.strip()
    else:
        subprocess.run(cmd, shell=True, check=check)

def srt_time_to_sec(t):
    """Convert SRT timestamp to seconds."""
    h, m, rest = t.split(':')
    s, ms = rest.split(',')
    return int(h) * 3600 + int(m) * 60 + int(s) + int(ms) / 1000.0

def sec_to_srt_time(sec):
    """Convert seconds to SRT timestamp."""
    h = int(sec // 3600)
    m = int((sec % 3600) // 60)
    s = int(sec % 60)
    ms = int((sec - int(sec)) * 1000)
    return f"{h:02d}:{m:02d}:{s:02d},{ms:03d}"

def get_audio_duration(path):
    """Get audio file duration in seconds."""
    cmd = f'ffprobe -v error -show_entries format=duration -of default=nw=1:nk=1 "{path}"'
    return float(run_cmd(cmd, capture=True))

# ============================================================================
# Step 1: Download Video
# ============================================================================

def download_video(url, output_dir):
    """Download video from YouTube using yt-dlp."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    output_template = output_dir / "%(title)s.%(ext)s"
    cmd = f'yt-dlp -f "bestvideo[height<=720]+bestaudio/best[height<=720]" --merge-output-format mp4 -o "{output_template}" "{url}"'
    run_cmd(cmd)
    
    # Find the downloaded file
    mp4_files = list(output_dir.glob("*.mp4"))
    if mp4_files:
        return str(mp4_files[-1])
    raise FileNotFoundError("No MP4 file found after download")

# ============================================================================
# Step 2: Speech-to-Text (Whisper.cpp)
# ============================================================================

def extract_audio(video_path, output_path):
    """Extract audio from video as 16kHz mono WAV for Whisper."""
    cmd = f'ffmpeg -y -i "{video_path}" -ar 16000 -ac 1 -c:a pcm_s16le "{output_path}"'
    run_cmd(cmd)
    return output_path

def transcribe_whisper_cpp(audio_path, output_dir, model="medium", whisper_cpp_path=None):
    """Transcribe audio using whisper.cpp."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    if whisper_cpp_path is None:
        whisper_cpp_path = os.path.expanduser("~/.openclaw/workspace/whisper.cpp")
    
    model_path = f"{whisper_cpp_path}/models/ggml-{model}.bin"
    if not os.path.exists(model_path):
        print(f"Downloading whisper model: {model}")
        run_cmd(f"cd {whisper_cpp_path} && bash models/download-ggml-model.sh {model}")
    
    base_name = Path(audio_path).stem
    output_base = output_dir / base_name
    
    cmd = f'cd {whisper_cpp_path} && ./main -m "{model_path}" -f "{audio_path}" -osrt -of "{output_base}"'
    run_cmd(cmd)
    
    srt_path = f"{output_base}.srt"
    return srt_path

def parse_srt(srt_path):
    """Parse SRT file into list of segments."""
    with open(srt_path, encoding='utf-8') as f:
        data = f.read()
    
    pattern = r'(\d+)\n(\d{2}:\d{2}:\d{2},\d{3}) --> (\d{2}:\d{2}:\d{2},\d{3})\n(.+?)(?=\n\n|\Z)'
    matches = re.findall(pattern, data, re.DOTALL)
    
    segments = []
    for idx, start, end, text in matches:
        segments.append({
            'id': idx,
            'start': start,
            'end': end,
            'duration': srt_time_to_sec(end) - srt_time_to_sec(start),
            'text': text.replace('\n', ' ').strip()
        })
    return segments

# ============================================================================
# Step 3: Translation
# ============================================================================

def translate_openai(segments, api_key, model="gpt-4o", batch_size=20):
    """Translate segments using OpenAI API."""
    import openai
    client = openai.OpenAI(api_key=api_key)
    
    def translate_batch(batch):
        prompt = """You are a professional subtitle translator. Translate the following subtitles into Traditional Chinese (Taiwan).
REQUIREMENTS:
1. Adaptive Translation: Keep the meaning but fit the time duration. Be concise and natural (spoken style).
2. Output JSON ONLY: Return a list of objects with 'id' and 'text_zh'. No markdown, no explanation.
3. Content:
"""
        input_list = [{"id": seg["id"], "duration": f"{seg['duration']:.1f}s", "text": seg["text"]} for seg in batch]
        prompt += json.dumps(input_list, ensure_ascii=False)
        
        response = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            response_format={"type": "json_object"}
        )
        
        result = json.loads(response.choices[0].message.content)
        return result.get('translations', result) if isinstance(result, dict) else result
    
    results = []
    batches = [segments[i:i+batch_size] for i in range(0, len(segments), batch_size)]
    
    with ThreadPoolExecutor(max_workers=5) as executor:
        futures = {executor.submit(translate_batch, b): b for b in batches}
        for i, future in enumerate(as_completed(futures)):
            print(f"Translated batch {i+1}/{len(batches)}")
            batch_result = future.result()
            for res in batch_result:
                seg = next((s for s in segments if str(s['id']) == str(res.get('id'))), None)
                if seg:
                    seg['text_zh'] = res.get('text_zh', '')
                    results.append(seg)
    
    results.sort(key=lambda x: int(x['id']))
    return results

def translate_openclaw(segments, batch_size=20):
    """Translate segments using OpenClaw's configured LLM (via CLI)."""
    
    def translate_batch(batch):
        prompt = """You are a professional subtitle translator. Translate the following subtitles into Traditional Chinese (Taiwan).
REQUIREMENTS:
1. Adaptive Translation: Keep the meaning but fit the time duration. Be concise and natural (spoken style).
2. Output JSON ONLY: Return a list of objects with 'id' and 'text_zh'. No markdown, no explanation.
3. Content:
"""
        input_list = [{"id": seg["id"], "duration": f"{seg['duration']:.1f}s", "text": seg["text"]} for seg in batch]
        prompt += json.dumps(input_list, ensure_ascii=False)
        
        # Use openclaw CLI to send message and get response
        import tempfile
        with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
            f.write(prompt)
            prompt_file = f.name
        
        try:
            # Call openclaw gateway API via CLI
            cmd = f'openclaw send --file "{prompt_file}" --wait --json 2>/dev/null'
            result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
            
            if result.returncode != 0:
                # Fallback: use curl to gateway API
                import urllib.request
                import urllib.parse
                
                gateway_url = os.environ.get("OPENCLAW_GATEWAY_URL", "http://localhost:18789")
                gateway_token = os.environ.get("OPENCLAW_GATEWAY_TOKEN", "")
                
                headers = {"Content-Type": "application/json"}
                if gateway_token:
                    headers["Authorization"] = f"Bearer {gateway_token}"
                
                data = json.dumps({"message": prompt, "model": "gemini-high"}).encode()
                req = urllib.request.Request(f"{gateway_url}/api/chat", data=data, headers=headers)
                
                with urllib.request.urlopen(req, timeout=120) as resp:
                    response_data = json.loads(resp.read().decode())
                    text = response_data.get("content", response_data.get("text", ""))
            else:
                text = result.stdout
            
            # Parse JSON from response
            json_match = re.search(r'\[.*\]', text, re.DOTALL)
            if json_match:
                return json.loads(json_match.group(0))
            return json.loads(text)
        finally:
            os.unlink(prompt_file)
    
    results = []
    batches = [segments[i:i+batch_size] for i in range(0, len(segments), batch_size)]
    
    for i, batch in enumerate(batches):
        print(f"Translated batch {i+1}/{len(batches)} (via OpenClaw)")
        batch_result = translate_batch(batch)
        for res in batch_result:
            seg = next((s for s in segments if str(s['id']) == str(res.get('id'))), None)
            if seg:
                seg['text_zh'] = res.get('text_zh', '')
                results.append(seg)
    
    results.sort(key=lambda x: int(x['id']))
    return results

def translate_gemini(segments, api_key, model="gemini-1.5-pro", batch_size=20):
    """Translate segments using Google Gemini API."""
    import google.generativeai as genai
    genai.configure(api_key=api_key)
    model_client = genai.GenerativeModel(model)
    
    def translate_batch(batch):
        prompt = """You are a professional subtitle translator. Translate the following subtitles into Traditional Chinese (Taiwan).
REQUIREMENTS:
1. Adaptive Translation: Keep the meaning but fit the time duration. Be concise and natural (spoken style).
2. Output JSON ONLY: Return a list of objects with 'id' and 'text_zh'. No markdown, no explanation.
3. Content:
"""
        input_list = [{"id": seg["id"], "duration": f"{seg['duration']:.1f}s", "text": seg["text"]} for seg in batch]
        prompt += json.dumps(input_list, ensure_ascii=False)
        
        response = model_client.generate_content(prompt)
        text = response.text
        json_match = re.search(r'\[.*\]', text, re.DOTALL)
        if json_match:
            return json.loads(json_match.group(0))
        return json.loads(text)
    
    results = []
    batches = [segments[i:i+batch_size] for i in range(0, len(segments), batch_size)]
    
    for i, batch in enumerate(batches):
        print(f"Translated batch {i+1}/{len(batches)}")
        batch_result = translate_batch(batch)
        for res in batch_result:
            seg = next((s for s in segments if str(s['id']) == str(res.get('id'))), None)
            if seg:
                seg['text_zh'] = res.get('text_zh', '')
                results.append(seg)
    
    results.sort(key=lambda x: int(x['id']))
    return results

# ============================================================================
# Step 4: Text-to-Speech (Edge-TTS)
# ============================================================================

def generate_tts(segments, output_dir, voice=DEFAULT_TTS_VOICE):
    """Generate TTS audio for each segment using edge-tts."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    for i, seg in enumerate(segments):
        out_path = output_dir / f"seg_{int(seg['id']):04d}.mp3"
        if out_path.exists():
            continue
        
        text = seg.get('text_zh', '').strip()
        if not text:
            continue
        
        cmd = f'edge-tts --text "{text}" --voice {voice} --write-media "{out_path}"'
        run_cmd(cmd, check=False)
        
        if (i + 1) % 20 == 0:
            print(f"TTS progress: {i+1}/{len(segments)}")
    
    print(f"TTS completed: {len(segments)} segments")
    return output_dir

# ============================================================================
# Step 5: Audio Alignment
# ============================================================================

def align_audio(segments, audio_dir, output_path):
    """Align TTS audio to original timestamps."""
    audio_dir = Path(audio_dir)
    output_path = Path(output_path)
    
    # Sort segments by start time
    segments = sorted(segments, key=lambda x: srt_time_to_sec(x['start']))
    
    concat_list = []
    last_end = 0.0
    temp_files = []
    
    for seg in segments:
        seg_id = int(seg['id'])
        start = srt_time_to_sec(seg['start'])
        end = srt_time_to_sec(seg['end'])
        target_duration = max(0.0, end - start)
        
        audio_path = audio_dir / f"seg_{seg_id:04d}.mp3"
        if not audio_path.exists():
            continue
        
        # Add silence gap if needed
        if start > last_end + 0.01:
            gap = start - last_end
            silence_path = audio_dir / f"silence_{int(last_end*1000)}.wav"
            if not silence_path.exists():
                cmd = f'ffmpeg -y -f lavfi -i anullsrc=channel_layout=mono:sample_rate={SILENCE_SAMPLE_RATE} -t {gap} "{silence_path}"'
                run_cmd(cmd)
                temp_files.append(silence_path)
            concat_list.append(str(silence_path))
            last_end = start
        
        # Adjust audio speed if needed
        orig_duration = get_audio_duration(str(audio_path))
        speed = 1.0
        if orig_duration > target_duration and target_duration > 0.05:
            speed = min(MAX_TTS_SPEEDUP, orig_duration / target_duration)
        
        adj_path = audio_dir / f"seg_{seg_id:04d}_adj.wav"
        if abs(speed - 1.0) > 0.01:
            cmd = f'ffmpeg -y -i "{audio_path}" -filter:a "atempo={speed}" -ar {SILENCE_SAMPLE_RATE} "{adj_path}"'
        else:
            cmd = f'ffmpeg -y -i "{audio_path}" -ar {SILENCE_SAMPLE_RATE} "{adj_path}"'
        run_cmd(cmd)
        temp_files.append(adj_path)
        
        adj_duration = get_audio_duration(str(adj_path))
        concat_list.append(str(adj_path))
        last_end += adj_duration
        
        # Add padding if audio is shorter than target
        pad = max(0.0, target_duration - adj_duration)
        if pad > 0.01:
            pad_path = audio_dir / f"pad_{seg_id:04d}.wav"
            cmd = f'ffmpeg -y -f lavfi -i anullsrc=channel_layout=mono:sample_rate={SILENCE_SAMPLE_RATE} -t {pad} "{pad_path}"'
            run_cmd(cmd)
            temp_files.append(pad_path)
            concat_list.append(str(pad_path))
            last_end += pad
    
    # Concatenate all audio
    list_file = audio_dir / "concat_list.txt"
    with open(list_file, 'w') as f:
        for path in concat_list:
            f.write(f"file '{path}'\n")
    
    cmd = f'ffmpeg -y -f concat -safe 0 -i "{list_file}" -c copy "{output_path}"'
    run_cmd(cmd)
    
    print(f"Aligned audio saved to: {output_path}")
    return output_path

# ============================================================================
# Step 6: Video Synthesis
# ============================================================================

def create_chinese_srt(segments, output_path):
    """Create Chinese SRT file from translated segments."""
    with open(output_path, 'w', encoding='utf-8') as f:
        for i, seg in enumerate(segments, 1):
            f.write(f"{i}\n{seg['start']} --> {seg['end']}\n{seg.get('text_zh', '')}\n\n")
    return output_path

def synthesize_video(video_path, audio_path, srt_path, output_path):
    """Combine video with Chinese audio and subtitles."""
    cmd = f'''ffmpeg -y -i "{video_path}" -i "{audio_path}" -i "{srt_path}" \
        -map 0:v -map 1:a -map 2:s \
        -c:v copy -c:a aac -c:s mov_text \
        -metadata:s:s:0 language=chi \
        "{output_path}"'''
    run_cmd(cmd)
    print(f"Output video: {output_path}")
    return output_path

def synthesize_subtitle_only(video_path, srt_path, output_path, language="chi"):
    """Add subtitles to video without changing audio."""
    cmd = f'''ffmpeg -y -i "{video_path}" -i "{srt_path}" \
        -map 0:v -map 0:a -map 1:s \
        -c:v copy -c:a copy -c:s mov_text \
        -metadata:s:s:0 language={language} \
        "{output_path}"'''
    run_cmd(cmd)
    return output_path

# ============================================================================
# Main Pipeline
# ============================================================================

def run_pipeline(args):
    """Run the complete pipeline."""
    work_dir = Path(args.output_dir)
    work_dir.mkdir(parents=True, exist_ok=True)
    
    # Step 1: Download or use existing video
    if args.input.startswith('http'):
        print("Step 1: Downloading video...")
        video_path = download_video(args.input, work_dir / "downloads")
    else:
        video_path = args.input
    print(f"Video: {video_path}")
    
    # Step 2: Transcribe
    print("Step 2: Transcribing audio...")
    audio_path = work_dir / "audio.wav"
    extract_audio(video_path, str(audio_path))
    srt_path = transcribe_whisper_cpp(
        str(audio_path), 
        work_dir / "subtitles",
        model=args.whisper_model,
        whisper_cpp_path=args.whisper_cpp_path
    )
    segments = parse_srt(srt_path)
    print(f"Transcribed: {len(segments)} segments")
    
    # Save English SRT
    english_srt = work_dir / "subtitles" / "english.srt"
    run_cmd(f'cp "{srt_path}" "{english_srt}"')
    
    # Step 3: Translate
    print("Step 3: Translating to Chinese...")
    if args.translator == "openai":
        api_key = args.api_key or os.environ.get("OPENAI_API_KEY")
        segments = translate_openai(segments, api_key, model=args.translation_model)
    elif args.translator == "gemini":
        api_key = args.api_key or os.environ.get("GOOGLE_API_KEY")
        segments = translate_gemini(segments, api_key, model=args.translation_model)
    elif args.translator == "openclaw":
        segments = translate_openclaw(segments)
    
    # Save translated data
    translated_json = work_dir / "translated.json"
    with open(translated_json, 'w', encoding='utf-8') as f:
        json.dump(segments, f, ensure_ascii=False, indent=2)
    
    # Create Chinese SRT
    chinese_srt = work_dir / "subtitles" / "chinese.srt"
    create_chinese_srt(segments, str(chinese_srt))
    print(f"Chinese SRT: {chinese_srt}")
    
    # Step 4: Generate TTS
    print("Step 4: Generating TTS audio...")
    tts_dir = generate_tts(segments, work_dir / "tts", voice=args.tts_voice)
    
    # Step 5: Align audio
    print("Step 5: Aligning audio...")
    aligned_audio = work_dir / "aligned_chinese.wav"
    align_audio(segments, tts_dir, aligned_audio)
    
    # Step 6: Synthesize videos
    print("Step 6: Synthesizing videos...")
    output_dir = work_dir / "output"
    output_dir.mkdir(exist_ok=True)
    
    # Chinese dubbed version
    chinese_dubbed = output_dir / "chinese_dubbed.mp4"
    synthesize_video(video_path, str(aligned_audio), str(chinese_srt), str(chinese_dubbed))
    
    # Chinese subtitle only version (original audio)
    chinese_sub_only = output_dir / "chinese_subtitles_only.mp4"
    synthesize_subtitle_only(video_path, str(chinese_srt), str(chinese_sub_only))
    
    # English subtitle version
    english_sub = output_dir / "english_subtitles.mp4"
    synthesize_subtitle_only(video_path, str(english_srt), str(english_sub), language="eng")
    
    print("\n" + "="*50)
    print("Pipeline completed!")
    print("="*50)
    print(f"Chinese dubbed:     {chinese_dubbed}")
    print(f"Chinese subtitles:  {chinese_sub_only}")
    print(f"English subtitles:  {english_sub}")

# ============================================================================
# CLI
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="YouTube to Chinese Dubbing Pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic usage with YouTube URL
  python y2c.py https://youtube.com/watch?v=xxx -o ./output --api-key YOUR_KEY
  
  # Use local video file
  python y2c.py video.mp4 -o ./output --api-key YOUR_KEY
  
  # Use Gemini for translation
  python y2c.py video.mp4 -o ./output --translator gemini --api-key YOUR_GEMINI_KEY
  
  # Use OpenClaw (no API key needed if OpenClaw is running)
  python y2c.py video.mp4 -o ./output --translator openclaw
  
  # Custom TTS voice
  python y2c.py video.mp4 -o ./output --tts-voice zh-CN-XiaoxiaoNeural
"""
    )
    
    parser.add_argument("input", help="YouTube URL or local video file path")
    parser.add_argument("-o", "--output-dir", default="./y2c_output", help="Output directory")
    parser.add_argument("--api-key", help="API key for translation (or set OPENAI_API_KEY/GOOGLE_API_KEY env)")
    parser.add_argument("--translator", choices=["openai", "gemini", "openclaw"], default="openai", help="Translation provider")
    parser.add_argument("--translation-model", default=None, help="Translation model (default: gpt-4o for OpenAI, gemini-1.5-pro for Gemini)")
    parser.add_argument("--whisper-model", default="medium", help="Whisper model size (tiny/base/small/medium/large)")
    parser.add_argument("--whisper-cpp-path", default=None, help="Path to whisper.cpp directory")
    parser.add_argument("--tts-voice", default=DEFAULT_TTS_VOICE, help="Edge-TTS voice name")
    
    args = parser.parse_args()
    
    # Set default translation model based on provider
    if args.translation_model is None:
        if args.translator == "openai":
            args.translation_model = "gpt-4o"
        elif args.translator == "gemini":
            args.translation_model = "gemini-1.5-pro"
        # openclaw uses configured model, no default needed
    
    run_pipeline(args)

if __name__ == "__main__":
    main()
