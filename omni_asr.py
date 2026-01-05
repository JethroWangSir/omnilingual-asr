import os
import time
import librosa
from omnilingual_asr.models.inference.pipeline import ASRInferencePipeline

# 設定 GPU
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

# 初始化模型
pipeline = ASRInferencePipeline(model_card="omniASR_LLM_7B")

# 定義列表檔案的路徑
list_file_path = "/share/nas169/jethrowang/common_voice_15_0/subset/audio/audio.list"

# 讀取檔案並將每一行轉換為 list 元素
audio_files = []
if os.path.exists(list_file_path):
    with open(list_file_path, 'r', encoding='utf-8') as f:
        audio_files = [line.strip() for line in f if line.strip()]
else:
    print(f"Error: File not found at {list_file_path}")

# 檢查是否有讀取到檔案
if audio_files:
    print(f"Loaded {len(audio_files)} audio files.")

    # --- 步驟 1: 計算音訊總長度 (Total Duration) ---
    print("Calculating total audio duration...")
    total_audio_duration = 0.0
    
    for audio_path in audio_files:
        try:
            # librosa 0.10+ 版本使用 path=，舊版本使用 filename=
            # 這裡使用通用寫法，若報錯請嘗試改為 filename=audio_path
            duration = librosa.get_duration(path=audio_path)
            total_audio_duration += duration
        except Exception as e:
            # 兼容舊版 librosa 或處理讀取錯誤
            try:
                duration = librosa.get_duration(filename=audio_path)
                total_audio_duration += duration
            except:
                print(f"Warning: Could not get duration for {audio_path}. Error: {e}")

    print(f"Total Audio Duration: {total_audio_duration:.2f} seconds")

    # --- 步驟 2: 執行推論並計時 (Inference Time) ---
    print("Starting transcription...")
    start_time = time.time()  # 開始計時

    transcriptions = pipeline.transcribe(audio_files, batch_size=1)

    end_time = time.time()    # 結束計時
    
    # --- 步驟 3: 計算並印出 RTF ---
    inference_time = end_time - start_time
    
    print("-" * 30)
    print(f"Total Inference Time: {inference_time:.2f} seconds")
    
    if total_audio_duration > 0:
        rtf = inference_time / total_audio_duration
        print(f"Real-Time Factor (RTF): {rtf:.4f}")
    else:
        print("RTF: N/A (Total duration is 0)")
    print("-" * 30)

    print(f'Transcriptions: {transcriptions}')

else:
    print("No audio files to transcribe.")