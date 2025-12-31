import os
from omnilingual_asr.models.inference.pipeline import ASRInferencePipeline

os.environ['CUDA_VISIBLE_DEVICES'] = '0'

pipeline = ASRInferencePipeline(model_card="omniASR_LLM_7B")

audio_files = [
    "/share/nas169/jethrowang/common_voice_15_0/cache/test_split/downloads/extracted/f77daaa68b69d0bfb9cf104affec0454a77b4e0554f322e516781a1b17f6272b/zh-TW_test_0/common_voice_zh-TW_17368075.mp3",
    "/share/nas169/jethrowang/common_voice_15_0/cache/test_split/downloads/extracted/f77daaa68b69d0bfb9cf104affec0454a77b4e0554f322e516781a1b17f6272b/zh-TW_test_0/common_voice_zh-TW_17368077.mp3"
]

transcriptions = pipeline.transcribe(audio_files, batch_size=2)

print(f'Transcriptions: {transcriptions}')