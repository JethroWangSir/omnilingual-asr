from datasets import load_dataset
from omnilingual_asr.models.inference.pipeline import ASRInferencePipeline

gv33 = load_dataset("hanamizuki-ai/genshin-voice-v3.3-mandarin")
print(next(iter(gv33)))

# # Convert to pipeline input format
# audio_data = [{"waveform": x["array"], "sample_rate": x["sampling_rate"]}
#               for x in batch["audio"]]
# print(f'Audio Data:\n{audio_data}')

# # Run inference
# pipeline = ASRInferencePipeline(model_card="omniASR_LLM_7B")
# transcriptions = pipeline.transcribe(audio_data, batch_size=2)

# # Display results
# for i, (transcription, original_text) in enumerate(zip(transcriptions, batch["raw_text"]), 1):
#     print(f"\n Sample {i}:")
#     print(f"   Ground Truth: {original_text}")
#     print(f"   Predicted:    {transcription}")