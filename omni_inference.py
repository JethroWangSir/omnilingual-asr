from datasets import load_dataset
from omnilingual_asr.models.inference.pipeline import ASRInferencePipeline

# Load dataset for a specific language (e.g., Ligurian)
omni_dataset = load_dataset("facebook/omnilingual-asr-corpus", "lij_Latn", split="train", streaming=True)
# print(next(iter(omni_dataset)))
batch = next(omni_dataset.iter(5))
print(f'Batch:\n{batch}')

# Convert to pipeline input format
audio_data = [{"waveform": x["array"], "sample_rate": x["sampling_rate"]}
              for x in batch["audio"]]
print(f'Audio Data:\n{audio_data}')

# Run inference
pipeline = ASRInferencePipeline(model_card="omniASR_LLM_7B")
transcriptions = pipeline.transcribe(audio_data, batch_size=2)

# Display results
for i, (transcription, original_text) in enumerate(zip(transcriptions, batch["raw_text"]), 1):
    print(f"\n Sample {i}:")
    print(f"   Ground Truth: {original_text}")
    print(f"   Predicted:    {transcription}")