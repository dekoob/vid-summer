import time
import torch
from transformers import AutoModel, AutoTokenizer, pipeline


# # model dir
# local_directory = "models/whisper-th-medium-combined"

# # Load the model and tokenizer from the local path
# local_model = AutoModel.from_pretrained(local_directory)
# local_tokenizer = AutoTokenizer.from_pretrained(local_directory)


# Create the ASR pipeline using the local model and tokenizer
def main():
    MODEL_NAME = "models/whisper-th-small-combined"
    lang = "th"
    device = "mps" if torch.backends.mps.is_available() else "cpu"

    t0 = time.perf_counter()
    pipe = pipeline(
        task="automatic-speech-recognition",
        model=MODEL_NAME,
        chunk_length_s=30,
        device=device,
    )
    setup_time = time.perf_counter() - t0

    # Perform ASR with the created pipe.
    pipe.model.config.forced_decoder_ids = pipe.tokenizer.get_decoder_prompt_ids(
        language=lang,
        task="transcribe",
    )

    t0 = time.perf_counter()
    result = pipe("data/test/converted/thlang.wav")
    inference_time = time.perf_counter() - t0

    text = result["text"]
    print("Transcription:", text)

    total = setup_time + inference_time
    print(f"Timing: setup={setup_time:.2f}s inference={inference_time:.2f}s total={total:.2f}s")

if __name__ == "__main__":
    main()