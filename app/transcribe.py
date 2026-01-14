import whisper
import os
import tempfile
from pathlib import Path
from convert_to_audio import convert_file, _ffprobe_audio_info


def prepare_audio_for_transcription(input_file: str, sample_rate: int = 16000) -> tuple[str, bool]:
    """Prepare WAV for transcription and return (wav_path, created_temp).

    If the input is already 16kHz mono 16-bit WAV (pcm_s16le) this returns the
    original path and created_temp=False. Otherwise it converts into a temp
    WAV and returns its path with created_temp=True.
    """
    input_path = Path(input_file)
    info = _ffprobe_audio_info(input_path)
    if info:
        sr, ch, codec = info
        if sr == sample_rate and ch == 1 and input_path.suffix.lower() == ".wav" and codec == "pcm_s16le":
            return str(input_path), False

    tmp = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
    tmp.close()
    out_path = Path(tmp.name)
    out, ok, err = convert_file(input_path, out_path, sample_rate=sample_rate, overwrite=True)
    if not ok:
        # clean up temp file if conversion failed
        try:
            out_path.unlink()
        except Exception:
            pass
        raise RuntimeError(f"Conversion failed: {err}")

    return str(out), True


def transcribe_audio(file_path: str) -> str:
    # Load the Whisper model
    model = whisper.load_model("base")

    # Transcribe the audio file
    result = model.transcribe(file_path)

    return result["text"]

def save_transcription_to_file(transcription: str, output_file: str):
    with open(output_file, 'w') as f:
        f.write(transcription)

def main(input_audio_path: str = None):
    if not input_audio_path:
        raise ValueError("input_audio_path must be provided")

    transcription_output_path = "transcription.txt"

    wav_path = None
    temp_created = False
    try:
        wav_path, temp_created = prepare_audio_for_transcription(input_audio_path)
        # Transcribe
        transcription = transcribe_audio(wav_path)
        save_transcription_to_file(transcription, transcription_output_path)
    finally:
        # Clean up only if we created a temporary file
        if temp_created and wav_path and Path(wav_path).exists():
            try:
                os.remove(wav_path)
            except Exception:
                pass

    return transcription

if __name__ == "__main__":
    input_audio = "data/test/thlang.mp4" # Replace with your audio file path
    transcription_result = main(input_audio)
    print("Transcription:", transcription_result)