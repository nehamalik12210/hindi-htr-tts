"""
Phase 6 — Audio format converter: WAV → MP3 / OGG.
Uses pydub + ffmpeg (installed in Modal image via apt).
"""

from pathlib import Path
from pydub import AudioSegment


def convert_audio(input_wav: str, output_format: str, output_path: str) -> str:
    """
    Convert a WAV file to the specified format.
    Supported formats: wav, mp3, ogg
    """
    fmt = output_format.lower().strip()

    if fmt == "wav":
        # Just copy
        if input_wav != output_path:
            import shutil
            shutil.copy2(input_wav, output_path)
        return output_path

    if fmt not in ("mp3", "ogg"):
        raise ValueError(f"Unsupported audio format: {fmt}. Supported: wav, mp3, ogg")

    audio = AudioSegment.from_wav(input_wav)

    if fmt == "mp3":
        audio.export(output_path, format="mp3", bitrate="192k")
    elif fmt == "ogg":
        audio.export(output_path, format="ogg", codec="libvorbis")

    return output_path
