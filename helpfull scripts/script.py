"""
Text-to-Speech script using OpenAI's API.

Note: Whisper is OpenAI's speech-to-text model (audio -> text).
This script does the opposite: text -> audio, using OpenAI's
text-to-speech model (gpt-4o-mini-tts).

Setup:
    pip install openai
    export OPENAI_API_KEY="your-api-key-here"   # (Windows: set OPENAI_API_KEY=...)

Usage:
    python text_to_speech.py
    python text_to_speech.py --text "Hello there!" --output hello.mp3
    python text_to_speech.py --input-file script.txt --voice nova
"""

import argparse
from pathlib import Path
from openai import OpenAI


def text_to_speech(
    text: str,
    output_path: str = "speech.mp3",
    model: str = "gpt-4o-mini-tts",
    voice: str = "coral",
    instructions: str | None = None,
) -> Path:
    """
    Convert text to an audio file using OpenAI's TTS API.

    Args:
        text: The text to convert to speech.
        output_path: Where to save the resulting audio file (.mp3, .wav, etc.)
        model: TTS model to use ("gpt-4o-mini-tts", "tts-1", or "tts-1-hd").
        voice: Voice to use (e.g. "alloy", "coral", "nova", "onyx", "shimmer").
        instructions: Optional steering instructions, e.g. "Speak slowly and calmly."
                       (only supported by gpt-4o-mini-tts)

    Returns:
        Path to the saved audio file.
    """
    client = OpenAI()  # reads OPENAI_API_KEY from environment
    speech_file_path = Path(output_path)

    kwargs = dict(model=model, voice=voice, input=text)
    if instructions and model == "gpt-4o-mini-tts":
        kwargs["instructions"] = instructions

    with client.audio.speech.with_streaming_response.create(**kwargs) as response:
        response.stream_to_file(speech_file_path)

    return speech_file_path


def main():
    parser = argparse.ArgumentParser(description="Convert text to an audio file using OpenAI TTS.")
    parser.add_argument("--text", type=str, help="Text to convert directly.")
    parser.add_argument("--input-file", type=str, help="Path to a .txt file containing the text.")
    parser.add_argument("--output", type=str, default="speech.mp3", help="Output audio file path.")
    parser.add_argument("--model", type=str, default="gpt-4o-mini-tts",
                         choices=["gpt-4o-mini-tts", "tts-1", "tts-1-hd"])
    parser.add_argument("--voice", type=str, default="coral",
                         help="alloy, ash, ballad, coral, echo, fable, nova, onyx, sage, shimmer, verse")
    parser.add_argument("--instructions", type=str, default=None,
                         help="Optional tone/style instructions (gpt-4o-mini-tts only).")
    args = parser.parse_args()

    if args.input_file:
        text = Path(args.input_file).read_text(encoding="utf-8")
    elif args.text:
        text = args.text
    else:
        text = "Hello! This is a test of OpenAI's text to speech API."
        print("No --text or --input-file given, using a default test sentence.")

    path = text_to_speech(
        text=text,
        output_path=args.output,
        model=args.model,
        voice=args.voice,
        instructions=args.instructions,
    )
    print(f"Saved audio to: {path.resolve()}")


if __name__ == "__main__":
    main()