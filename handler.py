import base64
import io
import time

import numpy as np
import soundfile as sf
import runpod

from bayan_tts import get_generator


def audio_to_base64(audio, sample_rate=24000):

    buffer = io.BytesIO()

    sf.write(
        buffer,
        audio,
        sample_rate,
        format="WAV",
        subtype="PCM_16",
    )

    buffer.seek(0)

    return base64.b64encode(
        buffer.read()
    ).decode("utf-8")


def handler(job):

    job_input = job.get("input", {})

    text = job_input.get("text")

    if not text:
        return {
            "error": "No text supplied."
        }

    temperature = job_input.get(
        "temperature",
        0.6
    )

    top_p = job_input.get(
        "top_p",
        0.95
    )

    repetition_penalty = job_input.get(
        "repetition_penalty",
        1.1
    )

    max_new_tokens = job_input.get(
        "max_new_tokens",
        500
    )

    print("=" * 60)
    print("New generation request")
    print(f"Text length: {len(text)}")
    print("=" * 60)

    start_time = time.time()

    try:

        generator = get_generator()

        audio = generator.generate(
            text=text,
            temperature=float(temperature),
            top_p=float(top_p),
            repetition_penalty=float(
                repetition_penalty
            ),
            max_new_tokens=int(
                max_new_tokens
            ),
        )

        elapsed = time.time() - start_time

        audio_duration = (
            len(audio) / 24000
        )

        encoded_audio = audio_to_base64(
            audio,
            24000
        )

        print(
            f"Generation completed in "
            f"{elapsed:.2f} seconds"
        )

        print(
            f"Audio duration: "
            f"{audio_duration:.2f} seconds"
        )

        return {
            "audio_base64": encoded_audio,
            "sample_rate": 24000,
            "audio_duration": audio_duration,
            "generation_time": elapsed,
        }

    except Exception as e:

        print(
            f"Generation failed: {type(e).__name__}: {e}"
        )

        return {
            "error": str(e),
            "error_type": type(e).__name__,
        }

if __name__ == "__main__":
    runpod.serverless.start({
        "handler": handler
    })




