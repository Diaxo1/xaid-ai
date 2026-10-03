import os
import shutil
import subprocess
import tempfile
import time
from pathlib import Path

import boto3
import requests
import runpod


REAL_ESRGAN_DIR = Path(
    os.environ.get("REAL_ESRGAN_DIR", "/opt/Real-ESRGAN")
)


def progress(job, percent: int, message: str) -> None:
    """Send progress to RunPod without breaking the render if reporting fails."""
    try:
        runpod.serverless.progress_update(
            job,
            f"{percent}% — {message}",
        )
    except Exception as exc:
        print(
            f"[XAID] progress update failed: {exc}",
            flush=True,
        )


def download(url: str, target: Path) -> None:
    """Download the source video to the worker's temporary directory."""
    print(f"[XAID] downloading input: {url}", flush=True)

    with requests.get(
        url,
        stream=True,
        timeout=(30, 600),
    ) as response:
        response.raise_for_status()

        with target.open("wb") as handle:
            for chunk in response.iter_content(
                chunk_size=4 * 1024 * 1024
            ):
                if chunk:
                    handle.write(chunk)


def upload(bucket: str, key: str, source: Path) -> None:
    """Upload the final enhanced MP4 to S3-compatible storage."""
    endpoint = os.environ.get("XAID_S3_ENDPOINT")

    s3 = boto3.client(
        "s3",
        endpoint_url=endpoint or None,
        region_name=os.environ.get("XAID_S3_REGION", "auto"),
        aws_access_key_id=os.environ["XAID_S3_ACCESS_KEY_ID"],
        aws_secret_access_key=os.environ["XAID_S3_SECRET_ACCESS_KEY"],
    )

    print(
        f"[XAID] uploading result: s3://{bucket}/{key}",
        flush=True,
    )

    s3.upload_file(
        str(source),
        bucket,
        key,
        ExtraArgs={"ContentType": "video/mp4"},
    )


def run_command(command: list[str], cwd: Path | None = None) -> None:
    """Run a command and raise a useful error if it fails."""
    print(
        "[XAID] running:",
        " ".join(command),
        flush=True,
    )

    process = subprocess.Popen(
        command,
        cwd=str(cwd) if cwd else None,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
    )

    if process.stdout:
        for line in process.stdout:
            print(line, end="", flush=True)

    return_code = process.wait()

    if return_code != 0:
        raise RuntimeError(
            f"Command exited with code {return_code}: {' '.join(command)}"
        )


def run_realesrgan(
    input_path: Path,
    output_dir: Path,
    scale: int,
    denoise: float,
    tile: int,
    model: str,
    job,
) -> Path:
    """Run the official Real-ESRGAN video inference script."""
    command = [
        "python",
        str(REAL_ESRGAN_DIR / "inference_realesrgan_video.py"),
        "-i",
        str(input_path),
        "-n",
        model,
        "-o",
        str(output_dir),
        "-s",
        str(scale),
        "-dn",
        str(denoise),
        "--suffix",
        "xaid",
        "--tile",
        str(tile),
        "--extract_frame_first",
    ]

    print(
        "[XAID] running:",
        " ".join(command),
        flush=True,
    )

    process = subprocess.Popen(
        command,
        cwd=str(REAL_ESRGAN_DIR),
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
    )

    current_progress = 10
    last_progress_time = time.monotonic()
    progress(job, current_progress, "AI enhancement in progress")

    if process.stdout:
        for line in process.stdout:
            print(line, end="", flush=True)

            lower_line = line.lower()

            if "inference:" in lower_line or "frame" in lower_line:
                next_progress = min(88, current_progress + 1)
                now = time.monotonic()

                # RunPod progress updates are asynchronous. Throttle them so
                # Real-ESRGAN's frame-by-frame output cannot flood the
                # progress endpoint or race with the final job result.
                if next_progress > current_progress and (
                    now - last_progress_time >= 1.0
                    or next_progress >= 88
                ):
                    current_progress = next_progress
                    last_progress_time = now
                    progress(
                        job,
                        current_progress,
                        "AI enhancement in progress",
                    )

    return_code = process.wait()

    if return_code != 0:
        raise RuntimeError(
            f"Real-ESRGAN exited with code {return_code}"
        )

    outputs = sorted(output_dir.glob("*.mp4"))

    if not outputs:
        raise RuntimeError(
            "Real-ESRGAN completed but produced no MP4 output."
        )

    return outputs[0]


def get_effective_settings(
    settings: dict,
) -> tuple[int, float, int, str, float, str]:
    """Convert XAID UI settings into actual worker processing settings."""
    preset = str(settings.get("preset", "Auto Enhance"))

    detail = float(settings.get("detail", 72))
    noise = float(settings.get("noise", 18))
    sharpen = float(settings.get("sharpen", 42))

    detail = max(0.0, min(100.0, detail))
    noise = max(0.0, min(100.0, noise))
    sharpen = max(0.0, min(100.0, sharpen))

    # Presets affect the actual GPU render.
    if preset == "Gaming":
        detail = max(detail, 78.0)
        noise = min(noise, 12.0)
        sharpen = max(sharpen, 58.0)
    elif preset == "Clean":
        detail = max(detail - 8.0, 35.0)
        noise = max(noise, 42.0)
        sharpen = min(sharpen, 30.0)
    elif preset == "Detail":
        detail = max(detail, 88.0)
        noise = min(noise, 10.0)
        sharpen = max(sharpen, 72.0)

    scale_text = str(settings.get("scale", "2×"))

    # "Original" means no upscaling. Real-ESRGAN can still enhance
    # at native output dimensions by using an outscale of 1.
    if scale_text == "Original":
        scale = 1
    elif "4" in scale_text:
        scale = 4
    else:
        scale = 2

    fps_text = str(settings.get("fps", "Original"))
    if fps_text == "Original":
        output_fps = 0
    else:
        try:
            output_fps = max(1, min(120, int(fps_text)))
        except ValueError:
            output_fps = 0

    # Supported Real-ESRGAN models.
    model = str(settings.get("model", "realesr-general-x4v3"))

    allowed_models = {
        "realesr-general-x4v3",
        "RealESRGAN_x4plus",
        "realesr-animevideov3",
        "RealESRGAN_x4plus_anime_6B",
    }

    if model not in allowed_models:
        model = "realesr-general-x4v3"

    # Real-ESRGAN's denoise control is 0..1.
    denoise = noise / 100.0

    # Detail + sharpen are applied in the final FFmpeg stage.
    sharpen_strength = (
        (detail / 100.0) * 0.75
        + (sharpen / 100.0) * 1.25
    )
    sharpen_strength = max(0.0, min(2.0, sharpen_strength))

    return (
        scale,
        denoise,
        output_fps,
        preset,
        sharpen_strength,
        model,
    )

def finalize_video(
    enhanced_path: Path,
    original_path: Path,
    final_path: Path,
    output_fps: int,
    sharpen_strength: float,
    job,
) -> Path:
    """Restore original audio and apply final FPS/sharpen processing."""
    filters: list[str] = []

    if output_fps > 0:
        filters.append(f"fps={output_fps}")

    if sharpen_strength > 0.01:
        # FFmpeg unsharp amount is intentionally kept moderate to avoid
        # turning compression/noise into harsh edges.
        amount = min(2.0, max(0.1, sharpen_strength))
        filters.append(
            f"unsharp=5:5:{amount:.2f}:5:5:0"
        )

    command = [
        "ffmpeg",
        "-y",
        "-i",
        str(enhanced_path),
        "-i",
        str(original_path),
        "-map",
        "0:v:0",
        "-map",
        "1:a?",
    ]

    if filters:
        command.extend(["-vf", ",".join(filters)])

    command.extend(
        [
            "-c:v",
            "libx264",
            "-preset",
            "medium",
            "-crf",
            "18",
            "-pix_fmt",
            "yuv420p",
            "-c:a",
            "aac",
            "-b:a",
            "192k",
            "-movflags",
            "+faststart",
            "-shortest",
            str(final_path),
        ]
    )

    progress(job, 90, "Finalizing video and restoring audio")
    run_command(command)

    if not final_path.exists() or final_path.stat().st_size == 0:
        raise RuntimeError("FFmpeg completed but produced no final MP4.")

    return final_path


def handler(job):
    """Main RunPod Serverless job handler."""
    job_input = job.get("input") or {}

    job_id = str(
        job.get("id")
        or job_input.get("job_id")
        or "unknown"
    )

    input_url = job_input.get("input_url")
    bucket = job_input.get("output_bucket")
    output_key = job_input.get("output_key")
    settings = job_input.get("settings") or {}

    if not input_url:
        raise ValueError("input_url is required")

    if not bucket:
        raise ValueError("output_bucket is required")

    if not output_key:
        raise ValueError("output_key is required")

    (
        scale,
        denoise,
        output_fps,
        preset,
        sharpen_strength,
        model,
    ) = get_effective_settings(settings)

    print(
        "[XAID] selected Real-ESRGAN model:", model,
        flush=True,
    )

    print(
        "[XAID] effective settings:",
        {
            "preset": preset,
            "scale": scale,
            "fps": output_fps or "original",
            "denoise": round(denoise, 3),
            "sharpen": round(sharpen_strength, 3),
            "model": model,
        },
        flush=True,
    )

    tile = int(os.environ.get("XAID_TILE", "0"))

    work_dir = Path(
        tempfile.mkdtemp(prefix=f"xaid-{job_id}-")
    )

    input_path = work_dir / "input.mp4"
    output_dir = work_dir / "results"
    output_dir.mkdir(parents=True, exist_ok=True)

    final_path = work_dir / "enhanced-final.mp4"

    try:
        progress(job, 2, "Downloading video")
        download(input_url, input_path)

        progress(job, 8, "Starting AI enhancement")

        enhanced_path = run_realesrgan(
            input_path=input_path,
            output_dir=output_dir,
            scale=scale,
            denoise=denoise,
            tile=tile,
            model=model,
            job=job,
        )

        progress(job, 89, "AI enhancement complete")

        finalize_video(
            enhanced_path=enhanced_path,
            original_path=input_path,
            final_path=final_path,
            output_fps=output_fps,
            sharpen_strength=sharpen_strength,
            job=job,
        )

        progress(job, 94, "Uploading enhanced video")

        upload(
            bucket,
            output_key,
            final_path,
        )

        return {
            "job_id": job_id,
            "status": "completed",
            "output_bucket": bucket,
            "output_key": output_key,
            "progress": 100,
            "message": "Enhancement complete",
        }

    finally:
        shutil.rmtree(
            work_dir,
            ignore_errors=True,
        )
