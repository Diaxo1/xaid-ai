import os
import shutil
import subprocess
import tempfile
from pathlib import Path

import boto3
import requests
import runpod


REAL_ESRGAN_DIR = Path(os.environ.get("REAL_ESRGAN_DIR", "/opt/Real-ESRGAN"))


def progress(job_id: str, percent: int, message: str) -> None:
    """Send progress to RunPod without turning the whole handler into a stream."""
    try:
        runpod.serverless.progress_update(job_id, percent, message)
    except Exception as exc:
        # Progress reporting must never kill an otherwise valid render.
        print(f"[XAID] progress update failed: {exc}", flush=True)


def download(url: str, target: Path) -> None:
    print(f"[XAID] downloading input: {url}", flush=True)

    with requests.get(url, stream=True, timeout=(30, 600)) as response:
        response.raise_for_status()
        with target.open("wb") as handle:
            for chunk in response.iter_content(chunk_size=4 * 1024 * 1024):
                if chunk:
                    handle.write(chunk)


def upload(bucket: str, key: str, source: Path) -> None:
    endpoint = os.environ.get("XAID_S3_ENDPOINT")

    s3 = boto3.client(
        "s3",
        endpoint_url=endpoint or None,
        region_name=os.environ.get("XAID_S3_REGION", "auto"),
        aws_access_key_id=os.environ["XAID_S3_ACCESS_KEY_ID"],
        aws_secret_access_key=os.environ["XAID_S3_SECRET_ACCESS_KEY"],
    )

    print(f"[XAID] uploading result: s3://{bucket}/{key}", flush=True)

    s3.upload_file(
        str(source),
        bucket,
        key,
        ExtraArgs={"ContentType": "video/mp4"},
    )


def run_realesrgan(
    input_path: Path,
    output_dir: Path,
    scale: int,
    denoise: float,
    tile: int,
    job_id: str,
) -> Path:
    """Run the official Real-ESRGAN video inference script."""

    command = [
        "python",
        str(REAL_ESRGAN_DIR / "inference_realesrgan_video.py"),
        "-i",
        str(input_path),
        "-n",
        "realesr-general-x4v3",
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
    ]

    print("[XAID] running:", " ".join(command), flush=True)

    process = subprocess.Popen(
        command,
        cwd=str(REAL_ESRGAN_DIR),
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
    )

    # Real-ESRGAN's tqdm output is not guaranteed to expose a clean percent
    # value, so use coarse progress while keeping the UI moving.
    current_progress = 10
    progress(job_id, current_progress, "AI enhancement in progress")

    if process.stdout:
        for line in process.stdout:
            print(line, end="", flush=True)
            if "inference:" in line.lower() or "frame" in line.lower():
                current_progress = min(88, current_progress + 1)
                progress(job_id, current_progress, "AI enhancement in progress")

    return_code = process.wait()

    if return_code != 0:
        raise RuntimeError(f"Real-ESRGAN exited with code {return_code}")

    outputs = sorted(output_dir.glob("*.mp4"))
    if not outputs:
        raise RuntimeError("Real-ESRGAN completed but produced no MP4 output.")

    return outputs[0]


def handler(job):
    job_input = job.get("input") or {}
    job_id = str(job.get("id") or job_input.get("job_id") or "unknown")

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

    scale_text = str(settings.get("scale", "2×"))
    scale = 4 if "4" in scale_text else 2

    noise = float(settings.get("noise", 18))
    denoise = max(0.0, min(1.0, noise / 100.0))

    # Keep this worker focused on the first real AI milestone:
    # Real-ESRGAN upscale + denoise + audio-preserving MP4 output.
    # FPS/frame interpolation will be added as a separate processing stage.
    tile = int(os.environ.get("XAID_TILE", "0"))

    work_dir = Path(tempfile.mkdtemp(prefix=f"xaid-{job_id}-"))
    input_path = work_dir / "input.mp4"
    output_dir = work_dir / "results"
    output_dir.mkdir(parents=True, exist_ok=True)

    try:
        progress(job_id, 2, "Downloading video")
        download(input_url, input_path)

        progress(job_id, 8, "Starting AI enhancement")
        enhanced_path = run_realesrgan(
            input_path=input_path,
            output_dir=output_dir,
            scale=scale,
            denoise=denoise,
            tile=tile,
            job_id=job_id,
        )

        progress(job_id, 92, "Uploading enhanced video")
        upload(bucket, output_key, enhanced_path)

        progress(job_id, 100, "Enhancement complete")

        return {
            "job_id": job_id,
            "status": "completed",
            "output_bucket": bucket,
            "output_key": output_key,
            "progress": 100,
            "message": "Enhancement complete",
        }
    finally:
        shutil.rmtree(work_dir, ignore_errors=True)


if __name__ == "__main__":
    runpod.serverless.start({"handler": handler})
