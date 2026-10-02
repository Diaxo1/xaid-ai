import os
import re
import shutil
import subprocess
import tempfile
from pathlib import Path

import requests
import runpod
import boto3


REAL_ESRGAN_DIR = Path(os.environ.get("REAL_ESRGAN_DIR", "/opt/Real-ESRGAN"))


def update(message, progress):
    return {"progress": max(0, min(100, int(progress))), "message": message}


def download(url: str, target: Path):
    with requests.get(url, stream=True, timeout=120) as response:
        response.raise_for_status()
        with target.open("wb") as handle:
            for chunk in response.iter_content(chunk_size=1024 * 1024):
                if chunk:
                    handle.write(chunk)


def upload(bucket: str, key: str, source: Path):
    s3 = boto3.client(
        "s3",
        endpoint_url=os.environ.get("XAID_S3_ENDPOINT"),
        region_name=os.environ.get("XAID_S3_REGION", "us-east-1"),
        aws_access_key_id=os.environ["XAID_S3_ACCESS_KEY_ID"],
        aws_secret_access_key=os.environ["XAID_S3_SECRET_ACCESS_KEY"],
    )
    s3.upload_file(
        str(source),
        bucket,
        key,
        ExtraArgs={"ContentType": "video/mp4"},
    )


def handler(job):
    job_input = job.get("input", {})
    job_id = job_input.get("job_id", job.get("id", "unknown"))
    input_url = job_input.get("input_url")
    bucket = job_input.get("output_bucket")
    output_key = job_input.get("output_key")
    settings = job_input.get("settings") or {}

    if not input_url or not bucket or not output_key:
        raise ValueError("input_url, output_bucket and output_key are required")

    scale_text = str(settings.get("scale", "2×"))
    scale = 4 if "4" in scale_text else 2
    fps_text = str(settings.get("fps", "60"))
    try:
        target_fps = float(fps_text)
    except ValueError:
        target_fps = None

    noise = float(settings.get("noise", 18))
    denoise = max(0.0, min(1.0, noise / 100.0))

    model = "realesr-general-x4v3"
    work_dir = Path(tempfile.mkdtemp(prefix=f"xaid-{job_id}-"))
    input_path = work_dir / "input.mp4"
    output_dir = work_dir / "results"
    output_dir.mkdir(parents=True, exist_ok=True)

    try:
        print(f"[XAID] downloading {input_url}")
        download(input_url, input_path)

        command = [
            "python",
            str(REAL_ESRGAN_DIR / "inference_realesrgan_video.py"),
            "-i", str(input_path),
            "-n", model,
            "-o", str(output_dir),
            "-s", str(scale),
            "-dn", str(denoise),
            "--suffix", "xaid",
        ]

        if target_fps and target_fps > 0:
            command.extend(["--fps", str(target_fps)])

        command.extend(["--tile", os.environ.get("XAID_TILE", "0")])

        process = subprocess.Popen(
            command,
            cwd=str(REAL_ESRGAN_DIR),
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )

        last_progress = 5
        if process.stdout:
            for line in process.stdout:
                print(line, end="")
                match = re.search(r"(\d+)%", line)
                if match:
                    frame_progress = int(match.group(1))
                    last_progress = 5 + int(frame_progress * 0.85)
                    yield update("AI enhancement in progress", last_progress)

        return_code = process.wait()
        if return_code != 0:
            raise RuntimeError(f"Real-ESRGAN exited with code {return_code}")

        outputs = sorted(output_dir.glob("*.mp4"))
        if not outputs:
            raise RuntimeError("Real-ESRGAN completed but produced no MP4 output.")

        source_output = outputs[0]
        yield update("Uploading enhanced video", 94)
        upload(bucket, output_key, source_output)

        yield update("Enhancement complete", 100)
        return {
            "job_id": job_id,
            "output_key": output_key,
            "progress": 100,
            "message": "Enhancement complete",
        }
    finally:
        shutil.rmtree(work_dir, ignore_errors=True)


runpod.serverless.start({"handler": handler})
