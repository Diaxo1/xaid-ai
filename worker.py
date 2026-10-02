import argparse, json, os, re, subprocess, sys, time
from pathlib import Path

def write_job(job_file, **updates):
    try:
        job = json.loads(job_file.read_text(encoding="utf-8"))
        job.update(updates)
        job["updatedAt"] = int(time.time() * 1000)
        job_file.write_text(json.dumps(job, indent=2), encoding="utf-8")
    except Exception:
        pass

def progress(job_file, value):
    value = max(0, min(100, int(value)))
    write_job(job_file, status="completed" if value >= 100 else "processing", progress=value)
    print(f"XAID_PROGRESS:{value}", flush=True)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--job", required=True)
    args = parser.parse_args()
    payload = json.loads(args.job)
    input_path = Path(payload["inputPath"])
    output_path = Path(payload["outputPath"])
    job_file = Path(payload["jobFile"])
    settings = payload.get("settings", {})

    if not input_path.exists():
        raise FileNotFoundError(input_path)

    root = Path(os.environ.get("XAID_REAL_ESRGAN_ROOT", ""))
    script = root / "inference_realesrgan_video.py"
    if not script.exists():
        raise RuntimeError("XAID_REAL_ESRGAN_ROOT is not configured. Point it at your cloned Real-ESRGAN repository.")

    scale_match = re.search(r"([0-9]+(?:\.[0-9]+)?)", settings.get("scale", "2×"))
    scale = float(scale_match.group(1)) if scale_match else 2.0
    model = "RealESRGAN_x4plus" if scale >= 4 else "RealESRGAN_x2plus"
    out_scale = 4 if scale >= 4 else 2

    output_path.parent.mkdir(parents=True, exist_ok=True)
    write_job(job_file, status="processing", progress=2)

    command = [sys.executable, str(script), "-i", str(input_path), "-o", str(output_path.parent), "-n", model, "-s", str(out_scale), "--suffix", "enhanced"]
    process = subprocess.Popen(command, cwd=str(root), stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1)
    progress(job_file, 5)

    last = 5
    for line in process.stdout or []:
        print(line.rstrip(), flush=True)
        match = re.search(r"(\d{1,3})%", line)
        if match:
            mapped = 5 + int(int(match.group(1)) * 0.9)
            if mapped > last:
                last = mapped
                progress(job_file, mapped)

    code = process.wait()
    if code != 0:
        raise RuntimeError(f"Real-ESRGAN exited with code {code}.")

    candidates = sorted(output_path.parent.glob("*_enhanced.mp4"), key=lambda p: p.stat().st_mtime, reverse=True)
    if not candidates:
        raise RuntimeError("Real-ESRGAN completed but produced no MP4 output.")
    produced = candidates[0]
    if produced.resolve() != output_path.resolve():
        if output_path.exists(): output_path.unlink()
        produced.replace(output_path)
    progress(job_file, 100)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--job", required=True)
    raw = parser.parse_args()
    try:
        main()
    except Exception as exc:
        try:
            payload = json.loads(raw.job)
            write_job(Path(payload["jobFile"]), status="failed", error=str(exc), progress=0)
        except Exception:
            pass
        print(f"XAID_ERROR:{exc}", file=sys.stderr, flush=True)
        raise
