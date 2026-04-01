#!/usr/bin/env python3
"""
Bonsai Fund — Bonsai Instance Launch Manager

Downloads the Bonsai-8B model and manages parallel llama-server instances.
Works with any GGUF-compatible llama.cpp build (llama-server, llama-prism, etc.).

Usage:
    python3 bonsai_fund/launch.py download-model
    python3 bonsai_fund/launch.py start --count 7
    python3 bonsai_fund/launch.py status
    python3 bonsai_fund/launch.py stop
    python3 bonsai_fund/launch.py restart --count 7
    python3 bonsai_fund/launch.py restart --count 7 --force
"""

from __future__ import annotations
import os, sys, time, subprocess, argparse, urllib.request, shutil
from pathlib import Path
from dataclasses import dataclass

# ---------------------------------------------------------------------------
# Config (use skill-level config)
# ---------------------------------------------------------------------------
SKILL_DIR = Path(__file__).parent.parent
sys.path.insert(0, str(SKILL_DIR))

try:
    from bonsai_fund import config
except ImportError:
    config = None

MODEL_NAME     = "Bonsai-8B.gguf"
PORT_BASE      = 8090
HF_REPO        = "lmstudio-community/Bonsai-8B-GGUF"
HF_FILENAME    = "bonsai-8b-v1-q1_0.gguf"
MAX_INSTANCES  = 7


def _cfg(key, default):
    return getattr(config, key, default) if config else default


# ---------------------------------------------------------------------------
# Model download
# ---------------------------------------------------------------------------

def get_model_path() -> Path:
    base = _cfg("MODEL_PATH", Path.home() / ".local" / "share" / "llama.cpp" / "bonsai")
    base.mkdir(parents=True, exist_ok=True)
    return base / HF_FILENAME


def get_llama_server() -> str:
    return _cfg("LLAMA_SERVER_BIN", str(Path.home() / "bin" / "llama-server"))


def download_model(force: bool = False) -> Path:
    dest = get_model_path()
    if dest.exists() and not force:
        size_mb = dest.stat().st_size / 1e6
        print(f"Model already exists: {dest} ({size_mb:.0f} MB)")
        return dest

    dest.parent.mkdir(parents=True, exist_ok=True)

    # Try HuggingFace hub CLI first
    try:
        from huggingface_hub import hf_hub_download
        print(f"Downloading via huggingface_hub...")
        path = hf_hub_download(repo_id=HF_REPO, filename=HF_FILENAME, local_dir=dest.parent)
        shutil.move(path, dest)
        size_mb = dest.stat().st_size / 1e6
        print(f"Downloaded: {dest} ({size_mb:.0f} MB)")
        return dest
    except ImportError:
        pass

    # Fallback: direct download via urllib
    url = f"https://huggingface.co/{HF_REPO}/resolve/main/{HF_FILENAME}"
    print(f"Downloading {url}...")
    print(f"This is ~1.2 GB — may take several minutes...")

    class DownloadProgress:
        def __init__(self): self.last = 0
        def __call__(self, block, size, total):
            pct = block * size / total * 100
            if pct - self.last >= 5:
                print(f"  {pct:.0f}%", end="\r")
                self.last = pct

    urllib.request.urlretrieve(url, dest, DownloadProgress())
    print(f"\nDownloaded: {dest} ({dest.stat().st_size / 1e6:.0f} MB)")
    return dest


def install_llama_cpp(force: bool = False) -> str:
    """Download a pre-built llama-server binary."""
    binary = Path(get_llama_server())
    if binary.exists() and not force:
        print(f"llama-server already at {binary}")
        return str(binary)

    binary.parent.mkdir(parents=True, exist_ok=True)

    # Detect OS and arch
    import platform, urllib.request, tarfile, zipfile, shutil

    system = platform.system().lower()  # linux, darwin
    machine = platform.machine().lower()  # x86_64, arm64, aarch64

    if system == "darwin" and machine in ("arm64", "aarch64"):
        url = "https://github.com/ggml-org/llama.cpp/releases/download/b3650/llama-cli-mac-arm64-CUDA.zip"
        fname = "llama-cli-mac-arm64"
    elif system == "linux" and machine == "x86_64":
        url = "https://github.com/ggml-org/llama.cpp/releases/download/b3650/llama-linux-x64.tar.gz"
        fname = "llama-server"
    else:
        print(f"No pre-built binary for {system}/{machine}. Building from source...")
        print("Run: git clone https://github.com/ggml-org/llama.cpp && cd llama.cpp && mkdir build && cmake .. && cmake --build build --target llama-server")
        return ""

    print(f"Downloading llama-server for {system}/{machine}...")
    tmp = Path("/tmp/llama_server_install")
    tmp.mkdir(exist_ok=True)
    archive = tmp / url.split("/")[-1]

    urllib.request.urlretrieve(url, archive)
    if archive.suffix == ".zip":
        with zipfile.ZipFile(archive) as z:
            z.extractall(binary.parent)
            # find binary
            for f in binary.parent.iterdir():
                if fname in f.name or "llama" in f.name:
                    if not f.name.endswith(".zip"):
                        shutil.move(str(f), str(binary))
    else:
        with tarfile.open(archive) as t:
            t.extractall(binary.parent)
        for f in (binary.parent / "llama-server").iterdir() if hasattr((binary.parent / "llama-server"), "iterdir") else []:
            pass
        src = tmp / fname
        if not src.exists():
            src = next((f for f in binary.parent.rglob(fname) if f.is_file()), None)
            if src:
                shutil.copy(src, binary)
        else:
            shutil.copy(src, binary)

    binary.chmod(0o755)
    print(f"Installed: {binary}")
    return str(binary)


# ---------------------------------------------------------------------------
# Instance management
# ---------------------------------------------------------------------------

@dataclass
class Instance:
    id: int
    port: int
    pid: int | None
    status: str  # UP, DOWN, LAUNCHING


def get_instances() -> list[Instance]:
    """Read instance state from pid file."""
    pid_dir = SKILL_DIR / "pids"
    instances = []
    for i in range(MAX_INSTANCES):
        pid_file = pid_dir / f"instance_{i}.pid"
        port = PORT_BASE + i
        if pid_file.exists():
            try:
                pid = int(pid_file.read_text().strip())
                # Check if still alive
                try:
                    os.kill(pid, 0)
                    status = "UP"
                except OSError:
                    status = "DOWN"
                instances.append(Instance(id=i, port=port, pid=pid, status=status))
            except:
                instances.append(Instance(id=i, port=port, pid=None, status="DOWN"))
        else:
            instances.append(Instance(id=i, port=port, pid=None, status="DOWN"))
    return instances


def launch_instance(instance_id: int, model_path: Path, port: int) -> int | None:
    binary = get_llama_server()
    pid_dir = SKILL_DIR / "pids"
    pid_dir.mkdir(exist_ok=True)
    pid_file = pid_dir / f"instance_{instance_id}.pid"

    # Kill existing if any
    if pid_file.exists():
        try:
            old_pid = int(pid_file.read_text().strip())
            os.kill(old_pid, 9)
        except:
            pass

    if not Path(model_path).exists():
        return None

    cmd = [
        binary,
        "-m", str(model_path),
        "-mg", "1",               # tensor type 41 for 1-bit
        "-c", "2048",              # context window
        "-b", "512",               # batch size
        "--host", "127.0.0.1",
        "--port", str(port),
        "-t", "4",                 # threads
    ]

    try:
        proc = subprocess.Popen(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        pid_file.write_text(str(proc.pid))
        return proc.pid
    except Exception as e:
        print(f"Failed to launch instance {instance_id}: {e}")
        return None


def start_instances(count: int, force: bool = False) -> list[Instance]:
    model = get_model_path()
    if not model.exists():
        print("Model not found. Run: bonsai_fund/launch.py download-model")
        return []
    binary = get_llama_server()
    if not Path(binary).exists():
        print(f"llama-server not found at {binary}")
        print("Run: bonsai_fund/launch.py install-llama")
        return []

    launched = []
    for i in range(min(count, MAX_INSTANCES)):
        pid = launch_instance(i, model, PORT_BASE + i)
        status = "UP" if pid else "DOWN"
        launched.append(Instance(id=i, port=PORT_BASE + i, pid=pid, status=status))
        time.sleep(0.5)  # stagger startup
    return launched


def stop_instances():
    pid_dir = SKILL_DIR / "pids"
    stopped = 0
    for pf in pid_dir.glob("instance_*.pid"):
        try:
            pid = int(pf.read_text().strip())
            os.kill(pid, 9)
            stopped += 1
        except:
            pass
        pf.unlink()
    return stopped


def gpu_memory_str() -> str:
    try:
        r = subprocess.run(["nvidia-smi", "--query-gpu=memory.used,memory.total",
                           "--format=csv,noheader"], capture_output=True, text=True)
        used, total = map(int, r.stdout.strip().split(","))
        return f"{used}/{total} MiB ({total-used} free)"
    except:
        return "unknown"


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Bonsai Instance Manager")
    sub = parser.add_subparsers(dest="cmd")

    sub.add_parser("download-model")
    install = sub.add_parser("install-llama")
    install.add_argument("--force", action="store_true")
    start = sub.add_parser("start")
    start.add_argument("--count", type=int, default=7)
    stop = sub.add_parser("stop")
    restart = sub.add_parser("restart")
    restart.add_argument("--count", type=int, default=7)
    restart.add_argument("--force", action="store_true")
    st = sub.add_parser("status")

    args = parser.parse_args()

    if args.cmd == "download-model":
        download_model()

    elif args.cmd == "install-llama":
        install_llama_cpp(force=getattr(args, "force", False))

    elif args.cmd == "start":
        instances = start_instances(args.count)
        up = sum(1 for i in instances if i.status == "UP")
        print(f"Started {up}/{len(instances)} instances. Run 'status' to verify.")

    elif args.cmd == "stop":
        n = stop_instances()
        print(f"Stopped {n} instances.")

    elif args.cmd == "restart":
        stop_instances()
        time.sleep(1)
        instances = start_instances(args.count, force=args.force)
        up = sum(1 for i in instances if i.status == "UP")
        print(f"Restarted {up}/{len(instances)} instances.")

    elif args.cmd == "status":
        instances = get_instances()
        vram = gpu_memory_str()
        print(f"\nBonsai Swarm: {len([i for i in instances if i.status == 'UP'])}/{len(instances)} UP")
        print(f"GPU memory: {vram}\n")
        for i in instances:
            print(f"  Instance {i.id} @ port {i.port}: {i.status}  PID={i.pid}")

    else:
        parser.print_help()


if __name__ == "__main__":
    main()
