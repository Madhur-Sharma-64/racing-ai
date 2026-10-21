"""Overnight ZORO training — waits for current train_zoro to finish, then runs 200 more gens.

Open a NEW terminal and run:
    py train_zoro_overnight.py
"""
import subprocess
import time
import os


ZORO_MODEL = "models/ZORO.npz"


def is_training_running():
    """Check if another train_zoro.py process is running via tasklist."""
    try:
        result = subprocess.run(
            ["tasklist", "/FI", "IMAGENAME eq python.exe", "/FO", "CSV"],
            capture_output=True, text=True, timeout=10
        )
        # Also check 'py.exe' launcher
        result2 = subprocess.run(
            ["tasklist", "/FI", "IMAGENAME eq py.exe", "/FO", "CSV"],
            capture_output=True, text=True, timeout=10
        )
        # Check wmic for command line details
        result3 = subprocess.run(
            ["wmic", "process", "where", "name='python.exe'", "get", "commandline"],
            capture_output=True, text=True, timeout=10
        )
        combined = result3.stdout.lower()
        # My own PID's command line will have 'train_zoro_overnight' not 'train_zoro.py'
        # Count occurrences of train_zoro.py (not overnight)
        lines = [l.strip() for l in combined.split('\n') if 'train_zoro.py' in l and 'overnight' not in l]
        return len(lines) > 0
    except Exception:
        return False


if __name__ == '__main__':
    print("=" * 60)
    print("  ZORO OVERNIGHT WATCHER")
    print("=" * 60)
    print(f"  Monitoring for running train_zoro.py process...")
    print(f"  Once it finishes → auto-launch 200 more generations")
    print("=" * 60)

    # First, confirm there IS a training process running
    if not is_training_running():
        print("\n[!] No train_zoro.py process detected.")
        print("[*] Jumping straight to 200-gen training...\n")
    else:
        print("\n[*] Training detected! Waiting for it to finish...")
        while is_training_running():
            now = time.strftime("%H:%M:%S")
            print(f"    [{now}] still training... checking again in 60s", flush=True)
            time.sleep(60)
        print(f"\n[✓] First batch complete!")

    print("\n[*] Starting 200-gen round in 5 seconds...")
    time.sleep(5)

    print("=" * 60)
    print("  LAUNCHING ZORO — 200 GENERATIONS")
    print("=" * 60)

    subprocess.run([
        "py", "train_zoro.py",
        "--generations", "200",
        "--pop-size", "60",
        "--save-path", "models/ZORO.npz",
        "--seed-model", "models/ZORO.npz",
    ])

    print("\n[✓] 200-gen batch complete! Good morning ☀️")
