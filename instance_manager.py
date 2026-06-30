"""
instance_manager.py — Start, monitor, and kill Balatrobot instances
===================================================================
"""

import os
import signal
import atexit
import subprocess
import time
import requests

from emulator.bridge.backend import LiveBackend, SimBackend

BALATROBOT_VERSION = "1.4.1"

BALATROBOT_FLAGS = [
    "--fast",
    "--no-shaders",
    "--fps-cap", "1000",
    "--gamespeed", "4",
    "--animation-fps", "1000",
]

HEALTH_TIMEOUT  = 30
HEALTH_INTERVAL = 2


class BalatrobotManager:
    """Starts and manages N Balatrobot instances, one per port.

    In emulator mode, no processes are launched — SimBackend is used instead.
    """

    def __init__(self, ports: list, emulator: bool = False, simplified: bool = False):
        self.ports     = ports
        self.emulator  = emulator
        self.simplified = simplified
        self.processes = {}
        atexit.register(self.kill_all)
        signal.signal(signal.SIGINT,  self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)

    def _signal_handler(self, sig, frame):
        print("\nShutdown signal received, killing Balatrobot instances...")
        self.kill_all()
        exit(0)

    def start_instance(self, port: int) -> bool:
        if port in self.processes:
            self.kill_instance(port)

        cmd = ["uvx", f"balatrobot=={BALATROBOT_VERSION}", "serve", "--port", str(port)] + BALATROBOT_FLAGS
        print(f"  [port {port}] Starting: {' '.join(cmd)}")

        try:
            proc = subprocess.Popen(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            self.processes[port] = proc
            return True
        except FileNotFoundError:
            print(f"  [port {port}]  'uvx' not found. Is Balatrobot installed?")
            return False
        except Exception as e:
            print(f"  [port {port}]  Failed to start: {e}")
            return False

    def wait_healthy(self, port: int) -> bool:
        url      = f"http://127.0.0.1:{port}"
        deadline = time.time() + HEALTH_TIMEOUT
        print(f"  [port {port}] Waiting for health check...", end="", flush=True)

        while time.time() < deadline:
            try:
                r = requests.post(url, json={
                    "jsonrpc": "2.0", "method": "health", "params": {}, "id": 1
                }, timeout=3)
                if r.json().get("result", {}).get("status") == "ok":
                    print("OK")
                    return True
            except Exception:
                pass
            print(".", end="", flush=True)
            time.sleep(HEALTH_INTERVAL)

        print(f"  timeout after {HEALTH_TIMEOUT}s")
        return False

    def start(self) -> list:
        """Start instances and return backends. Exits on failure."""
        if self.emulator:
            return [SimBackend(simplified=self.simplified, fast=True) for _ in self.ports]

        print(f"Starting {len(self.ports)} Balatrobot instances...\n")
        for port in self.ports:
            self.start_instance(port)
            time.sleep(1)
        print()
        for port in self.ports:
            if not self.wait_healthy(port):
                print("\n Not all instances healthy. Check Balatrobot installation.")
                self.kill_all()
                exit(1)
        return [LiveBackend(port=port) for port in self.ports]

    def kill_instance(self, port: int):
        proc = self.processes.get(port)
        if proc and proc.poll() is None:
            try:
                if os.name == "nt":
                    proc.send_signal(signal.CTRL_BREAK_EVENT)
                else:
                    os.killpg(os.getpgid(proc.pid), signal.SIGTERM)
                proc.wait(timeout=5)
            except ProcessLookupError:
                pass
            except Exception:
                try:
                    os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
                except Exception:
                    proc.kill()
        self.processes.pop(port, None)

    def kill_all(self):
        for port in list(self.processes.keys()):
            self.kill_instance(port)
        print("All Balatrobot instances stopped.")
