"""Docker sandboxes for the `bash` edit interface (the CLM paper's way of editing the context file).

One container per agent. Nothing of the host is mounted:
- /ctx is an in-memory folder of the container. Before every command the harness writes the live file into it
  (`docker exec -i … cat > /ctx/context.txt`, after removing whatever is at that path); after the command it reads
  it back through `docker exec` (only a regular file, at most `max_file_bytes`, decoded leniently). A link or a
  folder planted there can only point inside the container.
- /work is the scratch workspace, in memory: the container's own (one agent), or a Docker tmpfs volume shared by
  the containers of a world. `reset_scratch` empties /work, /tmp and /ctx (all but the live file).
- /notes (joint worlds with sharing=notes): a Docker tmpfs volume mounted in every container of a world, never
  emptied; `read_tree` lists what is in it (for the logs).
- After every command every process except the container's own two (an init that reaps, and `sleep infinity`) is
  killed, so nothing keeps running between commands (no delayed writes landing during another agent's turn).
Limits: no network, read-only root, all capabilities dropped, no privilege escalation, processes / memory / CPU,
file size and open files (ulimits), tmpfs sizes, the command time (`timeout`), and the output kept per stream.
Runs as the host user. The Wikipedia graph never enters a container: navigation stays in the harness.
"""
from __future__ import annotations

import atexit
import os
import subprocess
import threading
import time
import uuid

LIVE: dict[str, "Sandbox"] = {}
VOLUMES: set[str] = set()
FILE = "/ctx/context.txt"
KILL_OTHERS = ("import os, signal, sys\n"
               "keep = {os.getpid(), os.getppid(), *map(int, sys.argv[1].split())}\n"
               "n = 0\n"
               "for p in os.listdir('/proc'):\n"
               "    if p.isdigit() and int(p) not in keep:\n"
               "        try: os.kill(int(p), signal.SIGKILL); n += 1\n"
               "        except OSError: pass\n"
               "print(n)\n")
READ_FILE = (f"if [ -L {FILE} ] || [ ! -f {FILE} ]; then echo NOT_A_FILE >&2; exit 3; fi; "
             f"s=$(stat -c %s {FILE}); if [ \"$s\" -gt \"$1\" ]; then echo TOO_BIG $s >&2; exit 4; fi; cat {FILE}")


def docker(args: list[str], timeout: float = 60, data: bytes | None = None) -> subprocess.CompletedProcess:
    return subprocess.run(["docker", *args], capture_output=True, timeout=timeout, input=data)


def check_available(image: str) -> None:
    """Raise a clear error when Docker or the image is missing."""
    try:
        info = docker(["image", "inspect", image], timeout=30)
    except FileNotFoundError:
        raise ValueError("ctx.interface=bash needs Docker (not installed)") from None
    if info.returncode != 0:
        err = info.stderr.decode(errors="replace").lower()
        if "permission denied" in err:
            raise ValueError("ctx.interface=bash: no access to Docker; add your user to the docker group and log in "
                             "again, or run through `sg docker -c '...'`")
        raise ValueError(f"ctx.interface=bash: image {image} not found; build it with "
                         f"`docker build -t {image} experiments/context_file/sandbox`")


def image_id(image: str) -> str | None:
    out = docker(["image", "inspect", image, "--format", "{{.Id}}"], timeout=30)
    return out.stdout.decode().strip() or None if out.returncode == 0 else None


def shared_volume(size: str) -> str:
    """A tmpfs Docker volume for a world's shared /work (removed by remove_volume / at exit)."""
    name = f"sbx-work-{uuid.uuid4().hex[:12]}"
    opts = f"size={size},uid={os.getuid()},gid={os.getgid()}"
    out = docker(["volume", "create", "--driver", "local", "--opt", "type=tmpfs", "--opt", "device=tmpfs", "--opt", f"o={opts}", name])
    if out.returncode != 0:
        raise RuntimeError(f"could not create the shared workspace: {out.stderr.decode(errors='replace')[:300]}")
    VOLUMES.add(name)
    return name


def remove_volume(name: str) -> None:
    docker(["volume", "rm", "-f", name], timeout=60)
    VOLUMES.discard(name)


MARK = b"\n__cmd_end__"


def _drain(stream, cap: int, sink: dict, key: str) -> None:
    """Keep the first `cap` bytes and the last 256 (where the wrapper's marker is), count the rest."""
    kept, tail, total = bytearray(), b"", 0
    for chunk in iter(lambda: stream.read(65536), b""):
        total += len(chunk)
        if len(kept) < cap:
            kept += chunk[:cap - len(kept)]
        tail = (tail + chunk)[-256:]
    sink[key], sink[key + "_bytes"], sink[key + "_tail"] = bytes(kept), total, tail


class Sandbox:
    def __init__(self, settings, volume: str | None = None, notes: str | None = None):
        self.s, self.volume, self.notes = settings, volume, notes
        self.name, self.id, self.keep, self.restarts = f"sbx-{uuid.uuid4().hex[:12]}", None, "1", 0

    def start(self) -> None:
        s = self.s
        tmp = lambda path, size: ["--tmpfs", f"{path}:rw,nosuid,nodev,size={size},mode=1777"]
        args = ["run", "-d", "--rm", "--init", "--name", self.name, "--network", "none", "--read-only",
                "--cap-drop", "ALL", "--security-opt", "no-new-privileges",
                "--pids-limit", str(s.get("pids_limit", 128)), "--memory", str(s.get("memory", "512m")),
                "--cpus", str(s.get("cpus", 1)), "--user", f"{os.getuid()}:{os.getgid()}",
                "--ulimit", f"fsize={int(s.get('max_file_bytes', 8 * 2**20))}",
                "--ulimit", f"nofile={int(s.get('nofile', 256))}:{int(s.get('nofile', 256))}",
                *tmp("/tmp", s.get("tmpfs_size", "64m")), *tmp("/ctx", s.get("ctx_size", "16m"))]
        args += ["-v", f"{self.volume}:/work"] if self.volume else tmp("/work", s.get("tmpfs_size", "64m"))
        args += ["-v", f"{self.notes}:/notes"] if self.notes else []
        out = docker(args + ["-w", "/work", str(s.image), "sleep", "infinity"])
        if out.returncode != 0:
            raise RuntimeError(f"could not start the sandbox: {out.stderr.decode(errors='replace').strip()[:400]}")
        self.id = out.stdout.decode().strip()
        LIVE[self.name] = self
        listed = docker(["exec", self.name, "python3", "-c", "import os; print(*[p for p in os.listdir('/proc') if p.isdigit() and int(p) != os.getpid()])"])
        self.keep = listed.stdout.decode().strip() or "1"        # the init and sleep infinity, nothing else yet

    def alive(self) -> bool:
        out = docker(["inspect", "-f", "{{.State.Running}}", self.name], timeout=30)
        return out.returncode == 0 and out.stdout.decode().strip() == "true"

    def restart(self) -> None:
        """A command broke the container (e.g. killed its idle process): start a fresh one, same settings. Its /ctx,
        /tmp and a private /work are lost; a world's shared /work volume is kept. Counted in `restarts`."""
        self.stop()
        self.name, self.restarts = f"sbx-{uuid.uuid4().hex[:12]}", self.restarts + 1
        self.start()

    def put(self, text: str) -> None:
        """Write the live file (whatever was at that path is removed first, even a read-only folder). If that fails,
        the container is replaced and the write tried once more."""
        script = f"chmod -R u+w {FILE} 2>/dev/null; rm -rf {FILE} && cat > {FILE}"
        for attempt in (1, 2):
            if not self.alive():
                self.restart()
            out = docker(["exec", "-i", self.name, "sh", "-c", script], data=text.encode("utf-8"))
            if out.returncode == 0:
                return
            if attempt == 1:
                self.restart()
        raise RuntimeError(f"could not write the context file: {out.stderr.decode(errors='replace')[:300]}")

    def get(self) -> tuple[str | None, str | None]:
        """(text, None), or (None, problem) when the path is not a regular file or is too large."""
        limit = int(self.s.get("max_file_bytes", 8 * 2**20))
        out = docker(["exec", self.name, "sh", "-c", READ_FILE, "sh", str(limit)])
        if out.returncode != 0:
            err = out.stderr.decode(errors="replace").strip()
            return None, ("/ctx/context.txt is not a regular file" if "NOT_A_FILE" in err else
                          f"/ctx/context.txt is larger than {limit} bytes" if "TOO_BIG" in err else f"/ctx/context.txt could not be read")      # never Docker's own error text
        return out.stdout.decode("utf-8", errors="replace"), None

    def run(self, command: str) -> dict:
        """Run one command (timeout, output kept up to output_bytes per stream), then kill every other process."""
        limit, cap = int(self.s.get("command_timeout_s", 120)), int(self.s.get("output_bytes", 65536))
        started, sink = time.time(), {}
        # the sweep runs inside the same exec, so a background job cannot hold the output streams open
        # (the count goes to stderr after a marker: no path the command could have planted something at)
        wrapper = ('timeout -k 5 "$1" bash -c "$2"; rc=$?; k=$(python3 -c "$3" "$4" 2>/dev/null); '
                   'printf "\\n__cmd_end__%s\\n" "$k" >&2; exit $rc')
        proc = subprocess.Popen(["docker", "exec", "-w", "/work", self.name, "sh", "-c", wrapper, "sh", str(limit), command, KILL_OTHERS, self.keep],
                                stdin=subprocess.DEVNULL, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        readers = [threading.Thread(target=_drain, args=(proc.stdout, cap, sink, "stdout")),
                   threading.Thread(target=_drain, args=(proc.stderr, cap, sink, "stderr"))]
        for r in readers: r.start()
        try:
            code = proc.wait(timeout=limit + 30)
        except subprocess.TimeoutExpired:
            proc.kill(); code = proc.wait()
        for r in readers: r.join(timeout=10)
        elapsed = time.time() - started
        killed, err = -1, sink.get("stderr", b"")
        tail = sink.get("stderr_tail", b"")
        if MARK in tail:
            try:
                killed = int(tail.rsplit(MARK, 1)[1].split(b"\n")[0] or 0)
            except ValueError:
                killed = -1
        if MARK in err:
            err = err.rsplit(MARK, 1)[0]
        broken = not self.alive()
        leftover = 0 if broken else self.sweep()   # normally 0: the sweep inside the command's own exec already ran
        return {"exit_code": code, "stdout": sink.get("stdout", b"").decode("utf-8", errors="replace"),
                "stderr": err.decode("utf-8", errors="replace"), "container_died": broken,
                "stdout_bytes": sink.get("stdout_bytes", 0), "stderr_bytes": max(0, sink.get("stderr_bytes", 0) - (len(tail) - tail.rfind(MARK) if MARK in tail else 0)),
                "timed_out": code in (124, 137, -9) and elapsed >= limit - 1, "seconds": round(elapsed, 2),
                "killed_processes": killed, "leftover_processes": leftover}

    def read_tree(self, path: str, cap: int = 20000) -> dict:
        """{relative path: {bytes, sha1, text (the first `cap` characters, decoded leniently)}} for every regular file
        under `path`; {} if it cannot be read."""
        script = ("import hashlib, json, os, sys\n"
                  "root, cap, out = sys.argv[1], int(sys.argv[2]), {}\n"
                  "for d, _, fs in os.walk(root):\n"
                  "    for f in fs:\n"
                  "        q = os.path.join(d, f)\n"
                  "        if os.path.islink(q) or not os.path.isfile(q): continue\n"
                  "        b = open(q, 'rb').read(8 * 2**20)\n"
                  "        out[os.path.relpath(q, root)] = {'bytes': os.path.getsize(q), 'sha1': hashlib.sha1(b).hexdigest(),\n"
                  "                                         'text': b[:cap * 4].decode('utf-8', 'replace')[:cap]}\n"
                  "print(json.dumps(out))\n")
        if not self.alive():
            return {}
        out = docker(["exec", self.name, "python3", "-c", script, path, str(cap)], timeout=60)
        try:
            import json
            return json.loads(out.stdout.decode("utf-8", errors="replace") or "{}")
        except ValueError:
            return {}

    def sweep(self) -> int:
        """Kill every process but PID 1; returns how many were killed."""
        out = docker(["exec", self.name, "python3", "-c", KILL_OTHERS, self.keep], timeout=60)
        try:
            return int(out.stdout.decode().strip() or 0)
        except ValueError:
            return -1

    def reset_scratch(self) -> None:
        """Empty /work, /tmp, /dev/shm and /ctx (all but the live file)."""
        if not self.alive():
            self.restart()
        self.sweep()
        docker(["exec", self.name, "sh", "-c", "find /work /tmp /dev/shm -mindepth 1 -delete; find /ctx -mindepth 1 ! -path /ctx/context.txt -delete"], timeout=60)

    def stop(self) -> None:
        if self.id is not None:
            docker(["rm", "-f", self.name], timeout=60)
            self.id = None
        LIVE.pop(self.name, None)


def stop_all() -> None:
    for box in list(LIVE.values()):
        try:
            box.stop()
        except Exception:
            pass
    for name in list(VOLUMES):
        try:
            remove_volume(name)
        except Exception:
            pass


atexit.register(stop_all)
