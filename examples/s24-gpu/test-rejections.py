"""Run assertion-based Vulkan upload/view rejection checks in child processes."""
import json
import os
import re
import subprocess
import sys


def main():
    if len(sys.argv) != 2:
        raise SystemExit("usage: test-rejections.py /path/to/llama-s24-bench")
    if os.name == "nt":
        # Keep expected native assertion failures from opening OS crash dialogs.
        import ctypes
        ctypes.windll.kernel32.SetErrorMode(0x0001 | 0x0002 | 0x8000)
    else:
        import resource
        resource.setrlimit(resource.RLIMIT_CORE, (0, 0))
    results = []
    for case in ("invalid-scale", "invalid-support", "partial-upload", "view"):
        child = subprocess.run([sys.argv[1], "--negative", case], capture_output=True, timeout=40,
                               creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
        stdout = child.stdout.decode("utf-8", errors="replace")
        stderr = child.stderr.decode("utf-8", errors="replace")
        ready = any(line == json.dumps({"case": case, "type": "rejection_probe_ready"}, separators=(",", ":"))
                    for line in stdout.splitlines())
        passed = (child.returncode != 0 and ready and "GGML_ASSERT" in stderr and
                  re.search("s24", stderr, re.IGNORECASE) is not None and
                  "Backend accepted forbidden operation" not in stdout)
        results.append({"case": case, "passed": passed, "returncode": child.returncode})
        if not passed:
            print(stdout, file=sys.stderr)
            print(stderr, file=sys.stderr)
            print(json.dumps({"status": "FAIL", "cases": results}))
            return 1
    print(json.dumps({"status": "PASS", "cases": results}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
