"""Real child-process contract failures, including Release/NDEBUG builds."""
import ctypes
import os
from pathlib import Path
import subprocess
import sys
import unittest


def child(name, case):
    path = Path(os.environ["Q4KP_TEST_LIBRARY"]).resolve()
    directories = []
    if hasattr(os, "add_dll_directory"):
        for directory in (path.parent, Path(os.environ.get("Q4KP_TEST_RUNTIME_DIR", path.parent))):
            directories.append(os.add_dll_directory(str(directory)))
    dll = ctypes.CDLL(str(path))
    callback_type = ctypes.CFUNCTYPE(None, ctypes.c_char_p)

    @callback_type
    def aborted(message):
        # Use ggml's supported callback to exit without Windows crash dialogs
        # or dump files. A silent return or an access violation cannot pass.
        os.write(2, message + b"\n")
        os._exit(86)

    dll.ggml_set_abort_callback.argtypes = [callback_type]
    dll.ggml_set_abort_callback(aborted)
    fn = getattr(dll, name)
    fn.argtypes = [ctypes.c_int, ctypes.c_void_p, ctypes.c_size_t,
                  ctypes.c_void_p, ctypes.c_void_p, ctypes.c_int, ctypes.c_int]
    fn.restype = None
    buffers = [ctypes.create_string_buffer(8192) for _ in range(3)]
    ptrs = [(ctypes.addressof(buf) + 31) & ~31 for buf in buffers]
    gemm = name.endswith("gemm")
    args = [256, ptrs[0], 8, ptrs[1], ptrs[2], 4 if gemm else 1, 8]
    mutations = {
        "zero_n": (0, 0), "negative_n": (0, -256), "unaligned_n": (0, 255),
        "zero_nc": (6, 0), "unaligned_nc": (6, 7),
        "zero_nr": (5, 0), "unaligned_nr": (5, 3 if gemm else 2),
        "null_s": (1, None), "null_x": (3, None), "null_y": (4, None),
        "unaligned_s": (1, ptrs[0] + 1), "unaligned_x": (3, ptrs[1] + 1),
        "unaligned_y": (4, ptrs[2] + 1),
        "overlap_s_x": (1, ptrs[1]), "overlap_s_y": (1, ptrs[2]),
        "wrapped_s": (1, ctypes.c_size_t(-32).value),
        "wrapped_x": (3, ctypes.c_size_t(-32).value),
        "wrapped_y": (4, ctypes.c_size_t(-32).value),
        "short_stride": (2, 7), "overflow_stride": (2, ctypes.c_size_t(-1).value),
    }
    index, value = mutations[case]
    args[index] = value
    fn(*args)
    raise SystemExit("invalid compute unexpectedly returned")


class ContractTests(unittest.TestCase):
    def test_invalid_compute_aborts_before_isa_or_memory_access(self):
        names = ("q4kp_gemv", "q4kp_gemm", "q4kp_vnni_gemv", "q4kp_vnni_gemm",
                 "q4kp_vnni_original_gemv", "q4kp_vnni_original_gemm", "q4kp_wide_gemv")
        common = ("zero_n", "negative_n", "unaligned_n", "zero_nc", "unaligned_nc",
                  "zero_nr", "unaligned_nr", "null_s", "null_x", "null_y",
                  "unaligned_s", "unaligned_x", "unaligned_y", "overlap_s_x",
                  "overlap_s_y", "wrapped_s", "wrapped_x", "wrapped_y")
        for name in names:
            cases = common + (("short_stride", "overflow_stride") if name.endswith("gemm") else ())
            for case in cases:
                with self.subTest(kernel=name, case=case):
                    result = subprocess.run([sys.executable, "-B", str(Path(__file__).resolve()),
                                             name, case], capture_output=True, timeout=10)
                    self.assertEqual(result.returncode, 86, result.stderr.decode(errors="replace"))
                    self.assertIn(b"GGML_ASSERT(", result.stderr)
                    self.assertNotIn(b"supported()", result.stderr)


if __name__ == "__main__":
    if len(sys.argv) == 3:
        child(*sys.argv[1:])
    else:
        unittest.main()
