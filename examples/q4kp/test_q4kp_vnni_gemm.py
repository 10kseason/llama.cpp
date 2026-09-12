"""VNNI GEMM versus original AVX2, P6 AVX2, and independent scalar FMA."""
import ctypes
import os
import unittest

import numpy as np

import support
import test_q4kp_kernel as fixtures


class VNNIGemmTests(unittest.TestCase):
    cases = 0
    output_floats = 0

    @classmethod
    def setUpClass(cls):
        cls.dll = support.library()
        if not cls.dll.q4kp_vnni_supported():
            raise unittest.SkipTest("VNNI ISA unavailable")
        # Optional standalone candidate library permits checking this kernel
        # before integrating it into the backend. Normally use the CMake DLL.
        candidate = os.environ.get("Q4KP_VNNI_GEMM_LIBRARY")
        cls.candidate = ctypes.CDLL(candidate) if candidate else cls.dll
        cls.gemm = cls.candidate.q4kp_vnni_gemm
        cls.gemm.argtypes = [ctypes.c_int, ctypes.c_void_p, ctypes.c_size_t,
                            ctypes.c_void_p, ctypes.c_void_p, ctypes.c_int,
                            ctypes.c_int]
        cls.gemm.restype = None

    def compare(self, n, nr, nc, packed, q8):
        original, activations = packed.copy(), q8.copy()
        recoded = packed.copy()
        self.assertEqual(self.dll.q4kp_recode(recoded.ctypes.data, recoded.nbytes), 0)
        recoded_before = recoded.copy()
        stride = nc + 3
        outs = [np.full((nr + 2, stride), np.nan, dtype=np.float32)
                for _ in range(5)]
        for fn, matrix, out in ((self.dll.q4kp_original_gemm, packed, outs[0]),
                               (self.dll.q4kp_gemm, recoded, outs[1]),
                               (self.gemm, recoded, outs[2]),
                               (self.dll.q4kp_scalar_gemm, packed, outs[3]),
                               (self.dll.q4kp_vnni_original_gemm, packed, outs[4])):
            fn(n, out[1:].ctypes.data, stride, matrix.ctypes.data,
               q8.ctypes.data, nr, nc)
        for name, out in zip(("P6", "VNNI", "scalar", "VNNI original"), outs[1:]):
            np.testing.assert_array_equal(out.view(np.uint32), outs[0].view(np.uint32), name)
        self.assertTrue(np.isfinite(outs[2][1:-1, :nc]).all())
        self.assertTrue(np.isnan(outs[2][[0, -1]]).all())
        self.assertTrue(np.isnan(outs[2][1:-1, nc:]).all())
        np.testing.assert_array_equal(packed, original)
        np.testing.assert_array_equal(q8, activations)
        np.testing.assert_array_equal(recoded, recoded_before)
        type(self).cases += 1
        type(self).output_floats += nr * nc

    def test_body_tail_multiblock_and_prompt_batch(self):
        shapes = [(n, nr, nc) for n in (256, 512, 4096)
                  for nr in (4, 16, 20, 32) for nc in (8, 40)]
        shapes.append((512, 512, 16))
        for n, nr, nc in shapes:
            for seed in (46, 357):
                with self.subTest(n=n, nr=nr, nc=nc, seed=seed):
                    rng = np.random.default_rng(seed)
                    packed, _ = fixtures.fixtures(n, nc, seed)
                    values = rng.integers(-128, 128, (nr, n // 256, 256),
                                          dtype=np.int16).astype(np.int8)
                    delta = rng.uniform(-.3, .3, (nr, n // 256)).astype(np.float32)
                    self.compare(n, nr, nc, packed, fixtures.raw_q8x4(values, delta))

    def test_integer_extrema_and_all_scales(self):
        # Simultaneous maximum six-bit scales and signed Q8 extremes expose
        # an erroneous int16 scale/madd step after dpbusd immediately.
        for code in (0, 255, 0xF0, 0x0F, 0x87):
            packed, _ = fixtures.fixtures(512, 16, code)
            packed[:, 128:] = code
            packed[:, 32:128] = 255
            for value in (-128, -1, 0, 1, 127):
                values = np.full((20, 2, 256), value, dtype=np.int8)
                self.compare(512, 20, 16, packed,
                             fixtures.raw_q8x4(values, np.ones((20, 2), dtype=np.float32)))
        packed, _ = fixtures.fixtures(256, 512)
        scales = np.broadcast_to(np.arange(64, dtype=np.uint8)[:, None, None],
                                 (64, 8, 8)).copy()
        packed[:, 32:128] = fixtures.pack_metadata(scales, 63 - scales)
        values = np.resize(np.array([-128, 127, -1, 1], dtype=np.int8), (20, 1, 256))
        self.compare(256, 20, 512, packed,
                     fixtures.raw_q8x4(values, np.ones((20, 1), dtype=np.float32)))

    def test_half_edges_and_cancellation(self):
        packed, _ = fixtures.fixtures(512, 16)
        halves = np.array([0., -0., 2 ** -24, -(2 ** -24),
                           2 ** -14, .125, -128, 65504], dtype="<f2")
        packed[:, :16] = halves.view(np.uint8)
        packed[:, 16:32] = halves[::-1].copy().view(np.uint8)
        values = np.resize(np.array([-128, 127, -1, 1], dtype=np.int8), (20, 2, 256))
        self.compare(512, 20, 16, packed,
                     fixtures.raw_q8x4(values, np.ones((20, 2), dtype=np.float32)))


if __name__ == "__main__":
    unittest.main()
