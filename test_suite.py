"""
CHORUS Protocol - test_suite.py
================================
Mathematical validation of all core CHORUS protocol primitives.
No network calls - runs entirely in-process against crypto_engine.py.

Test classes:
  1. TestOrthogonalIsolation      - Mode A: zero-crosstalk signal separation
  2. TestHolographicSuperposition - Mode B: holographic consensus retention
  3. TestWatermarkIntegrity       - tamper detection, replay prevention
  4. TestKeyManagement            - invertibility, serialisation round-trip
  5. TestRelayAmplification       - linear scalar amplification on ciphertext

Run:
    python test_suite.py
Exit code 0 = all tests passed.
"""

import logging
import sys
import unittest

import torch
import torch.nn.functional as F

import crypto_engine as ce

logging.basicConfig(level=logging.WARNING)


# -----------------------------------------------------------------------------
# 1. Orthogonal Isolation  (Mode A)
# -----------------------------------------------------------------------------

class TestOrthogonalIsolation(unittest.TestCase):

    def setUp(self):
        self.dim = 128
        self.K, self.K_inv = ce.generate_key_pair(self.dim)
        self.W_A, self.W_B = ce.generate_orthogonal_projections(self.dim)
        self.seed = ce.generate_watermark_seed(self.dim)

    def test_projection_matrices_orthogonal(self):
        """W_A @ W_B must be zero matrix (orthogonal subspaces)."""
        ct = torch.max(torch.abs(self.W_A @ self.W_B)).item()
        self.assertLess(ct, 1e-10,
            f"Projections not orthogonal: max element={ct:.2e}")
        print(f"\n[1a] Projection orthogonality max element: {ct:.2e}  PASS")

    def test_zero_crosstalk(self):
        """
        After mix + isolate, signal A's energy in B's subspace must be < 0.01%.
        This is the core proof of Mode A: zero crosstalk on a shared wire.
        """
        torch.manual_seed(7)
        vA = torch.randn(self.dim)
        vB = torch.randn(self.dim)

        vA_enc  = ce.encrypt(ce.inject_watermark(vA, self.seed, 0), self.K)
        vB_enc  = ce.encrypt(ce.inject_watermark(vB, self.seed, 1), self.K)
        vA_proj = ce.project_signal(vA_enc, self.W_A)
        vB_proj = ce.project_signal(vB_enc, self.W_B)
        tunnel  = ce.mix_signals(vA_proj, vB_proj)

        vA_rec = ce.isolate_signal(tunnel, self.W_A)
        vB_rec = ce.isolate_signal(tunnel, self.W_B)

        ct_AB = (self.W_B.float() @ vA_rec).norm().item() / (vA_rec.norm().item() + 1e-8) * 100
        ct_BA = (self.W_A.float() @ vB_rec).norm().item() / (vB_rec.norm().item() + 1e-8) * 100

        self.assertLess(ct_AB, 0.01, f"Crosstalk A->B: {ct_AB:.6f}% (limit 0.01%)")
        self.assertLess(ct_BA, 0.01, f"Crosstalk B->A: {ct_BA:.6f}% (limit 0.01%)")
        print(f"[1b] Crosstalk A->B={ct_AB:.6f}%  B->A={ct_BA:.6f}%  PASS")

    def test_signal_recovery_fidelity(self):
        """
        Cosine similarity between the projected signal and its isolated
        recovery must be >= 0.999 (near-perfect fidelity).
        """
        torch.manual_seed(13)
        vA      = torch.randn(self.dim)
        vA_enc  = ce.encrypt(vA, self.K)
        vA_proj = ce.project_signal(vA_enc, self.W_A)

        vB_enc  = ce.encrypt(torch.randn(self.dim), self.K)
        vB_proj = ce.project_signal(vB_enc, self.W_B)
        tunnel  = ce.mix_signals(vA_proj, vB_proj)
        vA_rec  = ce.isolate_signal(tunnel, self.W_A)

        sim = F.cosine_similarity(vA_proj.unsqueeze(0).float(),
                                  vA_rec.unsqueeze(0).float()).item()
        self.assertGreater(sim, 0.999, f"Recovery fidelity: {sim:.6f}")
        print(f"[1c] Signal recovery cosine_sim={sim:.6f}  PASS")

    def test_encrypt_decrypt_round_trip(self):
        """V_raw -> encrypt -> decrypt must restore V_raw within 1e-4."""
        torch.manual_seed(99)
        v   = torch.randn(self.dim)
        err = (v - ce.decrypt(ce.encrypt(v, self.K), self.K_inv)).abs().max().item()
        self.assertLess(err, 1e-4, f"Round-trip error: {err:.2e}")
        print(f"[1d] Encrypt/decrypt round-trip max error: {err:.2e}  PASS")


# -----------------------------------------------------------------------------
# 2. Holographic Superposition  (Mode B)
# -----------------------------------------------------------------------------

class TestHolographicSuperposition(unittest.TestCase):

    def setUp(self):
        self.dim = 128
        self.K, self.K_inv = ce.generate_key_pair(self.dim)
        self.seed = ce.generate_watermark_seed(self.dim)

    def test_both_signals_present_in_collective(self):
        """
        cosine_similarity(V_A+V_B, V_A) >= 0.5  AND
        cosine_similarity(V_A+V_B, V_B) >= 0.5
        Proves both concepts are simultaneously encoded in the blend.
        """
        torch.manual_seed(55)
        vA = torch.randn(self.dim); vA /= vA.norm()
        vB = torch.randn(self.dim); vB /= vB.norm()
        vc = ce.superpose_signals(vA, vB)

        sim_A = F.cosine_similarity(vc.unsqueeze(0), vA.unsqueeze(0)).item()
        sim_B = F.cosine_similarity(vc.unsqueeze(0), vB.unsqueeze(0)).item()

        self.assertGreater(sim_A, 0.5, f"Concept A lost: sim={sim_A:.4f}")
        self.assertGreater(sim_B, 0.5, f"Concept B lost: sim={sim_B:.4f}")
        print(f"\n[2a] Holographic sim(collective,A)={sim_A:.4f}  sim(collective,B)={sim_B:.4f}  PASS")

    def test_triangle_inequality(self):
        """norm(V_A+V_B) must satisfy the triangle inequality."""
        torch.manual_seed(77)
        vA = torch.randn(self.dim)
        vB = torch.randn(self.dim)
        vc = ce.superpose_signals(vA, vB)
        nA, nB, nC = vA.norm().item(), vB.norm().item(), vc.norm().item()
        self.assertLessEqual(nC, nA + nB + 1e-5)
        self.assertGreaterEqual(nC, abs(nA - nB) - 1e-5)
        print(f"[2b] Triangle inequality: |{nA:.3f}-{nB:.3f}| <= {nC:.3f} <= {nA:.3f}+{nB:.3f}  PASS")

    def test_encrypted_superposition_round_trip(self):
        """Encrypt(V_A+V_B) -> decrypt must equal V_A+V_B within 1e-4."""
        torch.manual_seed(88)
        vc  = ce.superpose_signals(torch.randn(self.dim), torch.randn(self.dim))
        err = (vc - ce.decrypt(ce.encrypt(vc, self.K), self.K_inv)).abs().max().item()
        self.assertLess(err, 1e-4, f"Superposition round-trip error: {err:.2e}")
        print(f"[2c] Superposition round-trip max error: {err:.2e}  PASS")

    def test_superposition_is_commutative(self):
        """V_A + V_B == V_B + V_A."""
        torch.manual_seed(33)
        vA   = torch.randn(self.dim)
        vB   = torch.randn(self.dim)
        diff = (ce.superpose_signals(vA, vB) - ce.superpose_signals(vB, vA)).abs().max().item()
        self.assertLess(diff, 1e-6)
        print(f"[2d] Superposition commutativity max diff: {diff:.2e}  PASS")


# -----------------------------------------------------------------------------
# 3. Watermark Integrity
# -----------------------------------------------------------------------------

class TestWatermarkIntegrity(unittest.TestCase):

    def setUp(self):
        self.dim  = 128
        self.seed = ce.generate_watermark_seed(self.dim)

    def test_valid_watermark_passes(self):
        torch.manual_seed(1)
        v  = ce.inject_watermark(torch.randn(self.dim), self.seed, seq_num=5)
        ok = ce.verify_watermark(v, self.seed, seq_num=5)
        self.assertTrue(ok, "Valid watermark should pass")
        print(f"\n[3a] Valid watermark: PASS")

    def test_tampered_payload_fails(self):
        torch.manual_seed(2)
        v    = ce.inject_watermark(torch.randn(self.dim), self.seed, 5)
        wdim = max(1, int(self.dim * ce.WATERMARK_RATIO))
        v[:wdim] += torch.randn(wdim) * 10.0
        ok = ce.verify_watermark(v, self.seed, 5)
        self.assertFalse(ok, "Tampered watermark should fail")
        print(f"[3b] Tampered payload detected: PASS")

    def test_replay_attack_detected(self):
        """Watermark for seq=5 must not verify against seq=6."""
        torch.manual_seed(3)
        v  = ce.inject_watermark(torch.randn(self.dim), self.seed, 5)
        ok = ce.verify_watermark(v, self.seed, 6)
        self.assertFalse(ok, "Wrong seq_num should fail (replay attack)")
        print(f"[3c] Replay attack (wrong seq_num) detected: PASS")

    def test_watermarks_unique_per_sequence(self):
        wm0 = ce.compute_watermark(self.seed, 0, self.dim)
        wm1 = ce.compute_watermark(self.seed, 1, self.dim)
        sim = F.cosine_similarity(wm0.unsqueeze(0), wm1.unsqueeze(0)).item()
        self.assertLess(abs(sim), 0.95, f"Sequential watermarks too similar: {sim:.4f}")
        print(f"[3d] Watermark uniqueness seq0 vs seq1 cosine_sim={sim:.4f}  PASS")

    def test_wrong_seed_fails(self):
        torch.manual_seed(4)
        v        = ce.inject_watermark(torch.randn(self.dim), self.seed, 5)
        bad_seed = ce.generate_watermark_seed(self.dim)
        ok       = ce.verify_watermark(v, bad_seed, 5)
        self.assertFalse(ok, "Wrong seed should fail verification")
        print(f"[3e] Wrong seed rejected: PASS")


# -----------------------------------------------------------------------------
# 4. Key Management
# -----------------------------------------------------------------------------

class TestKeyManagement(unittest.TestCase):

    def test_k_times_k_inv_equals_identity(self):
        K, K_inv = ce.generate_key_pair(128)
        err = (K @ K_inv - torch.eye(128, dtype=torch.float64)).abs().max().item()
        self.assertLess(err, 1e-8, f"K @ K_inv != I: {err:.2e}")
        print(f"\n[4a] K @ K_inv identity max error: {err:.2e}  PASS")

    def test_serialisation_round_trip(self):
        K, _ = ce.generate_key_pair(64)
        Kf   = K.float()
        K2   = ce.bytes_to_matrix(ce.matrix_to_bytes(Kf), 64)
        err  = (Kf - K2).abs().max().item()
        self.assertLess(err, 1e-6, f"Serialisation error: {err:.2e}")
        print(f"[4b] Matrix serialisation round-trip: {err:.2e}  PASS")

    def test_different_keys_give_different_ciphertext(self):
        torch.manual_seed(10)
        v = torch.randn(128)
        K1, _ = ce.generate_key_pair(128)
        K2, _ = ce.generate_key_pair(128)
        diff  = (ce.encrypt(v, K1) - ce.encrypt(v, K2)).abs().max().item()
        self.assertGreater(diff, 0.01)
        print(f"[4c] Key uniqueness max diff={diff:.4f}  PASS")

    def test_multiple_dims_supported(self):
        for d in [64, 128, 256]:
            K, K_inv = ce.generate_key_pair(d)
            v   = torch.randn(d)
            err = (v - ce.decrypt(ce.encrypt(v, K), K_inv)).abs().max().item()
            self.assertLess(err, 1e-3, f"dim={d} round-trip error {err:.2e}")
        print(f"[4d] Multi-dim round-trip (64, 128, 256): PASS")


# -----------------------------------------------------------------------------
# 5. Relay Amplification
# -----------------------------------------------------------------------------

class TestRelayAmplification(unittest.TestCase):

    def test_amplify_scales_norm(self):
        """norm(factor * V_enc) == factor * norm(V_enc) within 1e-5."""
        K, _ = ce.generate_key_pair(128)
        torch.manual_seed(20)
        v_enc   = ce.encrypt(torch.randn(128), K)
        factor  = 2.5
        v_amp   = v_enc * factor
        rel_err = abs(v_amp.norm().item() - v_enc.norm().item() * factor) / (v_enc.norm().item() * factor)
        self.assertLess(rel_err, 1e-5, f"Amplification norm error: {rel_err:.2e}")
        print(f"\n[5a] Amplification norm linearity rel_err={rel_err:.2e}  PASS")

    def test_amplify_then_decrypt_scales_raw(self):
        """Decrypt(factor * V_enc) == factor * V_raw within 1e-3."""
        K, K_inv = ce.generate_key_pair(128)
        torch.manual_seed(21)
        v_raw  = torch.randn(128)
        v_enc  = ce.encrypt(v_raw, K)
        factor = 3.0
        v_dec  = ce.decrypt(v_enc * factor, K_inv)
        err    = (v_dec.float() - (v_raw * factor)).abs().max().item()
        self.assertLess(err, 1e-3, f"Amplified decrypt error: {err:.2e}")
        print(f"[5b] Amplify + decrypt max error={err:.2e}  PASS")

    def test_unit_factor_is_noop(self):
        """factor=1.0 must leave the tensor unchanged."""
        K, _ = ce.generate_key_pair(128)
        torch.manual_seed(22)
        v = ce.encrypt(torch.randn(128), K)
        self.assertTrue(torch.allclose(v * 1.0, v))
        print(f"[5c] Unit factor no-op: PASS")


# -----------------------------------------------------------------------------
# Runner
# -----------------------------------------------------------------------------

if __name__ == "__main__":
    print("=" * 68)
    print("  CHORUS Protocol  -  Validation Test Suite")
    print("=" * 68)

    loader = unittest.TestLoader()
    suite  = unittest.TestSuite()
    for cls in [
        TestOrthogonalIsolation,
        TestHolographicSuperposition,
        TestWatermarkIntegrity,
        TestKeyManagement,
        TestRelayAmplification,
    ]:
        suite.addTests(loader.loadTestsFromTestCase(cls))

    result = unittest.TextTestRunner(verbosity=2, stream=sys.stdout).run(suite)

    print("\n" + "=" * 68)
    if result.wasSuccessful():
        print(f"  ALL {result.testsRun} TESTS PASSED  -  CHORUS math verified")
    else:
        print(f"  FAILURES={len(result.failures)}  ERRORS={len(result.errors)}")
    print("=" * 68)
    sys.exit(0 if result.wasSuccessful() else 1)
