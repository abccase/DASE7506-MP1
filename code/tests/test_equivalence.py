"""Guards for the generalised student model.

Two things are checked here that the supplied contract tests do not cover:

1. Numerical equivalence with the classroom baseline.  The student model is
   self-contained and uses separate q/k/v projections, whereas `model.py` fuses
   them into one `qkv` projection.  Because both consume the RNG stream
   differently, equivalence must be checked with *identical weights*: the
   baseline's fused weight is split into q/k/v and loaded into the student model.
   This validates the computation path, which is what the comparisons rely on.

2. Parameter counts of the candidate configs, so the numbers quoted in
   specs/001-param-realloc-gqa/research.md cannot silently drift from the code.

Run: python -m unittest discover -s tests -v
"""
import json
import unittest
from pathlib import Path

import torch

from model import GPT as BaselineGPT
from student import build_model

CONFIG_DIR = Path(__file__).resolve().parent.parent / 'configs'

MHA_CONFIG = dict(vocab=2048, width=32, heads=4, depth=2, context=256, kv_heads=4)
MQA_CONFIG = dict(vocab=2048, width=32, heads=4, depth=2, context=256, kv_heads=1)

# Cross-implementation tolerance.  The baseline multiplies with one fused
# [3*width, width] matrix while the student model uses three [width, width]
# matrices; the results agree mathematically but not bit-for-bit in float32.
EQUIVALENCE_ATOL = 1e-5
EQUIVALENCE_RTOL = 1e-4

# Expected parameter totals from research.md R2.  Verified analytically as
# Total = 2306*d + L*(attn + mlp + 4d) and against the measured baseline.
EXPECTED_PARAMETERS = {
    'base.json': 1_088_256,
    # Depth-leaning arm of the equal-parameter frontier (rejected by measurement).
    'c1.json': 989_184,
    'c3.json': 1_056_272,
    'c4.json': 1_170_176,
    'c5.json': 1_004_352,
    'c6.json': 1_039_620,
    # Width-leaning arm, found by searching the frontier in the other direction.
    'w1.json': 1_084_176,
    'w2.json': 1_069_152,
    'w3.json': 1_083_532,
    'w4.json': 1_111_120,
    'w5.json': 1_090_980,
    'w6.json': 1_153_856,
    'w7.json': 1_121_568,
    'w8.json': 1_030_956,
    # Head-count sweep at the winning W7 shape.  Multi-head attention keeps
    # 4*d^2 + 4*d parameters per block regardless of the head count, so all three
    # must equal w7.json exactly; only the head_dim changes.
    'w7h2.json': 1_121_568,
    'w7h7.json': 1_121_568,
    'w7h14.json': 1_121_568,
    # Multi-head depth ladder at ~the baseline parameter count, used to test how
    # the width/depth allocation optimum moves with the number of processed
    # training targets.  Widths are the largest multiples of heads=4 that keep
    # the total inside the baseline's +/-3% band.
    'mhal5.json': 1_082_396,
    'mhal6.json': 1_097_280,
    'mhal7.json': 1_079_700,
    'mhal8.json': 1_116_096,
}


def baseline_state_for_student(baseline):
    """Map a baseline state_dict onto the student model's layout.

    `model.py` computes q, k and v with a single `qkv` linear whose output is
    viewed as (3, heads, head_dim), so rows [0:d], [d:2d] and [2d:3d] are the
    q, k and v weights respectively.
    """
    state = {
        'token.weight': baseline.token.weight.detach().clone(),
        'pos.weight': baseline.pos.weight.detach().clone(),
        'norm.weight': baseline.norm.weight.detach().clone(),
        'norm.bias': baseline.norm.bias.detach().clone(),
        'head.weight': baseline.head.weight.detach().clone(),
    }
    for index, block in enumerate(baseline.blocks):
        prefix = f'blocks.{index}.'
        width = block.qkv.weight.shape[1]
        for offset, name in enumerate(('q', 'k', 'v')):
            rows = slice(offset * width, (offset + 1) * width)
            state[prefix + f'attn.{name}.weight'] = block.qkv.weight.detach()[rows].clone()
            state[prefix + f'attn.{name}.bias'] = block.qkv.bias.detach()[rows].clone()
        state[prefix + 'attn.proj.weight'] = block.proj.weight.detach().clone()
        state[prefix + 'attn.proj.bias'] = block.proj.bias.detach().clone()
        for norm in ('norm1', 'norm2'):
            state[prefix + f'{norm}.weight'] = getattr(block, norm).weight.detach().clone()
            state[prefix + f'{norm}.bias'] = getattr(block, norm).bias.detach().clone()
        state[prefix + 'mlp.0.weight'] = block.mlp[0].weight.detach().clone()
        state[prefix + 'mlp.0.bias'] = block.mlp[0].bias.detach().clone()
        state[prefix + 'mlp.2.weight'] = block.mlp[2].weight.detach().clone()
        state[prefix + 'mlp.2.bias'] = block.mlp[2].bias.detach().clone()
    return state


class EquivalenceTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(2)
        torch.manual_seed(17)
        self.baseline = BaselineGPT(dict(MHA_CONFIG)).eval()
        self.student = build_model(dict(MHA_CONFIG)).eval()

    def test_state_dict_keys_match_the_mapping(self):
        state = baseline_state_for_student(self.baseline)
        self.assertEqual(set(state), set(self.student.state_dict()))

    def test_multi_head_path_reproduces_the_baseline(self):
        state = baseline_state_for_student(self.baseline)
        self.student.load_state_dict(state, strict=True)
        ids = torch.randint(0, 2048, (2, 12))
        with torch.no_grad():
            expected = self.baseline(ids)
            observed = self.student(ids)
        difference = (expected - observed).abs().max().item()
        self.assertLess(
            difference, EQUIVALENCE_ATOL,
            msg=f'student MHA logits differ from baseline by {difference:.3e}')

    def test_multi_head_path_reproduces_the_baseline_log_probs(self):
        state = baseline_state_for_student(self.baseline)
        self.student.load_state_dict(state, strict=True)
        ids = torch.randint(0, 2048, (2, 12))
        with torch.no_grad():
            expected = self.baseline.predict_log_probs(ids)
            observed = self.student.predict_log_probs(ids)
        torch.testing.assert_close(
            observed, expected, atol=EQUIVALENCE_ATOL, rtol=EQUIVALENCE_RTOL)

    def test_mqa_path_is_causal_and_normalized(self):
        """The official contract tests only exercise the default (multi-head) path."""
        model = build_model(dict(MQA_CONFIG)).eval()
        ids = torch.randint(0, 2048, (2, 12))
        changed = ids.clone()
        changed[:, 7:] = (changed[:, 7:] + 19) % 2048
        with torch.no_grad():
            original = model.predict_log_probs(ids)
            perturbed = model.predict_log_probs(changed)
        self.assertEqual(tuple(original.shape), (2, 12, 2048))
        torch.testing.assert_close(
            original.logsumexp(-1), torch.zeros(2, 12), atol=1e-6, rtol=1e-6)
        torch.testing.assert_close(
            original[:, :7], perturbed[:, :7], atol=1e-6, rtol=1e-6)


class ParameterCountTests(unittest.TestCase):
    def test_candidate_configs_match_the_research_table(self):
        for name, expected in EXPECTED_PARAMETERS.items():
            with self.subTest(config=name):
                config = json.loads((CONFIG_DIR / name).read_text())
                model = build_model(config)
                observed = sum(p.numel() for p in model.parameters())
                self.assertEqual(
                    observed, expected,
                    msg=f'{name}: got {observed}, research.md R2 says {expected}')

    def test_kv_heads_defaults_to_multi_head(self):
        """configs/baseline.json has no kv_heads field and must stay usable."""
        with_kv = build_model(dict(MHA_CONFIG))
        without = build_model({k: v for k, v in MHA_CONFIG.items() if k != 'kv_heads'})
        self.assertEqual(
            sum(p.numel() for p in with_kv.parameters()),
            sum(p.numel() for p in without.parameters()))

    def test_baseline_config_reproduces_the_measured_parameter_count(self):
        config = json.loads((CONFIG_DIR / 'baseline.json').read_text())
        model = build_model(config)
        self.assertEqual(sum(p.numel() for p in model.parameters()), 1_088_256)


class ConfigValidationTests(unittest.TestCase):
    def test_non_divisible_kv_heads_is_rejected(self):
        """heads=4, kv_heads=3 would broadcast to 3 heads and fail opaquely inside SDPA."""
        with self.assertRaises(ValueError):
            build_model({**MHA_CONFIG, 'heads': 4, 'kv_heads': 3})

    def test_out_of_range_kv_heads_is_rejected(self):
        with self.assertRaises(ValueError):
            build_model({**MHA_CONFIG, 'kv_heads': 8})
        with self.assertRaises(ValueError):
            build_model({**MHA_CONFIG, 'kv_heads': 0})


if __name__ == '__main__':
    unittest.main()