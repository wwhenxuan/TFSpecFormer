import inspect
import sys
import unittest
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tests.support import tiny_config
from tfsformer.layers import MemoryAttention


class MemoryAttentionTest(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(0)
        self.config = tiny_config()
        self.batch, self.seq = 2, 6
        self.hidden = torch.randn(self.batch, self.seq, self.config.d_model)
        self.mask = torch.zeros(self.batch, 1, 1, self.seq)
        self.positions = torch.arange(self.seq).unsqueeze(0)

    def test_module_does_not_depend_on_fla_or_token_ids(self):
        source = Path(inspect.getfile(MemoryAttention)).read_text()
        self.assertNotIn("flash_attn", source)
        self.assertNotIn("import fla", source)
        self.assertNotIn("past_key_values", source)
        self.assertNotIn("input_ids", inspect.signature(MemoryAttention.forward).parameters)

    def test_shared_memory_is_broadcast_and_used(self):
        module = MemoryAttention(self.config, use_rope=True).eval()
        self.assertEqual(tuple(module.memory.shape), (self.config.num_heads, self.config.d_kv))
        first = module(self.hidden, self.mask, self.positions).hidden_states
        with torch.no_grad():
            module.memory.add_(1.0)
        second = module(self.hidden, self.mask, self.positions).hidden_states
        self.assertFalse(torch.allclose(first, second))

    def test_headwise_gate_changes_the_output(self):
        module = MemoryAttention(self.config, use_rope=False).eval()
        first = module(self.hidden, self.mask).hidden_states
        head_dim = self.config.d_kv
        gate_rows = [(head + 1) * (head_dim + 1) - 1 for head in range(self.config.num_heads)]
        with torch.no_grad():
            module.q.weight[gate_rows] *= 10
        second = module(self.hidden, self.mask).hidden_states
        self.assertFalse(torch.allclose(first, second))

    def test_sdpa_matches_eager_and_backward_reaches_memory(self):
        module = MemoryAttention(self.config, use_rope=True).eval()
        module.config._attn_implementation = "sdpa"
        sdpa_out = module(self.hidden, self.mask, self.positions).hidden_states
        module.config._attn_implementation = "eager"
        eager_out = module(self.hidden, self.mask, self.positions).hidden_states
        self.assertTrue(torch.allclose(sdpa_out, eager_out, atol=1e-4, rtol=1e-4))

        module.train()
        module(self.hidden, self.mask, self.positions).hidden_states.sum().backward()
        self.assertIsNotNone(module.memory.grad)
        self.assertTrue(torch.isfinite(module.memory.grad).all())


if __name__ == "__main__":
    unittest.main()
