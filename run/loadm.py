"""One model loader for every stage, so precision is a flag rather than an edit in three files.

Qwen2.5-VL-7B needs 16.6 GB in bf16 on a 16 GB card. Loading it in 4-bit is the only way it fits,
but then a 3B-bf16 vs 7B-4bit comparison changes scale AND precision together and neither can be
blamed for a difference. `--load-4bit` therefore exists so the 3B can be re-run under the same
quantisation, making the scale axis a controlled comparison.
"""
from __future__ import annotations
import torch
from transformers import AutoModelForImageTextToText


def load(model_id, dev="cuda", four_bit=False, attn="eager"):
    kw = dict(dtype=torch.bfloat16, device_map=dev, attn_implementation=attn)
    if four_bit:
        from transformers import BitsAndBytesConfig
        kw["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True, bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16, bnb_4bit_use_double_quant=True)
        kw.pop("dtype")
    return AutoModelForImageTextToText.from_pretrained(model_id, **kw)
