from torchtitan.models.llava_onevision.model import LlavaOnevisionForConditionalGeneration
from transformers import AutoConfig

__all__ = [LlavaOnevisionForConditionalGeneration]

llava_onevision_configs = {
    # prob need to change variable names such as `dim`, `n_kv_heads` ...
    '7B': AutoConfig.from_pretrained("llava-hf/llava-onevision-qwen2-7b-ov-hf")
}