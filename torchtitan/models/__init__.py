# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from torchtitan.models.llama import llama3_configs, Transformer
from torchtitan.models.llava_onevision import llava_onevision_configs, LlavaOnevisionForConditionalGeneration


models_config = {
    "llama3": llama3_configs,
    "llava-hf/llava-onevision-qwen2-7b-ov-hf": llava_onevision_configs,
}

model_name_to_cls = {
    "llama3": Transformer,
    "llava-hf/llava-onevision-qwen2-7b-ov-hf": LlavaOnevisionForConditionalGeneration,
}

model_name_to_tokenizer = {
    "llama3": "tiktoken",
}
