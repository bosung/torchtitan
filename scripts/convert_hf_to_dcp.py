import os
import torch
from transformers import AutoConfig, AutoProcessor, LlavaOnevisionForConditionalGeneration
import torch.distributed.checkpoint as DCP

# Initialize model and accelerator

model_name = "llava-hf/llava-onevision-qwen2-7b-ov-hf"

#config = AutoConfig.from_pretrained(model_name)
#config.text_config.rope_scaling = {'type': 'dynamic', 'factor': 1.0} os.environ['MASTER_PORT'] = '12355'
#config.text_config.rope_scaling = {'type': 'linear', 'factor': 1.0}
processor = AutoProcessor.from_pretrained(model_name)
tokenizer = processor.tokenizer
processor.tokenizer.add_special_tokens({"additional_special_tokens": ['<|act|>']})

model = LlavaOnevisionForConditionalGeneration.from_pretrained(
    model_name,
    torch_dtype=torch.bfloat16,
    attn_implementation=None
    #config=config
    )
model.resize_token_embeddings(len(processor.tokenizer))

# TODO: Convert and save the distributed checkpoint
# Save the distributed checkpoint
output_dir = "./distributed_checkpoint"

#DCP.save({"model": model.state_dict()}, storage_writer=DCP.filesystem.FileSystemWriter(output_dir, thread_count=1))
#DCP.save(model.state_dict(), storage_writer=DCP.filesystem.FileSystemWriter(output_dir, thread_count=1))
DCP.save(model.state_dict(), checkpoint_id=output_dir)

print(f"Distributed checkpoint saved at {output_dir}")