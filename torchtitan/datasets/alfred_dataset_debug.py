from torchtitan.datasets import build_hf_data_loader, build_tokenizer, build_hf_processor
from huggingface_hub import snapshot_download
from torchtitan.logging import init_logger, logger

init_logger()

model_name = "llava-hf/llava-onevision-qwen2-7b-ov-hf"

processor = build_hf_processor(model_name)
tokenizer = processor.tokenizer
img_data_dir = snapshot_download(repo_id="bosungkim/alfred-small-img", repo_type="dataset", allow_patterns="*.tar", local_dir='data/alfred-small-img')
processor.tokenizer.add_special_tokens({"additional_special_tokens": ['<|act|>']})

# TODO make fancier
from torchtitan.datasets.hf_datasets import DPAwareDataLoader
from torchtitan.datasets.alfred_dataset import ALFREDDataset
dataset = ALFREDDataset(processor=processor, img_data_dir=img_data_dir)
dp_rank=0
data_loader = DPAwareDataLoader(dp_rank, dataset, 
                                batch_size=1,
                                world_size=1)

for batch in data_loader:
    print(batch['input_ids'])