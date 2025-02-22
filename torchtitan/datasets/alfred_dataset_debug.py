from torchtitan.datasets import build_hf_data_loader, build_tokenizer, build_hf_processor
from huggingface_hub import snapshot_download
from torchtitan.logging import init_logger, logger

init_logger()

model_name = "llava-hf/llava-onevision-qwen2-7b-ov-hf"

processor = build_hf_processor(model_name)
tokenizer = processor.tokenizer
#img_data_dir = snapshot_download(repo_id="bosungkim/alfred-small-img", repo_type="dataset", allow_patterns="*.tar", local_dir='data/alfred-small-img')
#img_data_dir = img_data_dir = '/root/torchtitan/data/alfred-full/data'
traj_data_dir = 'data/alfred/train_small_traj'
img_data_dir = 'data/alfred/train_small_img'
processor.tokenizer.add_special_tokens({"additional_special_tokens": ['<|act|>']})

# from torchtitan.datasets.hf_datasets import DPAwareDataLoader
from torchtitan.datasets.alfred_dataset import ALFREDDataset, AlfredDataLoader
dataset = ALFREDDataset(processor=processor, 
traj_data_dir=traj_data_dir,img_data_dir=img_data_dir, max_seq_len=32768)
dp_rank=0
data_loader = AlfredDataLoader(dp_rank, dataset, 
                                batch_size=1,
                                world_size=1)

for batch in data_loader:
    print(batch['input_ids'])