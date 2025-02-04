import pickle
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional

import torch
from torch.distributed.checkpoint.stateful import Stateful
from torch.utils.data import IterableDataset
from torchdata.stateful_dataloader import StatefulDataLoader

from torchtitan.datasets.tokenizer import Tokenizer
from torchtitan.logging import logger

from datasets import Dataset
from transformers import PreTrainedTokenizerFast


class MyDataset(IterableDataset, Stateful):
    def __init__(
        self,
        #data: List,
        tokenizer: Tokenizer,
        seq_len: int = 2048,
        world_size: int = 1,
        rank: int = 0,
        infinite: bool = False,
    ) -> None:
        self.dataset_name = "toy_dataset"
       
        self._tokenizer = tokenizer
        self.seq_len = seq_len
        
        conversation = [
            {

            "role": "user",
            "content": [
                {"type": "text", "text": "What are these?"},
                {"type": "image"},
                ],
            },
        ]
        prompt = processor.apply_chat_template(conversation, add_generation_prompt=True)

        image_file = "http://images.cocodataset.org/val2017/000000039769.jpg"
        raw_image = Image.open(requests.get(image_file, stream=True).raw)
        inputs = processor(images=raw_image, text=prompt, return_tensors='pt').to(0, torch.float16)


        # Variables for checkpointing
        self._sample_idx = 0
        self._all_tokens: List[int] = []
        self._data = [prompt]

    def _get_data_iter(self):
        if self._sample_idx == 0:
            return iter(self._data)

        if isinstance(self._data, Dataset) and self._sample_idx == len(self._data):
            return iter([])

        return iter(self._data.skip(self._sample_idx))

    def __iter__(self):
        while True:
            for sample_text in self._get_data_iter():
                if isinstance(self._tokenizer, PreTrainedTokenizerFast):
                    sample_tokens = self._tokenizer.encode(sample_text)
                else:
                    sample_tokens = self._tokenizer.encode(sample_text, bos=True, eos=True)
                
                self._sample_idx += 1

                x = torch.LongTensor(sample_tokens)
                    
                yield x

            if not self.infinite:
                logger.warning(f"Dataset {self.dataset_name} has run out of data")
                break
            else:
                # Reset offset for the next iteration
                self._sample_idx = 0
                logger.warning(f"Dataset {self.dataset_name} is being re-looped")

    def load_state_dict(self, state_dict):
        self._sample_idx = state_dict["sample_idx"]
        self._all_tokens = state_dict["token_buffer"]

    def state_dict(self):
        return {"token_buffer": self._all_tokens, "sample_idx": self._sample_idx}
