# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import pickle
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional

import torch
from torch.distributed.checkpoint.stateful import Stateful
from torch.utils.data import IterableDataset
from torchdata.stateful_dataloader import StatefulDataLoader

from torchtitan.datasets.tokenizer import Tokenizer
from torchtitan.logging import logger

from datasets import Dataset, load_dataset
from datasets.distributed import split_dataset_by_node

from scripts.long_context.eval.needle.utils import load_context, insert_needle

from transformers import PreTrainedTokenizerFast

@dataclass
class DatasetConfig:
    path: str
    loader: Callable
    text_processor: Callable


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
        self.dataset_name = "my_dataset"
        #self._data = split_dataset_by_node(ds, rank, world_size)
       
        self._tokenizer = tokenizer
        self.seq_len = seq_len
        self.infinite = infinite
        #self._text_processor = text_processor

        # Variables for checkpointing
        self._sample_idx = 0
        self._all_tokens: List[int] = []

        needle = "\nThe best thing to do in San Francisco is eat a sandwich and sit in Dolores Park on a sunny day.\n"
        depth = 0.5
        _context = load_context(fpath="scripts/long_context/eval/needle/PaulGrahamEssays/*.txt", ctx_len=seq_len)
        context = ""
        # # of entire context tokens = 148836
        for _ in range((seq_len//148836)+1):
            context += _context

        context = insert_needle(context, needle, depth=depth)
        needle_idx = context.find("The best thing to do in San Francisco is")
        logger.info("Context has %d chars, needle inserted at %d char location:\n" % (len(context), needle_idx))
        logger.info(context[needle_idx - 150: needle_idx + 150]) # look at how the needle is inserted 
        prompt ="\n<|im_start|> This is a very long story book: <book> %s </book>.\n" % context
        question = "What is the best thing to do in San Francisco?"
        prompt += "Based on the content of the book, Question: %s\nAnswer:" % question
        self._data = [prompt]

    def _get_data_iter(self):
        if self._sample_idx == 0:
            return iter(self._data)

        if isinstance(self._data, Dataset) and self._sample_idx == len(self._data):
            return iter([])

        return iter(self._data.skip(self._sample_idx))

    def __iter__(self):
        #max_buffer_token_len = 1 + self.seq_len

        while True:
            for sample in self._get_data_iter():
                # Use the dataset-specific text processor
                #sample_text = self._text_processor(sample)
                sample_text = sample
                if isinstance(self._tokenizer, PreTrainedTokenizerFast):
                    sample_tokens = self._tokenizer.encode(sample_text)
                else:
                    sample_tokens = self._tokenizer.encode(sample_text, bos=True, eos=True)
                #self._all_tokens.extend(sample_tokens)
                self._sample_idx += 1

                #while len(self._all_tokens) >= max_buffer_token_len:
                #x = torch.LongTensor(self._all_tokens[:max_buffer_token_len])
                x = torch.LongTensor(sample_tokens)
                    # update tokens to the remaining tokens
                    #self._all_tokens = self._all_tokens[max_buffer_token_len:]
                #input = x[:-1]
                #label = x[1:]
                #yield input, label
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


class DPAwareDataLoader(StatefulDataLoader, Stateful):
    """
    A wrapper around the StatefulDataLoader that ensures that the state is stored only once per DP rank.
    """

    def __init__(self, dp_rank: int, hf_ds: IterableDataset, batch_size: int):
        super().__init__(hf_ds, batch_size)
        self._dp_rank = dp_rank
        self._rank_id = f"dp_rank_{dp_rank}"

    def state_dict(self) -> Dict[str, Any]:
        # Store state only for dp rank to avoid replicating the same state across other dimensions
        return {self._rank_id: pickle.dumps(super().state_dict())}

    def load_state_dict(self, state_dict: Dict[str, Any]) -> None:
        # State being empty is valid
        if not state_dict:
            return

        if self._rank_id not in state_dict:
            logger.warning(
                f"DataLoader state is empty for dp rank {self._dp_rank}, expected key {self._rank_id}"
            )
            return
        super().load_state_dict(pickle.loads(state_dict[self._rank_id]))


def build_hf_data_loader(
    dataset_name: str,
    dataset_path: Optional[str],
    tokenizer: Tokenizer,
    batch_size: int,
    seq_len: int,
    world_size: int,
    rank: int,
    infinite: bool = True,
):
    """Build a data loader for HuggingFace datasets."""
    hf_ds = HuggingFaceDataset(
        dataset_name, dataset_path, tokenizer, seq_len, world_size, rank, infinite
    )
    return DPAwareDataLoader(rank, hf_ds, batch_size=batch_size)
