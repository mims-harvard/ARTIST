from typing import Any, Dict, List

from torch.utils.data import Dataset, default_collate


from torch.nn.utils.rnn import pad_sequence
from transformers import PreTrainedTokenizerBase
import transformers
import numpy as np

from src.utils import get_logger
logger = get_logger(__name__)

class ListDataset(Dataset):

    def __init__(self, data):
        super().__init__()
        self.data = data

    def __getitem__(self, index) -> Dict:
        return self.data[index]
    
    def __len__(self) -> int:
        return len(self.data)
    


class TokenizePadAndCollate(object):
    MAX_LEN = 256       

    "A collate function that tokenizes, pads and collates a batch of samples"
    def __init__(self, tokenizer : PreTrainedTokenizerBase, features_to_tokenize) -> None:
        self.tokenizer = tokenizer
        self.features_to_tokenize = features_to_tokenize
        logger.warning(f"Max length of tokenized input is {self.MAX_LEN}")
    
    def __call__(self, batch: List[Dict]) -> Dict[str, Any]:
       return self.tokenize_features(batch)
    
    def tokenize_features(self, batch):
        tokenized_batch = {}
        for key, _value in batch[0].items():
            if key in self.features_to_tokenize:
                if isinstance(batch[0][key], list):
                    tokenized_batch[key] = pad_sequence([self.tokenizer(item[key], padding="do_not_pad", truncation=True,
                                                        return_tensors="pt")["input_ids"] for item in batch], batch_first=True)
                else:
                    tokenized_batch[key] = self.tokenizer([item[key] for item in batch], padding="do_not_pad", truncation=True,
                                                        return_tensors="pt", max_length=self.MAX_LEN)["input_ids"]
            else:
                tokenized_batch[key] = [item[key] for item in batch]
        return tokenized_batch

def simple_list_collate(batches):
    batch = {}
    for key, _value in batches[0].items():
        batch[key] = [item[key] for item in batches]
    return batch

def ts_padded_collate(batches, min_ts_len=600):
    """
    Custom collate function that handles time series padding for batching.
    This ensures all time series in a batch have the same length.
    """
    batch = {}
    
    # Handle time series padding separately
    if 'ts' in batches[0]:
        ts_list = [item['ts'] for item in batches]
        
        # Check if we have 2TS or 1TS
        if isinstance(ts_list[0], (list, tuple)):
            # 2TS case: [[ts1, ts2], [ts1, ts2], ...]
            # Find max length for each series type, enforce minimum
            max_len_1 = max(max(len(ts_pair[0]) for ts_pair in ts_list), min_ts_len)
            max_len_2 = max(max(len(ts_pair[1]) for ts_pair in ts_list), min_ts_len)
            
            padded_ts = []
            for ts_pair in ts_list:
                # Pad first series
                ts1 = np.array(ts_pair[0])
                if len(ts1) < max_len_1:
                    ts1 = np.pad(ts1, (0, max_len_1 - len(ts1)), mode='constant', constant_values=0)
                
                # Pad second series
                ts2 = np.array(ts_pair[1])
                if len(ts2) < max_len_2:
                    ts2 = np.pad(ts2, (0, max_len_2 - len(ts2)), mode='constant', constant_values=0)
                
                padded_ts.append([ts1, ts2])
            
            batch['ts'] = padded_ts
        else:
            # 1TS case: [ts1, ts2, ...]
            # Enforce minimum length for PatchTST
            max_len = max(max(len(ts) for ts in ts_list), min_ts_len)
            
            padded_ts = []
            for ts in ts_list:
                ts_array = np.array(ts)
                if len(ts_array) < max_len:
                    ts_array = np.pad(ts_array, (0, max_len - len(ts_array)), mode='constant', constant_values=0)
                padded_ts.append(ts_array)
            
            batch['ts'] = padded_ts
    
    # Handle other fields normally (as lists, no padding)
    for key, _value in batches[0].items():
        if key != 'ts':  # Already handled above
            batch[key] = [item[key] for item in batches]
    
    return batch

def smart_tokenizer_and_embedding_resize(
    special_tokens_dict: Dict,
    tokenizer: transformers.PreTrainedTokenizer,
    model: transformers.PreTrainedModel,
):
    """Resize tokenizer and embedding.

    Note: This is the unoptimized version that may make your embedding size not be divisible by 64.
    """
    num_new_tokens = tokenizer.add_special_tokens(special_tokens_dict)
    model.resize_token_embeddings(len(tokenizer))

    if num_new_tokens > 0:
        input_embeddings = model.get_input_embeddings().weight.data
        output_embeddings = model.get_output_embeddings().weight.data

        input_embeddings_avg = input_embeddings[:-num_new_tokens].mean(
            dim=0, keepdim=True)
        output_embeddings_avg = output_embeddings[:-num_new_tokens].mean(
            dim=0, keepdim=True)

        input_embeddings[-num_new_tokens:] = input_embeddings_avg
        output_embeddings[-num_new_tokens:] = output_embeddings_avg


