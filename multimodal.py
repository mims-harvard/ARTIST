import os
from typing import Dict, Optional, List

import numpy as np
from pytorch_lightning.utilities.types import EVAL_DATALOADERS
from torch.utils.data import DataLoader, IterableDataset
from transformers import AutoTokenizer, DataCollatorWithPadding
from random import shuffle

from tasks import Task, MultimodalMixin
from utils_general import write_jsonl, read_jsonl, get_logger
from utils_data import ListDataset, TokenizePadAndCollate, simple_list_collate, ts_padded_collate
from sklearn.preprocessing import StandardScaler
import pickle
import ast
import string
import re
import torch
from torch.utils.data import WeightedRandomSampler
logger = get_logger(__name__)

def make_balanced_sampler(dataset) -> WeightedRandomSampler:
    # Assumes dataset[i]["label_index"] exists and is an int in [0..C-1]
    labels = torch.tensor([dataset[i]["label_index"] for i in range(len(dataset))], dtype=torch.long)

    class_counts = torch.bincount(labels)
    class_weights = 1.0 / class_counts.float().clamp(min=1)  # avoid div-by-zero
    sample_weights = class_weights[labels]

    return WeightedRandomSampler(
        weights=sample_weights,
        num_samples=len(sample_weights),
        replacement=True
    )


# Is this right? Should I be tokenizing elsewhere?
class MultimodalDataset(ListDataset):
    def __init__(self, data, tokenizer = None, 
                 context_columns=["context"],
                 ts_column="ts",
                 label_column="label",
                 context_prefix="",
                 index_from_options=False,
                 **kwargs):
        
        self.tokenizer = tokenizer
        self.index_from_options = index_from_options
        
        self.context_columns = context_columns
        self.ts_column = ts_column
        self.label_column = label_column
        self.context_prefix = context_prefix

        super().__init__(data)

    def __getitem__(self, index) -> Dict:
        data = super().__getitem__(index)
        context = self.context_prefix + " ".join([str(data[col]) for col in self.context_columns])
        if self.index_from_options:
            options = data["options"]
            label = options[data["answer_index"]]
        else:
            label = data[self.label_column]
            
        return {
            "context" : context,
            "ts" : np.array(data[self.ts_column]),
            "label" : label,
        }

class MultimodalMCQDataset(ListDataset):
    def __init__(self, data, tokenizer, 
                 context_prefix="",
                 ts_column="ts",
                 contrastive_column=None,
                 label_column="source",
                 options_column="options",
                 metadata_column='metadata',
                 description_column='description',
                 use_metadata=True,
                 shuffle_labels=False,
                 context_columns=["context"],
                 format_abc_mcq=False,
                 encoder_name=None, 
                 scale_ts=False,     
                 task_name='ETI',
                 partition=None,
                 w_cot=False,
                 **kwargs):
        
        self.tokenizer = tokenizer
        self.partition = partition
        self.context_prefix = context_prefix
        self.ts_column = ts_column
        self.label_column = label_column
        self.options_column = options_column
        self.shuffle_labels = shuffle_labels
        self.contrastive_column = contrastive_column
        self.context_columns = context_columns
        self.metadata_column = metadata_column
        self.description_column = description_column
        self.use_metadata = use_metadata
        self.format_abc_mcq = format_abc_mcq
        self.task_name = task_name
        if self.task_name == "TOY_DS_EASY":
            self.other_keys = ["span", "difficulty","question"]
        else:
            self.other_keys = ["ts_qid","uuid","category","question", "options_raw"]
        self.encoder_name = encoder_name
        self.scale_ts = scale_ts #whether to scale the ts or not
        self.task_name = task_name # ETI, 1TS, 2TS
        self.w_cot = w_cot
        if contrastive_column:
            self.other_keys.append(contrastive_column)

        super().__init__(data)

    def fit_scaler(self, data):
        # Collect all real (unpadded) time points from all series
        ts_list = []
        for item in data:
            ts = item[self.ts_column]
            if self.contrastive_column:
                ts = ts + item[self.contrastive_column]
            ts_arr = np.array(ts)
            if ts_arr.ndim == 1:
                ts_arr = ts_arr[:, None]    # (T,1)
            ts_list.append(ts_arr)
        # Concatenate along time axis: [total_points, n_features]
        all_points = np.concatenate(ts_list, axis=0)
        THRESH = 1e10  # Adjust to your real-world max
        filtered_points = all_points[np.abs(all_points) < THRESH]

        self.scaler = self.scaler.fit(filtered_points.reshape(-1, 1))

    def normalize_ts(self, ts_arr):
        if ts_arr.ndim == 1:
                ts_arr = ts_arr[:, None]

        mean = ts_arr.mean(axis=0, keepdims=True)
        std = ts_arr.std(axis=0, keepdims=True) + 1e-8  # avoid div by zero

        ts_scaled = (ts_arr - mean) / std

        # If univariate, flatten
        if ts_scaled.shape[1] == 1:
            ts_scaled = ts_scaled[:, 0]
        
        return ts_scaled

    def extract_question_and_context(self, text):
        """
        Extract the clinical context and question from a single ECG prompt text.
        
        Returns (clinical_context, question). If something is not found, returns None for it.
        """
        # --- Clinical Context ---
        # Grab everything after "Clinical Context:" up until either a blank line
        # or "Your task" (non-greedy)
        context_match = re.search(
            r"Clinical Context:\s*(.+?)(?=\n\s*\n|Your task)",
            text,
            flags=re.IGNORECASE | re.DOTALL,
        )
        clinical_context = context_match.group(1).strip() if context_match else None

        # --- Question ---
        # Grab everything after "Question:" up until a blank line or "Instructions:"
        question_match = re.search(
            r"Question:\s*(.+?)(?=\n\s*\n|Instructions:)",
            text,
            flags=re.IGNORECASE | re.DOTALL,
        )
        question = question_match.group(1).strip() if question_match else None

        return clinical_context, question
        
    def __getitem__(self, index) -> Dict:
        only_stats_ds = ['ETI', 'TIMERBED_RCW','TIMERBED_ECG', 'TOY_DS_EASY', 'TOY_DS_MEDIUM', 'TOY_DS_HARD', 'TOY_DS_MIXED', 'TOY_DS_W_MULTIPLE_SEGS']
        data = super().__getitem__(index)
        if self.options_column != '':
            options = data[self.options_column].copy()
        else:
            options = []
        timestamps = {}
        cot_l = None
        if self.w_cot and self.partition != "test" and "cot_in_format" in data:
            cot_l = data["cot_in_format"]

        if self.task_name in ["ECG_QA_S_VERIFY", "ECG_QA_S_QUERY","ECG_QA_MIXED"]:
            pre_prompt = data["pre_prompt"]
            clinical_context, question = self.extract_question_and_context(pre_prompt)
            # post_prompt = data["post_prompt"]
            ts_text = data["time_series_text_filtered"]
            ts_arr = np.array(data["filtered_ts"])

            num_points = ts_arr.shape[0]
            stats_text = f"number of points: {num_points}"
            context = pre_prompt + "\n" + ts_text + "\n" + stats_text
           
            options = data["possible_answers"]
            label_index = options.index(data["correct_answer"])
            full_options = [f"{chr(ord('A') + i)}) {x}" for i, x in enumerate(options)]
            post_prompt = "Based on your analysis of the ECG data, select your answer from the following options:" + "\n" + "\n".join(full_options) + "\n  Make sure that your last word is the answer. You MUST end your response with 'Answer: '"
            context = context + "\n" + post_prompt
            answer = f"{chr(ord('A') + label_index)}"

            timestamps = {}

            base =  {
                "context" : context,
                "ts" : ts_arr,
                "label" : answer,
                "options" : options,
                "options_raw": data["possible_answers"],
                "label_index" : label_index,
                "cot_in_format" : cot_l,
                "timestamps" : timestamps,
                'raw_ts' : ts_arr,  
                "clinical_context" : clinical_context,
                "question" : question,
                "ts_text" : ts_text,
            }
        elif self.task_name == "SLEEPQA":
            pre_prompt = data["pre_prompt"]
            clinical_context = ""
            # clinical_context, question = self.extract_question_and_context(pre_prompt)
            question = "You are given a 30-second EEG time series segment. Your task is to classify the sleep stage based on analysis of the data."
            post_prompt = data["post_prompt"]
            ts_text = data["time_series_text"][0]
            ts_arr = np.array(data["time_series"])

            num_points = ts_arr.shape[0]
            stats_text = f"number of points: {num_points}"
            context = pre_prompt + "\n" + ts_text + "\n" + stats_text + "\n" + post_prompt
           
            options = ['Wake',
                        'Non-REM stage 1',
                        'Non-REM stage 2',
                        'Non-REM stage 3',
                        'REM sleep',
                        'Movement']#data["possible_answers"]
            label_index = options.index(data["correct_answer"])
            answer = string.ascii_uppercase[label_index]

            timestamps = {}

            base =  {
                "context" : context,
                "ts" : ts_arr,
                "label" : answer,
                "options" : options,
                "options_raw": options,
                "label_index" : label_index,
                "cot_in_format" : cot_l,
                "timestamps" : timestamps,
                'raw_ts' : ts_arr,  
                "clinical_context" : clinical_context,
                "question" : question,
                "ts_text" : ts_text,
            }
        elif self.task_name == "SLEEPQA_BINARY":
            pre_prompt = data["pre_prompt"]
            clinical_context = ""
            # question = "You are given a 30-second EEG time series segment. Your task is to classify the sleep stage based on analysis of the data."
            post_prompt = data["post_prompt"]
            # ts_text = data["time_series_text"][0]
            ts_arr = np.array(data["time_series"])
            context = pre_prompt + "\n" + post_prompt
            label_index = 0 if  data["correct_option"] == 'A' else 1
            # label_index = options.index(data["correct_option"])
            # answer = string.ascii_uppercase[label_index]
            timestamps = {}
            base =  {
                "context" : context,
                "ts" : ts_arr,
                "label" : data["correct_option"],
                "cot_in_format" : cot_l,
                "label_index" : label_index,
            }
        else:
            context = "\n".join([str(data[col]) for col in self.context_columns])
            context = self.context_prefix + context
            context = "The question is: " + context

            if self.task_name not in only_stats_ds:
                if self.description_column in data:
                    description = data[self.description_column]
                    context = "The time series description is: " + description + "\n" + context

                if self.use_metadata:
                    metadata = data.get(self.metadata_column, {})
                    if metadata:
                        # if isinstance(metadata, str):
                        #     import ast
                        #     metadata = ast.literal_eval(metadata)
                        
                        # # Format metadata nicely
                        # if isinstance(metadata, dict) and metadata:
                        #     metadata_parts = []
                        #     for key, value in metadata.items():
                        #         # Capitalize first letter of key for readability
                        #         formatted_key = key.capitalize()
                        #         metadata_parts.append(f"{formatted_key}: {value}")
                        #         if 'date' in key.lower():
                        #             timestamps[key] = value
                            
                        metadata_text = f"Metadata: " + metadata
                        context = metadata_text + "\n" + context

                
                # Add time series statistics for non-ETI tasks
                ts = data[self.ts_column]
                ts_arr = np.array(ts)
                mean_ts = np.mean(ts_arr)
                std_ts = np.std(ts_arr)
                min_ts = np.min(ts_arr)
                max_ts = np.max(ts_arr)
                num_points = ts_arr.shape[0]
                stats_text = f"TS Info (mean, std, min, max, total points): {mean_ts:.3f}, {std_ts:.3f}, {min_ts:.3f}, {max_ts:.3f}, {num_points}"
                context = stats_text + "\n" + context
            elif self.task_name == "ECG_QA": # this dataset is already normalized. 
                ts = data[self.ts_column]
                ts_arr = np.array(ts)
                num_points = ts_arr.shape[0]
                stats_text = f"number of points: {num_points}"
                context = stats_text + "\n" + context
            else:
                ts = data[self.ts_column]
                ts_arr = np.array(ts)
                mean_ts = np.mean(ts_arr)
                std_ts = np.std(ts_arr)
                min_ts = np.min(ts_arr)
                max_ts = np.max(ts_arr)
                num_points = ts_arr.shape[0]
                metadata_text = f"TS Info (mean, std, min, max, total points): {mean_ts:.3f}, {std_ts:.3f}, {min_ts:.3f}, {max_ts:.3f}, {num_points}"
                context = metadata_text + "\n" + context
                if self.task_name == "RCW":
                    context = "The time series data is an underwater audio segment sampled at 2000 Hz, with normalized amplitude values." + "\n" + context
                elif self.task_name == "ECG":
                    context = "The time series data is an ECG segment sampled at 300 Hz, with values in millivolts (mV)." + "\n" + context
            
            if self.shuffle_labels:
                shuffle(options)
            
            if "answer_index" in data and self.task_name != "ETI":
                label_index = data["answer_index"]
                if self.shuffle_labels:
                    print("WARNING: You are relying on a precomputed answer index but also shuffling the labels. This is probably not what you want.")
            elif self.task_name in ["TOY_DS_EASY", "TOY_DS_MEDIUM", "TOY_DS_HARD", "TOY_DS_MIXED", "TOY_DS_W_MULTIPLE_SEGS"]:
                label_index = data["correct_index"]
            elif self.task_name == 'TSQA':
                if data['question_format'] == 'multiple_choice':    
                    answer = data[self.label_column]
                    letters_l = ['A', 'B', 'C', 'D','E','F','G','H','I','J','K']
                    label_index = letters_l.index(answer)
                else:
                    answer = data[self.label_column]
                    letters_l = ['True', 'False']
                    label_index = letters_l.index(answer)
            elif self.task_name == "TSQA_TF":
                answer = data[self.label_column]
                letters_l = ['True', 'False']
                label_index = letters_l.index(answer)
            elif self.task_name == "TRQA" or self.task_name == "TRQA_MIXED":
                if data['question_type'] == 'multiple_choices':
                    answer = data[self.label_column]
                    letters_l = ['A', 'B', 'C', 'D']
                    label_index = letters_l.index(answer)
                else: # true/false
                    answer = data[self.label_column]
                    letters_l = ['T', 'F']
                    label_index = letters_l.index(answer)
            else:
                label_index = options.index(data[self.label_column])
            
        
            if self.format_abc_mcq:
                full_options = [f"{chr(ord('A') + i)}) {x}" for i, x in enumerate(options)]
                options = [f"{chr(ord('A') + i)}" for i, x in enumerate(options)]
                context = context + "\n" + "\n".join(full_options)
                answer = options[label_index]
            elif self.task_name == "TSQA" or self.task_name == "TSQA_TF":
                context = context + ": \n" + "\n".join(options)
                if data['question_format'] == 'multiple_choice':
                    answer = data[self.label_column]
                    # letters_l = ['A', 'B', 'C', 'D','E','F','G','H','I','J','K']
                    # label_index = letters_l.index(answer)
                # else:
                #     answer = data[self.label_column]
                #     letters_l = ['True', 'False']
                #     answer = letters_l.index(answer)
               
            else:
                answer = chr(ord('A') + label_index)

            ts = data[self.ts_column]
            ts_arr = np.array(ts)  # shape [T] 
            ts_scaled = self.normalize_ts(ts_arr) if self.scale_ts else ts_arr


            ts_scaled2 = None
            if self.contrastive_column: # 2TS 
                ts2 = data[self.contrastive_column]
                ts_arr2 = np.array(ts2)  # shape [T] 
                ts_scaled2 = self.normalize_ts(ts_arr2) if self.scale_ts else ts_arr2
            

            if self.contrastive_column and ts_scaled.shape[0] != ts_scaled2.shape[0]:
                max_len = max(ts_scaled.shape[0], ts_scaled2.shape[0])
                # Pad shorter series with zeros
                if ts_scaled.shape[0] < max_len:
                    pad_len = max_len - ts_scaled.shape[0]
                    ts_scaled = np.pad(ts_scaled, (0, pad_len), mode='constant', constant_values=0)
                
                if ts_scaled2.shape[0] < max_len:
                    pad_len = max_len - ts_scaled2.shape[0]  
                    ts_scaled2 = np.pad(ts_scaled2, (0, pad_len), mode='constant', constant_values=0)
                
            
            ts_total = ts_scaled if ts_scaled2 is None else [ts_scaled, ts_scaled2]        

            base =  {
                "context" : context,
                "ts" : ts_total,
                "label" : answer,
                "options" : options,
                "options_raw": data[self.options_column],
                "label_index" : label_index,
                "cot_in_format" : cot_l,
                "timestamps" : timestamps,
                'raw_ts' : ts_arr,  
            }
            # if self.task_name == "TOY_DS_EASY":
            #     base["span"] = data["span"]
            #     base["difficulty"] = data["difficulty"]

            for key in self.other_keys:
                if key in data:
                    base[key] = data[key]   
            
            if self.task_name in ["TOY_DS_W_MULTIPLE_SEGS","TOY_DS_EASY", "TOY_DS_MEDIUM", "TOY_DS_HARD", "TOY_DS_MIXED"]:
                only_segs = []
                segs_w_info = data["segments"]
                for seg_w_info in segs_w_info:
                    seg_s = seg_w_info["start"]
                    seg_e = seg_w_info["end"]
                    only_segs.append([seg_s, seg_e])
                
                base["gt_segs"] = only_segs
        
        return base

class MultimodalOpenDataset(ListDataset):
    """Dataset for open-ended QA (not multiple choice)"""
    def __init__(self, data, tokenizer, 
                 context_prefix="",
                 ts_column="timeseries",
                 contrastive_column=None,
                 label_column="output",  # Changed from "source" 
                 metadata_column='metadata',
                 description_column='description',
                 use_metadata=False,
                 context_columns=["input"],  # Changed from "context"
                 encoder_name=None, 
                 scale_ts=True,     
                 task_name='pretrain',
                 partition=None,
                 w_cot=False,
                 **kwargs):
        
        self.tokenizer = tokenizer
        self.partition = partition
        self.context_prefix = context_prefix
        self.ts_column = ts_column
        self.label_column = label_column
        self.contrastive_column = contrastive_column
        self.context_columns = context_columns
        self.metadata_column = metadata_column
        self.description_column = description_column
        self.use_metadata = use_metadata
        self.other_keys = []#["ts_qid","uuid","category","question", "full_answer"]
        self.encoder_name = encoder_name
        self.scale_ts = scale_ts
        self.task_name = task_name
        self.w_cot = w_cot
        if contrastive_column:
            self.other_keys.append(contrastive_column)

        super().__init__(data)

    def normalize_ts(self, ts_arr):
        if ts_arr.ndim == 1:
            ts_arr = ts_arr[:, None]

        mean = float(ts_arr.mean())
        std = float(ts_arr.std()) + 1e-8

        ts_scaled = (ts_arr - mean) / std

        # If univariate, flatten
        if ts_scaled.shape[1] == 1:
            ts_scaled = ts_scaled[:, 0]
        elif ts_scaled.shape[0] == 1:
            ts_scaled = ts_scaled[0]
        
        return ts_scaled, mean, std

    def __getitem__(self, index) -> Dict:
        data = super().__getitem__(index)
        question = data["clean_question"]
        # Build context (question)
        context = "\n".join([str(data[col]) for col in self.context_columns])
        question = "The question is: " + question
        context = "Time series additional information (metadata): " + self.context_prefix + context
        cot_l = None
        if self.w_cot:
            cot_l = data["cot_in_format"]

        # Get the open-ended answer
        answer = data[self.label_column]

        # Process time series
        ts = data[self.ts_column]
        ts_arr = np.array(ts)
        min_val = ts_arr.min()
        max_val = ts_arr.max()
        
        if self.scale_ts:
            ts_scaled, mean, std = self.normalize_ts(ts_arr)
        else:
            ts_scaled = ts_arr
            mean = float(ts_arr.mean())
            std = float(ts_arr.std())

        # Format statistics nicely with limited decimal places
        metadata = f"Time series statistics: Min={min_val:.3f}, Max={max_val:.3f}, Mean={mean:.3f}, Std={std:.4f}, Total points={len(ts_arr)}"

        # context = context.replace(': <ts><ts/>', f'. {metadata}')
        # context = context.replace('</ts>', '')
        # context = context.replace('<ts>', '')
        context = question + "\n" + context + "\n" + metadata

        base = {
            "context": context,
            "ts": ts_scaled,
            "cot_in_format": cot_l,
            "label": answer,  # Full open-ended answer,
            "raw_ts": ts_arr,
        }

        for key in self.other_keys:
            if key in data:
                base[key] = data[key]   
    
        return base

class MultimodalTask(Task, MultimodalMixin):

    def __init__(self, ts_column:str = "series",
                       context_columns:List[str] = ["description_tiny", "metadata"],
                       label_column:str = "description",
                       description_column:str = "description",
                       cache_path:str = "data/processed/synthetic_descriptions",
                       batch_size:int = 16,
                       context_prefix:str = "",
                       index_from_options:bool = False,
                       contrastive_column:Optional[str] = None,
                       metadata_column='metadata',
                       **kwargs):
        
        self.cache_path = cache_path
        self.ts_column = ts_column
        self.context_columns = context_columns
        self.label_column = label_column
        self.description_column = description_column
        self.context_prefix = context_prefix    
        self.index_from_options = index_from_options
        self.contrastive_column = contrastive_column
        self.cache_path = cache_path
        self.metadata_column = metadata_column

        self.tokenizer = None #TODO Add if ever needed
        self.batch_size = batch_size

        super(Task, self).__init__()
        super(MultimodalMixin, self).__init__(**kwargs)
     
    def get_dataloader(self, partition):
        data = self.load(partition)   
        dataset = MultimodalMCQDataset(
            data, 
            tokenizer=self.tokenizer,
            context_columns=self.context_columns,
            context_prefix=self.context_prefix,
            ts_column=self.ts_column,
            format_abc_mcq=self.format_abc_mcq,         
            label_column=self.label_column,
            options_column=self.options_column,
            metadata_column=self.metadata_column,
            description_column=self.description_column,
            contrastive_column=self.contrastive_column,
            shuffle_labels=self.shuffle_labels,
            encoder_name=self.encoder_name,
            scale_ts=self.scale_ts,
            task_name=self.task_name,
            partition=partition,
        )
        
        if self.tokenizer:
            data_collator = TokenizePadAndCollate(
                tokenizer=self.tokenizer,
                features_to_tokenize=["context", "label", "options"]
            )
        else:
            data_collator = ts_padded_collate  # Use custom collate for time series padding, return to simple_list_collate if needed
            
        return DataLoader(dataset, batch_size=self.batch_size, collate_fn=data_collator, shuffle=True)
    
    def val_dataloader(self) -> DataLoader:
        return self.get_dataloader("val")
    
    def train_dataloader(self) -> DataLoader:
        return self.get_dataloader("train")
    
    def test_dataloader(self) -> DataLoader:
        return self.get_dataloader("test")
    
    def scaler_path(self):
        """Put scaler in the cache directory"""
        return os.path.join(self.cache_path, f"scaler_{self.encoder_name}.pkl")
    
    def _save_scaler(self, scaler):
        """Save scaler to disk"""
        if scaler is not None:
            with open(self.scaler_path, 'wb') as f:
                pickle.dump(scaler, f)
            print(f"Saved scaler to {self.scaler_path}")
    
    def _load_scaler(self):
        """Load scaler from disk if it exists"""
        if os.path.exists(self.scaler_path):
            with open(self.scaler_path, 'rb') as f:
                scaler = pickle.load(f)
            print(f"Loaded existing scaler from {self.scaler_path}")
            return scaler
        return None
    
    def load(self, partition):
        path = os.path.join(self.cache_path, partition + ".json")
        if not os.path.exists(path):
            path = os.path.join(self.cache_path, partition + ".jsonl")
            if not os.path.exists(path):
                logger.info("Attempting to cache data to {}".format(self.cache_path))
                if hasattr(self, "cache"):
                    self.cache()
                else:
                    raise NotImplementedError("You need to implement a cache method if the data doesn't already exist")
        return read_jsonl(path)
    
    def cache(self):
        os.makedirs(self.cache_path, exist_ok=True)
    
class MultimodalMCQTask(MultimodalTask):

    def __init__(self, num_classes:int,
                 options_column:str = "options",
                 format_abc_mcq:bool = False,
                 shuffle_labels: bool = False,
                 encoder_name: str = "totem", 
                 scale_ts: bool = False, 
                 use_metadata: bool = True,
                 metadata_column: str ='metadata',
                 task_name: str = 'ETI',         
                 w_cot: bool = False,
                 balanced_sampler: bool = False,
                 **kwargs):
        
        self.num_classes = num_classes
        self.options_column = options_column
        self.format_abc_mcq = format_abc_mcq
        self.shuffle_labels = shuffle_labels
        self.encoder_name = encoder_name
        self.scale_ts = scale_ts
        self.task_name = task_name
        self.metadata_column = metadata_column
        self.use_metadata = use_metadata
        self.w_cot = w_cot
        self.balanced_sampler = balanced_sampler
        self.dataset_kwargs = {}
        for key in ['encoder_name', 'scale_ts']:
            if key in kwargs:
                self.dataset_kwargs[key] = kwargs.pop(key)

        super().__init__(metadata_column=metadata_column, **kwargs)
    
    def get_dataloader(self, partition):
        data = self.load(partition)
        dataset_args = {
            'tokenizer': self.tokenizer,
            'context_columns': self.context_columns,
            'context_prefix': self.context_prefix,
            'ts_column': self.ts_column,
            'format_abc_mcq': self.format_abc_mcq,
            'label_column': self.label_column,
            'options_column': self.options_column,
            'contrastive_column': self.contrastive_column,
            'shuffle_labels': self.shuffle_labels,
            'encoder_name': self.encoder_name,
            'scale_ts':self.scale_ts,
            'task_name':self.task_name,
            'metadata_column':self.metadata_column,
            'use_metadata':self.use_metadata,
            'w_cot': self.w_cot,
            'partition':partition,
            **self.dataset_kwargs,  # Add any extra dataset kwargs
        }
        
        # Handle scaler for train/val/test
        if partition == "train":
            dataset_args['fit_scaler_bool'] = True
        else:
            dataset_args['fit_scaler_bool'] = False
            if hasattr(self, '_train_scaler'):
                dataset_args['scaler'] = self._train_scaler
        
        dataset = MultimodalMCQDataset(data, **dataset_args)
        
        # Save training scaler
        if partition == "train" and hasattr(dataset, 'scaler'):
            self._train_scaler = dataset.scaler
        
        if self.tokenizer:
            data_collator = TokenizePadAndCollate(
                tokenizer=self.tokenizer,
                features_to_tokenize=["context", "label", "options"]
            )
        else:
            data_collator = ts_padded_collate  # Use custom collate for time series padding, return to simple_list_collate if needed
            
        if partition == "test" or partition == "val":
            return DataLoader(dataset, batch_size=self.batch_size, collate_fn=data_collator, shuffle=False)
        else:
            if self.balanced_sampler:
                sampler = make_balanced_sampler(dataset)
                return DataLoader(dataset, batch_size=self.batch_size, collate_fn=data_collator, shuffle=False, sampler=sampler)
            else:
                return DataLoader(dataset, batch_size=self.batch_size, collate_fn=data_collator, shuffle=True)


class MultimodalOpenTask(MultimodalTask):
    """Task for open-ended questions (not multiple choice)"""

    def __init__(self, 
                 encoder_name: str = "totem", 
                 scale_ts: bool = False, 
                 use_metadata: bool = False,
                 metadata_column: str = 'metadata',
                 task_name: str = 'pretrain',
                 context_columns: List[str] = ["input"],  # Default to "question" for open QA
                 label_column: str = "output",  # Default to "answer" for open QA
                 # Accept but ignore MCQ-specific parameters for compatibility
                 shuffle_labels: bool = False,
                 format_abc_mcq: bool = False,
                 options_column: str = "options",
                 num_classes: int = None,
                 w_cot: bool = False,
                 **kwargs):
        
        self.encoder_name = encoder_name
        self.scale_ts = scale_ts
        self.task_name = task_name
        self.metadata_column = metadata_column
        self.use_metadata = use_metadata
        self.num_classes = num_classes
        
        # Store MCQ parameters (even though we ignore them)
        self.shuffle_labels = shuffle_labels
        self.format_abc_mcq = format_abc_mcq  
        self.options_column = options_column
        self.w_cot = w_cot

        self.dataset_kwargs = {}
        for key in ['encoder_name', 'scale_ts']:
            if key in kwargs:
                self.dataset_kwargs[key] = kwargs.pop(key)

        # Override defaults for open QA
        kwargs['context_columns'] = context_columns  
        kwargs['label_column'] = label_column
        
        super().__init__(**kwargs)
    
    def get_dataloader(self, partition):
        data = self.load(partition)
        dataset_args = {
            'tokenizer': self.tokenizer,
            'context_columns': self.context_columns,
            'context_prefix': self.context_prefix,
            'ts_column': self.ts_column,
            'label_column': self.label_column,
            'contrastive_column': self.contrastive_column,
            'encoder_name': self.encoder_name,
            'scale_ts': self.scale_ts,
            'task_name': self.task_name,
            'metadata_column': self.metadata_column,
            'use_metadata': self.use_metadata,
            'partition': partition,
            'w_cot': self.w_cot,
            **self.dataset_kwargs,  # Add any extra dataset kwargs
        }
        
        # Handle scaler for train/val/test
        # if partition == "train":
        #     dataset_args['fit_scaler_bool'] = True
        # else:
        #     dataset_args['fit_scaler_bool'] = False
        #     if hasattr(self, '_train_scaler'):
        #         dataset_args['scaler'] = self._train_scaler
        
        dataset = MultimodalOpenDataset(data, **dataset_args)
        
        # Save training scaler
        # if partition == "train" and hasattr(dataset, 'scaler'):
        #     self._train_scaler = dataset.scaler
        
        if self.tokenizer:
            data_collator = TokenizePadAndCollate(
                tokenizer=self.tokenizer,
                features_to_tokenize=["context", "label"]  # No "options" for open QA
            )
        else:
            data_collator = ts_padded_collate  
            
        return DataLoader(dataset, batch_size=self.batch_size, collate_fn=data_collator, shuffle=True)


class TSQAOpenTask(MultimodalOpenTask):
    """Task for TSQA dataset using open-ended format (for alignment training)"""
    
    def __init__(self, **kwargs):
        kwargs.setdefault('ts_column', 'series')
        kwargs.setdefault('context_columns', ['clean_question'])
        kwargs.setdefault('label_column', 'answer')
        kwargs.setdefault('task_name', 'TSQA_OPEN')
        kwargs.setdefault('use_metadata', True)
        super().__init__(**kwargs)
    
    def load(self, partition):
        path = os.path.join(self.cache_path, partition + ".json")
        if not os.path.exists(path):
            path = os.path.join(self.cache_path, partition + ".jsonl")
            if not os.path.exists(path):
                logger.info("Attempting to cache data to {}".format(self.cache_path))
                if hasattr(self, "cache"):
                    self.cache()
                else:
                    raise NotImplementedError("You need to implement a cache method if the data doesn't already exist")
        
        data = read_jsonl(path)
        
        # Process TSQA data for open-ended format
        processed_data = []
        for i, item in enumerate(data):
            # Create comprehensive metadata
            metadata = {
                'application_domain': item.get('application_domain', ''),
                'task_type': item.get('task_type', ''),
                'question_format': item.get('question_format', 'open_ended_question'),
                'QA_list': item.get('QA_list', ''),
            }
            
            # Clean up answer by removing extra quotes and spaces
            answer = item['answer']
            if isinstance(answer, str):
                answer = answer.strip()  # Remove leading/trailing spaces
                if answer.startswith('"') and answer.endswith('"'):
                    answer = answer[1:-1].strip()  # Remove surrounding quotes and strip again
            
            processed_item = {
                'series': item['series'],  # Time series data  
                'clean_question': item.get('clean_question', item.get('question', '')),
                'question': item.get('question', ''),
                'answer': answer,
                'application_domain': item.get('application_domain', ''),
                # Map to expected column names for MultimodalOpenDataset
                'input': item.get('clean_question', item.get('question', '')),  # question as input
                'output': answer,  # answer as output
                'timeseries': item['series'],  # time series data
                'cot_in_format': item.get('cot_in_format', None),
            }
            processed_data.append(processed_item)
        
        return processed_data
        

class TimerbedRCWTask(MultimodalMCQTask):
    """Task for TIMERBED Right Whale Call dataset"""
    
    def __init__(self, **kwargs):
        kwargs.setdefault('ts_column', 'ecg')
        kwargs.setdefault('context_columns', ['question'])
        kwargs.setdefault('label_column', 'label_new')
        kwargs.setdefault('options_column', 'options')
        kwargs.setdefault('num_classes', 2)
        kwargs.setdefault('task_name', 'RCW')
          
        super().__init__(**kwargs)
    
    def load(self, partition):
        path = os.path.join(self.cache_path, partition + ".json")
        if not os.path.exists(path):
            if hasattr(self, "cache"):
                self.cache()
            else:
                raise NotImplementedError("You need to implement a cache method")
        
        data = read_jsonl(path)
        
        # Add required fields that MultimodalMCQDataset expects
        for i, item in enumerate(data):
            item.setdefault('metadata', {'record': item.get('record', '')})
            item.setdefault('description', '')
            item.setdefault('ts_qid', item.get('record', f'rcw_{i}'))
            item.setdefault('uuid', f"rcw_{item.get('record', i)}")
            item.setdefault('category', 'right_whale_call')
            item.setdefault('options_raw', item.get('options', []))
            item.setdefault('label', item.get('label_new'))
            item.setdefault('cot_in_format', item.get('cot_in_format', None))
        
        return data


class TimerbedECGTask(MultimodalMCQTask):
    """Task for TIMERBED ECG classification dataset"""
    
    def __init__(self, **kwargs):
        kwargs.setdefault('ts_column', 'ecg')
        kwargs.setdefault('context_columns', ['question'])
        kwargs.setdefault('label_column', 'label_new')
        kwargs.setdefault('options_column', 'options')
        kwargs.setdefault('num_classes', 3)
        kwargs.setdefault('task_name', 'ECG')
        kwargs.setdefault('use_metadata', True)
        super().__init__(**kwargs)
    
    def load(self, partition):
        path = os.path.join(self.cache_path, partition + ".json")
        if not os.path.exists(path):
            if hasattr(self, "cache"):
                self.cache()
            else:
                raise NotImplementedError("You need to implement a cache method")
        
        data = read_jsonl(path)
        
        # Add required fields that MultimodalMCQDataset expects
        for i, item in enumerate(data):
            # Create metadata from ECG fields
            metadata = {
                'record': item.get('record', ''),
                'segment_id': item.get('segment_id', ''),
                'start': item.get('start', 0),
                'end': item.get('end', 0)
            }
            item.setdefault('metadata', metadata)
            item.setdefault('description', f"ECG segment from record {metadata.get('record', 'unknown')}")
            item.setdefault('ts_qid', item.get('segment_id', f'ecg_{i}'))
            item.setdefault('uuid', f"ecg_{item.get('record', i)}_{item.get('segment_id', i)}")
            item.setdefault('category', 'ecg_classification')
            item.setdefault('options_raw', item.get('options', []))
            item.setdefault('label', item.get('label_new'))
        
        return data
        


class SquareOrSine(MultimodalMCQTask):

    def cache(self):
        np.random.seed(0)
        if not os.path.exists(self.cache_path):
            os.makedirs(self.cache_path)        

        N_SAMPLES_TRAIN=10000
        N_SAMPLES_VAL=1000
        N_SAMPLES_TEST=1000

        partitions = [("train", N_SAMPLES_TRAIN), ("val", N_SAMPLES_VAL), ("test", N_SAMPLES_TEST)]
        
        def _cache_partition(partition, n_samples):
            data = []
            for _i in range(n_samples):

                if np.random.rand() > 0.5:
                    source = "square"
                    desc = "This synthetic time series represents a square wave with additive noise."
                    wave = self._square_wave()

                else:
                    source = "sine"
                    desc = "This synthetic time series represents a sine wave with additive noise."
                    wave = self._sine_wave()

                data.append({
                "context": "",
                "signal": wave,
                "label": source,
                "options": ["square", "sine"],
                })
            
            with open(os.path.join(self.cache_path, partition + ".json"), "w") as f:
                write_jsonl(f, data)
               
        list(map(lambda x: _cache_partition(*x), partitions))

    def _square_wave(self):
        # Generate a square wave with random amplitude and frequency
        # and additive noise
        n = np.random.randint(1, 10)
        amplitude = np.random.uniform(0, 1)
        frequency = np.random.uniform(5, 20)
        noise = np.random.uniform(0, 1,1000)
        x = np.linspace(0, 1, 1000)
        y = amplitude * np.sin(2 * np.pi * frequency * x) 
        y = np.where(y > 0, 1, -1) + noise
        return y

        
    def _sine_wave(self):
        # Generate a square wave with random amplitude and frequency
        # and additive noise
        n = np.random.randint(1, 10)
        amplitude = np.random.uniform(0, 1)
        frequency = np.random.uniform(5, 20)
        noise = np.random.uniform(0, 1,1000)
        x = np.linspace(0, 1, 1000)
        y = amplitude * np.sin(2 * np.pi * frequency * x) 
        y += noise
        return y
