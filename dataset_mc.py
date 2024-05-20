from torch.utils.data import Dataset
import pandas as pd

choices = ["A", "B", "C", "D", "E", "F", "G", "H"]


def format_example(df, idx, include_answer=True, options=4):
    prompt = df.loc[idx]["Question"]

    for j in range(options):
        prompt += f"\n{choices[j]}. {df.loc[idx][choices[j]]}"

    prompt += "\nAnswer:"
    if include_answer:
        prompt += f" {df.loc[idx]['Answer']}\n\n"

    return prompt


def gen_prompt(train_df, k=-1, prompt_start=''):
    prompt = prompt_start
    if k == -1:
        k = train_df.shape[0]
    for i in range(k):
        prompt += format_example(train_df, i)
    return prompt


class base_dataset(Dataset):
    def __init__(self, data_path, options=4, k=5, prompt_start=''):
        self.train_df = pd.read_json(f"{data_path}/train.jsonl", lines=True, dtype=str)
        self.df = pd.read_json(f"{data_path}/test.jsonl", lines=True, dtype=str)
        self.options = options
        self.k = min(k, len(self.train_df))
        self.prompt_start = prompt_start
        self.len = len(self.df)
    
    def __getitem__(self, idx):
        prompt_end = format_example(self.df, idx, include_answer=False, options=self.options)
        train_prompt = gen_prompt(self.train_df, self.k, self.prompt_start)
        prompt = train_prompt + prompt_end
        return prompt, self.df.loc[idx]['Answer']
    
    def __len__(self):
        return self.len


class multisubject_dataset(Dataset):
    def __init__(self, data_path, options=4, k=5, prompt_start=''):
        self.options = options
        self.k = k
        self.prompt_start = prompt_start

        self.df = pd.read_json(f"{data_path}/test.jsonl", lines=True, dtype=str)
        train_df = pd.read_json(f"{data_path}/train.jsonl", lines=True, dtype=str)
        self.train_df_cats = {k: train_df[train_df.Type==k].reset_index(drop=True) for k in set(train_df.Type)}
        self.len = len(self.df)
    
    def __getitem__(self, idx):
        prompt_end = format_example(self.df, idx, include_answer=False, options=self.options)
        train_prompt = gen_prompt(self.train_df_cats[self.df.loc[idx].Type], self.k, self.prompt_start)
        prompt = train_prompt + prompt_end
        return prompt, self.df.loc[idx]['Answer']
    
    def __len__(self):
        return self.len


def get_dataset(dataset, options=4, k=5):
    if dataset == 'gsm8k':
        return base_dataset('data/gsm8k-mc', options, k, "The following are multiple choice questions (with answers) about grade school math.\n\n")
    if dataset == 'math':
        return multisubject_dataset('data/math-mc', options, k, "The following are multiple choice questions (with answers) about high school math.\n\n")
    if dataset == 'pythonio':
        return base_dataset('data/pythonio-mc', options, k, "The following are multiple choice questions (with answers) about Python program reasoning.\n\n")
    raise NotImplementedError()
