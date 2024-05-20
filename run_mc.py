import argparse
import torch
import numpy as np
import os
from tqdm.auto import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM, T5ForConditionalGeneration
from dataset_mc import get_dataset, choices


id2choice = {i: c for i, c in enumerate(choices)}


parser = argparse.ArgumentParser()
parser.add_argument("--model", "-m", type=str, default="google/flan-t5-small")
parser.add_argument("--dataset", type=str, default="pythonio")
parser.add_argument("--options", type=int, default=4)
parser.add_argument("--ntrain", "-k", type=int, default=5)

# args = parser.parse_args('--model /amax/ziyinzhang/projects/models/mt5s/mtk-3b7'.split())
args = parser.parse_args()
dataset = get_dataset(args.dataset, args.options, args.ntrain)
save_dir = f"predictions/{args.dataset}/"
print(args)
device = 'cuda:0'
model_name = args.model.split('/')[-1].lower()
os.makedirs(save_dir, exist_ok=True)


def is_decoder(model_name):
    return False if any(m in model_name for m in ['t5', 'tk', 'ul2']) else True

def get_option_ids(model_name, tokenizer):
    if any(m in model_name for m in ['llama-3', 'qwen1.5', 'bloom', 'gemma', 'pythia', 'phi-2', 'phi-1']):
        return [tokenizer(f" {choices[j]}", add_special_tokens=False).input_ids[0] for j in range(args.options)]
    if any(m in model_name for m in ['t5', 'tk', 'ul2', 'llama-2', 'phi-3', 'mistral', 'chatglm3']):
        return [tokenizer(choices[j], add_special_tokens=False).input_ids[0] for j in range(args.options)]
    raise NotImplementedError()

def load_model():
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(args.model, torch_dtype=torch.bfloat16, device_map='auto', trust_remote_code=True) if is_decoder(model_name) \
        else T5ForConditionalGeneration.from_pretrained(args.model, torch_dtype=torch.bfloat16, device_map='auto')
    return model.eval(), tokenizer

model, tokenizer = load_model()
option_ids = get_option_ids(model_name, tokenizer)
print(f"Option ids: {option_ids}")


@torch.no_grad()
def eval(args, model, tokenizer, dataset):
    cors = []
    preds = []
    all_probs = []

    for i, (prompt, label) in tqdm(enumerate(dataset)):
       
        input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to(device)

        if is_decoder(model_name):
            logits = model(input_ids=input_ids).logits[0,-1]
        else:
            decoder_input_ids = tokenizer("", return_tensors="pt").input_ids.to(device)
            decoder_input_ids = model._shift_right(decoder_input_ids)
            logits = model(
                input_ids=input_ids, decoder_input_ids=decoder_input_ids
            ).logits[0,-1]

        probs = (
            torch.nn.functional.softmax(
                torch.tensor([logits[id] for id in option_ids]).float(),
                dim=0,
            )
            .detach().cpu().numpy()
        )
        pred = id2choice[np.argmax(probs)]

        cor = pred == label
        cors.append(cor)
        preds.append(pred)

        all_probs.append(probs)

        if i == len(dataset) - 1:
            break

    acc = np.mean(cors)
    cors = np.array(cors)

    all_probs = np.array(all_probs)

    return cors, acc, all_probs, preds

def eval_subject(dataset):
    cors, acc, probs, preds = eval(args, model, tokenizer, dataset)
    test_df = dataset.df
    test_df["Prediction"] = preds
    test_df["Correct"] = cors.astype(np.int64)
    for j in range(probs.shape[1]):
        choice = choices[j]
        test_df[f"Choice_{choice}_prob"] = probs[:, j]
    print(f"Accuracy of {model_name} on {args.dataset}: {acc*100:.2f}")
    return test_df

# main
result = eval_subject(dataset)
result.to_json(f"{save_dir}/{model_name}.jsonl", orient='records', lines=True)