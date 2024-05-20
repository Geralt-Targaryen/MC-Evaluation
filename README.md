# MC Evaluation

This is the repo for the paper Multiple-Choice Questions are Efficient and Robust LLM Evaluators.

## Data

In `data.tar.gz`, there are three folders and five jsonl files:

- `gsm8k-mc`
- `math-mc`
- `pythonio-mc`
- `gsm8k-test-candidates.jsonl`, `gsm8k-trian-candidates.jsonl`
- `math-test-candidates.jsonl`, `math-trian-candidates.jsonl`
- `pythonio-candidates.jsonl`

In each of the three folders, there are one `test.jsonl` and one `train.jsonl`, which are the multiple-choice questions used in our paper.

The other five jsonl files contain the complete candidate pool for each problem generated by us, represented as a list of strings. For example:

```
{
    "question":"Natalia sold clips to 48 of her friends in April, and then she sold half as many clips in May. How many clips did Natalia sell altogether in April and May?",
    "candidates":["72","84","96","292","4896","36","60","144","48","6","1800","30040"]
}
```

The first candidate in the list is the ground-truth answer.

## Evaluation

To evaluate a model on one of the three datasets, run:

```
python run_mc.py --dataset gsm8k --model google/flan-t5-small
```

where the dataset can be either of `gsm8k`, `math`, and `pythonio`. The model argument can be a model name on Hugging Face, or a local directory.
