import time
import json
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

models = {
    "NEWS MODEL": "../saved_models/l3cube_news",
    "TITLE MODEL": "../saved_models/l3cube_title"
}

def load_model(MODEL_PATH):
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_PATH).to(DEVICE)
    model.eval()

    with open(f"{MODEL_PATH}/label_map.json", "r") as f:
        label2id = json.load(f)

    id2label = {int(v): k for k, v in label2id.items()}

    return tokenizer, model, id2label


def classify_single(news_dict, tokenizer, model, id2label):
    title = news_dict.get("TITLE", "")
    description = news_dict.get("DESCRIPTION", "")
    text = title + " " + description

    inputs = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        max_length=256,
        padding=True
    )

    inputs = {k: v.to(DEVICE) for k, v in inputs.items()}

    with torch.no_grad():
        outputs = model(**inputs)
        pred = torch.argmax(outputs.logits, dim=1).item()

    return id2label[pred]


def classify_batch(news_list, tokenizer, model, id2label):
    texts = [
        item.get("TITLE", "") + " " + item.get("DESCRIPTION", "")
        for item in news_list
    ]

    inputs = tokenizer(
        texts,
        return_tensors="pt",
        truncation=True,
        max_length=256,
        padding=True
    )

    inputs = {k: v.to(DEVICE) for k, v in inputs.items()}

    with torch.no_grad():
        outputs = model(**inputs)
        preds = torch.argmax(outputs.logits, dim=1).cpu().numpy()

    return [id2label[int(p)] for p in preds]

with open("../dataset_l3cube/input.json", "r") as f:
    data = json.load(f)

samples = data[:30]

for model_name, model_path in models.items():

    print(f"Running for: {model_name}")

    tokenizer, model, id2label = load_model(model_path)

    start = time.time()

    single_results = []
    for sample in samples:
        single_results.append(classify_single(sample, tokenizer, model, id2label))

    end = time.time()
    single_time = end - start

    print("\nSingle Inference:")
    print(f"Total Time for 30 samples: {single_time:.4f} seconds")

    start = time.time()

    batch_results = classify_batch(samples, tokenizer, model, id2label)

    end = time.time()
    batch_time = end - start

    print("\nBatch Inference:")
    print(f"Total Time for 30 samples: {batch_time:.4f} seconds")