import json

path = r"/content/drive/MyDrive/steering_contrastive/results_qwen3B.jsonl"

correct = 0
total = 0
with open(path, 'r', encoding='utf-8') as f:
    for line in f:
        obj = json.loads(line)
        if obj["correct"] == True:
            correct += 1
        total += 1

print(correct/total)
