import json

path = r"/content/drive/MyDrive/steering_contrastive/results_qwen3B.jsonl"
line_num = 55 # <-- change this (1-indexed)

with open(path, "r", encoding="utf-8") as f:
    for i, line in enumerate(f, 1):
        if i == line_num:
            obj = json.loads(line)

            # Print the entire JSON object pretty
            print(json.dumps(obj, indent=2, ensure_ascii=False))

            # OR print only the model output field with real newlines
            print("\n" + "="*40 + " OUTPUT_TEXT " + "="*40 + "\n")
            try:
                print(obj["output_text"])   # <-- this automatically converts \n properly
            except:
                print(obj["full_text"])
            break
