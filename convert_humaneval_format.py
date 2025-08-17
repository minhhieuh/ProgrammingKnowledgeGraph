import json
import os

root = "experiment_results/gpt-4o-mini_humaneval_1754247720"
base_dir = os.path.basename(root)

dest = f"humaneval_results/{base_dir}"
if not os.path.exists(dest):
    os.makedirs(dest, exist_ok=True)

# for all jsonl files in the root, read each line, convert keep the task_id and generated_code, rename generated_code to completion
for file in os.listdir(root):
    if file.endswith(".jsonl"):
        new_jsonl = []
        with open(os.path.join(root, file), "r") as f:
            for line in f:
                data = json.loads(line)
                new_jsonl.append({
                    "task_id": data["task_id"],
                    "completion": data["generated_code"]
                })
        with open(os.path.join(dest, file), "w") as f:
            for line in new_jsonl:
                f.write(json.dumps(line) + "\n")
                