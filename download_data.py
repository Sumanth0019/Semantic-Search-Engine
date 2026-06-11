from datasets import load_dataset
import os
from collections import defaultdict

os.makedirs("data", exist_ok=True)

# Remove old files
for f in os.listdir("data"):
    if f.endswith(".txt"):
        os.remove(f"data/{f}")

print("Downloading full SQuAD dataset...")
dataset = load_dataset("squad", split="train")  # all 87,000 rows

seen_contexts = set()
topic_counts = defaultdict(int)
saved = 0
MAX_PER_TOPIC = 8   # max passages per topic — forces diversity
TARGET = 400        # stop after 400 unique passages

for row in dataset:
    if saved >= TARGET:
        break

    context = row["context"].strip()
    title = row["title"]

    # Skip if context already seen
    if context in seen_contexts:
        continue

    # Skip if this topic already has enough passages
    if topic_counts[title] >= MAX_PER_TOPIC:
        continue

    seen_contexts.add(context)
    topic_counts[title] += 1

    safe_title = title.replace(" ", "_").replace("/", "_")
    fname = f"data/{safe_title}_{saved:04d}.txt"
    with open(fname, "w", encoding="utf-8") as f:
        f.write(context)
    saved += 1

print(f"\nSaved         : {saved} unique passages")
print(f"Unique topics : {len(topic_counts)}")
print(f"\nTop 15 topics:")
for title, count in sorted(topic_counts.items(),
                           key=lambda x: -x[1])[:15]:
    print(f"  {title:<40} {count} passages")
