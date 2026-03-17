import json
import math
import glob
import pandas as pd
from collections import Counter

CATEGORY = "Industrial_and_Scientific"
ALPHA = 1.0

train_fp = glob.glob(f"./data/Amazon/train/{CATEGORY}*11.csv")[0]
info_fp = glob.glob(f"./data/Amazon/info/{CATEGORY}*.txt")[0]
out_fp = f"./data/Amazon/index/{CATEGORY}.item_value.json"

print("train_fp:", train_fp)
print("info_fp:", info_fp)
print("out_fp:", out_fp)

# 读取训练集
df = pd.read_csv(train_fp)
print("Columns:", df.columns.tolist())

if "item_sid" not in df.columns:
    raise ValueError("训练集里没有 item_sid 列，当前方案不能直接用。")

# 统计目标 item_sid 频次
item_sids = df["item_sid"].astype(str).str.strip()
cnt = Counter(item_sids)

print(f"num_samples = {len(item_sids)}")
print(f"num_unique_item_sid_in_train = {len(cnt)}")

# 读取 info_file，拿到所有合法 sid
all_sids = []
with open(info_fp, "r", encoding="utf-8") as f:
    for line in f:
        parts = line.rstrip("\n").split("\t")
        if len(parts) >= 1:
            sid = parts[0].strip()
            if sid:
                all_sids.append(sid)

all_sids = sorted(set(all_sids))
print(f"num_all_legal_sids = {len(all_sids)}")

# 对数平滑，映射到 [1, 1+ALPHA]
max_log = max(math.log1p(v) for v in cnt.values()) if len(cnt) > 0 else 1.0

item_value = {}
for sid in all_sids:
    c = cnt.get(sid, 0)
    value = 1.0 + ALPHA * (math.log1p(c) / max_log) if max_log > 0 else 1.0
    item_value[sid] = round(float(value), 6)

# 保存
with open(out_fp, "w", encoding="utf-8") as f:
    json.dump(item_value, f, ensure_ascii=False, indent=2)

print(f"Saved to: {out_fp}")

vals = list(item_value.values())
print("value range:", min(vals), "to", max(vals))

print("\nTop-20 frequent target items:")
for sid, c in cnt.most_common(20):
    print(f"{sid}\tcount={c}\tvalue={item_value[sid]}")
