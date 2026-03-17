import json
import math
import numpy as np
from fire import Fire


def normalize_sid(x: str) -> str:
    return str(x).strip().strip('"').strip()


def extract_target_sid(sample):
    """
    尽量兼容你现有结果 json 的字段。
    calc.py 里大概率用的是 output 作为真实标签。
    """
    target = sample.get("output", None)
    if target is None:
        raise ValueError(f"sample has no 'output' field: {sample.keys()}")

    if isinstance(target, list):
        # 有些数据可能 output 是 list，只取第一个
        target = target[0]
    return normalize_sid(target)


def extract_predict_list(sample):
    pred = sample.get("predict", None)
    if pred is None:
        raise ValueError(f"sample has no 'predict' field: {sample.keys()}")

    if not isinstance(pred, list):
        raise ValueError(f"'predict' is not list, got type={type(pred)}")

    return [normalize_sid(x) for x in pred]


def hr_at_k(target, pred_list, k):
    return 1.0 if target in pred_list[:k] else 0.0


def ndcg_at_k(target, pred_list, k):
    topk = pred_list[:k]
    if target not in topk:
        return 0.0
    rank = topk.index(target)  # 0-based
    return 1.0 / math.log2(rank + 2)


def evaluate_subset(samples, topk_list=(1, 3, 5, 10, 20, 50)):
    hrs = []
    ndcgs = []

    for k in topk_list:
        hr_vals = []
        ndcg_vals = []
        for s in samples:
            target = s["target_sid"]
            preds = s["predict_list"]
            hr_vals.append(hr_at_k(target, preds, k))
            ndcg_vals.append(ndcg_at_k(target, preds, k))
        hrs.append(float(np.mean(hr_vals)) if hr_vals else 0.0)
        ndcgs.append(float(np.mean(ndcg_vals)) if ndcg_vals else 0.0)

    return list(topk_list), ndcgs, hrs


def main(
    result_path: str,
    item_value_path: str,
    split_mode: str = "median",   # median / tertile
):
    with open(result_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    with open(item_value_path, "r", encoding="utf-8") as f:
        item_value = json.load(f)

    enriched = []
    missing_value_cnt = 0

    for sample in data:
        target_sid = extract_target_sid(sample)
        predict_list = extract_predict_list(sample)
        value = float(item_value.get(target_sid, 1.0))
        if target_sid not in item_value:
            missing_value_cnt += 1

        enriched.append({
            "target_sid": target_sid,
            "predict_list": predict_list,
            "value": value,
        })

    values = np.array([x["value"] for x in enriched], dtype=float)

    print(f"num_samples = {len(enriched)}")
    print(f"missing_value_cnt = {missing_value_cnt}")
    print(f"value_min = {values.min():.6f}")
    print(f"value_max = {values.max():.6f}")
    print(f"value_mean = {values.mean():.6f}")
    print(f"value_median = {np.median(values):.6f}")

    # 全体样本
    topk, ndcg_all, hr_all = evaluate_subset(enriched)

    print("\n=== ALL SAMPLES ===")
    print("TopK :", topk)
    print("NDCG :", ndcg_all)
    print("HR   :", hr_all)

    if split_mode == "median":
        thr = float(np.median(values))
        high_group = [x for x in enriched if x["value"] >= thr]
        low_group = [x for x in enriched if x["value"] < thr]

        print(f"\nSplit by median = {thr:.6f}")
        print(f"high_group size = {len(high_group)}")
        print(f"low_group  size = {len(low_group)}")

        topk, ndcg_high, hr_high = evaluate_subset(high_group)
        topk, ndcg_low, hr_low = evaluate_subset(low_group)

        print("\n=== HIGH-VALUE GROUP ===")
        print("TopK :", topk)
        print("NDCG :", ndcg_high)
        print("HR   :", hr_high)

        print("\n=== LOW-VALUE GROUP ===")
        print("TopK :", topk)
        print("NDCG :", ndcg_low)
        print("HR   :", hr_low)

    elif split_mode == "tertile":
        q1 = float(np.quantile(values, 1/3))
        q2 = float(np.quantile(values, 2/3))

        low_group = [x for x in enriched if x["value"] < q1]
        mid_group = [x for x in enriched if q1 <= x["value"] < q2]
        high_group = [x for x in enriched if x["value"] >= q2]

        print(f"\nSplit by tertile: q1={q1:.6f}, q2={q2:.6f}")
        print(f"low_group  size = {len(low_group)}")
        print(f"mid_group  size = {len(mid_group)}")
        print(f"high_group size = {len(high_group)}")

        topk, ndcg_low, hr_low = evaluate_subset(low_group)
        topk, ndcg_mid, hr_mid = evaluate_subset(mid_group)
        topk, ndcg_high, hr_high = evaluate_subset(high_group)

        print("\n=== LOW-VALUE GROUP ===")
        print("TopK :", topk)
        print("NDCG :", ndcg_low)
        print("HR   :", hr_low)

        print("\n=== MID-VALUE GROUP ===")
        print("TopK :", topk)
        print("NDCG :", ndcg_mid)
        print("HR   :", hr_mid)

        print("\n=== HIGH-VALUE GROUP ===")
        print("TopK :", topk)
        print("NDCG :", ndcg_high)
        print("HR   :", hr_high)

    else:
        raise ValueError(f"Unsupported split_mode: {split_mode}")


if __name__ == "__main__":
    Fire(main)