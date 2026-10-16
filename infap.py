import numpy as np
import pandas as pd

import json

#load csv
df = pd.read_csv("msrvtt_data/MSRVTT_JSFUSION_test.csv")
EPS = 1e-5

#print(df.head())

#get the column named video_id
#videos = df["video_id"] #pandas Series. Get the column named "video_id"
videos = df["video_id"].to_list()
#print(videos[:10])

#load json
with open("fire_msrvtt_dataset.json") as f:
    data = json.load(f)

annotations = data["annotations"]
print(type(annotations))

#build judged
judged = {}
for a in annotations:
    query = a["query"]
    video_id = a["video_id"]
    label = a["label"]

    if query not in judged:
        judged[query] = {}
    if label == "relevant":
        judged[query][video_id] = 1
    else:
        judged[query][video_id] = 0

#load similarity matrices
models = ["clip4clip", "collaborative-experts", "ssb"]
sims = {}
for model in models:
    sims[model] = np.load(
        f"data/predictions/msrvtt/{model}/sim_test.npy"
    )

rankings = {}
for model in models:
    model_rankings = []

    for i in range(len(df)):
        ranking_indices = np.argsort(-sims[model][i])

        ranking = []
        for idx in ranking_indices:
            ranking.append(videos[idx])

        model_rankings.append(ranking)

    rankings[model] = model_rankings

#build pool
pools = []
for i in range(len(df)):
    pool = set() #use set instead of list

    for model in models:
        pool.update(rankings[model][i][:10])

    pool.add(df.iloc[i]["video_id"])

    pools.append(pool)

#print(len(pools))
#print(len(pools[0]))
#print(list(pools[0])[:10]) #use list() to convert a set to a list.

#row = df.iloc[0] #extract the first row.

#row["sentence"] the words in the current query
#judeged.get(..., {}) empty otherwise
def qrels_for(row, judged):
    q = judged.get(row["sentence"], {}).copy()
    q[row["video_id"]] = 1

    return q

def recall_at(ranking, qrels, k):
    for video in ranking[:k]:
        if qrels.get(video) == 1:
            return 1.0
    return 0.0

def average_precision(ranking, qrels):
    total_relevant = sum(qrels.values())

    if total_relevant == 0:
        return np.nan

    hits = 0
    score = 0.0

    for rank, video in enumerate(ranking, start = 1):
        if qrels.get(video) == 1:
            hits += 1
            precision = hits/rank
            score += precision

    return score/total_relevant

def inf_ap(ranking, qrels, pool):
    n_rel_judged = sum(qrels.values())

    if n_rel_judged == 0:
        return np.nan

    total = 0.0
    in_pool = 0
    rel = 0
    nonrel = 0

    for k, video in enumerate(ranking, start=1):
        label = qrels.get(video)

        if label == 1:
            if k ==1:
                total += 1.0

            else:
                total += (1.0/k
                          + ((k - 1) / k)
                            * (in_pool / (k - 1))
                            * ((rel + EPS) / (rel + nonrel + 2 * EPS)))

        if video in pool:
            in_pool += 1

        if label == 1:
            rel += 1
        elif label == 0:
            nonrel += 1

    return total / n_rel_judged

orig_ap = []
fire_ap = []
r1_orig = []
r1_fire = []
r10_fire = []
infap_scores = []
sampled_infap_scores = []

for model in models:
    for i in range(len(df)):
        row = df.iloc[i]

        ranking = rankings[model][i]

        # Original MSR-VTT: only one ground-truth video
        gold = {row["video_id"]: 1}

        # FIRE: original GT + FIRE judgments
        qrels = qrels_for(row, judged)

        pool_qrels = {}
        for video, label in qrels.items():
            if video in pools[i]:
                pool_qrels[video] = label

        orig_ap.append(
            average_precision(ranking, gold)
        )

        fire_ap.append(
            average_precision(ranking, qrels)
        )

        r1_orig.append(
            recall_at(ranking, gold, 1)
        )

        r1_fire.append(
            recall_at(ranking, qrels, 1)
        )

        r10_fire.append(
            recall_at(ranking, qrels, 10)
        )

        infap_scores.append(
            inf_ap(ranking, pool_qrels, pools[i])
        )

    print("\nModel:", model)
    print("R@1 original:", np.mean(r1_orig))
    print("R@1 FIRE:", np.mean(r1_fire))
    print("R@10 FIRE:", np.mean(r10_fire))

    print("mAP original:", np.nanmean(orig_ap))
    print("mAP FIRE:", np.nanmean(fire_ap))

    print("infAP:", np.nanmean(infap_scores))