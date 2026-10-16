import numpy as np
import pandas as pd

import json
import random
from collections import defaultdict

df = pd.read_csv("msrvtt_data/MSRVTT_JSFUSION_test.csv")
EPS = 1e-5

#print(df.head())

#get the column named video_id
#videos = df["video_id"] #pandas Series. Get the column named "video_id"
videos = df["video_id"].to_list()
#print(videos[:10])

with open("fire_msrvtt_dataset.json") as f:
    data = json.load(f)

annotations = data["annotations"]

first = annotations[0]
#print(type(first))
query = first["query"]
video_ids = first["video_id"]
label = first["label"]

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

#sim = np.load("data/predictions/msrvtt/clip4clip/sim_test.npy")
models = ["clip4clip", "collaborative-experts", "ssb"]
sims = {}
for model in models:
    sims[model] = np.load(
        f"data/predictions/msrvtt/{model}/sim_test.npy"
    )


"""
ranking_indices = np.argsort(-sim[0])
print(ranking_indices[:10])

top_10_videos = []

for idx in ranking_indices[:10]:
    top_10_videos.append(videos[idx])


rankings = []
for i in range(len(df)):
    ranking_indices = np.argsort(-sim[i])

    ranking = []
    for idx in ranking_indices:
        ranking.append(videos[idx])

    rankings.append(ranking)

pool = set(rankings[0][:10])
pool.add(df.iloc[0]["video_id"])
"""

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

"""
pool = set()
for model in models:
    pool.update(rankings[model][0][:10]) # add several elements

pool.add(df.iloc[0]["video_id"])
"""

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

row = df.iloc[0] #extract the first row.

#row["sentence"] the words in the current query
#judeged.get(..., {}) empty otherwise
def qrels_for(row, judged):
    q = judged.get(row["sentence"], {}).copy()
    q[row[video_id]] = 1

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

