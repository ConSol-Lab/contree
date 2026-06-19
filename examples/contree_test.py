from pycontree import ConTree
import pandas as pd
from sklearn.metrics import accuracy_score
import time

print(f"{'dataset':<12s}     {'time':>8s} {'errors':>6s} {'acc':>6s} {'sz':>4s}")
print("-" * (12 + 1 + 3 + 1 + 8 + 1 + 6 + 1 + 6 + 1 + 4))

correct = {
    3: {
        "avila": 4329, "bank": 19, "bean": 1406, "bidding": 37,
        "eeg": 3495, "fault": 494, "htru": 275, "magic": 2567, 
        "occupancy": 47, "page": 125, "raisin": 76, "rice": 189,
        "room": 62, "segment": 208, "skin": 6161, "wilt": 18 
    },
    4: {
        "bank": 0, "bidding": 16, "occupancy": 26, "page": 91,
        "room": 11, "segment": 76, "skin": 2067, "wilt": 2
    }
}

for max_depth, runs in correct.items():
    for dataset, correct_score in runs.items():
        df = pd.read_csv(f"train-datasets/{dataset}.txt", sep=" ", header=None)
        X = df[df.columns[1:]]
        y = df[0]

        contree = ConTree(max_depth=max_depth, time_limit=14400)
        start = time.time()
        contree.fit(X, y)
        duration = time.time() - start
        contree_ypred = contree.predict(X)
        accuracy = accuracy_score(y, contree_ypred)
        misclassifications = sum(contree_ypred != y)
        size = contree.get_num_branching_nodes()
        
        print(f"{dataset:<12s} d={max_depth} {duration:8.2f} {misclassifications:6d} {accuracy*100:6.1f} {size:4d}")
        assert(misclassifications == correct_score)