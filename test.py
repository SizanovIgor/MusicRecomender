import pandas as pd
import numpy as np

max_epoch = 20
k = 32
batch_size = 4096

test = pd.read_csv(
    "/home/igor/Documents/MusicRecomendaton/data/test.csv",
    sep=",",
    header=0,
    names=["user", "mbid", "artis", "play", "plays_log", "user_id", "artist_id"]
)

user_cnt = test["user_id"].nunique()
artist_cnt = test["artist_id"].nunique()

ui = test["user_id"].values
ai = test["artist_id"].values
rs = test["plays_log"].values

for epoch in range(max_epoch):
    Q = np.load(f"/home/igor/Documents/MusicRecomendaton/study/ArtistEpoch{epoch}.npy")
    P = np.load(f"/home/igor/Documents/MusicRecomendaton/study/UsersEpoch{epoch}.npy")
    num = len(test)
    epoch_loss = []
    for batch_start in range(0, num, batch_size):
        batch_end = batch_size + batch_start
        batch_end = min(batch_end, num)
        ui_b = ui[batch_start:batch_end]
        ai_b = ai[batch_start:batch_end]
        rs_b = rs[batch_start:batch_end]
        error = np.sum(P[ui_b] * Q[ai_b], axis=1)
        error = rs_b - error
        epoch_loss.append(np.mean(error**2))
    loss = np.mean(epoch_loss)
    print(f"eposh: {epoch} loss: {loss:.4f}")
