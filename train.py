import pandas as pd
import numpy as np
import random

user_cnt = 358198
artist_cnt = 67788
max_epoch = 20
k = 32
batch_size = 4096
def calc_loss(P, Q, ui, ai, rs, L, Y):
    predict = np.sum(P[ui] * Q[ai], axis=1)
    error = rs - predict
    grad_p = (error[:, np.newaxis] * Q[ai] - L * P[ui]) * Y
    grad_q = (error[:, np.newaxis] * P[ui] - L * Q[ai]) * Y 
    P[ui] += grad_p
    Q[ai] += grad_q
    return np.mean(error**2)


def train(Y, L, P, Q, ui, ai, rs):
    num_lessons = len(ui)
    for epoch in range(max_epoch):
        shuffle = np.random.permutation(num_lessons)
        ui_shuffled = ui[shuffle]
        ai_shuffled = ai[shuffle]
        rs_shuffled = rs[shuffle]
        epoch_loss = []
        for start_batch in range(0, num_lessons, batch_size):
            end_batch = min(start_batch + batch_size, num_lessons)
            ui_b = ui_shuffled[start_batch:end_batch]
            ai_b = ai_shuffled[start_batch:end_batch]
            rs_b = rs_shuffled[start_batch:end_batch]

            predict = np.sum(P[ui_b] * Q[ai_b], axis=1)
            error = rs_b - predict
            error_col = error[:, np.newaxis]
            grad_p = (error_col * Q[ai_b] - L * P[ui_b]) * Y
            grad_q = (error_col * P[ui_b] - L * Q[ai_b]) * Y 
            P[ui_b] += grad_p
            Q[ai_b] += grad_q
            epoch_loss.append(np.mean(error**2))
        loss = np.mean(epoch_loss)
        print(f"epoch: {epoch}, loss: {loss:.4f}")
        np.save(f"study/UsersEpoch{epoch}", P)
        np.save(f"study/ArtistEpoch{epoch}", Q)
        
r_mat = pd.read_csv(
    "/home/igor/Documents/MusicRecomendaton/data/train.csv",
    sep=",",
    header=0,
    names=["user", "mbid", "artis", "play", "plays_log", "user_id", "artist_id"]
)

user_ids = np.array(r_mat["user_id"].values)
artist_ids = np.array(r_mat["artist_id"].values)
real_score = np.array(r_mat["plays_log"].values)

P = np.random.uniform(-0.1, 0.1, size=(user_cnt, k))
Q = np.random.uniform(-0.1, 0.1, size=(artist_cnt, k))

train(0.01, 0.02, P, Q, user_ids, artist_ids, real_score)
