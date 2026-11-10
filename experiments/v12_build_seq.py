"""Per-event lifelong sequences for the DIN / SIM baselines (drup/din.py): for every user, the last L standard-log
events strictly before the cut-off (2022-04-22 00:00 UTC+8) on videos OUTSIDE the random pool, with the content key
codes (tag, author, music) of the video and the engagement signal. The random log is never read.

The tables of experiments/v9_build_lifelong.py keep only user x key-value counts; the attention models need the order of
the events, so this builder keeps the events themselves, streamed chunk by chunk (per user, only the last L events are
kept in memory at any time).

Output (data/raw/seq_<version>.npz): users (sorted raw ids), tag / author / music (int32, (U, L), codes, 0 = missing),
sig (int8, 2 * click + long view), video and time_ms (int64, for audits and the leakage tests only; the models do not
read them), length (int32), n_events (int64: all pre-cut out-of-pool events of the user), vocab_tag (str), vocab_author
and vocab_music (int64): the value of code c is vocab[c - 1]. Events are right-aligned and time-ordered.

27K (streams the archive; about 40 minutes on the full standard log of experiments/v9_build_lifelong.py):
    python3 experiments/v12_build_seq.py --archive data/raw/dl_kr27k/KuaiRand-27K.tar.gz --version 27k
    (the pool and the dataset's 27K users are the KuaiRand-Pure files of data/raw/KuaiRand-Pure/data)
1K (reads the CSV files of data/raw/KuaiRand-1K/data):
    python3 experiments/v12_build_seq.py --data_dir data/raw/KuaiRand-1K/data --version 1k --pool_suffix 1k
--max_rows N (smoke runs only): reads at most N rows of every file.
"""
import argparse
import os
import subprocess
import sys
import time

import numpy as np
import pandas as pd

CUT = 1650556800000          # 2022-04-22 00:00 UTC+8
L_DEFAULT = 200
MISSING_TAG = "<NA>"
ARCHIVE_27K = {"features": "KuaiRand-27K/data/video_features_basic_27k.csv",
               "logs": ["KuaiRand-27K/data/log_standard_4_08_to_4_21_27k_part1.csv",
                        "KuaiRand-27K/data/log_standard_4_08_to_4_21_27k_part2.csv",
                        "KuaiRand-27K/data/log_standard_4_22_to_5_08_27k_part1.csv",
                        "KuaiRand-27K/data/log_standard_4_22_to_5_08_27k_part2.csv"]}
LOG_COLS = ["user_id", "video_id", "time_ms", "is_click", "long_view"]


def chunks(archive, data_dir, name, usecols, dtype, chunksize, max_rows):
    """Chunks of one member: a file of data_dir, or a member of the tar.gz (streamed with tar)."""
    if data_dir:
        src = os.path.join(data_dir, os.path.basename(name))
        p = None
    else:
        p = subprocess.Popen(["tar", "xzOf", archive, name], stdout=subprocess.PIPE, bufsize=1 << 24)
        src = p.stdout
    n = 0
    try:
        for ch in pd.read_csv(src, usecols=usecols, dtype=dtype, chunksize=chunksize):
            if max_rows and n + len(ch) > max_rows:
                ch = ch.iloc[:max_rows - n]
            n += len(ch)
            yield ch
            if max_rows and n >= max_rows:
                break
    finally:
        if p is not None:
            p.kill()
            p.wait()


def first_tag(col):
    return col.where(col.notna(), MISSING_TAG).astype(str).str.split(",").str[0]


def pool_ids(pool_dir, suffix):
    return np.sort(pd.read_csv(os.path.join(pool_dir, f"log_random_4_22_to_5_08_{suffix}.csv"),
                               usecols=["video_id"]).video_id.unique())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--archive", default=None, help="tar.gz with the 27K members (ARCHIVE_27K)")
    ap.add_argument("--data_dir", default=None, help="directory with the 1K CSV files (instead of --archive)")
    ap.add_argument("--version", default="27k")
    ap.add_argument("--pool_dir", default=os.path.join("data", "raw", "KuaiRand-Pure", "data"))
    ap.add_argument("--pool_suffix", default="pure")
    ap.add_argument("--L", type=int, default=L_DEFAULT)
    ap.add_argument("--chunksize", type=int, default=4_000_000)
    ap.add_argument("--max_rows", type=int, default=0)
    ap.add_argument("--members", nargs="*", default=None,
                    help="archive members: the features file, then the standard logs (default ARCHIVE_27K)")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    if bool(a.archive) == bool(a.data_dir):
        ap.error("give exactly one of --archive and --data_dir")
    out = a.out or os.path.join("data", "raw", f"seq_{a.version}.npz")
    t0 = time.time()
    if a.archive:
        feat_name, log_names = ARCHIVE_27K["features"], ARCHIVE_27K["logs"]
        if a.members:
            feat_name, log_names = a.members[0], a.members[1:]
    else:
        feat_name = f"video_features_basic_{a.pool_suffix}.csv"
        log_names = [f"log_standard_4_08_to_4_21_{a.pool_suffix}.csv", f"log_standard_4_22_to_5_08_{a.pool_suffix}.csv"]
    pool = pool_ids(a.pool_dir, a.pool_suffix)
    # 1. video features of every video (ids and key values); missing values are 0 codes
    parts = []
    for ch in chunks(a.archive, a.data_dir, feat_name, ["video_id", "author_id", "music_id", "tag"],
                     {"tag": str}, a.chunksize, a.max_rows):
        parts.append(pd.DataFrame({"v": ch.video_id.values.astype(np.int64),
                                   "a": pd.to_numeric(ch.author_id, errors="coerce").fillna(-1).astype(np.int64).values,
                                   "m": pd.to_numeric(ch.music_id, errors="coerce").fillna(-1).astype(np.int64).values,
                                   "t": first_tag(ch.tag).values}))
    feat = pd.concat(parts, ignore_index=True).drop_duplicates("v").sort_values("v").reset_index(drop=True)
    del parts
    vocab_tag = np.array(sorted(set(feat.t.values) - {MISSING_TAG}), dtype=object)
    vocab_author = np.unique(feat.a.values[feat.a.values >= 0])
    vocab_music = np.unique(feat.m.values[feat.m.values >= 0])
    tcode = pd.Index(vocab_tag.astype(str)).get_indexer(feat.t.values) + 1
    acode = np.searchsorted(vocab_author, feat.a.values) + 1
    mcode = np.searchsorted(vocab_music, feat.m.values) + 1
    acode = np.where(feat.a.values >= 0, acode, 0)
    mcode = np.where(feat.m.values >= 0, mcode, 0)
    vid_sorted = feat.v.values
    print(f"features: {len(vid_sorted)} videos, {len(vocab_tag)} tags, {len(vocab_author)} authors, "
          f"{len(vocab_music)} music ids [{time.time() - t0:.0f}s]", flush=True)
    # 2. standard log: pre-cut events on non-pool videos; per user the last L events, merged chunk by chunk
    buf = pd.DataFrame({"user_id": np.zeros(0, np.int64), "time_ms": np.zeros(0, np.int64),
                        "video": np.zeros(0, np.int64), "tag": np.zeros(0, np.int32), "author": np.zeros(0, np.int32),
                        "music": np.zeros(0, np.int32), "sig": np.zeros(0, np.int8)})
    n_events = pd.Series(dtype=np.int64)
    n_rows = 0
    for name in log_names:
        for ch in chunks(a.archive, a.data_dir, name, LOG_COLS, None, a.chunksize, a.max_rows):
            n_rows += len(ch)
            v = ch.video_id.values.astype(np.int64)
            t = ch.time_ms.values.astype(np.int64)
            keep = (t < CUT) & ~np.isin(v, pool)
            if not keep.any():
                continue
            v, t = v[keep], t[keep]
            u = ch.user_id.values[keep].astype(np.int64)
            c = ch.is_click.values[keep].astype(np.int8)
            lv = ch.long_view.values[keep].astype(np.int8)
            pos = np.searchsorted(vid_sorted, v)
            pos_c = np.minimum(pos, len(vid_sorted) - 1)
            hit = vid_sorted[pos_c] == v
            new = pd.DataFrame({"user_id": u, "time_ms": t, "video": v,
                                "tag": np.where(hit, tcode[pos_c], 0).astype(np.int32),
                                "author": np.where(hit, acode[pos_c], 0).astype(np.int32),
                                "music": np.where(hit, mcode[pos_c], 0).astype(np.int32),
                                "sig": (2 * c + lv).astype(np.int8)})
            cnt = new.user_id.value_counts()
            n_events = n_events.add(cnt, fill_value=0).astype(np.int64)
            new = new.sort_values(["user_id", "time_ms"], kind="stable").groupby("user_id", sort=False).tail(a.L)
            buf = pd.concat([buf, new], ignore_index=True).sort_values(["user_id", "time_ms"], kind="stable")
            buf = buf.groupby("user_id", sort=False).tail(a.L)
        print(f"  {name}: {n_rows} rows, {len(buf)} kept events [{time.time() - t0:.0f}s]", flush=True)
    # 2b. author and music vocabularies are pruned to the values of the pool videos and of the kept events (the
    # features file has millions of values; an unused value has no embedding to learn)
    pool_rows = np.isin(feat.v.values, pool)
    for key, vocab, col in (("author", vocab_author, acode), ("music", vocab_music, mcode)):
        used = np.zeros(len(vocab) + 1, bool)
        used[np.unique(col[pool_rows])] = True
        used[np.unique(buf[key].values)] = True
        used[0] = False
        newc = np.zeros(len(vocab) + 1, np.int32)
        newc[used] = np.arange(1, int(used.sum()) + 1, dtype=np.int32)
        buf[key] = newc[buf[key].values]
        if key == "author":
            vocab_author = vocab[used[1:]]
        else:
            vocab_music = vocab[used[1:]]
    print(f"vocabularies pruned: {len(vocab_author)} authors, {len(vocab_music)} music ids", flush=True)
    # 3. right-aligned (U, L) tables
    users = np.sort(buf.user_id.unique()) if len(buf) else np.zeros(0, np.int64)
    U, L = len(users), a.L
    ui = np.searchsorted(users, buf.user_id.values)
    back = buf.groupby("user_id", sort=False).cumcount(ascending=False).values
    col = L - 1 - back
    res = {"users": users.astype(np.int64), "L": np.array(L), "cut_ms": np.array(CUT), "version": np.array(a.version),
           "n_rows": np.array(n_rows), "vocab_tag": vocab_tag.astype(str), "vocab_author": vocab_author.astype(np.int64),
           "vocab_music": vocab_music.astype(np.int64)}
    for k, dt in (("tag", np.int32), ("author", np.int32), ("music", np.int32), ("sig", np.int8),
                  ("video", np.int64), ("time_ms", np.int64)):
        arr = np.zeros((U, L), dtype=dt)
        if U:
            arr[ui, col] = buf[k].values
        res[k if k != "video" else "video"] = arr
    res["length"] = np.bincount(ui, minlength=U).astype(np.int32) if U else np.zeros(0, np.int32)
    res["n_events"] = n_events.reindex(users).fillna(0).astype(np.int64).values if U else np.zeros(0, np.int64)
    np.savez_compressed(out, **res)
    print(f"saved {out}: {U} users, {int(res['length'].sum())} events kept, L={L} ({time.time() - t0:.0f}s)", flush=True)


if __name__ == "__main__":
    sys.exit(main())
