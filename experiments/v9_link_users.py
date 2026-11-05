"""Link KuaiRand-27K (or -1K) user ids to KuaiRand-Pure user ids through the random-exposure logs, which
are the same events in every version: join on (time_ms, video_id) and require a one-to-one mapping.
Writes data/raw/user_map_<version>_pure.csv (columns: user_<version>, user_pure, n_events).
python3 experiments/v9_link_users.py --archive data/raw/dl_kr27k/KuaiRand-27K.tar.gz --member <random log member>
"""
import argparse
import subprocess

import pandas as pd


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--archive", required=True)
    ap.add_argument("--member", required=True)
    ap.add_argument("--version", default="27k")
    ap.add_argument("--pure", default="data/raw/KuaiRand-Pure/data/log_random_4_22_to_5_08_pure.csv")
    a = ap.parse_args()
    p = subprocess.Popen(["tar", "xzOf", a.archive, a.member], stdout=subprocess.PIPE)
    r = pd.read_csv(p.stdout, usecols=["user_id", "video_id", "time_ms"])
    p.wait()
    q = pd.read_csv(a.pure, usecols=["user_id", "video_id", "time_ms"])
    j = r.merge(q, on=["time_ms", "video_id"], suffixes=("_v", "_pure"))
    pairs = j.groupby(["user_id_v", "user_id_pure"]).size().rename("n_events").reset_index()
    one_v = pairs.user_id_v.is_unique
    one_p = pairs.user_id_pure.is_unique
    print(f"{a.version} random rows {len(r)}, Pure random rows {len(q)}, joined {len(j)} "
          f"({len(j)/len(r):.4f}); users {r.user_id.nunique()} -> linked {pairs.user_id_v.nunique()}; "
          f"one-to-one: {one_v and one_p}; identical ids: {(pairs.user_id_v == pairs.user_id_pure).mean():.4f}")
    pairs.rename(columns={"user_id_v": f"user_{a.version}", "user_id_pure": "user_pure"}).to_csv(
        f"data/raw/user_map_{a.version}_pure.csv", index=False)


if __name__ == "__main__":
    main()
