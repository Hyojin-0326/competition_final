#!/usr/bin/env python3
import os
import glob
import subprocess
import argparse
import numpy as np

from main_gridsearch import load_cached_results, RESULTS_DIR, CLASSES
import postprocess_utils as pp
import smoothing_test as smt

def save_events(event_ids, filtered_counts, out_dir):
    os.makedirs(out_dir, exist_ok=True)
    for idx, eid in enumerate(event_ids):
        fn = os.path.join(out_dir, f"event_{eid:05d}.txt")
        with open(fn, "w") as fw:
            for cid, cnt in enumerate(filtered_counts[idx]):
                fw.write(f"{cid:02d}\t{CLASSES[cid]}\t{cnt}\n")

def main():
    parser = argparse.ArgumentParser(
        description="Generate event_*.txt with postprocess & smoothing"
    )
    parser.add_argument("--idea", type=int, required=True,
                        help="post-processing idea (현재는 1만 지원)")
    parser.add_argument("--win", type=int, required=True,
                        help="smoothing window size")
    parser.add_argument("--k", type=int, required=True,
                        help="hysteresis K (0 이면 skip)")
    parser.add_argument("--cached", action="store_true",
                        help="results 디렉토리에 캐시만 로드")
    args = parser.parse_args()

    # 1) 캐시 확인
    cache_list = glob.glob(os.path.join(RESULTS_DIR, "event_*.txt"))
    if args.cached and cache_list:
        # 이미 캐시가 있으면 inference skip
        event_ids, raw_counts, raw_boxes = load_cached_results(RESULTS_DIR)
    else:
        # inference & cache 생성
        subprocess.run(["python3", "cache_raw.py"], check=True)
        event_ids, raw_counts, raw_boxes = load_cached_results(RESULTS_DIR)

    # 2) 후처리 (idea1 고정 파라미터)
    if args.idea != 1:
        raise NotImplementedError("현재는 --idea 1만 지원합니다")
    weights = np.array([1.0, 0.9, 0.8, 0.6, 0.6], dtype=float)
    T = len(event_ids)
    raw = np.zeros((T, len(CLASSES)), dtype=int)
    for t in range(T):
        raw[t] = pp.weighted_vote_union(
            raw_counts[t], weights, thr=1.0, min_hits=1
        )

    # 3) 스무딩 & 히스테리시스
    sm = smt.temporal_mode_smoothing(raw, window=args.win)
    filt = smt.hysteresis_filter(sm, K=args.k) if args.k > 0 else sm

    # 4) 결과 저장
    out_dir = "/home/aistore51/git/custom_output"
    save_events(event_ids, filt, out_dir)
    print(f"[INFO] Saved {T} files → {out_dir}")

if __name__ == "__main__":
    main()
