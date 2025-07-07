import os
import csv

# 1) 경로 설정 (절대경로)
OUTPUTS_ROOT = "/home/aistore51/git/outputs"
GT_ROOT      = "/home/aistore51/git/output"
RESULT_CSV   = "/home/aistore51/git/eval_results.csv"

# 2) groundtruth 로딩: event_id -> [counts...]
def load_counts_from_file(path):
    counts = []
    with open(path, 'r') as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) >= 3:
                counts.append(int(parts[-1]))
    return counts

print(">> GT 파일 로딩 시작...")
gt_dict = {}
for fn in sorted(os.listdir(GT_ROOT)):
    if fn.startswith("event_") and fn.endswith(".txt"):
        path = os.path.join(GT_ROOT, fn)
        gt_dict[fn] = load_counts_from_file(path)
        print(f"  - Loaded GT: {fn} ({len(gt_dict[fn])} classes)")

print(f">> 총 {len(gt_dict)}개의 GT 파일 로드 완료.\n")

# 3) 평가 함수
def evaluate_setting(setting_dir):
    print(f">> Evaluating setting: {setting_dir}")
    scores = []
    total_files = 0
    for fn in sorted(os.listdir(setting_dir)):
        if not (fn.startswith("event_") and fn.endswith(".txt")):
            continue
        total_files += 1
        inf_path = os.path.join(setting_dir, fn)
        inf_counts = load_counts_from_file(inf_path)

        if fn not in gt_dict:
            print(f"  [WARN] GT 없음: {fn}")
            continue

        gt_counts = gt_dict[fn]
        mae = sum(abs(g - i) for g, i in zip(gt_counts, inf_counts))
        scores.append(mae)

    avg_mae = sum(scores) / len(scores) if scores else float("nan")
    print(f"  -> 처리된 이벤트: {total_files}, 비교된 이벤트: {len(scores)}, 평균 MAE: {avg_mae:.3f}\n")
    return avg_mae

# 4) 전체 탐색 & 결과 저장
print(">> 전체 설정 평가 시작...\n")
results = []
for cfg_name in sorted(os.listdir(OUTPUTS_ROOT)):
    cfg_path = os.path.join(OUTPUTS_ROOT, cfg_name)
    if not os.path.isdir(cfg_path):
        continue
    print(f"## CFG 디렉토리: {cfg_name}")
    for setting in sorted(os.listdir(cfg_path)):
        setting_dir = os.path.join(cfg_path, setting)
        if os.path.isdir(setting_dir):
            score = evaluate_setting(setting_dir)
            results.append((cfg_name, setting, score))

# 5) 결과 CSV로 저장
print(">> 결과 CSV 저장 중...")
with open(RESULT_CSV, "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["cfg", "setting", "mean_abs_error"])
    for cfg, setting, score in results:
        writer.writerow([cfg, setting, f"{score:.3f}"])
print(f">> 평가 끝! 결과: {RESULT_CSV}")
