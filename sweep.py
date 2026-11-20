"""여러 설정을 차례로 학습하고 결과를 표 하나로 비교한다. (설정마다 03_train.py를 따로 실행)

  python sweep.py --data data/qa_all.jsonl --chunks data/batch/chunks.jsonl            # 기본 설정 묶음
  python sweep.py --data data/qa_merged.jsonl --chunks data/batch/chunks.jsonl --seeds 42 7
  python sweep.py ... --only G0 Q1 Q2                                                  # 일부만

주의: split이 group인 행(G*)과 question인 행(Q*)은 test 집합이 달라 손실을 서로 비교할 수 없다. 같은 묶음끼리만 비교한다.
      seed는 구역/질문 분할도 바꾼다. 여러 seed의 평균은 '분할 변동까지 포함한' 값이다.
"""
import argparse
import json
import os
import subprocess
import sys

BASE = ["--epochs", "2", "--lr", "5e-5", "--n-eval", "100", "--no-repeat", "3"]
CONFIGS = {
    # 본 적 없는 구역에 대한 일반화
    "G0": ("group", BASE, False),
    "G1": ("group", ["--epochs", "3", "--lr", "5e-5", "--n-eval", "100", "--no-repeat", "3"], True),
    # 학습한 구역의 새 질문 (매뉴얼 활용 시나리오)
    "Q0": ("question", BASE, False),
    "Q1": ("question", ["--epochs", "3", "--lr", "5e-5", "--n-eval", "100", "--no-repeat", "3"], True),
    "Q2": ("question", ["--epochs", "3", "--lr", "5e-5", "--n-eval", "100", "--no-repeat", "3", "--pretrain-epochs", "6"], True),
    "Q3": ("question", ["--epochs", "4", "--lr", "3e-5", "--n-eval", "100", "--no-repeat", "3", "--pretrain-epochs", "10", "--pretrain-lr", "1e-4"], True),
}


def run(cmd):
    print("\n$", " ".join(cmd), flush=True)
    return subprocess.run(cmd).returncode


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", default="data/qa_all.jsonl")
    ap.add_argument("--chunks", default="data/batch/chunks.jsonl", help="원문 이어 학습용 조각 파일")
    ap.add_argument("--model", default="skt/kogpt2-base-v2")
    ap.add_argument("--seeds", type=int, nargs="+", default=[42])
    ap.add_argument("--only", nargs="+", default=None)
    ap.add_argument("--root", default="runs/sweep")
    a = ap.parse_args()
    rows = {}
    for name, (split, extra, use_pre) in CONFIGS.items():
        if a.only and name not in a.only:
            continue
        for seed in a.seeds:
            out = os.path.join(a.root, f"{name}_s{seed}")
            cmd = [sys.executable, "03_train.py", "--model", a.model, "--data", a.data, "--out", out,
                   "--split", split, "--seed", str(seed)] + extra
            if use_pre:
                cmd += ["--pretrain-data", a.chunks]
            if run(cmd) != 0:
                print(f"  [실패] {name} seed {seed}"); continue
            rows.setdefault(name, []).append(json.load(open(os.path.join(out, "metrics.json"), encoding="utf-8")))
    if not rows:
        sys.exit("완료된 실행이 없다")
    print("\n" + "=" * 92)
    print(f"{'설정':<5}{'분할':<9}{'원문학습':<8}{'test 손실':>10}{'F1':>8}{'기준선':>8}{'수치재현':>9}{'val최저에폭':>12}{'실행수':>7}")
    mean = lambda xs: sum(xs) / len(xs)
    for name, ms in rows.items():
        pre = "O" if ms[0].get("pretrain") else "-"
        best = [m["eval_losses"].index(min(m["eval_losses"])) + 1 for m in ms]
        print(f"{name:<5}{ms[0]['split']:<9}{pre:<8}{mean([m['final_test_loss'] for m in ms]):>10.3f}"
              f"{mean([m['final_f1'] for m in ms]):>8.3f}{mean([m['echo_f1'] for m in ms]):>8.3f}"
              f"{mean([m.get('num_recall', 0) for m in ms]):>9.3f}{mean(best):>12.1f}{len(ms):>7}")
    print("=" * 92)
    print("같은 분할끼리만 비교한다 (G*끼리, Q*끼리). F1이 기준선(질문 되풀이)을 넘는지, 수치 재현이 오르는지를 함께 본다.")


if __name__ == "__main__":
    main()
