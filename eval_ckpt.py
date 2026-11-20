"""중단된 학습의 체크포인트로 최종 평가만 다시 수행한다. (학습은 하지 않는다)

03_train.py가 학습은 마쳤지만([검증 손실 기록]까지 출력됨) 그 뒤 평가 도중 끊긴 경우에 쓴다.
저장된 체크포인트를 읽어 test 손실과 생성 F1을 재고, metrics.json과 final 폴더를 만든다.
학습 전(base) 수치와 검증 손실 기록은 끊긴 로그에 이미 찍혀 있으므로 인자로 받아 채워 넣는다.

사용 예
  python eval_ckpt.py --ckpt runs/tesla_ep4/ckpt/checkpoint-1148 --out runs/tesla_ep4 \
         --base-loss 4.4623 --base-f1 0.004 --echo-f1 0.238 \
         --eval-losses 2.3206 1.9862 1.8452 1.8344 --n-eval 50 --no-repeat 3
"""
import argparse
import importlib.util
import json
import math
import os
import shutil

import torch


def load_train_module(path="03_train.py"):
    """03_train.py의 함수를 그대로 재사용한다 (평가 조건을 똑같이 맞추기 위해)."""
    spec = importlib.util.spec_from_file_location("train_mod", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)          # main()은 __main__ 가드 안에 있어 실행되지 않는다
    return mod


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ckpt", required=True, help="평가할 체크포인트 폴더")
    ap.add_argument("--out", required=True, help="결과를 넣을 폴더 (splits/ 가 들어 있어야 한다)")
    ap.add_argument("--lang", default="ko")
    ap.add_argument("--max-len", type=int, default=256)
    ap.add_argument("--n-eval", type=int, default=50)
    ap.add_argument("--no-repeat", type=int, default=3)
    ap.add_argument("--num-beams", type=int, default=1)
    ap.add_argument("--n-show", type=int, default=5)
    # 끊긴 로그에서 읽어 넣을 값들 (metrics.json 을 완성하기 위한 것일 뿐 계산에는 쓰지 않는다)
    ap.add_argument("--base-loss", type=float, required=True, help="[학습 전] test 손실")
    ap.add_argument("--base-f1", type=float, required=True, help="[학습 전] 생성 F1")
    ap.add_argument("--echo-f1", type=float, required=True, help="[기준선] 질문 되풀이 F1")
    ap.add_argument("--eval-losses", type=float, nargs="+", required=True, help="[검증 손실 기록]")
    ap.add_argument("--peak-gb", type=float, default=0.0, help="[메모리] 학습 중 최대 할당")
    a = ap.parse_args()

    t = load_train_module()
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # 1) 체크포인트에서 토크나이저와 모델을 읽는다 (학습 때와 같은 방식)
    tok = t.load_tokenizer(a.ckpt, "kogpt2")
    model = t.load_model(a.ckpt, tok).to(device).eval()
    print(f"[로드] {a.ckpt} ({sum(p.numel() for p in model.parameters()):,} 파라미터, {device})")

    # 2) 학습 때 저장해 둔 test 분할을 그대로 쓴다 (같은 문항으로 재야 비교가 된다)
    test_path = os.path.join(a.out, "splits", "test.jsonl")
    rows = t.read_jsonl(test_path)
    items, dropped = t.encode(rows, tok, a.lang, a.max_len)
    print(f"[test] {len(items)}개 (길이 초과 제외 {dropped}개)  <- {test_path}")
    ds = t.QADataset(items)
    collator = t.Collator(tok.pad_token_id)

    # 3) test 손실
    final_loss = t.eval_loss(model, ds, collator, device)
    print(f"[test 손실] {final_loss:.4f} (ppl {math.exp(final_loss):.1f})")

    # 4) 생성 평가 (학습 때와 같은 디코딩 설정)
    dec = dict(no_repeat=a.no_repeat, beams=a.num_beams)
    print(f"[생성 평가] {a.n_eval}문항 x 2회 (반복 억제 유무). 몇 분 걸린다...")
    preds, f1_plain, num_plain = t.gen_eval(model, tok, items, a.lang, device, a.n_eval, 1.0, **dec)
    _, f1_rep, _ = t.gen_eval(model, tok, items, a.lang, device, a.n_eval, 1.15, **dec)

    print(f"\n[학습 후] test 손실 {final_loss:.4f} (ppl {math.exp(final_loss):.1f}) | "
          f"생성 F1 {f1_plain:.3f} (반복 억제 1.15: {f1_rep:.3f}) | "
          f"수치 재현율 {num_plain[0]:.3f} ({num_plain[1]}문항)")
    print(f"[개선] 손실 {a.base_loss:.3f} -> {final_loss:.3f}, "
          f"F1 {a.base_f1:.3f} -> {f1_plain:.3f} (질문 되풀이 기준선 {a.echo_f1:.3f})")

    print("\n[생성 예시] (test 질문: 학습에 쓰지 않은 구역)")
    for i in range(min(a.n_show, len(preds))):
        it = items[i]
        print(f"\n  Q   : {it['question']}\n  정답: {it['ref']}\n  학습후: {preds[i][:150]}")

    # 5) 저장 (03_train.py 와 같은 형식)
    final_dir = os.path.join(a.out, "final")
    os.makedirs(final_dir, exist_ok=True)
    for fn in os.listdir(a.ckpt):
        if fn in ("optimizer.pt", "scheduler.pt", "rng_state.pth", "trainer_state.json", "training_args.bin"):
            continue                       # 학습 재개용 파일은 추론에 필요 없다
        shutil.copy2(os.path.join(a.ckpt, fn), os.path.join(final_dir, fn))
    tok.save_pretrained(final_dir)

    t.write_jsonl(os.path.join(a.out, "test_preds.jsonl"),
                  [dict(question=it["question"], ref=it["ref"], base="", pred=p)
                   for it, p in zip(items, preds)])
    json.dump(dict(model="skt/kogpt2-base-v2", lang=a.lang, split="group",
                   n_train=len(t.read_jsonl(os.path.join(a.out, "splits", "train.jsonl"))),
                   n_val=len(t.read_jsonl(os.path.join(a.out, "splits", "val.jsonl"))),
                   n_test=len(items),
                   base_test_loss=a.base_loss, final_test_loss=final_loss,
                   base_f1=a.base_f1, echo_f1=a.echo_f1, final_f1=f1_plain,
                   final_f1_rep115=f1_rep, num_recall=num_plain[0], num_recall_n=num_plain[1],
                   eval_losses=a.eval_losses,
                   args=dict(epochs=len(a.eval_losses), lr=5e-5, bs=4, grad_accum=4, seed=42,
                             no_repeat=a.no_repeat, num_beams=a.num_beams, optim="adafactor",
                             grad_ckpt=False, peak_gb=a.peak_gb),
                   pretrain=None, recovered_from=a.ckpt),
              open(os.path.join(a.out, "metrics.json"), "w", encoding="utf-8"), ensure_ascii=False, indent=2)
    print(f"\n[저장] {final_dir}, {a.out}/metrics.json, {a.out}/test_preds.jsonl")


if __name__ == "__main__":
    main()
