"""단계별 성공 점검 도구. 각 단계 실행 직후 해당 점검을 돌리고, PASS가 아니면 다음 단계로 가지 않는다.

  python check_step.py env                     # 0단계: 환경
  python check_step.py raw                     # 1단계: 01 실행 후 (추출 품질)
  python check_step.py chunks                  # 2단계: chunk + prepare 후 (submit 전 마지막 관문)
  python check_step.py batch                   # 3단계: submit/status 후
  python check_step.py qa                      # 4단계: collect 후 (데이터 품질)
  python check_step.py tok --model skt/kogpt2-base-v2   # 5단계: 토크나이저/모델 로드
  python check_step.py result --out runs/tesla_ko       # 7단계: 학습 결과 판정

종료 코드: PASS=0, WARN=0, FAIL=1  (쉘에서 && 로 이어 쓸 수 있다)
"""
import argparse
import collections
import importlib
import json
import os
import random
import re
import shutil
import sys

RAW, CHUNKS, STATE, DIR, QA = "data/raw/sections.jsonl", "data/batch/chunks.jsonl", "data/batch/state.json", "data/batch", "data/qa_all.jsonl"
HANGUL = re.compile(r"[가-힣]")
FAILS, WARNS = [], []


def ok(msg): print(f"  [PASS] {msg}")
def warn(msg): WARNS.append(msg); print(f"  [WARN] {msg}")
def fail(msg): FAILS.append(msg); print(f"  [FAIL] {msg}")


def rj(path):
    with open(path, encoding="utf-8") as f:
        return [json.loads(l) for l in f if l.strip()]


def need(path):
    if not os.path.exists(path):
        fail(f"{path} 가 없다. 이전 단계를 먼저 실행한다")
        finish()
    return path


def finish():
    print()
    if FAILS:
        print(f"결과: FAIL ({len(FAILS)}건) -> 아래를 해결하기 전에는 다음 단계로 가지 않는다")
        for m in FAILS:
            print("   -", m)
        sys.exit(1)
    print("결과: PASS" + (f" (경고 {len(WARNS)}건 확인 필요)" if WARNS else ""))
    sys.exit(0)


def norm_digits(s): return re.sub(r"\d+", "#", s.strip())


# ---------------------------------------------------------------- 0 env
def c_env(a):
    print("[0단계] 환경 점검")
    v = sys.version_info
    (ok if v >= (3, 9) else fail)(f"Python {v.major}.{v.minor}")
    for mod in ["pypdf", "requests", "bs4", "openai", "torch", "transformers", "accelerate"]:
        try:
            m = importlib.import_module(mod)
            ok(f"{mod} {getattr(m, '__version__', '')}")
        except Exception as e:
            (warn if mod in ("openai", "bs4") else fail)(f"{mod} 불러오기 실패: {type(e).__name__}")
    try:
        import torch
        if torch.cuda.is_available():
            p = torch.cuda.get_device_properties(0)
            ok(f"GPU {p.name} {p.total_memory/1024**3:.1f}GB, bf16 {torch.cuda.is_bf16_supported()}")
        else:
            fail("CUDA GPU를 찾지 못했다 (CPU 학습은 비현실적이다). 드라이버와 torch CUDA 빌드를 확인한다")
    except Exception:
        pass
    (ok if os.environ.get("OPENAI_API_KEY") else warn)("OPENAI_API_KEY " + ("설정됨" if os.environ.get("OPENAI_API_KEY") else "미설정 (submit 전에 필요)"))
    free = shutil.disk_usage(".").free / 1024**3
    (ok if free > 10 else warn)(f"여유 디스크 {free:.0f}GB")
    for f in ["01_fetch_manual.py", "02_make_dataset_batch.py", "03_train.py"]:
        (ok if os.path.exists(f) else fail)(f"{f} 존재")
    finish()


# ---------------------------------------------------------------- 1 raw
def c_raw(a):
    print("[1단계] 추출 결과 점검 (data/raw)")
    secs = rj(need(RAW))
    n = len(secs)
    tot = sum(len(s["text"]) for s in secs)
    (ok if n >= 20 else fail)(f"구역 {n}개 (20개 미만이면 분할이 불가능하다)")
    (ok if tot >= 100_000 else warn)(f"총 {tot:,}자 (매뉴얼 전체면 보통 수십만 자 이상)")
    allt = "".join(s["text"] for s in secs)
    hr = len(HANGUL.findall(allt)) / max(1, len(re.sub(r"\s", "", allt)))
    (ok if hr >= 0.25 else fail)(f"한글 비율 {hr:.0%} (25% 미만이면 한글 추출 실패 또는 영문 판본이다)")
    bad = len(re.findall(r"[�\u0000]", allt)) + len(re.findall(r"[-]", allt))
    (ok if bad / max(1, len(allt)) < 0.001 else fail)(f"깨진 글자/사설 영역 문자 {bad}개 (많으면 폰트 인코딩 문제)")
    # 남은 머리글/바닥글: 숫자 정규화 후 여러 구역에 반복되는 줄
    cnt = collections.Counter()
    for s in secs:
        for ln in {norm_digits(l) for l in s["text"].splitlines() if 3 < len(l.strip()) < 80}:
            cnt[ln] += 1
    rep = [(l, c) for l, c in cnt.most_common(8) if c >= max(5, 0.3 * n)]
    (ok if not rep else warn)("반복 줄 없음" if not rep else f"여러 구역에 반복되는 줄 {len(rep)}종: {rep[:3]}")
    paras = [p for s in secs for p in s["text"].split("\n\n")]
    avg = sum(len(p) for p in paras) / max(1, len(paras))
    (ok if 40 <= avg <= 800 else warn)(f"문단 평균 {avg:.0f}자 (40 미만: 줄이 안 이어짐, 800 초과: 문단 경계 소실)")
    print("\n  [표본 3개] 눈으로 읽히는지 확인한다 (아래가 어색하면 FAIL로 본다)")
    random.Random(0).shuffle(secs)
    for s in secs[:3]:
        print(f"   --- {s['id']} ---\n   {s['text'][:300]!r}")
    finish()


# ---------------------------------------------------------------- 2 chunks
def c_chunks(a):
    print("[2단계] chunk/prepare 점검 (submit 전 마지막 관문, 여기까지는 비용이 없다)")
    ch = rj(need(CHUNKS))
    n = len(ch)
    (ok if n >= 100 else warn)(f"조각 {n}개")
    groups = {c["group"] for c in ch}
    (ok if len(groups) >= 30 else warn)(f"그룹 {len(groups)}개 (30 미만이면 val/test가 구역 몇 개뿐이라 평가가 흔들린다. 01의 --pages-per-group을 2~3으로 줄인다)")
    dig = sum(1 for c in ch if sum(x.isdigit() for x in c["text"]) / max(1, len(c["text"])) > 0.3)
    (ok if dig / max(1, n) < 0.05 else warn)(f"숫자 비중 30% 초과 조각 {dig}개 (표/목차 찌꺼기 가능)")
    files = sorted(f for f in os.listdir(DIR) if f.startswith("batch_input_"))
    (ok if files else fail)(f"batch_input 파일 {len(files)}개")
    reqs, ids, models, chars = 0, set(), set(), 0
    for f in files:
        for r in rj(os.path.join(DIR, f)):
            reqs += 1; ids.add(r["custom_id"]); models.add(r["body"]["model"]); chars += len(json.dumps(r, ensure_ascii=False))
    (ok if reqs == len(ids) else fail)(f"요청 {reqs}건, custom_id 중복 {'없음' if reqs == len(ids) else '있음'}")
    (ok if len(models) == 1 else fail)(f"모델 {models}")
    est = chars * 0.85
    print(f"  입력 토큰 추정 약 {est/1e6:.2f}M (한글 글자당 0.7~1토큰 가정), 출력은 요청당 약 400토큰 추정 -> 총 {reqs*400/1e6:.2f}M")
    print("  비용 = (입력M x 입력단가 + 출력M x 출력단가) x Batch 할인율. 단가는 OpenAI 가격 페이지에서 확인한다 (확인이 필요하다)")
    print("  먼저 소량 시험: --max-requests 를 20으로 prepare 한 뒤 'submit --only 0' 으로 1개 파일만 제출하면 비용이 거의 없다")
    finish()


# ---------------------------------------------------------------- 3 batch
def c_batch(a):
    print("[3단계] Batch 제출/완료 점검")
    st = json.load(open(need(STATE), encoding="utf-8"))
    for i, f in enumerate(st["files"]):
        if not f.get("batch_id"):
            fail(f"[{i}] 아직 제출되지 않았다 (submit)"); continue
        s = f.get("status", "?")
        out = os.path.exists(os.path.join(DIR, f"output_{i:03d}.jsonl"))
        if s == "completed" and out:
            ok(f"[{i}] {f['batch_id']} completed, 결과 파일 있음")
        elif s in ("failed", "expired", "cancelled"):
            fail(f"[{i}] {s}. errors_{i:03d}.jsonl 과 status 출력의 오류 메시지를 확인한다")
        else:
            fail(f"[{i}] 상태 {s}. 아직 끝나지 않았다 (status를 다시 실행, 최대 24시간)")
        ep = os.path.join(DIR, f"errors_{i:03d}.jsonl")
        if os.path.exists(ep) and os.path.getsize(ep) > 0:
            warn(f"[{i}] 오류 파일에 {len(rj(ep))}건. 첫 줄: {open(ep, encoding='utf-8').readline()[:200]}")
    finish()


# ---------------------------------------------------------------- 4 qa
def c_qa(a):
    print("[4단계] 질문/답변 데이터 품질 점검 (학습 전 마지막 관문)")
    qa = rj(need(QA))
    ch = rj(need(CHUNKS))
    n = len(qa)
    yield_ = n / max(1, len(ch))
    (ok if n >= 1500 else warn if n >= 500 else fail)(f"통과 {n}쌍 (500 미만이면 학습 의미가 없고, 수천 쌍 미만이면 과적합 위험이 크다)")
    (ok if yield_ >= 1.5 else warn)(f"조각당 {yield_:.1f}쌍 (기대 2~4). 낮으면 collect 출력의 제외 사유를 보고 --min-ground, --max-a를 조정한다")
    g = collections.Counter(r["group"] for r in qa)
    (ok if len(g) >= 30 else warn)(f"그룹 {len(g)}개")
    top = g.most_common(1)[0]
    (ok if top[1] / n < 0.1 else warn)(f"한 그룹 최대 비중 {top[1]/n:.0%} ({top[0]})")
    hr = sum(bool(HANGUL.search(r["answer"])) for r in qa) / n
    (ok if hr > 0.9 else fail)(f"답변 한글 포함 비율 {hr:.0%}")
    aq = len({r["question"] for r in qa}); aa = len({r["answer"] for r in qa})
    (ok if aa / n > 0.9 else warn)(f"고유 답변 비율 {aa/n:.0%}, 고유 질문 {aq/n:.0%}")
    ql = sorted(len(r["question"]) for r in qa); al = sorted(len(r["answer"]) for r in qa)
    print(f"  질문 길이 중앙값 {ql[n//2]}자, 답변 길이 중앙값 {al[n//2]}자 (p95 {al[int(n*.95)]}자)")
    (ok if al[int(n * .95)] < 400 else warn)("답변 p95 400자 미만이어야 --max-len 256에 대부분 들어간다")
    generic = sum(1 for r in qa if re.search(r"어떻게 해야 합니까\??$|무엇입니까\??$", r["question"]) and len(r["question"]) < 14)
    (ok if generic / n < 0.05 else warn)(f"너무 짧고 모호한 질문 {generic}개")
    print("\n  [수동 점검] 아래 표본 10개를 원문과 대조한다. 숫자와 경고 문구가 틀린 것이 2개 이상이면 데이터를 다시 만든다.")
    for r in random.Random(1).sample(qa, min(10, n)):
        print(f"   [{r['group']}] Q: {r['question']}\n        A: {r['answer']}")
    finish()


# ---------------------------------------------------------------- 5 tok
def c_tok(a):
    print(f"[5단계] 토크나이저/모델 로드 점검 ({a.model})")
    try:
        import torch
        from transformers import AutoModelForCausalLM, PreTrainedTokenizerFast
    except Exception as e:
        fail(f"torch/transformers 불러오기 실패: {e}"); finish()
    try:
        tok = PreTrainedTokenizerFast.from_pretrained(a.model, bos_token="</s>", eos_token="</s>", unk_token="<unk>",
                                                      pad_token="<pad>", mask_token="<mask>")
    except Exception as e:
        fail(f"토크나이저 로드 실패(허브 접속/모델명 확인): {e}"); finish()
    s = "### 질문: 타이어 공기압은 얼마인가?\n### 답변: 42 psi이다."
    ids = tok(s, add_special_tokens=False)["input_ids"]
    back = tok.decode(ids)
    (ok if back.replace(" ", "") == s.replace(" ", "") else fail)(f"한글 왕복 복원: {back!r}")
    tpc = len(ids) / len(s)
    (ok if tpc < 0.9 else warn)(f"글자당 토큰 {tpc:.2f} (KoGPT2는 대체로 0.5~0.8)")
    (ok if tok.eos_token_id is not None and tok.pad_token_id is not None else fail)(f"eos {tok.eos_token_id}, pad {tok.pad_token_id}")
    try:
        m = AutoModelForCausalLM.from_pretrained(a.model, dtype=torch.float32)
    except TypeError:
        m = AutoModelForCausalLM.from_pretrained(a.model, torch_dtype=torch.float32)
    npar = sum(p.numel() for p in m.parameters())
    (ok if 100e6 < npar < 140e6 else warn)(f"파라미터 {npar:,}")
    (ok if len(tok) <= m.get_input_embeddings().num_embeddings else warn)(f"어휘 {len(tok)} / 임베딩 {m.get_input_embeddings().num_embeddings} (초과 시 03이 자동 확장)")
    if torch.cuda.is_available():
        m.to("cuda"); x = torch.tensor([ids]).cuda()
        loss = m(input_ids=x, labels=x).loss.item()
        (ok if loss == loss and loss < 15 else fail)(f"GPU 순전파 손실 {loss:.2f} (NaN/15 초과면 dtype 문제)")
        print(f"  GPU 메모리 사용 {torch.cuda.max_memory_allocated()/1024**3:.2f}GB")
    finish()


# ---------------------------------------------------------------- 7 result
def c_result(a):
    print(f"[7단계] 학습 결과 판정 ({a.out})")
    m = json.load(open(need(os.path.join(a.out, "metrics.json")), encoding="utf-8"))
    b, f_, ev = m["base_test_loss"], m["final_test_loss"], m["eval_losses"]
    (ok if f_ < b else fail)(f"test 손실 {b:.3f} -> {f_:.3f}")
    (ok if m["final_f1"] > m["base_f1"] else warn)(f"생성 F1 {m['base_f1']:.3f} -> {m['final_f1']:.3f}")
    if "echo_f1" in m:
        (ok if m["final_f1"] > m["echo_f1"] + 0.03 else warn)(f"질문 되풀이 기준선 F1 {m['echo_f1']:.3f} (모델 {m['final_f1']:.3f}). 기준선과 가까우면 내용을 배우지 못한 것이다")
    if len(ev) >= 2:
        best = ev.index(min(ev)) + 1
        (ok if best > 1 else warn)(f"val 손실 최저 에폭 {best}/{len(ev)}: {[round(x, 3) for x in ev]}")
        if best == 1:
            warn("1에폭에서 이미 최저다. 학습률(--lr 5e-5)이나 데이터 양을 점검한다")
        if best == len(ev):
            warn("마지막 에폭이 최저다. 더 학습할 여지가 있다 (--epochs 늘리기)")
    gap = abs(f_ - min(ev)) if ev else 0
    (ok if gap < 0.3 else warn)(f"val 최저와 test 손실 차 {gap:.2f} (크면 val/test 구역 편차가 크다는 뜻이다)")
    (ok if m["final_f1_rep115"] >= m["final_f1"] - 0.02 else warn)(f"반복 억제 1.15 F1 {m['final_f1_rep115']:.3f} (plain {m['final_f1']:.3f})")
    if "num_recall" in m:
        print(f"  수치 재현율 {m['num_recall']:.3f} (숫자가 든 정답 {m['num_recall_n']}문항 기준)")
    if m.get("split") == "question":
        print("  분할: question (같은 구역의 새 질문). group 분할 결과와 test 집합이 달라 손실과 F1을 서로 비교하지 않는다")
    if m.get("pretrain"):
        pr = m["pretrain"]
        print(f"  원문 이어 학습: 범위 {pr['scope']}, {pr['tokens']:,}토큰, {pr['epochs']}에폭 (QA 학습 전 test 손실 {m['base_test_loss']:.3f} -> {pr['test_loss_after']:.3f})")
    print(f"  n_train {m['n_train']} / val {m['n_val']} / test {m['n_test']}")
    (ok if m["n_train"] >= 1500 else warn)("train 1500쌍 이상이어야 의미 있는 비교가 된다")
    print("  F1은 글자 2-gram 겹침이라 수치/경고 정확도를 보장하지 않는다. 03 출력의 생성 예시를 읽고 숫자가 맞는지 직접 대조한다.")
    finish()


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sp = ap.add_subparsers(dest="cmd", required=True)
    for name, fn in [("env", c_env), ("raw", c_raw), ("chunks", c_chunks), ("batch", c_batch), ("qa", c_qa)]:
        sp.add_parser(name).set_defaults(fn=fn)
    p = sp.add_parser("tok"); p.add_argument("--model", default="skt/kogpt2-base-v2"); p.set_defaults(fn=c_tok)
    p = sp.add_parser("result"); p.add_argument("--out", default="runs/tesla_ko"); p.set_defaults(fn=c_result)
    a = ap.parse_args()
    a.fn(a)


if __name__ == "__main__":
    main()
