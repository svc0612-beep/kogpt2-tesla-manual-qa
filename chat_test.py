"""학습된 모델 두 개 이상을 같은 질문으로 나란히 비교하는 챗봇 테스트.

  # 1) 자동 점검 (질문 세트 30개, 결과는 CSV/JSON으로 저장)
  python chat_test.py --models Q0=runs/sweep/Q0_s42/final Q2=runs/sweep/Q2_s42/final --batch

  # 2) 직접 대화 (같은 질문을 모든 모델에게 보내 답을 나란히 본다)
  python chat_test.py --models Q0=runs/sweep/Q0_s42/final Q2=runs/sweep/Q2_s42/final --chat

  # 학습 질문과 비슷한 문항 표시(암기 여부 구분): --train-jsonl runs/sweep/Q2_s42/splits/train.jsonl

주의: 이 모델들은 한 번에 질문 하나만 받는 질의응답 모델이다. 이전 대화를 기억하지 않는다(멀티턴 아님).
      자동 채점은 키워드 포함 여부일 뿐이다. 최종 판단은 CSV의 '사람 점수' 열에 직접 매긴다.
"""
import argparse
import csv
import json
import os
import re
import sys
from collections import Counter

import torch
from transformers import AutoModelForCausalLM, PreTrainedTokenizerFast

PROMPT = "### 질문: {q}\n### 답변:"        # 03_train.py의 한글 형식과 같다
ABSTAIN = re.compile(r"알 수 없|모릅니다|모르|확인할 수 없|답변할 수 없|제공하지|범위를 벗어|매뉴얼에.{0,10}없|해당 정보")


def load(path, device):
    if "kogpt2" in path.lower():
        # KoGPT2 원본(허브)은 special token이 비어 있어 직접 지정해야 한글이 깨지지 않는다.
        # 학습된 폴더는 저장될 때 함께 들어 있지만, 같은 값을 줘도 문제되지 않는다.
        tok = PreTrainedTokenizerFast.from_pretrained(path, bos_token="</s>", eos_token="</s>",
                                                      unk_token="<unk>", pad_token="<pad>", mask_token="<mask>")
    else:
        tok = PreTrainedTokenizerFast.from_pretrained(path)   # AutoTokenizer는 한글이 깨질 수 있어 쓰지 않는다
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    try:
        model = AutoModelForCausalLM.from_pretrained(path, dtype=torch.float32)
    except TypeError:
        model = AutoModelForCausalLM.from_pretrained(path, torch_dtype=torch.float32)
    return tok, model.to(device).eval()


@torch.inference_mode()
def ask(tok, model, question, device, max_new=100, no_repeat=3, beams=1, temp=0.0, rep=1.0):
    enc = tok(PROMPT.format(q=question), return_tensors="pt", add_special_tokens=False,
              return_token_type_ids=False).to(device)
    ctx = getattr(model.config, "n_positions", None) or getattr(model.config, "max_position_embeddings", 1024)
    max_new = max(1, min(max_new, ctx - enc["input_ids"].shape[1]))
    kw = dict(max_new_tokens=max_new, num_beams=beams, repetition_penalty=rep,
              eos_token_id=tok.eos_token_id, pad_token_id=tok.pad_token_id)
    if temp > 0:
        kw.update(do_sample=True, temperature=temp, top_p=0.9)
    else:
        kw.update(do_sample=False)
    if no_repeat > 0:
        kw["no_repeat_ngram_size"] = no_repeat
    out = model.generate(**enc, **kw)
    text = tok.decode(out[0][enc["input_ids"].shape[1]:], skip_special_tokens=True)
    text = re.sub(r"^[\s#]+", "", text)                # 맨 앞의 '###'는 형식 흉내이므로 떼어낸다
    return re.split(r"\s*#{2,}", text)[0].strip()      # 그 뒤에 또 나오면 거기서 자른다


def repeat_flag(ans):
    """반복 루프 의심: 단어 3-gram 중 같은 것이 3번 이상이거나, 고유 3-gram 비율이 0.6 미만."""
    w = ans.split()
    tri = [tuple(w[i:i + 3]) for i in range(max(0, len(w) - 2))]
    if len(tri) < 4:
        return False
    c = Counter(tri)
    return max(c.values()) >= 3 or len(c) / len(tri) < 0.6


def score(item, ans):
    """반환: (통과 여부 또는 None, 키워드 점수 또는 None, 메모). None은 자동 채점 대상이 아님."""
    a = re.sub(r"\s+", "", ans)
    if item.get("abstain"):
        ok = bool(ABSTAIN.search(ans))
        return ok, 1.0 if ok else 0.0, "거절/모름 표현 있음" if ok else "모른다고 하지 않고 답을 지어냄 가능성"
    if not item["must"]:
        return None, None, ""
    hit = sum(any(re.sub(r"\s+", "", k) in a for k in grp) for grp in item["must"])
    sc = hit / len(item["must"])
    bad = [f for f in item.get("forbid", []) if re.sub(r"\s+", "", f) in a]
    if bad:
        return False, 0.0, f"금지 표현 포함: {bad}"
    return sc == 1.0, sc, ""


def bigram_f1(x, y):
    f = lambda s: Counter(re.sub(r"\s+", "", s)[i:i + 2] for i in range(max(0, len(re.sub(r'\s+', '', s)) - 1)))
    a, b = f(x), f(y)
    ov = sum((a & b).values())
    if not ov:
        return 0.0
    p, r = ov / sum(a.values()), ov / sum(b.values())
    return 2 * p * r / (p + r)


def run_batch(models, items, a, device, train_qs):
    rows = []
    for it in items:
        row = dict(id=it["id"], cat=it["cat"], question=it["q"], ref=it["ref"])
        if train_qs:
            sim = max(bigram_f1(it["q"], t) for t in train_qs)
            row["학습질문유사도"] = round(sim, 2)
        for name, (tok, model) in models.items():
            ans = ask(tok, model, it["q"], device, a.max_new, a.no_repeat, a.num_beams, a.temp)
            ok, sc, note = score(it, ans)
            row.update({f"{name}_답": ans, f"{name}_통과": "" if ok is None else int(ok),
                        f"{name}_점수": "" if sc is None else round(sc, 2), f"{name}_반복": int(repeat_flag(ans)),
                        f"{name}_길이": len(ans), f"{name}_메모": note, f"{name}_사람점수(0/1/2)": ""})
        rows.append(row)
        print(f"\n[{it['id']}/{it['cat']}] {it['q']}\n  참고: {it['ref']}")
        for name in models:
            flag = {1: "통과", 0: "미달", "": "수동"}[row[f"{name}_통과"]]
            rep = " 반복의심" if row[f"{name}_반복"] else ""
            print(f"  {name:<6}[{flag}{rep}] {row[f'{name}_답'][:200]}" + (f"   <{row[f'{name}_메모']}>" if row[f"{name}_메모"] else ""))
    return rows


def summarize(models, rows):
    print("\n" + "=" * 86)
    print("모델별 요약 (통과율은 자동 채점 대상 문항 기준. 사람이 읽고 확인하기 전에는 참고치다)")
    print(f"{'모델':<8}{'분류':<12}{'문항':>5}{'통과':>6}{'통과율':>8}{'평균점수':>9}{'반복의심':>9}{'평균길이':>9}")
    for name in models:
        for cat in ["fact", "number", "safety", "paraphrase", "outofscope", "stress", "전체"]:
            rs = [r for r in rows if cat == "전체" or r["cat"] == cat]
            scored = [r for r in rs if r[f"{name}_통과"] != ""]
            if not rs:
                continue
            ps = sum(r[f"{name}_통과"] for r in scored)
            ms = sum(float(r[f"{name}_점수"]) for r in scored) / max(1, len(scored))
            rate = f"{ps / len(scored):.0%}" if scored else "-"
            print(f"{name:<8}{cat:<12}{len(rs):>5}{ps:>6}{rate:>8}{ms:>9.2f}{sum(r[f'{name}_반복'] for r in rs):>9}{sum(r[f'{name}_길이'] for r in rs) / len(rs):>9.0f}")
    names = list(models)
    if len(names) == 2:
        A, B = names
        win = sum(1 for r in rows if r[f"{A}_통과"] == 1 and r[f"{B}_통과"] == 0)
        lose = sum(1 for r in rows if r[f"{A}_통과"] == 0 and r[f"{B}_통과"] == 1)
        print(f"\n{A}만 통과 {win}문항 / {B}만 통과 {lose}문항 (둘 다 통과 또는 둘 다 미달은 제외)")
    # 일관성: 같은 사실을 다르게 물은 쌍이 둘 다 맞았는지
    groups = {}
    for r in rows:
        if r["cat"] == "paraphrase":
            groups.setdefault(r["id"][:2], []).append(r)
    if groups:
        for name in models:
            both = sum(all(x[f"{name}_통과"] == 1 for x in g) for g in groups.values())
            print(f"{name}: 의역 쌍 {len(groups)}개 중 둘 다 맞은 쌍 {both}개")
    print("=" * 86)


def run_chat(models, a, device):
    print("질문을 입력하면 모든 모델이 답한다. 명령: /q 종료, /nr N (no_repeat), /beams N, /temp T (0=결정적), /max N")
    st = dict(no_repeat=a.no_repeat, beams=a.num_beams, temp=a.temp, max_new=a.max_new)
    while True:
        try:
            q = input("\n질문> ").strip()
        except (EOFError, KeyboardInterrupt):
            break
        if not q:
            continue
        if q in ("/q", "/quit", "quit", "exit"):
            break
        m = re.match(r"/(nr|beams|temp|max)\s+([\d.]+)", q)
        if m:
            k, v = m.group(1), float(m.group(2))
            if k == "nr": st["no_repeat"] = int(v)
            elif k == "beams": st["beams"] = int(v)
            elif k == "temp": st["temp"] = v
            else: st["max_new"] = int(v)
            print("  설정:", st); continue
        for name, (tok, model) in models.items():
            ans = ask(tok, model, q, device, st["max_new"], st["no_repeat"], st["beams"], st["temp"])
            print(f"  [{name}] {ans}" + ("   (반복 의심)" if repeat_flag(ans) else ""))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--models", nargs="+", required=True, help="이름=폴더 형식. 예: Q0=runs/sweep/Q0_s42/final")
    ap.add_argument("--batch", action="store_true", help="질문 세트를 자동으로 돌리고 CSV/JSON 저장")
    ap.add_argument("--chat", action="store_true", help="직접 질문 입력")
    ap.add_argument("--questions", default="chat_questions.json")
    ap.add_argument("--only", nargs="+", default=None, help="일부 분류나 문항만 (예: fact safety P1a)")
    ap.add_argument("--train-jsonl", default=None, help="학습 질문 파일: 학습 질문과 비슷한 문항을 표시한다")
    ap.add_argument("--out", default="runs/chat_test")
    ap.add_argument("--max-new", type=int, default=100)
    ap.add_argument("--no-repeat", type=int, default=3)
    ap.add_argument("--num-beams", type=int, default=1)
    ap.add_argument("--temp", type=float, default=0.0, help="0이면 결정적 생성")
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    if not (a.batch or a.chat):
        sys.exit("--batch 또는 --chat 중 하나를 지정한다")
    torch.manual_seed(a.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    models = {}
    for spec in a.models:
        if "=" not in spec:
            sys.exit(f"'이름=폴더' 형식이 아니다: {spec}")
        name, path = spec.split("=", 1)
        if not os.path.isdir(path) and "/" not in path:
            sys.exit(f"폴더가 없다: {path}  (허브 모델명이면 skt/kogpt2-base-v2 처럼 슬래시를 포함한다)")
        models[name] = load(path, device)
        print(f"[로드] {name}: {path} ({sum(p.numel() for p in models[name][1].parameters()):,} 파라미터, {device})")
    if a.batch:
        items = json.load(open(a.questions, encoding="utf-8"))
        if a.only:
            items = [i for i in items if i["cat"] in a.only or i["id"] in a.only]
        train_qs = None
        if a.train_jsonl:
            train_qs = [json.loads(l)["question"] for l in open(a.train_jsonl, encoding="utf-8") if l.strip()]
        rows = run_batch(models, items, a, device, train_qs)
        summarize(models, rows)
        os.makedirs(a.out, exist_ok=True)
        json.dump(rows, open(os.path.join(a.out, "chat_report.json"), "w", encoding="utf-8"), ensure_ascii=False, indent=1)
        with open(os.path.join(a.out, "chat_report.csv"), "w", encoding="utf-8-sig", newline="") as f:      # 엑셀에서 한글이 깨지지 않게 BOM을 붙인다
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            w.writeheader(); w.writerows(rows)
        print(f"\n[저장] {a.out}/chat_report.csv (사람 점수 열을 직접 채운다), {a.out}/chat_report.json")
    if a.chat:
        run_chat(models, a, device)


if __name__ == "__main__":
    main()
