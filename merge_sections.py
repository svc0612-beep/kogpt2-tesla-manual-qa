"""여러 차종의 sections.jsonl을 하나로 합친다. (id/group에 차종 접두사, 차종 간 중복 측정)

Tesla 매뉴얼은 차종이 달라도 공통 기능 설명이 대부분 같다.
그대로 합치면 2단계(OpenAI Batch)에서 같은 내용으로 질문을 네 번 만들게 되므로,
먼저 얼마나 겹치는지 재고, 필요하면 --dedup 으로 중복 문단을 빼고 합친다.

사용 예
  python merge_sections.py --inputs S=data/raw_S 3=data/raw_3 X=data/raw_X Y=data/raw_Y --out data/raw
  python merge_sections.py --inputs S=data/raw_S 3=data/raw_3 X=data/raw_X Y=data/raw_Y --out data/raw --dedup

출력 (--out 폴더)
  sections.jsonl : {"id","group","title","text","source","model"}  id/group 앞에 차종 글자가 붙는다
  manual.txt     : 사람이 확인하기 위한 전체 텍스트

중복 판정 방법
  문단에서 차종 이름(Model S / Model 3 / Model X / Model Y)을 <MODEL>로 바꾸고 공백을 지운 뒤 비교한다.
  "Model S에는 ..." 과 "Model Y에는 ..." 처럼 차종 이름만 다른 문단을 같은 문단으로 센다.
"""
import argparse
import collections
import json
import os
import re
import sys

MODEL_NAME = re.compile(r"Model\s*(?:S|3|X|Y)")     # 문단 비교 전에 지울 차종 이름
MIN_PARA = 30                                        # 이보다 짧은 문단은 중복 집계에서 뺀다 ("경고:" 같은 공통 조각)
FULL_NAME = {"S": "Model S", "3": "Model 3", "X": "Model X", "Y": "Model Y"}


def read_jsonl(path):
    with open(path, encoding="utf-8") as f:
        return [json.loads(l) for l in f if l.strip()]


def norm_para(p):
    """차종 이름과 공백을 지운 비교용 문자열. 차종 이름만 다른 문단을 같은 것으로 센다."""
    return re.sub(r"\s+", "", MODEL_NAME.sub("<MODEL>", p))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--inputs", nargs="+", required=True,
                    help="차종글자=폴더 형식. 예: S=data/raw_S 3=data/raw_3 X=data/raw_X Y=data/raw_Y")
    ap.add_argument("--out", default="data/raw", help="합친 결과를 넣을 폴더")
    ap.add_argument("--dedup", action="store_true",
                    help="앞 차종에 이미 나온 문단을 뒤 차종에서 제거한다 (기본은 측정만 하고 전부 남긴다)")
    ap.add_argument("--min-chars", type=int, default=200, help="제거 후 이 글자 수 미만이 된 구역은 버린다")
    a = ap.parse_args()

    # 1) 차종별로 읽는다. 순서가 중요하다: --dedup 일 때 먼저 온 차종이 공통 문단의 주인이 된다
    loaded = []
    for spec in a.inputs:
        if "=" not in spec:
            sys.exit(f"'차종글자=폴더' 형식이 아니다: {spec}")
        key, folder = spec.split("=", 1)
        path = os.path.join(folder, "sections.jsonl")
        if not os.path.exists(path):
            sys.exit(f"파일이 없다: {path} (01_fetch_manual.py 를 --out {folder} 로 먼저 실행한다)")
        secs = read_jsonl(path)
        loaded.append((key, folder, secs))
        print(f"[읽음] {FULL_NAME.get(key, key)}: 구역 {len(secs)}개, "
              f"{sum(len(s['text']) for s in secs):,}자  <- {path}")

    # 2) 문단이 몇 개 차종에 나타나는지 센다 (한 차종 안에서 같은 문단이 여러 번 나와도 1로 센다)
    owners = collections.defaultdict(set)            # 정규화 문단 -> 그 문단을 가진 차종 집합
    sample_len = {}                                  # 정규화 문단 -> 원래 길이 (글자 수 집계용)
    for key, _, secs in loaded:
        for s in secs:
            for p in s["text"].split("\n\n"):
                if len(p.strip()) < MIN_PARA:
                    continue
                k = norm_para(p)
                owners[k].add(key)
                sample_len.setdefault(k, len(p))

    n_para = len(owners)
    by_count = collections.Counter(len(v) for v in owners.values())
    chars_by_count = collections.Counter()
    for k, v in owners.items():
        chars_by_count[len(v)] += sample_len[k]
    total_chars = sum(chars_by_count.values())

    print(f"\n[중복 측정] 길이 {MIN_PARA}자 이상 고유 문단 {n_para:,}개 (차종 이름 차이는 무시)")
    print(f"{'나타난 차종 수':<14}{'문단 수':>9}{'비중':>8}{'글자 수':>12}{'글자 비중':>10}")
    for c in sorted(by_count):
        print(f"{c}개 차종에만{'':<4}{by_count[c]:>9}{by_count[c]/n_para:>8.0%}"
              f"{chars_by_count[c]:>12,}{chars_by_count[c]/total_chars:>10.0%}")
    shared = sum(v for c, v in by_count.items() if c >= 2)
    print(f"-> 2개 이상 차종에 겹치는 문단: {shared:,}개 ({shared/n_para:.0%})")
    uniq_only = by_count.get(1, 0)
    print(f"-> 한 차종에만 있는 문단: {uniq_only:,}개 ({uniq_only/n_para:.0%})  <- 차종 고유 정보")

    # 3) 합친다. --dedup 이면 앞 차종이 이미 가진 문단을 뒤 차종에서 뺀다
    seen = set()                                     # --dedup 일 때 이미 채택한 정규화 문단
    out, removed = [], collections.Counter()
    for key, folder, secs in loaded:
        model = FULL_NAME.get(key, key)
        for s in secs:
            paras = s["text"].split("\n\n")
            if a.dedup:
                keep = []
                for p in paras:
                    k = norm_para(p)
                    if len(p.strip()) >= MIN_PARA and k in seen:
                        removed[key] += len(p)       # 앞 차종에 있던 문단이므로 버린다
                        continue
                    if len(p.strip()) >= MIN_PARA:
                        seen.add(k)
                    keep.append(p)
                paras = keep
            text = "\n\n".join(x for x in paras if x.strip())
            if len(text) < a.min_chars:              # 제거 후 거의 빈 구역은 버린다
                continue
            out.append(dict(
                id=f"{key}-{s['id']}",               # 차종 접두사: 차종끼리 id가 겹치지 않게 한다
                group=f"{key}-{s['group']}",         # group 도 같이 바꾼다 (3단계 분할 단위)
                title=f"{model} {s.get('title', '')}".strip(),   # 제목에 차종을 넣는다 (2단계 프롬프트로 전달된다)
                text=text,
                source=s.get("source", folder),
                model=model,                         # 차종 필드 (나중에 차종별 평가에 쓴다)
            ))

    if a.dedup:
        print("\n[중복 제거] " + ", ".join(f"{FULL_NAME.get(k, k)} {v:,}자" for k, v in removed.items()))

    # 4) 저장 (01_fetch_manual.py 의 write_outputs 와 같은 형식)
    os.makedirs(a.out, exist_ok=True)
    with open(os.path.join(a.out, "sections.jsonl"), "w", encoding="utf-8") as f:
        for s in out:
            f.write(json.dumps(s, ensure_ascii=False) + "\n")
    with open(os.path.join(a.out, "manual.txt"), "w", encoding="utf-8") as f:
        for s in out:
            f.write(f"\n===== [{s['id']}] {s['title']} =====\n{s['text']}\n")

    tot = sum(len(s["text"]) for s in out)
    print(f"\n[저장] 구역 {len(out)}개, 글자 {tot:,}자 -> {a.out}/sections.jsonl, manual.txt")
    print("  차종별 구역 수: " + ", ".join(
        f"{m} {c}개" for m, c in collections.Counter(s["model"] for s in out).items()))
    print(f"  그룹 {len({s['group'] for s in out})}개 (3단계 train/val/test 분할 단위)")


if __name__ == "__main__":
    main()
