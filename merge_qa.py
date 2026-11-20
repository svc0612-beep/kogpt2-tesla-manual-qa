"""여러 라운드에서 만든 질문/답변 파일을 합친다. (질문 중복과 답변 중복 제거, 구역 정보 유지)

  python merge_qa.py data/qa_all.jsonl data2/qa_all.jsonl --out data/qa_merged.jsonl
"""
import argparse
import json
import re
import sys


def norm(s):
    return re.sub(r"[\s\.\,\?\!~…\"'`]", "", s).lower()


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("inputs", nargs="+")
    ap.add_argument("--out", default="data/qa_merged.jsonl")
    a = ap.parse_args()
    seen_q, seen_a, out = set(), set(), []
    for i, path in enumerate(a.inputs):
        kept = dup = 0
        for line in open(path, encoding="utf-8"):
            if not line.strip():
                continue
            r = json.loads(line)
            kq, ka = norm(r["question"]), norm(r["answer"])
            if kq in seen_q or ka in seen_a:
                dup += 1
                continue
            seen_q.add(kq); seen_a.add(ka)
            r["id"] = f"r{i}-{r.get('id', len(out))}"
            out.append(r); kept += 1
        print(f"[{path}] 채택 {kept}, 중복 제외 {dup}")
    if not out:
        sys.exit("합친 결과가 비었다")
    with open(a.out, "w", encoding="utf-8") as f:
        for r in out:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"[합계] {len(out)}쌍, 그룹 {len({r['group'] for r in out})}개 -> {a.out}")


if __name__ == "__main__":
    main()
