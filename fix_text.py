"""PDF 추출 텍스트에서 줄바꿈 때문에 갈라진 어절을 복원한다.

PDF는 줄 끝에서 단어가 잘리는데, 텍스트로 뽑으면 그 자리에 공백이 남는다.
  "충 돌", "있 습니다", "드라이 빙", "페이지 의"
학습에 쓰면 모델이 이 깨진 형태를 그대로 배우므로 복원한다.

복원 판단은 **문서 자체를 사전으로** 삼는다.
  "있 습니다"가 223곳 있는데 "있습니다"가 5,672곳 있다 -> 전자가 깨진 것이다.
붙인 형태가 문서에 훨씬 흔할 때만 고치므로, 원래 띄어 쓰는 말을 잘못 붙일 위험이 낮다.

사용 예
  python fix_text.py                             # data/raw/sections.jsonl 을 복원 (백업 자동)
  python fix_text.py --dry-run                   # 고치지 않고 무엇을 고칠지만 본다
  python fix_text.py --min-freq 5 --ratio 3      # 더 보수적으로 (기본 3 / 2)
"""
import argparse
import collections
import io
import json
import os
import re
import shutil

# 어절 경계에서 "짧은 조각 + 공백 + 짧은 조각" 을 찾는다. 앞뒤가 한글이면 어절 중간이므로 제외한다.
# 앞뒤 5글자까지 본다. 3글자까지만 보면 "마십 시오", "않 습니다" 같은
# 금지·부정 표현이 빠져나간다 (그 경고를 못 읽으면 Q&A 생성이 뒤집힌다).
PAIR = re.compile(r"(?<![가-힣])([가-힣]{1,5}) ([가-힣]{1,5})(?![가-힣])")
WORD = re.compile(r"[가-힣]+")


def read_jsonl(path):
    with io.open(path, encoding="utf-8") as f:
        return [json.loads(l) for l in f if l.strip()]


def build_dict(sections):
    """문서 전체의 한글 어절 빈도. 이 문서만의 사전으로 쓴다."""
    c = collections.Counter()
    for s in sections:
        c.update(WORD.findall(s["text"]))
    return c


def find_broken(sections, words, min_freq, ratio):
    """복원할 (앞조각, 뒷조각) -> 붙인 형태 목록을 만든다."""
    seen = collections.Counter()
    for s in sections:
        toks = re.split(r"(\s+)", s["text"])
        for i in range(0, len(toks) - 2, 2):
            a, sep, b = toks[i], toks[i + 1], toks[i + 2]
            if sep != " ":
                continue
            ma = re.search(r"([가-힣]{1,5})$", a or "")
            mb = re.match(r"([가-힣]{1,5})", b or "")
            if not (ma and mb):
                continue
            a, b = ma.group(1), mb.group(1)
            if words[a + b] >= min_freq:
                seen[(a, b)] += 1
    fix = {}
    for (a, b), cnt in seen.items():
        merged = a + b
        # 붙인 형태가 갈라진 형태보다 ratio 배 이상 흔할 때만 고친다
        if words[merged] >= cnt * ratio:
            fix[(a, b)] = merged
    return fix, seen


# 어절 끝의 한글 조각 / 어절 앞의 한글 조각.
# "주의하십시 오." 처럼 구두점이 붙어 있어도 한글 부분만 떼어 비교하기 위한 것이다.
TAIL = re.compile(r"([가-힣]{1,5})$")
HEAD = re.compile(r"^([가-힣]{1,5})")

# 문서 사전으로 판단할 수 없는 경우를 위한 문법 규칙.
# 한국어에서 종결어미와 조사는 반드시 앞말에 붙여 쓰므로, 빈도와 무관하게 복원해도 안전하다.
# (같은 문장이 네 개 매뉴얼에 반복되며 전부 같은 자리에서 깨지면 사전 판단이 불가능하다)
ENDING = re.compile(r"(십시|습니|합니|입니|됩니|칩니|읍니|니|세|어|아|여)$")   # 앞 조각의 끝
ENDING_TAIL = re.compile(r"^(오|다|요)$")                                  # 뒤 조각 전체
JOSA = re.compile(r"^(을|를|이|가|은|는|와|과|의|에|로|도|만|랑|께|야)$")      # 뒤 조각 전체(조사)


def grammar_merge(a, b):
    """사전에 없어도 문법상 붙여 써야 하는 조합이면 True."""
    if ENDING_TAIL.match(b) and ENDING.search(a):
        return True                                   # "분리하십시" + "오" -> "분리하십시오"
    if JOSA.match(b) and len(a) >= 2:
        return True                                   # "액추에이터" + "가" -> "액추에이터가"
    return False


def apply_fix(text, fix):
    """한 구역의 텍스트에서 깨진 어절을 복원한다. 문단 구분(\\n\\n)과 공백은 그대로 둔다.

    정규식 치환(re.sub)을 쓰지 않는 이유:
      re.sub 은 매치한 자리를 소비하므로 "펫 모 드" 에서 "펫 모"를 시도했다가 실패하면
      그 자리를 먹어버려 "모 드"를 못 본다. 연속으로 깨진 곳을 통째로 놓친다.
    그래서 어절 단위로 훑으면서, 실패하면 한 칸만 밀고 다시 본다.
    """
    parts = re.split(r"(\s+)", text)      # 공백도 보존한다: [말, 공백, 말, 공백, ...]
    n, i = 0, 0
    while i + 2 < len(parts):
        a, sep, b = parts[i], parts[i + 1], parts[i + 2]
        # 사이가 '한 칸 공백'일 때만 붙인다 (줄바꿈이면 문단 경계이므로 건드리지 않는다)
        ma = TAIL.search(a or "")
        mb = HEAD.match(b or "")
        key = (ma.group(1), mb.group(1)) if (ma and mb) else None
        ok = key in fix or (key and grammar_merge(*key))
        if sep == " " and ma and mb and ok:
            merged = fix.get(key) or (key[0] + key[1])
            # 앞 어절의 앞부분과 뒤 어절의 뒷부분(구두점 등)은 그대로 살린다
            parts[i] = a[:ma.start(1)] + merged + b[mb.end(1):]
            del parts[i + 1:i + 3]        # 공백과 뒷조각을 지운다
            n += 1
            continue                      # 같은 자리에서 다시 본다 (세 조각으로 갈린 것까지 잇는다)
        i += 2                            # 실패하면 다음 어절로 한 칸만 민다
    return "".join(parts), n


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--raw", default="data/raw/sections.jsonl")
    ap.add_argument("--min-freq", type=int, default=2, help="붙인 형태가 문서에 최소 몇 번 나와야 하는가")
    ap.add_argument("--ratio", type=float, default=2.0, help="붙인 형태가 갈라진 형태보다 몇 배 흔해야 하는가")
    ap.add_argument("--dry-run", action="store_true", help="고치지 않고 목록만 본다")
    ap.add_argument("--show", type=int, default=20, help="표본 몇 개를 보여줄 것인가")
    a = ap.parse_args()

    sections = read_jsonl(a.raw)
    words = build_dict(sections)
    print(f"[사전] 고유 어절 {len(words):,}개 (이 문서에서만 수집)")

    fix, seen = find_broken(sections, words, a.min_freq, a.ratio)
    total = sum(seen[k] for k in fix)
    print(f"[대상] 복원할 어절 {len(fix):,}종 / {total:,}곳\n")

    print(f"{'깨진 형태':<18}{'복원':<14}{'등장':>7}{'붙은 형태 빈도':>14}")
    for (x, y), cnt in seen.most_common():
        if (x, y) in fix and a.show > 0:
            print(f"{x + ' ' + y:<18}{fix[(x, y)]:<14}{cnt:>7}{words[fix[(x, y)]]:>14}")
            a.show -= 1

    # 복원 전후를 한 군데 보여준다 (눈으로 확인하기 위한 것)
    for s in sections:
        new, n = apply_fix(s["text"], fix)
        if n >= 3:
            i = next((k for k in range(len(s["text"])) if s["text"][k:k + 1] and PAIR.match(s["text"], k)), 0)
            print(f"\n[표본] 구역 {s['id']} ({n}곳 복원)")
            print(f"  전: ...{s['text'][i:i + 160]}...")
            print(f"  후: ...{new[i:i + 160]}...")
            break

    if a.dry_run:
        print("\n(--dry-run 이므로 파일을 고치지 않았다)")
        return

    # 원본을 백업하고 복원본을 쓴다
    backup = a.raw.replace(".jsonl", "_before_fix.jsonl")
    if not os.path.exists(backup):
        shutil.copy2(a.raw, backup)
        print(f"\n[백업] {backup}")

    fixed_total, out = 0, []
    for s in sections:
        new, n = apply_fix(s["text"], fix)
        fixed_total += n
        out.append(dict(s, text=new))
    with io.open(a.raw, "w", encoding="utf-8") as f:
        for s in out:
            f.write(json.dumps(s, ensure_ascii=False) + "\n")

    # 사람이 읽는 manual.txt 도 다시 쓴다
    man = os.path.join(os.path.dirname(a.raw), "manual.txt")
    with io.open(man, "w", encoding="utf-8") as f:
        for s in out:
            f.write(f"\n===== [{s['id']}] {s['title']} =====\n{s['text']}\n")

    chars = sum(len(s["text"]) for s in out)
    print(f"[저장] {fixed_total:,}곳 복원 -> {a.raw}, {man}")
    print(f"  구역 {len(out)}개, 글자 {chars:,}자")
    print(f"\n다음: python check_step.py raw   (추출 품질 재점검)")


if __name__ == "__main__":
    main()
