"""2단계: 매뉴얼 텍스트 -> 질문/답변 학습 데이터 (OpenAI Batch API)

흐름
  chunk    : sections.jsonl -> chunks.jsonl (문단 단위로 일정 길이 조각)
  prepare  : 조각마다 요청 1건씩 batch_input_XXX.jsonl 생성 (아직 비용 없음)
  submit   : 파일 업로드 + batch 생성 (여기서부터 과금, 완료까지 최대 24시간)
  status   : 진행 확인, 완료되면 결과 파일 자동 다운로드
  collect  : 결과 파싱 + 품질 필터 + 중복 제거 -> qa_all.jsonl
  (collect --output-file 로컬.jsonl 로 이미 받은 결과를 바로 파싱할 수도 있다)

사용 예
  export OPENAI_API_KEY=sk-...
  python 02_make_dataset_batch.py chunk   --lang ko
  python 02_make_dataset_batch.py prepare --lang ko --model gpt-4.1-mini --pairs-per-chunk 4
  python 02_make_dataset_batch.py submit
  python 02_make_dataset_batch.py status          # 완료될 때까지 가끔 실행
  python 02_make_dataset_batch.py collect

메모
  - Batch 입력 한 줄: {"custom_id","method":"POST","url":"/v1/chat/completions","body":{...}}
    한 파일은 한 모델만 쓸 수 있고, custom_id는 중복되면 안 된다. 결과 순서는 입력 순서와 다르므로 custom_id로 맞춘다.
  - 모델 이름은 계정과 시점에 따라 다르므로 --model 로 지정한다(기본값은 확인이 필요하다).
  - 'chunk'와 'prepare'는 인터넷 없이 실행된다. 요청 수와 대략의 입력 토큰을 먼저 확인하고 submit 한다.
"""
import argparse
import collections
import json
import os
import re
import sys
import time

try:                                    # 사내망/보안 프로그램의 자체 서명 인증서를 OS 인증서 저장소에서 읽는다 (pip install truststore)
    import truststore
    truststore.inject_into_ssl()
except ImportError:
    pass

DIR = "data/batch"
RAW = "data/raw/sections.jsonl"
CHUNKS = os.path.join(DIR, "chunks.jsonl")
STATE = os.path.join(DIR, "state.json")
QA_OUT = "data/qa_all.jsonl"

SYSTEM = {
    "ko": (
        "당신은 자동차 사용자 매뉴얼을 바탕으로 학습용 질의응답 데이터를 만드는 편집자이다.\n"
        "주어진 [본문]에 실제로 적힌 내용만 사용하여 질문과 답변 쌍을 만든다.\n"
        "규칙:\n"
        "1. 질문은 본문 없이도 이해되는 독립적인 문장으로 쓴다. '본문', '위 내용', '이 구역' 같은 표현을 쓰지 않는다.\n"
        "2. 질문 유형을 섞는다: 사용 방법, 의미나 정의, 주의와 경고, 수치와 조건, 문제 해결.\n"
        "3. 답변은 1~3문장으로 정확하게 쓰고, 숫자와 단위와 조건은 본문 그대로 옮긴다. 본문에 없는 내용을 지어내지 않는다.\n"
        "4. 본문에 쓸 만한 정보가 없으면 빈 목록을 돌려준다.\n"
        "5. 쪽 번호, 다른 항목을 참조하라는 안내('페이지의 175 참조' 등), 등급분류 번호, 목차와 색인 항목에 관한 질문은 만들지 않는다.\n"
        "6. [본문]의 제목 맨 앞에 차종(Model S, Model 3, Model X, Model Y)이 적혀 있다. 모든 질문에 그 차종 이름을 반드시 포함한다. "
        "예: 'Model X의 전면 에어백은 어떤 상황에서 작동합니까?' 제목에 차종이 없으면 차종을 넣지 않는다.\n"
        "7. 본문에 '또는', '이상', '미만', '~인 경우' 같은 조건이 붙어 있으면 답변에 빠짐없이 옮긴다. "
        "조건을 하나라도 생략하면 안 된다.\n"
        "8. 본문이 금지나 경고('~하지 마십시오', '~하면 안 됩니다', '절대로 ~')이면 답변도 반드시 "
        "금지·경고로 쓴다. 긍정이나 허용으로 바꾸지 않는다.\n"
        "9. 본문에 없는 조건, 수치, 결과를 덧붙이지 않는다. 본문 문장을 재구성하되 내용을 늘리지 않는다.\n"
        "10. 출력은 JSON 객체 하나만: {\"pairs\": [{\"question\": \"...\", \"answer\": \"...\"}]}"
    ),
    "en": (
        "You are an editor who builds Q&A training data from a car owner's manual.\n"
        "Use only information that is actually written in the given [PASSAGE].\n"
        "Rules:\n"
        "1. Write each question so it is understandable without the passage. Never say 'the passage', 'the text' or 'this section'.\n"
        "2. Mix question types: how to use, meaning or definition, cautions and warnings, numbers and conditions, troubleshooting.\n"
        "3. Answers are 1-3 precise sentences. Copy numbers, units and conditions exactly. Never invent facts.\n"
        "4. If the passage has no usable information, return an empty list.\n"
        "5. Output exactly one JSON object: {\"pairs\": [{\"question\": \"...\", \"answer\": \"...\"}]}"
    ),
}
USER_TMPL = {
    "ko": "[본문]\n제목: {title}\n{text}\n\n위 본문에서 질문과 답변 쌍을 최대 {n}개 만들어 JSON으로 출력한다.",
    "en": "[PASSAGE]\nTitle: {title}\n{text}\n\nCreate up to {n} question-answer pairs from the passage above and output them as JSON.",
}
BANNED = {
    "ko": re.compile(r"본문|위 내용|위 글|제시된|지문|이 구역|이 문단|해당 구역|위의 내용|페이지\s*의?\s*\d+"),
    "en": re.compile(r"the passage|the text|this section|this passage|above (text|passage)|according to the", re.I),
}
NUM = re.compile(r"\d[\d,]*(?:\.\d+)?")


# ------------------------------------------------------------------ 유틸
def read_jsonl(path):
    with open(path, encoding="utf-8") as f:
        return [json.loads(l) for l in f if l.strip()]


def write_jsonl(path, rows):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def norm(s):
    return re.sub(r"[\s\.\,\?\!~…\"'`]", "", s).lower()


def nums(s):
    return {n.replace(",", "").rstrip(".") for n in NUM.findall(s)}


def ngrams(s, n=3):
    s = norm(s)
    return {s[i:i + n] for i in range(max(0, len(s) - n + 1))}


# ------------------------------------------------------------------ 노이즈 필터
RATING = re.compile(r"등급분류번호|제작·배급업신고번호|이용등급")
INDEX_ITEM = re.compile(r"[가-힣A-Za-z0-9\)\(]\s*:\s*\d{1,3}(?:\s*,\s*\d{1,3})*(?![\d.\"”a-zA-Z/%°])")   # 정수 쪽 번호만 (소수, 단위가 붙은 제원은 제외)
NUMBERED = re.compile(r"(?:^|\s)\d{1,2}\.\s")
XREF = re.compile(r"페이지\s*의\s*\d+|페이지\s*\d+")
LEGAL = re.compile(r"FCC|ISED|저작권 및 상표권|오픈 ?소스|무선 주파수|B등급 디지털|혼신|인증 적합성|VIAQ|사고기록장치|제품배상책임|Free Sounds|상표 또는 등록 상표|Copyright|©")


def noise_kind(p, in_tail):
    """문단 하나가 학습에 쓸 수 없는 종류인지 판정한다. 해당 없으면 None."""
    if RATING.search(p):
        return "등급분류 표"
    if len(INDEX_ITEM.findall(p)) >= 8 and len(INDEX_ITEM.findall(p)) / max(1, len(p)) * 100 > 1.5:
        return "색인"
    if len(XREF.findall(p)) >= 2 and len(NUMBERED.findall(p)) >= 2 and len(p) < 400 and len(p) / len(NUMBERED.findall(p)) < 60:
        return "목차/쪽 참조 목록"         # "4. 키 카드 센서(키 페이지의 22) 5. 충전 포트(...페이지의 173)" 형태. 본문 문단은 번호 항목이 거의 없다
    if in_tail and LEGAL.search(p):
        return "법적 고지(뒷부분)"
    return None


def drop_noise(sections, tail_frac):
    """문단 단위로 노이즈를 제거한다. 법적 고지 규칙은 뒷부분(전체 구역의 마지막 tail_frac)에만 적용한다."""
    out, stat = [], collections.Counter()
    n = len(sections)
    for i, sec in enumerate(sections):
        in_tail = i >= n * (1 - tail_frac)
        keep = []
        for p in sec["text"].split("\n\n"):
            k = noise_kind(p, in_tail) if p.strip() else None
            if k:
                stat[k] += len(p)
            else:
                keep.append(p)
        if keep:
            out.append(dict(sec, text="\n\n".join(keep)))
    print("[노이즈 제거] " + (", ".join(f"{k} {v:,}자" for k, v in stat.items()) if stat else "해당 없음"))
    return out


# ------------------------------------------------------------------ chunk
def cmd_chunk(a):
    sections = read_jsonl(a.raw)
    if not a.keep_noise:
        sections = drop_noise(sections, a.tail_frac)
    chunks = []
    for sec in sections:
        paras = []
        for p in (x for x in sec["text"].split("\n\n") if x.strip()):
            paras.extend(split_long(p, a.max_chars))
        cur, cur_len, k = [], 0, 0

        def flush():
            nonlocal cur, cur_len, k
            text = "\n".join(cur).strip()
            if len(text) >= a.min_chars and len(nums_free(text)) > 0:
                chunks.append(dict(id=f"{sec['id']}-c{k:02d}", group=sec["group"], title=sec["title"], text=text))
                k += 1
            cur, cur_len = [], 0

        for p in paras:
            if cur and cur_len + len(p) > a.max_chars:
                flush()
            cur.append(p); cur_len += len(p) + 1
        if cur:
            flush()
    write_jsonl(CHUNKS, chunks)
    tot = sum(len(c["text"]) for c in chunks)
    print(f"[chunk] 조각 {len(chunks)}개, 평균 {tot // max(1, len(chunks))}자 -> {CHUNKS}")


def split_long(p, max_chars):
    """PDF 추출은 문단 경계가 사라지는 일이 많다. 너무 긴 문단은 문장 단위로 쪼갠다."""
    if len(p) <= max_chars:
        return [p]
    sents = re.split(r"(?<=[.!?。])\s+|(?<=다\.)\s+|(?<=요\.)\s+", p)
    out, cur = [], ""
    for s_ in sents:
        if cur and len(cur) + len(s_) + 1 > max_chars:
            out.append(cur); cur = s_
        else:
            cur = (cur + " " + s_).strip()
    if cur:
        out.append(cur)
    return out


def nums_free(text):
    """숫자 비율이 너무 높은 조각(표, 목차 찌꺼기)을 거르기 위한 글자만 추출"""
    return re.sub(r"[\d\s\W_]+", "", text)


# ------------------------------------------------------------------ prepare
def cmd_prepare(a):
    chunks = read_jsonl(CHUNKS)
    lang = a.lang
    reqs = []
    for c in chunks:
        n = max(1, min(a.pairs_per_chunk, len(c["text"]) // a.chars_per_pair))
        body = {
            "model": a.model,
            "messages": [
                {"role": "system", "content": SYSTEM[lang]},
                {"role": "user", "content": USER_TMPL[lang].format(title=c["title"], text=c["text"], n=n)},
            ],
            "response_format": {"type": "json_object"},
        }
        if a.temperature is not None:
            body["temperature"] = a.temperature      # 일부 모델은 temperature를 받지 않는다. 기본은 생략
        reqs.append({"custom_id": c["id"], "method": "POST", "url": "/v1/chat/completions", "body": body})
    ids = [r["custom_id"] for r in reqs]
    assert len(ids) == len(set(ids)), "custom_id가 중복되었다"
    files = []
    for i in range(0, len(reqs), a.max_requests):
        path = os.path.join(DIR, f"batch_input_{i // a.max_requests:03d}.jsonl")
        write_jsonl(path, reqs[i:i + a.max_requests])
        files.append(path)
    chars = sum(len(json.dumps(r, ensure_ascii=False)) for r in reqs)
    print(f"[prepare] 요청 {len(reqs)}건, 파일 {len(files)}개, 입력 글자수 합계 {chars:,}자 (토큰은 한글 기준 글자수의 약 0.7~1배, 영문은 약 0.25배로 추정)")
    print("  첫 요청 미리보기:")
    print("  " + json.dumps(reqs[0], ensure_ascii=False)[:400] + " ...")
    state = dict(model=a.model, lang=lang, files=[dict(file=f) for f in files])
    json.dump(state, open(STATE, "w", encoding="utf-8"), ensure_ascii=False, indent=2)
    print("  확인 후 submit 을 실행한다. (과금은 submit 시점부터)")


# ------------------------------------------------------------------ submit / status
def client():
    if not os.environ.get("OPENAI_API_KEY"):
        sys.exit("환경변수 OPENAI_API_KEY가 필요하다")
    from openai import OpenAI
    return OpenAI()


def load_state():
    if not os.path.exists(STATE):
        sys.exit("state.json이 없다. prepare를 먼저 실행한다")
    return json.load(open(STATE, encoding="utf-8"))


def save_state(st):
    json.dump(st, open(STATE, "w", encoding="utf-8"), ensure_ascii=False, indent=2)


def cmd_submit(a):
    st = load_state()
    cl = client()
    for i, f in enumerate(st["files"]):
        if a.only is not None and i != a.only:
            continue
        if f.get("batch_id"):
            print(f"[건너뜀] {f['file']} 이미 제출됨: {f['batch_id']}")
            continue
        up = cl.files.create(file=open(f["file"], "rb"), purpose="batch")
        b = cl.batches.create(input_file_id=up.id, endpoint="/v1/chat/completions", completion_window="24h",
                              metadata={"job": "tesla-manual-qa", "part": str(i)})
        f.update(input_file_id=up.id, batch_id=b.id, status=b.status)
        save_state(st)
        print(f"[제출] {f['file']} -> batch {b.id} ({b.status})")


def cmd_status(a):
    st = load_state()
    cl = client()
    for i, f in enumerate(st["files"]):
        if not f.get("batch_id"):
            print(f"[{i}] 미제출"); continue
        b = cl.batches.retrieve(f["batch_id"])
        f.update(status=b.status, output_file_id=b.output_file_id, error_file_id=b.error_file_id)
        rc = b.request_counts
        print(f"[{i}] {b.id} {b.status} | 완료 {getattr(rc, 'completed', '?')}/{getattr(rc, 'total', '?')} 실패 {getattr(rc, 'failed', '?')}")
        if b.status == "failed" and b.errors:
            print("   오류:", [getattr(e, "message", e) for e in (b.errors.data or [])][:3])
        if b.status in ("completed", "expired", "cancelled") and b.output_file_id:
            path = os.path.join(DIR, f"output_{i:03d}.jsonl")
            if not os.path.exists(path):
                open(path, "wb").write(cl.files.content(b.output_file_id).read())
                print(f"   결과 저장: {path}")
        if b.error_file_id:
            path = os.path.join(DIR, f"errors_{i:03d}.jsonl")
            if not os.path.exists(path):
                open(path, "wb").write(cl.files.content(b.error_file_id).read())
                print(f"   오류 파일 저장: {path}")
    save_state(st)


# ------------------------------------------------------------------ collect
def parse_pairs(content):
    content = content.strip()
    content = re.sub(r"^```(?:json)?\s*|\s*```$", "", content)
    obj = json.loads(content)
    if isinstance(obj, dict):
        obj = obj.get("pairs", [])
    if not isinstance(obj, list):
        raise ValueError("pairs가 목록이 아니다")
    return obj


def cmd_collect(a):
    st = json.load(open(STATE, encoding="utf-8")) if os.path.exists(STATE) else {}
    lang = a.lang or st.get("lang", "ko")
    chunks = {c["id"]: c for c in read_jsonl(CHUNKS)}
    paths = [a.output_file] if a.output_file else sorted(
        os.path.join(DIR, f) for f in os.listdir(DIR) if f.startswith("output_") and f.endswith(".jsonl"))
    if not paths:
        sys.exit("결과 파일이 없다. status로 완료 여부와 다운로드를 확인한다")
    drop = {"요청 실패": 0, "JSON 오류": 0, "질문 길이": 0, "답변 길이": 0, "본문 언급": 0,
            "숫자가 본문에 없음": 0, "본문과 거의 안 겹침": 0, "중복 질문": 0}
    seen, out, n_resp, empty = set(), [], 0, 0
    for path in paths:
        for row in read_jsonl(path):
            cid = row["custom_id"]
            resp = row.get("response") or {}
            if row.get("error") or resp.get("status_code") != 200 or cid not in chunks:
                drop["요청 실패"] += 1; continue
            n_resp += 1
            try:
                pairs = parse_pairs(resp["body"]["choices"][0]["message"]["content"])
            except Exception:
                drop["JSON 오류"] += 1; continue
            if not pairs:
                empty += 1
            ch = chunks[cid]
            ch_nums, ch_grams = nums(ch["text"]), ngrams(ch["text"])
            for j, p in enumerate(pairs):
                q, ans = str(p.get("question", "")).strip(), str(p.get("answer", "")).strip()
                if not (a.min_q <= len(q) <= a.max_q):
                    drop["질문 길이"] += 1; continue
                if not (a.min_a <= len(ans) <= a.max_a):
                    drop["답변 길이"] += 1; continue
                if BANNED[lang].search(q) or BANNED[lang].search(ans):
                    drop["본문 언급"] += 1; continue
                if not nums(ans) <= ch_nums:        # 답변의 숫자는 본문에 있어야 한다 (환각 방지)
                    drop["숫자가 본문에 없음"] += 1; continue
                ag = ngrams(ans)
                if ag and len(ag & ch_grams) / len(ag) < a.min_ground:
                    drop["본문과 거의 안 겹침"] += 1; continue
                k = norm(q)
                if k in seen:
                    drop["중복 질문"] += 1; continue
                seen.add(k)
                out.append(dict(id=f"{cid}-q{j}", group=ch["group"], chunk_id=cid, title=ch["title"],
                                question=q, answer=ans))
    write_jsonl(QA_OUT, out)
    print(f"[collect] 응답 {n_resp}건 (빈 응답 {empty}) -> 통과 {len(out)}쌍 -> {QA_OUT}")
    for k, v in drop.items():
        if v:
            print(f"  제외 - {k}: {v}")
    if out:
        import statistics
        print(f"  질문 평균 {statistics.mean(len(o['question']) for o in out):.0f}자, "
              f"답변 평균 {statistics.mean(len(o['answer']) for o in out):.0f}자, 그룹 {len({o['group'] for o in out})}개")
        print("  [표본]")
        for o in out[:3]:
            print(f"   Q: {o['question']}\n   A: {o['answer']}")


# ------------------------------------------------------------------ main
def main():
    global DIR, RAW, CHUNKS, STATE, QA_OUT
    pre = argparse.ArgumentParser(add_help=False)
    pre.add_argument("--data-dir", default="data")
    dd = pre.parse_known_args()[0].data_dir      # 기본 경로 상수를 먼저 바꿔야 하위 명령의 기본값에 반영된다
    DIR, RAW = os.path.join(dd, "batch"), os.path.join(dd, "raw", "sections.jsonl")
    CHUNKS, STATE, QA_OUT = os.path.join(DIR, "chunks.jsonl"), os.path.join(DIR, "state.json"), os.path.join(dd, "qa_all.jsonl")
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data-dir", default="data", help="작업 폴더 (기본 data). 두 번째 생성 라운드는 data2처럼 따로 둔다. 하위 명령 앞에 쓴다")
    sp = ap.add_subparsers(dest="cmd", required=True)

    p = sp.add_parser("chunk")
    p.add_argument("--raw", default=RAW)
    p.add_argument("--max-chars", type=int, default=1200, help="조각 최대 글자수 (영문은 1800 정도 권장)")
    p.add_argument("--min-chars", type=int, default=300)
    p.add_argument("--keep-noise", action="store_true", help="등급분류 표, 색인, 쪽 참조 목록, 뒷부분 법적 고지 제거를 끈다")
    p.add_argument("--tail-frac", type=float, default=0.1, help="법적 고지 규칙을 적용할 뒷부분 비율")
    p.set_defaults(fn=cmd_chunk)

    p = sp.add_parser("prepare")
    p.add_argument("--lang", choices=["ko", "en"], default="ko")
    p.add_argument("--model", default="gpt-4.1-mini", help="Batch를 지원하는 모델명 (계정에서 확인이 필요하다)")
    p.add_argument("--pairs-per-chunk", type=int, default=4)
    p.add_argument("--chars-per-pair", type=int, default=250, help="조각이 짧으면 질문 수를 줄이는 기준")
    p.add_argument("--temperature", type=float, default=None)
    p.add_argument("--max-requests", type=int, default=2000, help="파일당 요청 수 (한도를 넘으면 나눠 제출)")
    p.set_defaults(fn=cmd_prepare)

    p = sp.add_parser("submit")
    p.add_argument("--only", type=int, default=None, help="파일 번호 하나만 제출")
    p.set_defaults(fn=cmd_submit)

    p = sp.add_parser("status")
    p.set_defaults(fn=cmd_status)

    p = sp.add_parser("collect")
    p.add_argument("--lang", choices=["ko", "en"], default=None)
    p.add_argument("--output-file", default=None, help="이미 받은 결과 jsonl을 직접 지정")
    p.add_argument("--min-q", type=int, default=8)
    p.add_argument("--max-q", type=int, default=200)
    p.add_argument("--min-a", type=int, default=10)
    p.add_argument("--max-a", type=int, default=600)
    p.add_argument("--min-ground", type=float, default=0.10, help="답변 글자 3-gram 중 본문에 있는 비율의 하한")
    p.set_defaults(fn=cmd_collect)

    a = ap.parse_args()
    os.makedirs(DIR, exist_ok=True)
    a.fn(a)


if __name__ == "__main__":
    main()
