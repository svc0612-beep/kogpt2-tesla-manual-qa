"""가짜 Batch 결과 생성: 정상, 환각 숫자, 본문 언급, 깨진 JSON, 빈 응답, 실패 응답을 섞는다."""
import json, re, random, sys
random.seed(0)
chunks = [json.loads(l) for l in open("data/batch/chunks.jsonl", encoding="utf-8")]
QTPL = [("tire pressure", "What is the recommended cold tire pressure ({k})?"),
        ("charge limit", "What is the default charge limit ({k})?"),
        ("door alarm", "After how many seconds does the door alarm sound ({k})?"),
        ("Auto fold", "At what speed do the mirrors auto fold ({k})?"),
        ("overheat", "What temperature does cabin overheat protection hold ({k})?"),
        ("Traffic Aware", "What is the maximum cruise speed ({k})?"),
        ("blades", "When should the wiper blades be replaced ({k})?"),
        ("installation", "How long does a software installation take ({k})?")]
rows = []
for i, c in enumerate(chunks):
    sents = re.split(r"(?<=[.!?])\s+", c["text"])
    pairs = []
    for s in sents:
        for key, q in QTPL:
            if key in s:
                pairs.append({"question": q.format(k=f"{c['id']}-{len(pairs)}"), "answer": s.strip()})
    pairs = pairs[:3]
    mode = i % 11
    if mode == 3 and pairs:      # 환각: 본문에 없는 숫자
        pairs[0]["answer"] = pairs[0]["answer"] + " It is 999 psi."
    if mode == 5 and pairs:      # 본문 언급
        pairs[0]["question"] = "According to the text, " + pairs[0]["question"]
    if mode == 7:
        pairs = []
    content = json.dumps({"pairs": pairs}, ensure_ascii=False)
    if mode == 9:
        content = "```json\n" + content + "\n```"     # 코드펜스는 허용되어야 함
    if mode == 10:
        content = "{not json"
    row = {"id": f"batch_req_{i}", "custom_id": c["id"],
           "response": {"status_code": 200, "request_id": "x", "body": {"choices": [{"message": {"content": content}}]}}, "error": None}
    if mode == 8:
        row = {"id": f"batch_req_{i}", "custom_id": c["id"], "response": {"status_code": 500, "body": {}}, "error": None}
    rows.append(row)
random.shuffle(rows)                                   # 실제 결과도 입력 순서와 다르다
open("data/batch/output_000.jsonl", "w", encoding="utf-8").write("\n".join(json.dumps(r, ensure_ascii=False) for r in rows) + "\n")
print("mock rows", len(rows))
