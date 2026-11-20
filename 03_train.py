"""3단계: 질문/답변 데이터로 distilgpt2 또는 KoGPT2 미세조정 (전체 파라미터 학습)

  python 03_train.py --model distilgpt2          --data data/qa_all.jsonl   # 영문 매뉴얼(en_us)용
  python 03_train.py --model skt/kogpt2-base-v2  --data data/qa_all.jsonl   # 한글 매뉴얼(ko_kr)용

핵심 설계
  1) 분할: 질문 단위가 아니라 group(= 매뉴얼의 같은 구역) 단위로 train/val/test를 나눈다.
     같은 구역에서 나온 질문들이 양쪽에 섞이면 검증 손실이 거짓으로 좋아지기 때문이다.
  2) 형식: "### 질문: {q}\\n### 답변: {a}<EOS>"   (영문은 Question/Answer)
     prompt 부분의 label은 -100으로 가려서 '답변 + EOS'만 학습한다.
  3) 패딩: distilgpt2는 pad가 없어 eos를 pad로 쓰므로, 기본 collator를 쓰면 끝 표시(EOS)까지 가려질 수 있다.
     이 스크립트는 길이 기반으로 직접 패딩하여 EOS 라벨을 보존한다.
  4) 정밀도: 가중치는 fp32로 불러오고(transformers 5.x는 dtype 미지정 시 저장된 dtype을 따른다),
     GPU가 bf16을 지원하면 bf16 autocast를 쓴다. fp16 혼합정밀도는 쓰지 않는다.
  5) 비교: 학습 전(base) 모델의 test 손실과 생성 품질을 함께 재서 '학습 효과'를 수치로 본다.

한계(중요)
  - 82M~125M 모델은 수치와 경고 문구를 정확히 외우는 데 한계가 있다. 실제 안내 용도에는 검색(RAG)이 더 안전하다.
  - distilgpt2의 토크나이저는 영어 중심이다. 한글 데이터에는 KoGPT2를 쓴다. (KoGPT2 라이선스: CC-BY-NC-SA 4.0, 비상업)
"""
import argparse
import inspect
import json
import math
import os
import random
import re
import sys

import torch
from torch.utils.data import DataLoader, Dataset
from transformers import (AutoModelForCausalLM, AutoTokenizer, EarlyStoppingCallback, PreTrainedTokenizerFast,
                          Trainer, TrainingArguments, set_seed)

FMT = {
    "ko": ("### 질문: {q}\n### 답변:", " {a}"),
    "en": ("### Question: {q}\n### Answer:", " {a}"),
}


# ------------------------------------------------------------------ 데이터
def read_jsonl(path):
    with open(path, encoding="utf-8") as f:
        return [json.loads(l) for l in f if l.strip()]


def write_jsonl(path, rows):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def norm(s):
    return re.sub(r"[\s\.\,\?\!~…\"'`]", "", s).lower()


def split_by_group(rows, val_ratio, test_ratio, seed):
    groups = sorted({r["group"] for r in rows})
    random.Random(seed).shuffle(groups)
    n = len(groups)
    n_test = max(1, round(n * test_ratio)) if n >= 3 else 0
    n_val = max(1, round(n * val_ratio)) if n >= 3 else 0
    test_g, val_g = set(groups[:n_test]), set(groups[n_test:n_test + n_val])
    tr, va, te = [], [], []
    for r in rows:
        (te if r["group"] in test_g else va if r["group"] in val_g else tr).append(r)
    # 방어: 같은 질문 또는 같은 답변이 train과 val/test에 동시에 있으면 train에서 뺀다
    held_q = {norm(r["question"]) for r in va + te}
    held_a = {norm(r["answer"]) for r in va + te if len(r["answer"]) >= 20}   # 짧은 답("예" 등)은 제외
    before = len(tr)
    tr = [r for r in tr if norm(r["question"]) not in held_q and norm(r["answer"]) not in held_a]
    assert not (set(r["group"] for r in tr) & set(r["group"] for r in va + te)), "그룹이 분할 사이에 겹쳤다"
    return tr, va, te, before - len(tr)


def split_by_question(rows, val_ratio, test_ratio, seed):
    """질문 단위 분할. 같은 매뉴얼 구역의 다른 질문이 train에 남는다.
    '이미 학습한 구역의, 본 적 없는 질문'에 답하는지(= 매뉴얼 암기/활용)를 잰다. group 분할과 목적이 다르다."""
    idx = list(range(len(rows)))
    random.Random(seed).shuffle(idx)
    n = len(rows)
    n_test, n_val = max(1, round(n * test_ratio)), max(1, round(n * val_ratio))
    te_i, va_i = set(idx[:n_test]), set(idx[n_test:n_test + n_val])
    tr, va, te = [], [], []
    for i, r in enumerate(rows):
        (te if i in te_i else va if i in va_i else tr).append(r)
    held_q = {norm(r["question"]) for r in va + te}
    held_a = {norm(r["answer"]) for r in va + te if len(r["answer"]) >= 20}
    before = len(tr)
    tr = [r for r in tr if norm(r["question"]) not in held_q and norm(r["answer"]) not in held_a]
    return tr, va, te, before - len(tr)


def build_lm_items(chunks, tok, max_len):
    """매뉴얼 원문 조각을 max_len 길이 창으로 잘라 인과 언어모델 학습 항목(라벨 = 입력 전체)으로 만든다."""
    items = []
    for c in chunks:
        ids = tok(c["text"], add_special_tokens=False)["input_ids"]
        step = max_len - 1
        for st in range(0, len(ids), step):
            w = ids[st:st + step]
            if len(w) < 8:
                continue
            w = w + [tok.eos_token_id]
            items.append(dict(input_ids=w, labels=list(w)))
    return items


class QADataset(Dataset):
    def __init__(self, items):
        self.items = items

    def __len__(self):
        return len(self.items)

    def __getitem__(self, i):
        return self.items[i]


def encode(rows, tok, lang, max_len):
    pf, cf = FMT[lang]
    out, dropped = [], 0
    for r in rows:
        p = tok(pf.format(q=r["question"]), add_special_tokens=False)["input_ids"]
        c = tok(cf.format(a=r["answer"]), add_special_tokens=False)["input_ids"] + [tok.eos_token_id]
        if len(p) + len(c) > max_len:
            dropped += 1
            continue
        out.append(dict(input_ids=p + c, labels=[-100] * len(p) + c, ref=r["answer"], question=r["question"]))
    return out, dropped


class Collator:
    """길이 기반 패딩: pad id가 eos와 같아도 EOS 라벨이 가려지지 않는다."""
    def __init__(self, pad_id):
        self.pad_id = pad_id

    def __call__(self, batch):
        m = max(len(b["input_ids"]) for b in batch)
        ids = torch.full((len(batch), m), self.pad_id, dtype=torch.long)
        lab = torch.full((len(batch), m), -100, dtype=torch.long)
        att = torch.zeros((len(batch), m), dtype=torch.long)
        for i, b in enumerate(batch):
            n = len(b["input_ids"])
            ids[i, :n] = torch.tensor(b["input_ids"])
            lab[i, :n] = torch.tensor(b["labels"])
            att[i, :n] = 1
        return dict(input_ids=ids, labels=lab, attention_mask=att)


# ------------------------------------------------------------------ 모델
def load_tokenizer(name, kind):
    if kind == "kogpt2" or (kind == "auto" and "kogpt2" in name.lower()):
        # KoGPT2는 AutoTokenizer가 영어 GPT-2 토크나이저로 잘못 잡을 수 있어 special token을 명시한다.
        tok = PreTrainedTokenizerFast.from_pretrained(name, bos_token="</s>", eos_token="</s>", unk_token="<unk>",
                                                      pad_token="<pad>", mask_token="<mask>")
    else:
        tok = AutoTokenizer.from_pretrained(name)
        if tok.pad_token is None:
            tok.pad_token = tok.eos_token
    tok.padding_side = "right"
    return tok


def load_model(name, tok):
    try:
        model = AutoModelForCausalLM.from_pretrained(name, dtype=torch.float32)
    except TypeError:                       # transformers 4.5x 이하
        model = AutoModelForCausalLM.from_pretrained(name, torch_dtype=torch.float32)
    emb = model.get_input_embeddings().num_embeddings
    if len(tok) > emb:
        print(f"[임베딩 확장] 토크나이저 {len(tok)} > 모델 {emb}")
        model.resize_token_embeddings(len(tok))
    model.config.pad_token_id = tok.pad_token_id
    model.config.eos_token_id = tok.eos_token_id
    gc = model.generation_config
    gc.pad_token_id, gc.eos_token_id = tok.pad_token_id, tok.eos_token_id
    return model


# ------------------------------------------------------------------ 평가
@torch.no_grad()
def eval_loss(model, ds, collator, device, bs=16):
    model.eval()
    tot, cnt = 0.0, 0
    for batch in DataLoader(ds, batch_size=bs, collate_fn=collator):
        batch = {k: v.to(device) for k, v in batch.items()}
        n = int((batch["labels"][:, 1:] != -100).sum())
        if n == 0:
            continue
        tot += model(**batch).loss.float().item() * n
        cnt += n
    return tot / max(1, cnt)


def bigrams(s):
    s = re.sub(r"\s+", "", s)
    return [s[i:i + 2] for i in range(max(0, len(s) - 1))]


def f1(pred, ref):
    from collections import Counter
    a, b = Counter(bigrams(pred)), Counter(bigrams(ref))
    ov = sum((a & b).values())
    if not ov:
        return 0.0
    p, r = ov / sum(a.values()), ov / sum(b.values())
    return 2 * p * r / (p + r)


@torch.no_grad()
def answer(model, tok, question, lang, device, max_new=100, rep=1.0, no_repeat=0, beams=1):
    model.eval()
    prompt = FMT[lang][0].format(q=question)
    enc = tok(prompt, return_tensors="pt", add_special_tokens=False, return_token_type_ids=False).to(device)
    ctx = getattr(model.config, "n_positions", None) or getattr(model.config, "max_position_embeddings", 1024)
    max_new = max(1, min(max_new, ctx - enc["input_ids"].shape[1]))      # 모델의 최대 길이를 넘지 않게 한다
    kw = dict(max_new_tokens=max_new, do_sample=False, repetition_penalty=rep, num_beams=beams,
              eos_token_id=tok.eos_token_id, pad_token_id=tok.pad_token_id)
    if no_repeat > 0:
        kw["no_repeat_ngram_size"] = no_repeat      # 같은 n-gram 반복 금지 (토큰 단위 페널티보다 '반복 루프'를 직접 막는다)
    out = model.generate(**enc, **kw)
    text = tok.decode(out[0][enc["input_ids"].shape[1]:], skip_special_tokens=True)
    return re.split(r"\s*#{2,}", text)[0].strip()      # "###"뿐 아니라 "## 답변:" 같은 반복 시작도 자른다


NUMRE = re.compile(r"\d+(?:\.\d+)?")


def num_recall(preds, items):
    """정답에 들어 있는 숫자 중 생성 답변에도 나타난 비율의 평균 (숫자가 있는 정답만 대상). 반환: (값, 대상 수)"""
    vals = []
    for p, it in zip(preds, items):
        ref = set(NUMRE.findall(it["ref"]))
        if ref:
            vals.append(len(ref & set(NUMRE.findall(p))) / len(ref))
    return (sum(vals) / len(vals) if vals else 0.0), len(vals)


def gen_eval(model, tok, items, lang, device, n, rep=1.0, no_repeat=0, beams=1):
    items = items[:n]
    preds = [answer(model, tok, it["question"], lang, device, rep=rep, no_repeat=no_repeat, beams=beams) for it in items]
    score = sum(f1(p, it["ref"]) for p, it in zip(preds, items)) / max(1, len(preds))
    return preds, score, num_recall(preds, items)


# ------------------------------------------------------------------ main
def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", default="distilgpt2")
    ap.add_argument("--tokenizer-kind", choices=["auto", "kogpt2", "plain"], default="auto")
    ap.add_argument("--data", default="data/qa_all.jsonl")
    ap.add_argument("--lang", choices=["auto", "ko", "en"], default="auto")
    ap.add_argument("--out", default="runs/tesla")
    ap.add_argument("--epochs", type=float, default=5)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--bs", type=int, default=8)
    ap.add_argument("--grad-accum", type=int, default=2)
    ap.add_argument("--max-len", type=int, default=256)
    ap.add_argument("--val-ratio", type=float, default=0.1)
    ap.add_argument("--test-ratio", type=float, default=0.1)
    ap.add_argument("--patience", type=int, default=2)
    ap.add_argument("--n-eval", type=int, default=50, help="생성 평가에 쓸 test 질문 수")
    ap.add_argument("--n-show", type=int, default=5)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--split", choices=["group", "question"], default="group",
                    help="group: 본 적 없는 구역에 일반화하는가 / question: 학습한 구역의 새 질문에 답하는가(매뉴얼 활용 시나리오)")
    ap.add_argument("--pretrain-data", default=None, help="data/batch/chunks.jsonl 같은 원문 조각 파일: 지정하면 QA 학습 전에 원문으로 이어 학습한다")
    ap.add_argument("--pretrain-epochs", type=float, default=3)
    ap.add_argument("--pretrain-lr", type=float, default=5e-5)
    ap.add_argument("--pretrain-scope", choices=["auto", "train", "all"], default="auto",
                    help="train: train 구역의 원문만(group 분할의 평가 오염 방지) / all: 전체 원문. auto = group 분할이면 train, question 분할이면 all")
    ap.add_argument("--no-repeat", type=int, default=0, help="생성 시 no_repeat_ngram_size (0이면 끔, 3 권장)")
    ap.add_argument("--num-beams", type=int, default=1)
    ap.add_argument("--optim", default="adamw_torch",
                    help="옵티마이저. adamw_torch(기본, 상태 1.0GB) / adafactor(상태 거의 0, VRAM 2GB 환경용) / "
                         "adamw_bnb_8bit(0.25GB, bitsandbytes 필요)")
    ap.add_argument("--grad-ckpt", action="store_true",
                    help="그래디언트 체크포인팅: 활성값 메모리를 줄이고 속도를 20~30%% 희생한다 (VRAM이 모자랄 때)")
    a = ap.parse_args()

    set_seed(a.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    rows = read_jsonl(a.data)
    head = rows[:200]
    lang = a.lang if a.lang != "auto" else (
        "ko" if sum(bool(re.search("[가-힣]", r["answer"])) for r in head) > len(head) / 2 else "en")
    print(f"[설정] 모델 {a.model} | 언어 {lang} | 장치 {device} | 데이터 {len(rows)}쌍")

    # 1) 분할
    splitter = split_by_group if a.split == "group" else split_by_question
    tr, va, te, removed = splitter(rows, a.val_ratio, a.test_ratio, a.seed)
    unit = "그룹 단위" if a.split == "group" else "질문 단위(같은 구역의 다른 질문은 train에 있다)"
    print(f"[분할] train {len(tr)} / val {len(va)} / test {len(te)}  ({unit}, 중복 방어로 train에서 {removed}건 제외)")
    if not tr or not va or not te:
        sys.exit("분할 결과가 비었다. 그룹 수가 너무 적다. 데이터를 늘린다(그룹 3개 이상 필요).")
    for name, part in [("train", tr), ("val", va), ("test", te)]:
        write_jsonl(os.path.join(a.out, "splits", f"{name}.jsonl"), part)

    # 2) 토크나이저 + 인코딩 + 라벨 점검
    tok = load_tokenizer(a.model, a.tokenizer_kind)
    sample = [r["answer"] for r in rows[:100]]
    tpc = sum(len(tok(s, add_special_tokens=False)["input_ids"]) for s in sample) / max(1, sum(len(s) for s in sample))
    print(f"[토크나이저] 어휘 {len(tok)} | eos {tok.eos_token!r}({tok.eos_token_id}) pad {tok.pad_token!r}({tok.pad_token_id}) | 글자당 토큰 {tpc:.2f}")
    if lang == "ko" and tpc > 1.2:
        print("  경고: 한글이 글자당 1.2토큰을 넘는다. 영어 중심 토크나이저로 보인다. KoGPT2 사용을 권한다.")
    ds = {}
    for name, part in [("train", tr), ("val", va), ("test", te)]:
        items, dropped = encode(part, tok, lang, a.max_len)
        ds[name] = QADataset(items)
        rate = dropped / max(1, len(part))
        print(f"  {name}: {len(items)}개 사용, {dropped}개({rate:.1%}) 길이 초과로 제외" + ("  <- --max-len을 늘린다" if rate > 0.05 else ""))
    collator = Collator(tok.pad_token_id)
    b = collator([ds["train"][0]])
    kept = b["labels"][0][b["labels"][0] != -100]
    print("[라벨 점검] 모델이 읽는 글:", repr(tok.decode(b["input_ids"][0])))
    print("            학습되는 부분:", repr(tok.decode(kept)))
    assert kept[-1].item() == tok.eos_token_id, "마지막 학습 토큰이 EOS가 아니다"

    # 3) 모델 + 학습 전(base) 기준선
    model = load_model(a.model, tok).to(device)
    print(f"[모델] 파라미터 {sum(p.numel() for p in model.parameters()):,} | dtype {next(model.parameters()).dtype}")
    base_loss = eval_loss(model, ds["test"], collator, device)
    dec = dict(no_repeat=a.no_repeat, beams=a.num_beams)
    base_preds, base_f1, base_num = gen_eval(model, tok, ds["test"].items, lang, device, a.n_eval, 1.0, **dec)
    print(f"[학습 전] test 손실 {base_loss:.4f} (ppl {math.exp(base_loss):.1f}) | 생성 F1 {base_f1:.3f} | 수치 재현율 {base_num[0]:.3f}")
    echo_f1 = sum(f1(it["question"], it["ref"]) for it in ds["test"].items[:a.n_eval]) / max(1, min(a.n_eval, len(ds["test"].items)))
    print(f"[기준선] 질문을 그대로 되풀이했을 때 생성 F1 {echo_f1:.3f}  (모델 F1이 이 값에 가까우면 내용을 배운 것이 아니다)")

    # 3.5) (선택) 원문 이어 학습: 매뉴얼 본문 자체를 먼저 읽혀 사실을 모델에 넣는다
    use_bf16 = device == "cuda" and torch.cuda.is_bf16_supported()
    sig = inspect.signature(TrainingArguments.__init__).parameters
    tk = "processing_class" if "processing_class" in inspect.signature(Trainer.__init__).parameters else "tokenizer"
    pre_info, stage1_loss = None, None
    if a.pretrain_data:
        chunks = read_jsonl(a.pretrain_data)
        scope = a.pretrain_scope if a.pretrain_scope != "auto" else ("train" if a.split == "group" else "all")
        if scope == "train":
            tr_groups = {r["group"] for r in tr}
            chunks = [c for c in chunks if c["group"] in tr_groups]
        lm_items = build_lm_items(chunks, tok, a.max_len)
        if not lm_items:
            sys.exit("원문 이어 학습 데이터가 비었다. --pretrain-data 경로와 group 값을 확인한다")
        n_tok = sum(len(x["input_ids"]) for x in lm_items)
        print(f"[1단계: 원문 이어 학습] 조각 {len(chunks)}개 -> 창 {len(lm_items)}개, 약 {n_tok:,}토큰, 범위 {scope}, 에폭 {a.pretrain_epochs}, 학습률 {a.pretrain_lr}")
        if a.split == "question" and scope == "all":
            print("  주의: question 분할에서는 test 질문의 근거 원문도 읽힌다. '매뉴얼을 알고 있을 때 답하는가'를 재는 설정이다")
        pkw = dict(output_dir=os.path.join(a.out, "ckpt_pre"), num_train_epochs=a.pretrain_epochs,
                   per_device_train_batch_size=a.bs, gradient_accumulation_steps=1, learning_rate=a.pretrain_lr,
                   weight_decay=0.01, lr_scheduler_type="cosine", warmup_steps=5, max_grad_norm=1.0, bf16=use_bf16,
                   logging_steps=max(1, len(lm_items) // a.bs // 2), save_strategy="no", report_to="none",
                   seed=a.seed, remove_unused_columns=False,
                   optim=a.optim, gradient_checkpointing=a.grad_ckpt)
        pre_trainer = Trainer(model=model, args=TrainingArguments(**pkw), train_dataset=QADataset(lm_items),
                              data_collator=collator, **{tk: tok})
        pre_trainer.train()
        plosses = [h["loss"] for h in pre_trainer.state.log_history if "loss" in h]
        assert plosses and all(math.isfinite(x) for x in plosses), "원문 이어 학습 손실에 NaN/Inf가 있다"
        stage1_loss = eval_loss(model, ds["test"], collator, device)
        print(f"[1단계 후] test 손실(QA 형식, 아직 QA 학습 전) {base_loss:.4f} -> {stage1_loss:.4f}")
        pre_info = dict(chunks=len(chunks), windows=len(lm_items), tokens=n_tok, scope=scope,
                        epochs=a.pretrain_epochs, lr=a.pretrain_lr, test_loss_after=stage1_loss)

    # 4) 학습 (QA 형식)
    steps_per_epoch = math.ceil(len(ds["train"]) / (a.bs * a.grad_accum))
    total_steps = max(1, int(steps_per_epoch * a.epochs))
    kw = dict(
        output_dir=os.path.join(a.out, "ckpt"),
        num_train_epochs=a.epochs,
        per_device_train_batch_size=a.bs,
        per_device_eval_batch_size=a.bs * 2,
        gradient_accumulation_steps=a.grad_accum,
        learning_rate=a.lr,
        weight_decay=0.01,
        lr_scheduler_type="cosine",
        warmup_steps=max(1, int(0.05 * total_steps)),     # 정수로 계산 (버전에 상관없이 동작)
        max_grad_norm=1.0,
        bf16=use_bf16,
        logging_steps=max(1, steps_per_epoch // 4),
        save_strategy="epoch",
        save_total_limit=2,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        report_to="none",
        seed=a.seed,
        remove_unused_columns=False,
        optim=a.optim,                      # 옵티마이저 상태가 VRAM을 가장 많이 먹는다
        gradient_checkpointing=a.grad_ckpt, # 활성값을 다시 계산해 메모리를 아낀다
    )
    kw["eval_strategy" if "eval_strategy" in sig else "evaluation_strategy"] = "epoch"
    targs = TrainingArguments(**kw)
    trainer = Trainer(model=model, args=targs, train_dataset=ds["train"], eval_dataset=ds["val"],
                      data_collator=collator, callbacks=[EarlyStoppingCallback(early_stopping_patience=a.patience)],
                      **{tk: tok})
    print(f"[학습] 에폭당 {steps_per_epoch}스텝, 총 최대 {total_steps}스텝, bf16={use_bf16}, "
          f"optim={a.optim}, grad_ckpt={a.grad_ckpt}")
    if device == "cuda":
        torch.cuda.reset_peak_memory_stats()        # 여기부터의 최대 사용량만 재기 위해 초기화한다
    trainer.train()
    peak = torch.cuda.max_memory_allocated() / 1024**3 if device == "cuda" else 0.0
    if device == "cuda":
        print(f"[메모리] 학습 중 최대 할당 {peak:.2f}GB (여유가 많으면 --bs 를 올린다)")

    hist = trainer.state.log_history
    losses = [h["loss"] for h in hist if "loss" in h]
    assert losses and all(math.isfinite(x) for x in losses), "학습 손실에 NaN/Inf가 있다"
    assert max(losses) < 15, f"학습 손실 {max(losses):.1f}는 비정상(발산)이다. 학습률과 dtype을 점검한다"
    print("[검증 손실 기록]", [round(h["eval_loss"], 4) for h in hist if "eval_loss" in h])

    # 5) 학습 후 평가 (val로 고른 최적 체크포인트가 적용된 상태)
    final_loss = eval_loss(trainer.model, ds["test"], collator, device)
    preds_plain, f1_plain, num_plain = gen_eval(trainer.model, tok, ds["test"].items, lang, device, a.n_eval, 1.0, **dec)
    preds_rep, f1_rep, _ = gen_eval(trainer.model, tok, ds["test"].items, lang, device, a.n_eval, 1.15, **dec)
    print(f"[학습 후] test 손실 {final_loss:.4f} (ppl {math.exp(final_loss):.1f}) | 생성 F1 {f1_plain:.3f} (반복 억제 1.15: {f1_rep:.3f}) | 수치 재현율 {num_plain[0]:.3f} ({num_plain[1]}문항)")
    print(f"[개선] 손실 {base_loss:.3f} -> {final_loss:.3f}, F1 {base_f1:.3f} -> {f1_plain:.3f} (질문 되풀이 기준선 {echo_f1:.3f})")

    print("\n[생성 예시] (test 질문: " + ("학습에 쓰지 않은 구역" if a.split == "group" else "같은 구역의 본 적 없는 질문") + ")")
    for i in range(min(a.n_show, len(preds_plain))):
        it = ds["test"].items[i]
        print(f"\n  Q   : {it['question']}\n  정답: {it['ref']}\n  학습전: {base_preds[i][:150]}\n  학습후: {preds_plain[i][:150]}")

    # 6) 저장
    final_dir = os.path.join(a.out, "final")
    trainer.save_model(final_dir)
    tok.save_pretrained(final_dir)
    write_jsonl(os.path.join(a.out, "test_preds.jsonl"),
                [dict(question=it["question"], ref=it["ref"], base=b_, pred=p_) for it, b_, p_ in zip(ds["test"].items, base_preds, preds_plain)])
    json.dump(dict(model=a.model, lang=lang, split=a.split, n_train=len(ds["train"]), n_val=len(ds["val"]), n_test=len(ds["test"]),
                   base_test_loss=base_loss, final_test_loss=final_loss, base_f1=base_f1, echo_f1=echo_f1, final_f1=f1_plain,
                   final_f1_rep115=f1_rep, num_recall=num_plain[0], num_recall_n=num_plain[1],
                   eval_losses=[h["eval_loss"] for h in hist if "eval_loss" in h],
                   args=dict(epochs=a.epochs, lr=a.lr, bs=a.bs, grad_accum=a.grad_accum, seed=a.seed, no_repeat=a.no_repeat,
                             num_beams=a.num_beams, optim=a.optim, grad_ckpt=a.grad_ckpt, peak_gb=round(peak, 2)),
                   pretrain=pre_info),
              open(os.path.join(a.out, "metrics.json"), "w", encoding="utf-8"), ensure_ascii=False, indent=2)
    print(f"\n[저장] {final_dir}, {a.out}/metrics.json, {a.out}/test_preds.jsonl, {a.out}/splits/")
    print("사용 시 프롬프트 형식:", repr(FMT[lang][0].format(q="...")))


if __name__ == "__main__":
    main()
