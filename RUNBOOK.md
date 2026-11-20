# 단계별 실행표

실제로 이 순서로 진행했다. 각 단계 뒤 `check_step.py`가 PASS여야 다음으로 간다.

---

## 0. 환경

```bash
python -m venv .venv
.venv\Scripts\activate                      # Windows PowerShell
pip install -r requirements.txt
python check_step.py env
```

GPU용 torch는 CUDA 버전에 맞춰 따로 깔아야 한다 (`nvidia-smi`로 확인).

---

## 1. PDF → 텍스트 (차종별로 4번 반복)

tesla.com이 스크립트 접근을 **403으로 막는다.** 우회하지 말고 브라우저로 직접 받는다.

```bash
python 01_fetch_manual.py --mode local --pdf-path Owners_Manual_S.pdf --lang ko_kr --out-dir data/raw_S
python 01_fetch_manual.py --mode local --pdf-path Owners_Manual_3.pdf --lang ko_kr --out-dir data/raw_3
python 01_fetch_manual.py --mode local --pdf-path Owners_Manual_X.pdf --lang ko_kr --out-dir data/raw_X
python 01_fetch_manual.py --mode local --pdf-path Owners_Manual_Y.pdf --lang ko_kr --out-dir data/raw_Y
```

`manual.txt`를 직접 열어 문장이 이어지는지 본다.
표본이 어색하면 `--pages-per-group`, `--skip-first`를 조정해 재실행.

### 1-b. 4차종 합치기

```bash
python merge_sections.py
```

`id`와 `group`에 차종 접두어(`S_`, `3_`, `X_`, `Y_`)를 붙이고, 제목에 차종명을 넣는다.
차종 간 중복 문단 비율도 같이 측정해서 출력한다.

### 1-c. 쪼개진 단어 복원 ★ 중요

PDF 줄바꿈 때문에 `펫 모 드`, `제동 거 리`처럼 단어가 쪼개진다.
문서 자체를 사전으로 삼아(빈도 비교 + 한국어 어미·조사 규칙) 복원한다.

```bash
python fix_text.py --dry-run         # 무엇이 바뀔지 먼저 확인
python fix_text.py                   # 적용 — 원본은 sections_before_fix.jsonl로 백업
python check_step.py raw             # 쪼개진 단어 Count가 0이어야 통과
```

실제로 **15,596곳**이 복원됐다. `--min-freq`, `--ratio`로 판정 기준을 조절한다.

---

## 2. 조각내기 → Batch 요청 만들기

```bash
python 02_make_dataset_batch.py chunk   --max-chars 1200
python 02_make_dataset_batch.py prepare --lang ko --model gpt-4.1-mini --pairs-per-chunk 4
python check_step.py chunks
```

조각 1,527개가 나왔다. 조각 단위는 **문단**이고 **구역(group) 경계를 넘지 않는다.**

### 소량으로 먼저 시험할 때

```bash
python 02_make_dataset_batch.py prepare --lang ko --max-requests 20
python 02_make_dataset_batch.py submit --only 0
```

4단계까지 돌려보고 결과가 멀쩡하면 `--max-requests 2000`으로 prepare를 다시 한다.

---

## 3. 제출 → 완료 대기

```bash
export OPENAI_API_KEY=sk-...                  # Windows: $env:OPENAI_API_KEY="sk-..."
python 02_make_dataset_batch.py submit        # 여기서부터 과금
python 02_make_dataset_batch.py status        # completed가 될 때까지 가끔 실행
python check_step.py batch
```

> ⚠️ **`status`가 결과 파일을 다운로드한다.** `collect`는 이미 받아둔 파일만 읽는다.
> `status` 없이 `collect`를 돌리면 결과가 일부만 나온다.

Batch는 최대 24시간이지만 실제로는 수십 분 ~ 몇 시간이었다.

---

## 4. 수집 + 품질 필터

```bash
python 02_make_dataset_batch.py collect
python check_step.py qa
```

표본 10개를 원문과 직접 대조한다. 통과 수가 너무 적으면
출력의 제외 사유 집계를 보고 `--min-ground`, `--max-a`를 조정한다.

v1에서 **조건절 누락·금지문 반전·없는 수치 추가**가 발견돼
프롬프트에 규칙 4개를 추가하고 재생성했다 (v1 5,771쌍 → v2 4,929쌍).

---

## 5. 토크나이저·모델 로드 확인

```bash
python check_step.py tok --model skt/kogpt2-base-v2
```

한글 토큰/글자 비율이 1.2를 넘으면 경고가 뜬다.

---

## 6. 시험 학습 (1에폭, 약 30분)

```bash
python 03_train.py --model skt/kogpt2-base-v2 --data data/qa_all.jsonl \
       --out runs/smoke --epochs 1 --n-eval 10 --optim adafactor
```

출력에서 확인할 것: `[라벨 점검]`, `[학습 전]`/`[학습 후]` 손실, 길이 초과 제외 비율,
`[메모리] 학습 중 최대 할당`.

---

## 7. 본 학습 (2에폭, MX450에서 약 4시간 20분)

```bash
python 03_train.py --model skt/kogpt2-base-v2 --data data/qa_all.jsonl \
       --out runs/tesla_v2 --epochs 2 --bs 4 --grad-accum 4 \
       --lr 5e-5 --optim adafactor --no-repeat 3 --n-eval 50
python check_step.py result --out runs/tesla_v2
```

> ⚠️ **`--optim adafactor`가 없으면 2GB에서 OOM.** Adam 옵티마이저 상태가 약 1.0GB다.
> 그래도 모자라면 `--grad-ckpt`를 붙인다 (느려지는 대신 메모리를 아낀다).

> ⚠️ **학습이 유독 느리면 `nvidia-smi`로 다른 GPU 프로세스를 확인한다.**
> VS Code도 VRAM을 쓴다. 2GB에서 프로세스가 경쟁하면 몇 배로 느려진다.

### 학습이 중간에 끊겼을 때

체크포인트가 남아 있으면 재학습 없이 평가만 다시 할 수 있다.

```bash
python eval_ckpt.py --ckpt runs/tesla_v2/ckpt/checkpoint-492 --out runs/tesla_v2 \
       --base-loss 4.387 --base-f1 0.0007 --echo-f1 0.298 --eval-losses "2.323,2.140"
```

끊긴 로그에 찍혀 있던 값들을 인자로 넘기면 `metrics.json`과 `final/`을 만들어준다.

---

## 8. 평가

```bash
python chat_test.py --model runs/tesla_v2/final                    # 32문항 자동 채점
python chat_test.py --model runs/tesla_v2/final --chat             # 대화형
python chat_test.py --model skt/kogpt2-base-v2 --model runs/tesla_v2/final   # 학습 전/후 비교
```

결과: `runs/chat_test/chat_report.csv` (사람점수 열을 직접 채운다), `chat_report.json`

### 자동 지표만 믿지 말 것 ★

**수치 재현율은 "답변에 정답 숫자가 있는가"만 본다.
그 숫자가 질문에서 베낀 것인지 모델이 외운 것인지 구분하지 못한다.**

직접 확인하려면 `test_preds.jsonl`을 열어 정답 숫자가 질문에도 있는지 대조한다.

```python
# 정답 숫자 중 "질문에 없는데 답변에 있는" 것만이 진짜 외운 것이다
nums = set(re.findall(r"\d+(?:[.,]\d+)?", gold))
for n in nums:
    if n in pred:
        memorized = n not in question     # False면 베낀 것
```

실제로 세 모델 합쳐 정답 숫자 39개 중 **진짜 외운 것은 0개**였다.

---

## 비용 참고

| 항목 | 실제 |
|---|---|
| Batch 요청 | 1,527개 (gpt-4.1-mini, 50% 할인) |
| v1 + v2 재생성 | 합쳐서 2라운드 |
| 학습 | 전기값만 (로컬 GPU) |
