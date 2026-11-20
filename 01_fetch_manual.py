"""1단계: Tesla 사용자 매뉴얼 수집 -> 텍스트 추출

두 가지 방법을 지원한다.
  pdf  : Owners_Manual.pdf 한 개를 내려받아 pypdf로 텍스트를 추출한다. (권장: 요청 1회, 가장 단순)
  html : 웹 매뉴얼의 목차(index.html)에서 GUID-*.html 페이지를 따라가며 본문을 추출한다. (제목 단위 구조가 살아 있음)
  local: 이미 내려받은 PDF 파일(--pdf-path)만 변환한다. (사이트 접속이 막힌 환경)

사용 예
  python 01_fetch_manual.py --mode pdf  --model model3 --lang ko_kr
  python 01_fetch_manual.py --mode html --model model3 --lang en_us --max-pages 400
  python 01_fetch_manual.py --mode local --pdf-path Owners_Manual.pdf

출력 (data/raw/)
  sections.jsonl : {"id","group","title","text","source"}  한 줄이 한 구역(HTML=페이지, PDF=쪽 묶음)
  manual.txt     : 사람이 확인하기 위한 전체 텍스트

주의
  - 매뉴얼은 Tesla의 저작물이다. PDF 앞면에 "사전 서면 허락 없이 수정, 재생산, 복사할 수 없다"는 고지가 있다.
    개인 학습과 교육용 실험 범위에서 사용하고, 데이터나 모델 배포 여부는 별도로 확인해야 한다.
  - 사이트 약관과 robots.txt를 존중한다. html 모드는 robots.txt를 읽고, 요청 사이에 대기 시간을 둔다.
  - URL 규칙은 검색으로 확인한 형태(/ownersmanual/<model>/<lang>/)이다. 차종 폴더명(modely, models 등)은 확인이 필요하다.
"""
import argparse
import collections
import hashlib
import json
import os
import re
import sys
import time
import urllib.robotparser
from urllib.parse import urljoin, urlparse

try:                                    # 사내망/보안 프로그램의 자체 서명 인증서를 OS 인증서 저장소에서 읽는다 (pip install truststore)
    import truststore
    truststore.inject_into_ssl()
except ImportError:
    pass

BASE = "https://www.tesla.com/ownersmanual"
UA = "manual-ft-study/1.0 (personal research; contact: set --contact)"


# ------------------------------------------------------------------ 공통 정리
DOT_LEADER = re.compile(r"\.{4,}\s*\d+\s*$")           # 목차 줄: "충전 지침.......... 167"
ONLY_NUM = re.compile(r"^\s*\d{1,4}\s*$")               # 쪽 번호만 있는 줄
SENT_END = re.compile(r"[.!?。…:;)\]”’\"'다요음임함됨]$")  # 문장이 끝난 것으로 보는 글자


def clean_page_lines(pages):
    """쪽 위/아래 2줄에 반복되는 머리글/바닥글을 빈도로 찾아 제거하고, 목차 줄과 쪽 번호를 지운다.
    본문 중간의 반복 문장은 건드리지 않는다. (쪽이 20장 미만이면 머리글 제거를 하지 않는다)"""
    def key(ln):
        return re.sub(r"\d+", "#", ln)          # "167 | Model 3 매뉴얼"처럼 쪽 번호가 섞인 바닥글도 같은 줄로 센다
    def edges(p):
        ls = [l.strip() for l in p.splitlines() if l.strip()]
        return {key(l) for l in ls[:2] + ls[-2:]}
    repeated = set()
    if len(pages) >= 20:
        counter = collections.Counter()
        for p in pages:
            for ln in edges(p):
                counter[ln] += 1
        limit = max(5, int(len(pages) * 0.3))
        repeated = {ln for ln, c in counter.items() if c >= limit and len(ln) < 80}
    out = []
    for p in pages:
        keep = []
        for ln in p.splitlines():
            s = ln.strip()
            if not s or key(s) in repeated or ONLY_NUM.match(s) or DOT_LEADER.search(s):
                keep.append("")          # 빈 줄 = 문단 경계로 남긴다
                continue
            keep.append(s)
        out.append("\n".join(keep))
    return out


def join_wrapped(text):
    """PDF의 줄바꿈(문장 중간에서 끊김)을 이어 붙여 문단 단위로 만든다."""
    paras, cur = [], []
    for ln in text.splitlines():
        s = ln.strip()
        if not s:
            if cur:
                paras.append(" ".join(cur)); cur = []
            continue
        if cur and cur[-1].endswith("-") and not cur[-1].endswith(" -"):
            cur[-1] = cur[-1][:-1] + s          # 영문 하이픈 줄바꿈
            continue
        cur.append(s)
        if SENT_END.search(s) and len(s) < 40:   # 짧은 줄이 문장 끝이면 제목/항목이므로 끊는다
            paras.append(" ".join(cur)); cur = []
    if cur:
        paras.append(" ".join(cur))
    paras = [re.sub(r"[ \t ]+", " ", p).strip() for p in paras]
    return [p for p in paras if len(p) > 1]


def write_outputs(sections, outdir):
    os.makedirs(outdir, exist_ok=True)
    with open(os.path.join(outdir, "sections.jsonl"), "w", encoding="utf-8") as f:
        for s in sections:
            f.write(json.dumps(s, ensure_ascii=False) + "\n")
    with open(os.path.join(outdir, "manual.txt"), "w", encoding="utf-8") as f:
        for s in sections:
            f.write(f"\n===== [{s['id']}] {s['title']} =====\n{s['text']}\n")
    total = sum(len(s["text"]) for s in sections)
    print(f"[저장] 구역 {len(sections)}개, 글자 {total:,}자 -> {outdir}/sections.jsonl, manual.txt")
    if sections:
        print("[미리보기]", sections[len(sections) // 2]["text"][:200].replace("\n", " "))


# ------------------------------------------------------------------ PDF
def download(url, dest, contact):
    import requests
    os.makedirs(os.path.dirname(dest), exist_ok=True)
    if os.path.exists(dest) and os.path.getsize(dest) > 10_000:
        print(f"[재사용] {dest}")
        return dest
    print(f"[다운로드] {url}")
    r = requests.get(url, headers={"User-Agent": UA.replace("set --contact", contact)}, timeout=120, stream=True)
    r.raise_for_status()
    with open(dest, "wb") as f:
        for chunk in r.iter_content(1 << 20):
            f.write(chunk)
    print(f"  {os.path.getsize(dest)/1024**2:.1f} MB")
    return dest


def pdf_to_sections(path, pages_per_group=5, skip_first=3):
    from pypdf import PdfReader
    reader = PdfReader(path)
    raw = [(pg.extract_text() or "") for pg in reader.pages]
    print(f"[PDF] {len(raw)}쪽")
    pages = clean_page_lines(raw)
    sections = []
    for start in range(skip_first, len(pages), pages_per_group):
        block = "\n\n".join(pages[start:start + pages_per_group])
        paras = join_wrapped(block)
        text = "\n\n".join(paras)
        if len(text) < 200:
            continue
        sections.append(dict(id=f"pdf-p{start+1:04d}", group=f"pdf-p{start+1:04d}",
                             title=f"{start+1}~{min(start+pages_per_group, len(pages))}쪽",
                             text=text, source=os.path.basename(path)))
    return sections


# ------------------------------------------------------------------ HTML
def robots_ok(base_url, url, contact):
    rp = urllib.robotparser.RobotFileParser()
    root = f"{urlparse(base_url).scheme}://{urlparse(base_url).netloc}"
    rp.set_url(root + "/robots.txt")
    try:
        rp.read()
    except Exception as e:
        print(f"[robots.txt 읽기 실패: {e}] -> 보수적으로 진행하되 대기 시간을 지킨다")
        return True
    return rp.can_fetch(UA.replace("set --contact", contact), url)


def html_to_text(html):
    from bs4 import BeautifulSoup
    soup = BeautifulSoup(html, "html.parser")
    for tag in soup(["script", "style", "nav", "footer", "header", "aside", "noscript", "form", "button"]):
        tag.decompose()
    main = soup.find("main") or soup.find("article") or soup.find(id=re.compile("content|main", re.I)) or soup.body or soup
    title = (main.find(["h1", "h2"]) or soup.title)
    title = title.get_text(" ", strip=True) if title else ""
    lines = []
    for el in main.find_all(["h1", "h2", "h3", "h4", "p", "li", "td", "th", "dt", "dd"]):
        t = el.get_text(" ", strip=True)
        if not t:
            continue
        if el.name in ("h1", "h2", "h3", "h4"):
            lines.append("")
            lines.append(t)
        else:
            lines.append(("- " + t) if el.name == "li" else t)
    text = re.sub(r"\n{3,}", "\n\n", "\n".join(lines)).strip()
    return title, text


def crawl_html(model, lang, outdir, delay, max_pages, contact, base=BASE):
    import requests
    from bs4 import BeautifulSoup
    root = f"{base}/{model}/{lang}/"
    start = root + "index.html"
    if not robots_ok(root, start, contact):
        sys.exit("robots.txt가 이 경로의 수집을 허용하지 않는다. pdf 모드를 사용한다.")
    cache = os.path.join(outdir, "html_cache")
    os.makedirs(cache, exist_ok=True)
    sess = requests.Session()
    sess.headers["User-Agent"] = UA.replace("set --contact", contact)

    def get(url):
        fn = os.path.join(cache, hashlib.md5(url.encode()).hexdigest() + ".html")
        if os.path.exists(fn):
            return open(fn, encoding="utf-8").read()
        r = sess.get(url, timeout=60)
        r.raise_for_status()
        r.encoding = r.encoding or "utf-8"
        open(fn, "w", encoding="utf-8").write(r.text)
        time.sleep(delay)
        return r.text

    index_html = get(start)
    soup = BeautifulSoup(index_html, "html.parser")
    links = []
    for a in soup.find_all("a", href=True):
        href = a["href"].split("#")[0]
        if re.search(r"GUID-[0-9A-Fa-f-]+\.html$", href):
            u = urljoin(start, href)
            if u.startswith(root) and u not in links:
                links.append(u)
    print(f"[목차] GUID 페이지 {len(links)}개 발견 (최대 {max_pages}개 수집)")
    sections = []
    for i, u in enumerate(links[:max_pages], 1):
        try:
            title, text = html_to_text(get(u))
        except Exception as e:
            print(f"  실패 {u}: {e}")
            continue
        if len(text) < 150:
            continue
        gid = u.rsplit("GUID-", 1)[1][:-5][:8]
        sections.append(dict(id=f"html-{gid}", group=f"html-{gid}", title=title, text=text, source=u))
        if i % 25 == 0:
            print(f"  {i}/{min(len(links), max_pages)}")
    return sections


# ------------------------------------------------------------------ main
def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--mode", choices=["pdf", "html", "local"], default="pdf")
    ap.add_argument("--model", default="model3", help="차종 폴더명 (예: model3, modely, models, modelx) - 확인이 필요하다")
    ap.add_argument("--lang", default="ko_kr", help="예: ko_kr, en_us (distilgpt2는 영문 권장)")
    ap.add_argument("--pdf-path", default=None, help="local 모드: 이미 받은 PDF 경로")
    ap.add_argument("--out", default="data/raw")
    ap.add_argument("--delay", type=float, default=1.5, help="html 모드 요청 간격(초)")
    ap.add_argument("--max-pages", type=int, default=500)
    ap.add_argument("--pages-per-group", type=int, default=5, help="pdf 모드: 분할 단위 쪽 수")
    ap.add_argument("--skip-first", type=int, default=3, help="pdf 모드: 표지와 목차로 건너뛸 앞쪽 수")
    ap.add_argument("--contact", default="your-email", help="User-Agent에 넣을 연락처")
    ap.add_argument("--base", default=BASE, help="테스트용: 매뉴얼 루트 URL 교체")
    a = ap.parse_args()

    if a.mode == "html":
        sections = crawl_html(a.model, a.lang, a.out, a.delay, a.max_pages, a.contact, a.base)
    else:
        if a.mode == "local":
            if not a.pdf_path:
                sys.exit("--pdf-path가 필요하다")
            pdf = a.pdf_path
        else:
            pdf = download(f"{a.base}/{a.model}/{a.lang}/Owners_Manual.pdf",
                           os.path.join(a.out, f"{a.model}_{a.lang}_Owners_Manual.pdf"), a.contact)
        sections = pdf_to_sections(pdf, a.pages_per_group, a.skip_first)
    if not sections:
        sys.exit("추출된 텍스트가 없다. 스캔 PDF이거나 페이지 구조가 바뀌었을 수 있다. manual.txt와 원본을 확인한다.")
    write_outputs(sections, a.out)


if __name__ == "__main__":
    main()
