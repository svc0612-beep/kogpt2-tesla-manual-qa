"""실제 openai SDK 시그니처 확인 + submit/status를 가짜 클라이언트로 실행"""
import importlib.util, inspect, json, os, types, shutil
from openai import OpenAI
from openai.resources.batches import Batches
from openai.resources.files import Files
sig = inspect.signature(Batches.create).parameters
assert {"input_file_id", "endpoint", "completion_window", "metadata"} <= set(sig), sig.keys()
assert {"file", "purpose"} <= set(inspect.signature(Files.create).parameters)
assert hasattr(Files, "content") and hasattr(Batches, "retrieve")
print("SDK 시그니처 OK:", list(sig)[:6])

spec = importlib.util.spec_from_file_location("mk", "02_make_dataset_batch.py"); mk = importlib.util.module_from_spec(spec); spec.loader.exec_module(mk)
os.makedirs(mk.DIR, exist_ok=True)

class R:  # response with .read()
    def __init__(self, b): self.b = b
    def read(self): return self.b
class FakeClient:
    def __init__(s): s.n = 0; s.files = types.SimpleNamespace(create=s.fc, content=s.fcont); s.batches = types.SimpleNamespace(create=s.bc, retrieve=s.br)
    def fc(s, file, purpose): assert purpose == "batch" and file.read(10); return types.SimpleNamespace(id=f"file-{s.n}")
    def bc(s, input_file_id, endpoint, completion_window, metadata):
        assert endpoint == "/v1/chat/completions" and completion_window == "24h"; s.n += 1
        return types.SimpleNamespace(id=f"batch_{s.n}", status="validating")
    def br(s, bid):
        rc = types.SimpleNamespace(completed=10, total=10, failed=0)
        return types.SimpleNamespace(id=bid, status="completed", output_file_id="out-1", error_file_id=None, request_counts=rc, errors=None)
    def fcont(s, fid): return R(b'{"custom_id":"x"}\n')
mk.client = lambda: FakeClient()
st = json.load(open(mk.STATE)); [f.pop("batch_id", None) for f in st["files"]]; json.dump(st, open(mk.STATE, "w"))
for fn in os.listdir(mk.DIR):
    if fn.startswith("output_") and fn != "output_000.jsonl": os.remove(os.path.join(mk.DIR, fn))
backup = open(os.path.join(mk.DIR, "output_000.jsonl"), "rb").read()
os.remove(os.path.join(mk.DIR, "output_000.jsonl"))
mk.cmd_submit(types.SimpleNamespace(only=None))
st = json.load(open(mk.STATE)); assert all(f.get("batch_id") for f in st["files"]); print("제출 상태 저장 OK")
mk.cmd_submit(types.SimpleNamespace(only=None))      # 재실행해도 중복 제출 안 함
mk.cmd_status(types.SimpleNamespace())
assert os.path.exists(os.path.join(mk.DIR, "output_000.jsonl")); print("결과 다운로드 OK")
open(os.path.join(mk.DIR, "output_000.jsonl"), "wb").write(backup)   # 모의 결과 복원
