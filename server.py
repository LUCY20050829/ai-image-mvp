from __future__ import annotations

import base64
import concurrent.futures
import io
import zipfile
import json
import os
import queue
import re
import shutil
import subprocess
import sys
import tempfile
import threading
import time
import uuid
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, quote, urlparse

APP_ROOT = Path(__file__).resolve().parent
ROOT = APP_ROOT
PROJECT = APP_ROOT
STATIC = APP_ROOT / "static"
SCRIPT_ROOT = APP_ROOT / "workflow"
PROMPT_ASSETS = APP_ROOT / "prompt_assets"
APP_ASSETS = APP_ROOT / "assets"
FACE_LIBRARY = APP_ASSETS / "faces"
REFERENCE_ASSETS = APP_ASSETS / "reference_assets"
REFERENCE_INDEX_PATH = REFERENCE_ASSETS / "reference_index.json"
JOBS_DIR = APP_ROOT / "generated_outputs"
TASK_OUTPUTS_DIR = JOBS_DIR / "任务记录"
PACKAGE_OUTPUTS_DIR = JOBS_DIR / "打包下载"
SAVED_OUTPUTS_DIR = JOBS_DIR / "手动保存"
TEMP_OUTPUTS_DIR = Path(tempfile.gettempdir()) / "ai-image-mvp-runtime"
JOB_SUMMARY_LOG = JOBS_DIR / "job_summary.jsonl"
RUNTIME_EVENTS_LOG = JOBS_DIR / "runtime_events.jsonl"
TEST_RECORD_SYNC_SCRIPT = APP_ROOT / "测试记录" / "sync_test_records.py"
for path in [JOBS_DIR, TASK_OUTPUTS_DIR, PACKAGE_OUTPUTS_DIR, TEMP_OUTPUTS_DIR]:
    path.mkdir(parents=True, exist_ok=True)
sys.path.insert(0, str(SCRIPT_ROOT))
import run_layered_batch_images as base  # noqa: E402

CONTRACT_PATH = Path.home() / ".codex" / "skills" / "ai-book-layered-image-pipeline" / "references" / "prompt-contracts.md"
CONTRACT_TEXT = CONTRACT_PATH.read_text(encoding="utf-8") if CONTRACT_PATH.exists() else ""
TOTAL_PROMPT_PATH = PROMPT_ASSETS / "\u603b\u63d0\u793a\u8bcd_\u4fee\u6539\u7248.txt"

GRADE_PROMPT_FILES = {
    "lower_primary": PROMPT_ASSETS / "grade_lower_primary.txt",
    "middle_upper_primary": PROMPT_ASSETS / "grade_middle_upper_primary.txt",
    "junior_high": PROMPT_ASSETS / "grade_junior_high.txt",
}


def load_local_env() -> None:
    env_path = APP_ROOT / ".env"
    if not env_path.exists():
        return
    for line in env_path.read_text(encoding="utf-8", errors="ignore").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        os.environ.setdefault(key.strip(), value.strip().strip('"').strip("'"))


load_local_env()


CODEX_MODEL = os.getenv("WEB_AGENT_MODEL", "gpt-5.3-codex")
DEEPSEEK_MODEL = "deepseek-v4-flash"
GPT_MODEL = "gpt-5"
TOTAL_PROMPT_MODEL = os.getenv("TOTAL_PROMPT_MODEL", DEEPSEEK_MODEL)
DEEPSEEK_PROMPT_WORKERS = int(os.getenv("DEEPSEEK_PROMPT_WORKERS", "40"))
GPT_PROMPT_WORKERS = int(os.getenv("GPT_PROMPT_WORKERS", "25"))
IMAGE_WORKERS = int(os.getenv("IMAGE_WORKERS", "40"))
IMAGE_BATCH_SIZE = max(1, int(os.getenv("IMAGE_BATCH_SIZE", "8")))
IMAGE_BATCH_INTERVAL_SECONDS = max(0, int(os.getenv("IMAGE_BATCH_INTERVAL_SECONDS", "10")))
IMAGE_STALL_HINT_SECONDS = int(os.getenv("IMAGE_STALL_HINT_SECONDS", "90"))
IMAGE_RETRY_ROUNDS = max(0, int(os.getenv("IMAGE_RETRY_ROUNDS", "2")))
IMAGE_RETRY_DELAY_SECONDS = max(0, int(os.getenv("IMAGE_RETRY_DELAY_SECONDS", "30")))
API_RATE_PER_MINUTE = int(os.getenv("API_RATE_PER_MINUTE", "100"))
BATCH_POETS = [x.strip() for x in os.getenv("BATCH_POETS", "\u674e\u767d,\u675c\u752b").split(",") if x.strip()]
BATCH_COUNT = int(os.getenv("BATCH_COUNT", "3"))
BATCH_MODE = os.getenv("BATCH_MODE", "complete")


class RateLimiter:
    def __init__(self, limit: int, window_seconds: float = 60.0):
        self.limit = max(1, limit)
        self.window_seconds = window_seconds
        self.calls: list[float] = []
        self.lock = threading.Lock()

    def wait(self) -> None:
        while True:
            with self.lock:
                now = time.monotonic()
                self.calls = [t for t in self.calls if now - t < self.window_seconds]
                if len(self.calls) < self.limit:
                    self.calls.append(now)
                    return
                wait_for = self.window_seconds - (now - self.calls[0])
            time.sleep(max(0.05, wait_for))


_rate_limiters = {name: RateLimiter(API_RATE_PER_MINUTE) for name in ["codex", "deepseek", "gpt", "image2"]}
_base_post_json = base.post_json
_base_post_multipart = base.post_multipart


def _rate_key(path: str, payload: dict | None = None) -> str:
    model = str((payload or {}).get("model") or "").lower()
    if "/images/" in path or "image" in model:
        return "image2"
    if "codex" in model:
        return "codex"
    if "deepseek" in model:
        return "deepseek"
    if "gpt" in model:
        return "gpt"
    return "deepseek"


def post_json_rate_limited(path: str, payload: dict, headers: dict, timeout: int = 900) -> dict:
    _rate_limiters[_rate_key(path, payload)].wait()
    return _base_post_json(path, payload, headers, timeout=timeout)


def post_multipart_rate_limited(path: str, fields: dict, files: list[tuple[str, Path]]) -> dict:
    _rate_limiters[_rate_key(path, fields)].wait()
    return _base_post_multipart(path, fields, files)




def normalized_credentials() -> tuple[str, str]:
    app_id = os.getenv("TAL_MLOPS_APP_ID") or ""
    app_key = os.getenv("TAL_MLOPS_APP_KEY") or ""
    app_id = app_id.strip().strip('"').strip("'")
    app_key = app_key.strip().strip('"').strip("'")
    if not app_id or not app_key:
        raise RuntimeError("missing TAL_MLOPS_APP_ID/TAL_MLOPS_APP_KEY")
    prefix = app_id + ":"
    if app_key.startswith(prefix):
        app_key = app_key[len(prefix):]
    return app_id, app_key


base.credentials = normalized_credentials

base.post_json = post_json_rate_limited
base.post_multipart = post_multipart_rate_limited

AGENT_PARSE_SYSTEM = (
    "\u4f60\u662f\u201c\u8bd7\u5883\u6210\u56fe\u201dAI\u4ea4\u4e92\u56fe\u4e66\u5de5\u4f5c\u6d41\u7684\u4e3b agent\uff0c\u53ea\u8d1f\u8d23\u628a\u7528\u6237\u7684\u81ea\u7136\u8bed\u8a00\u8bf7\u6c42\u89e3\u6790\u6210\u53ef\u6267\u884c\u4efb\u52a1\u3002"
    "\u4f60\u4e0d\u5199\u63d0\u793a\u8bcd\uff0c\u4e0d\u751f\u56fe\uff0c\u4e0d\u89e3\u91ca\uff0c\u53ea\u8f93\u51fa JSON \u5bf9\u8c61\u3002"
    "JSON \u683c\u5f0f\u5fc5\u987b\u4e3a\uff1a{\"poet\":\"\",\"title\":\"\",\"count\":4,\"mode\":\"complete|background|person\",\"all_poems\":false}\u3002"
    "\u89c4\u5219\uff1a1. \u5982\u679c\u7528\u6237\u8bf4\u67d0\u8bd7\u4eba\u7684\u6240\u6709/\u5168\u90e8/\u5168\u5957\u8bd7\u6b4c\uff0c\u5219 all_poems=true\uff0ctitle \u7559\u7a7a\uff0cpoet \u5199\u8bd7\u4eba\u540d\u3002"
    "2. \u5982\u679c\u7528\u6237\u5199\u4e86\u300a\u8bd7\u540d\u300b\uff0c\u5219 all_poems=false\uff0ctitle \u5199\u8bd7\u540d\uff0cpoet \u80fd\u8bc6\u522b\u5c31\u5199\uff0c\u4e0d\u80fd\u8bc6\u522b\u7559\u7a7a\u3002"
    "3. \u7528\u6237\u660e\u786e\u8bf4\u80cc\u666f\u56fe/\u53ea\u8981\u80cc\u666f\u56fe\uff0cmode=background\uff1b\u660e\u786e\u8bf4\u4eba\u7269\u56fe/\u53ea\u8981\u4eba\u7269\u56fe\uff0cmode=person\uff1b\u8bf4\u5b8c\u6574\u9875\u9762/\u5408\u6210\u56fe/\u5b8c\u6574\u56fe\u6216\u672a\u6307\u5b9a\u65f6\uff0cmode=complete\u3002"
    "4. count \u662f\u6bcf\u9996\u8bd7\u9700\u8981\u751f\u6210\u7684\u5f20\u6570\uff1b\u7528\u6237\u6ca1\u8bf4\u5c31\u8fd4\u56de 4\uff1b\u4e0d\u8981\u8d85\u8fc7 20\u3002"
    "5. \u793a\u4f8b\uff1a\u201c\u751f\u6210\u674e\u767d\u6240\u6709\u8bd7\u6b4c\u7684\u80cc\u666f\u56fe\u201d -> {\"poet\":\"\u674e\u767d\",\"title\":\"\",\"count\":4,\"mode\":\"background\",\"all_poems\":true}\u3002"
    "6. \u793a\u4f8b\uff1a\u201c\u751f\u6210\u674e\u767d\u300a\u671b\u5929\u95e8\u5c71\u300b\u7684\u56fe\u7247\uff0c\u751f\u6210 10 \u5f20\u201d -> {\"poet\":\"\u674e\u767d\",\"title\":\"\u671b\u5929\u95e8\u5c71\",\"count\":10,\"mode\":\"complete\",\"all_poems\":false}\u3002"
)

jobs: dict[str, dict] = {}
events: dict[str, "queue.Queue[dict]"] = {}

def write_runtime_event(job_id: str, event: dict) -> None:
    job = jobs.get(job_id, {})
    record = {
        "job_id": job_id,
        "date": time.strftime("%Y-%m-%d"),
        "time": event.get("time") or time.strftime("%H:%M:%S"),
        "kind": event.get("kind"),
        "message": event.get("message"),
        "task_config": build_job_summary_config(job) if job else {},
    }
    if isinstance(event.get("image"), dict):
        record["image"] = {
            "name": event["image"].get("name"),
            "temporary": event["image"].get("temporary"),
        }
    for key in ["ok", "partial", "target_count", "success_count", "failed_count", "duration_seconds", "title", "out_dir"]:
        if key in event:
            record[key] = event.get(key)
    try:
        RUNTIME_EVENTS_LOG.parent.mkdir(parents=True, exist_ok=True)
        with RUNTIME_EVENTS_LOG.open("a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
    except Exception:
        pass


def emit(job_id: str, kind: str, message: str, **data) -> None:
    event = {"kind": kind, "message": message, "time": time.strftime("%H:%M:%S"), **data}
    jobs[job_id]["events"].append(event)
    events[job_id].put(event)
    write_runtime_event(job_id, event)





def build_job_summary_config(job: dict, data: dict | None = None) -> dict:
    request = job.get("request") if isinstance(job.get("request"), dict) else {}
    options = web_options_from_request(request) if request else {}
    poems = selected_poems_from_request(request)
    count = int(options.get("image_count") or 0) if options else 0
    return {
        "grade_band": options.get("grade_band") or "",
        "grade_label": options.get("grade_label") or "",
        "has_person": options.get("has_person") if options else None,
        "has_person_label": "\u6709\u4eba\u7269" if options.get("has_person") else "\u65e0\u4eba\u7269" if options else "",
        "has_poem": options.get("has_poem") if options else None,
        "has_poem_label": "\u6709\u8bd7\u8bcd" if options.get("has_poem") else "\u65e0\u8bd7\u8bcd" if options else "",
        "aspect_ratio": options.get("aspect_ratio") or "",
        "poem_count": len(poems),
        "images_per_poem": count,
        "selected_poems": [{"author": p.get("author", ""), "title": p.get("title", "")} for p in poems],
        "save_outputs": bool(job.get("save_outputs", False)),
    }

def write_job_summary(job_id: str, status: str, **data) -> None:
    job = jobs.get(job_id, {})
    created = float(job.get("created") or time.time())
    record = {
        "job_id": job_id,
        "status": status,
        "started_at": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(created)),
        "ended_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "duration_seconds": round(time.time() - created, 1),
        "target_count": int(job.get("expected_images") or 0),
        "success_count": len(data.get("images") or []),
        "task_config": build_job_summary_config(job, data),
        "image_metrics": summarize_image_metrics(job_id),
        **{k: v for k, v in data.items() if k != "images"},
    }
    try:
        JOB_SUMMARY_LOG.parent.mkdir(parents=True, exist_ok=True)
        with JOB_SUMMARY_LOG.open("a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
        sync_test_records_async()
    except Exception:
        pass



def sync_test_records_async() -> None:
    """Refresh the teacher-facing test workbook after a job summary is written.

    This is best-effort: an open Excel file or a script error must not fail image generation.
    """
    if not TEST_RECORD_SYNC_SCRIPT.exists():
        return

    def _run() -> None:
        try:
            subprocess.run(
                [sys.executable, str(TEST_RECORD_SYNC_SCRIPT)],
                cwd=str(APP_ROOT),
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                timeout=30,
                check=False,
            )
        except Exception:
            pass

    threading.Thread(target=_run, daemon=True).start()

def is_cancelled(job_id: str) -> bool:
    return bool(jobs.get(job_id, {}).get("cancelled"))


def ensure_not_cancelled(job_id: str) -> None:
    if is_cancelled(job_id):
        raise InterruptedError("任务已停止")


def norm(text: str) -> str:
    return re.sub(r"[《》\s・·/（）()【】\[\]「」『』,，.。:：;；!！?？]", "", str(text or ""))


def compact(text: str) -> str:
    return base.compact(text)


def safe_name(text: str) -> str:
    text = re.sub(r'[<>:"/\\|?*\x00-\x1f]', "_", str(text))
    text = re.sub(r"\s+", "", text)
    return text[:80] or "untitled"



def short_image_filename(name: str, sequence: int | None = None) -> str:
    raw = Path(str(name or "image.png")).name
    suffix = Path(raw).suffix or ".png"
    stem = Path(raw).stem
    match = re.match(r"^(\d+)_([^_]+)_(.+?)_web_v(\d+)(?:_(?:complete|final|composite|background|person))?$", stem)
    if match:
        index, _author, title, variant = match.groups()
        prefix = f"{int(sequence):03d}" if sequence else index
        return f"{prefix}_{safe_name(title)[:24]}_v{variant}{suffix}"
    if sequence:
        return f"{int(sequence):03d}_{safe_name(stem)[:24]}{suffix}"
    return f"{safe_name(stem)[:32]}{suffix}"

def parse_count(text: str) -> int:
    m = re.search(r"(\d+)\s*张", text)
    if m:
        return max(1, min(int(m.group(1)), 20))
    zh = {"一": 1, "二": 2, "两": 2, "三": 3, "四": 4, "五": 5, "六": 6, "七": 7, "八": 8, "九": 9, "十": 10}
    m = re.search(r"([一二两三四五六七八九十])\s*张", text)
    return zh.get(m.group(1), 4) if m else 4


def fallback_parse(text: str) -> dict:
    poet = ""
    title = ""
    all_poems = bool(re.search(r"(\u6240\u6709|\u5168\u90e8|\u5168\u5957|\u6574\u6279|\u8fd9\u4e2a\u8bd7\u4eba)", text))
    m = re.search("([\u4e00-\u9fa5]{1,4})\\s*\u300a([^\u300b]+)\u300b", text)
    if m:
        poet, title = m.group(1), m.group(2)
        for prefix in ["\u53ea\u751f\u6210", "\u8bf7\u751f\u6210", "\u6211\u8981", "\u8981", "\u751f\u6210"]:
            if poet.startswith(prefix):
                poet = poet[len(prefix):]
        if poet.startswith("\u6210") and len(poet) > 2:
            poet = poet[1:]
    else:
        m = re.search("\u300a([^\u300b]+)\u300b", text)
        if m:
            title = m.group(1)

    if all_poems and not poet:
        known_poets = ["\u9676\u6e0a\u660e", "\u674e\u767d", "\u675c\u752b", "\u738b\u7ef4", "\u5b5f\u6d69\u7136", "\u767d\u5c45\u6613", "\u674e\u5546\u9690", "\u675c\u7267", "\u674e\u8d3a", "\u738b\u660c\u9f84", "\u5218\u79b9\u9521", "\u97e9\u6108", "\u8d3a\u77e5\u7ae0", "\u865e\u4e16\u5357"]
        for name in known_poets:
            if name in text:
                poet = name
                break

    mode = "complete"
    asks_background = "\u80cc\u666f\u56fe" in text or "\u80cc\u666f" in text
    asks_person = "\u4eba\u7269\u56fe" in text or "\u4eba\u7269" in text
    asks_composite = "\u5408\u6210\u56fe" in text or "\u5b8c\u6574\u9875\u9762" in text or "\u5b8c\u6574\u56fe" in text
    background_only = asks_background and not asks_person and not asks_composite
    person_only = asks_person and not asks_background and not asks_composite
    if background_only:
        mode = "background"
    elif person_only:
        mode = "person"
    return {"poet": poet, "title": title, "count": parse_count(text), "mode": mode, "all_poems": all_poems}


def parse_with_agent(text: str) -> tuple[dict, str]:
    fallback = fallback_parse(text)
    try:
        app_id, app_key = base.credentials()
        payload = {
            "model": CODEX_MODEL,
            "messages": [
                {"role": "system", "content": AGENT_PARSE_SYSTEM},
                {"role": "user", "content": text},
            ],
            "stream": False,
        }
        if CODEX_MODEL.startswith("gpt-5"):
            payload["max_completion_tokens"] = 512
        else:
            payload["max_tokens"] = 512
        resp = base.post_json(
            "/openai-compatible/v1/chat/completions",
            payload,
            {"Authorization": f"Bearer {app_id}:{app_key}"},
            timeout=60,
        )
        content = resp["choices"][0]["message"]["content"]
        parsed = json.loads(re.search(r"\{.*\}", content, re.S).group(0))
        parsed["count"] = max(1, min(int(parsed.get("count") or fallback["count"]), 20))
        parsed["mode"] = parsed.get("mode") or fallback["mode"]
        parsed["all_poems"] = bool(parsed.get("all_poems") or fallback.get("all_poems"))
        return {**fallback, **{k: v for k, v in parsed.items() if v or k == "all_poems"}}, f"\u5df2\u7531 {CODEX_MODEL} \u5b8c\u6210\u4efb\u52a1\u89e3\u6790\u3002"
    except Exception as exc:
        return fallback, f"{CODEX_MODEL} \u89e3\u6790\u5931\u8d25\uff0c\u5df2\u4f7f\u7528\u672c\u5730\u89c4\u5219\u515c\u5e95\uff1a{exc}"


def face_search_dirs(grade_band: str = "") -> list[Path]:
    grade_band = normalize_grade_band(grade_band) if grade_band else ""
    mapping = {
        "lower_primary": [FACE_LIBRARY / "lower_primary"],
        "middle_upper_primary": [FACE_LIBRARY / "middle_upper_primary"],
        "junior_high": [FACE_LIBRARY / "junior_high", FACE_LIBRARY / "middle_upper_primary"],
    }
    dirs = mapping.get(grade_band, []) + [FACE_LIBRARY]
    result = []
    for path in dirs:
        if path.exists() and path not in result:
            result.append(path)
    return result


def iter_face_files(folder: Path):
    for suffix in ["*.png", "*.jpg", "*.jpeg", "*.webp"]:
        yield from folder.glob(suffix)


def find_face_fast(poet: str, grade_band: str = "") -> Path | None:
    if not FACE_LIBRARY.exists():
        return None
    poet = compact(poet)
    aliases = {"李白": "libai", "杜甫": "dufu", "王维": "wangwei", "白居易": "baijuyi"}
    for folder in face_search_dirs(grade_band):
        for path in iter_face_files(folder):
            if poet and poet in path.name:
                return path
        alias = aliases.get(poet)
        if alias:
            for path in iter_face_files(folder):
                if alias in path.name.lower():
                    return path
    return None

def infer_grade_level(text: str) -> str:
    if "小学" in text:
        return "小学"
    if "高中" in text:
        return "高中"
    if "初中" in text:
        return "初中"
    return "初中"


def infer_scene_type(text: str) -> str:
    if any(keyword in text for keyword in ["怀古", "想象", "历史", "典故"]):
        return "想象/历史"
    if any(keyword in text for keyword in ["写景", "山水", "田园", "边塞", "夜景"]):
        return "写景"
    return "叙事/抒情"


def infer_person_should_appear(text: str, mode: str) -> bool:
    if mode == "background":
        return False
    if any(keyword in text for keyword in ["不要人物", "不出现人物", "无人", "纯背景"]):
        return False
    return True


def infer_aspect_ratio(text: str) -> str:
    for ratio in ["16:9", "9:16", "4:3", "3:4", "1:1"]:
        if ratio in text:
            return ratio
    return "16:9"




def to_bool_value(value, default: bool = False) -> bool:
    if isinstance(value, bool):
        return value
    if value is None:
        return default
    return str(value).strip().lower() in {"1", "true", "yes", "y", "on", "?", "?", "??"}


def normalize_grade_band(value: str) -> str:
    text = compact(value or "")
    if text in {"lower_primary", "小低", "小学低年级", "12年级"}:
        return "lower_primary"
    if text in {'middle_upper_primary', 'middle_primary', 'upper_primary', '小中小高', '小中', '小高', '小学中高年级', '34年级', '56年级'}:
        return "middle_upper_primary"
    if text in {'junior_high', '初中', '789年级', '初中阶段'}:
        return "junior_high"
    return "junior_high"


GRADE_LABELS = {
    "lower_primary": "小低",
    "middle_upper_primary": "小中小高",
    "junior_high": "初中",
}

ASPECT_RATIO_RULES = {
    "1:1": "1:1 方图，适合头像、单个主体、局部特写或正方形卡片。",
    "4:3": "4:3 横图，适合课堂课件、普通插图和横向场景。",
    "3:4": "3:4 竖图，适合人物立绘、竖向插图和图书配图。",
    "16:9": "16:9 宽屏横图，适合章节头图、课件封面和大场景。",
    "9:16": "9:16 长屏竖图，适合移动端页面、竖版封面和长图。",
}


def normalize_aspect_ratio(value: str) -> str:
    text = str(value or "").strip()
    return text if text in ASPECT_RATIO_RULES else "16:9"


def web_options_from_request(data: dict | None) -> dict:
    data = data or {}
    grade_band = normalize_grade_band(data.get("grade_band") or data.get("grade_level") or "junior_high")
    aspect_ratio = normalize_aspect_ratio(data.get("aspect_ratio") or "16:9")
    has_person = to_bool_value(data.get("has_person"), True)
    reserve_text_default = to_bool_value(data.get("reserve_text_area"), True)
    has_poem = to_bool_value(data.get("has_poem"), reserve_text_default)
    try:
        image_count = max(1, min(int(data.get("image_count") or 4), 12))
    except Exception:
        image_count = 4
    return {
        "grade_band": grade_band,
        "grade_label": GRADE_LABELS.get(grade_band, "\u521d\u4e2d"),
        "aspect_ratio": aspect_ratio,
        "aspect_rule": ASPECT_RATIO_RULES[aspect_ratio],
        "has_person": has_person,
        "has_poem": has_poem,
        "needs_person_removal": not has_person,
        "needs_poem_text_removal": not has_poem,
        "image_count": image_count,
        "extra_requirement": str(data.get("extra_requirement") or data.get("user_description") or "").strip(),
    }


def build_generation_rules(options: dict) -> dict:
    return {
        "grade_band": options.get("grade_band"),
        "grade_label": options.get("grade_label"),
        "aspect_ratio": options.get("aspect_ratio"),
        "aspect_rule": options.get("aspect_rule"),
        "final_has_person": bool(options.get("has_person", True)),
        "final_has_poem_text": bool(options.get("has_poem", True)),
        "needs_person_removal": bool(options.get("needs_person_removal")),
        "needs_poem_text_removal": bool(options.get("needs_poem_text_removal")),
        "workflow_note": "\u5148\u751f\u6210\u542b\u5b8c\u6574\u6784\u56fe\u7684\u56fe\uff1b\u82e5\u6700\u7ec8\u9009\u62e9\u65e0\u4eba\u7269\u6216\u65e0\u8bd7\u8bcd\uff0c\u518d\u7528 image2 edit \u505a\u4e8c\u6b21\u5904\u7406\u3002",
        "person_removal_rule": "\u5982\u6700\u7ec8\u65e0\u4eba\u7269\uff0cDeepSeek \u4ecd\u9700\u5148\u8bbe\u8ba1\u4eba\u7269\u7ad9\u4f4d\uff0c\u5e76\u989d\u5916\u8f93\u51fa remove_person_prompt\uff0c\u8981\u6c42\u53bb\u6389\u4eba\u7269\u540e\u7528\u73af\u5883\u81ea\u7136\u8865\u5168\u539f\u4f4d\u7f6e\u3002",
        "poem_text_removal_rule": "\u5982\u6700\u7ec8\u65e0\u8bd7\u8bcd\u6587\u5b57\uff0cDeepSeek \u4ecd\u9700\u5148\u8bbe\u8ba1\u8bd7\u8bcd\u6587\u5b57\u533a\uff0c\u5e76\u989d\u5916\u8f93\u51fa remove_poem_text_prompt\uff0c\u8981\u6c42\u5220\u9664\u753b\u9762\u6587\u5b57\u5e76\u4fdd\u7559\u81ea\u7136\u7559\u767d\u3002",
        "extra_requirement": options.get("extra_requirement") or "",
    }


def selected_poems_from_request(data: dict | None) -> list[dict]:
    poems = (data or {}).get("selected_poems") or []
    if not isinstance(poems, list):
        return []
    clean = []
    for poem in poems:
        if not isinstance(poem, dict):
            continue
        title = compact(poem.get("title") or poem.get("name") or "")
        author = compact(poem.get("author") or poem.get("poet") or "")
        content = str(poem.get("content") or poem.get("poem") or "").strip()
        if title:
            clean.append({"title": title, "author": author, "content": content})
    return clean


def build_task_row(task: dict, user_text: str, web_options: dict | None = None, poem_content: str | None = None) -> dict:
    options = web_options or {}
    poet = compact(task.get("poet") or "")
    title = compact(task.get("title") or "\u672a\u547d\u540d\u4efb\u52a1")
    grade_level = options.get("grade_label") or infer_grade_level(user_text)
    aspect_ratio = options.get("aspect_ratio") or infer_aspect_ratio(user_text)
    scene_type = infer_scene_type(user_text)
    person_should_appear = True if web_options else infer_person_should_appear(user_text, task.get("mode") or "complete")
    poem_body = poem_content or user_text
    extra = options.get("extra_requirement") or ""
    main_visual = f"\u56f4\u7ed5\u300a{title}\u300b\u8bbe\u8ba1\u6559\u5b66\u914d\u56fe"
    if extra:
        main_visual += f"\uff1b\u8865\u5145\u8981\u6c42\uff1a{extra}"
    generation_rules = build_generation_rules(options) if web_options else {}
    return {
        "\u5e8f\u53f7": int(task.get("index") or 1),
        "\u8bd7\u4eba": poet,
        "\u8bd7\u540d": title,
        "\u8bd7\u6b4c\u5185\u5bb9": poem_body,
        "\u8bd7\u6b4c\u91ca\u4e49": poem_body,
        "\u8bd7\u4eba\u662f\u5426\u51fa\u73b0": "\u662f" if person_should_appear else "\u5426",
        "\u573a\u666f\u7c7b\u578b": scene_type,
        "\u73b0\u5b9e\u5c42": "\u6559\u5b66\u914d\u56fe\u4e3b\u573a\u666f\uff0c\u9002\u5408\u8bfe\u5802\u6216\u4ea4\u4e92\u56fe\u4e66\u4f7f\u7528",
        "\u60f3\u8c61/\u5386\u53f2\u5c42": "\u5982\u6d89\u53ca\u8bd7\u610f\u8054\u60f3\uff0c\u53ef\u5728\u8fdc\u666f\u6216\u5149\u5f71\u4e2d\u5f31\u5316\u5448\u73b0",
        "\u4e3b\u89c6\u89c9": main_visual,
        "\u80cc\u666f\u56fe\u751f\u6210\u8bf4\u660e": "\u5148\u751f\u6210\u542b\u5b8c\u6574\u6784\u56fe\u7684\u9875\u9762\u56fe\uff1a\u4eba\u7269\u7ad9\u4f4d\u548c\u8bd7\u8bcd\u6587\u5b57\u533a\u90fd\u8981\u8003\u8651\u5230\u3002\u6700\u7ec8\u662f\u5426\u5220\u9664\u4eba\u7269/\u8bd7\u8bcd\u6587\u5b57\u7531\u7ed3\u6784\u5316\u89c4\u5219\u51b3\u5b9a\u3002",
        "\u4eba\u7269\u59ff\u6001": "\u7ad9\u59ff\u81ea\u7136\uff0c\u4fbf\u4e8e\u878d\u5165\u4e3b\u753b\u9762" if person_should_appear else "\u65e0\u9700\u4eba\u7269",
        "\u4eba\u7269\u671d\u5411": "\u4fa7\u8eab\u9762\u5411\u753b\u9762\u4e3b\u4f53" if person_should_appear else "\u65e0\u9700\u4eba\u7269",
        "\u4eba\u7269\u52a8\u4f5c": "\u8f7b\u5fae\u52a8\u4f5c\uff0c\u7b26\u5408\u8bd7\u610f\u8868\u8fbe" if person_should_appear else "\u65e0\u9700\u4eba\u7269",
        "center_x": 0.5,
        "foot_y": 0.86,
        "height_ratio": 0.42,
        "pose": "standing" if person_should_appear else "none",
        "facing": "left" if person_should_appear else "none",
        "\u906e\u6321\u5173\u7cfb": "\u4eba\u7269\u53ef\u88ab\u524d\u666f\u8f7b\u5fae\u906e\u6321\uff0c\u589e\u5f3a\u771f\u5b9e\u611f",
        "\u9634\u5f71\u65b9\u5411": "\u4e0e\u4e3b\u5149\u6e90\u4e00\u81f4\uff0c\u81ea\u7136\u67d4\u548c",
        "\u878d\u5408\u6ce8\u610f": "\u786e\u4fdd\u4eba\u7269\u3001\u80cc\u666f\u3001\u8bd7\u8bcd\u533a\u5c42\u6b21\u6e05\u6670\uff0c\u4e0d\u906e\u6321\u4e3b\u4fe1\u606f",
        "\u8bd7\u8bcd\u533a\u4f4d\u7f6e": "\u753b\u9762\u7559\u767d\u5904",
        "\u8bd7\u8bcd\u533a\u65b9\u5f0f": "\u81ea\u7136\u7559\u767d\u5d4c\u5165",
        "\u5efa\u7b51\u7c7b\u522b": "",
        "\u5efa\u7b51\u5f62\u5236\u6ce8\u610f": "\u5982\u51fa\u73b0\u5efa\u7b51\uff0c\u9075\u5faa\u4e2d\u56fd\u53e4\u5178\u98ce\u683c",
        "\u82b1\u5349": "",
        "\u82b1\u5349\u5f62\u6001\u6ce8\u610f": "",
        "\u907f\u514d\u8bef\u753b": "\u907f\u514d\u73b0\u4ee3\u5143\u7d20\u3001\u989d\u5916\u6587\u5b57\u3001\u6c34\u5370\u548c\u8fc7\u5ea6\u9634\u90c1\u7684\u6c1b\u56f4",
        "\u5b66\u6bb5": grade_level,
        "\u56fe\u7247\u6bd4\u4f8b": aspect_ratio,
        "\u7528\u6237\u9700\u6c42": user_text,
        "\u7edf\u4e00\u98ce\u683c": True,
        "\u7ed3\u6784\u5316\u751f\u6210\u89c4\u5219": generation_rules,
        "\u6700\u7ec8\u662f\u5426\u9700\u8981\u4eba\u7269": "\u662f" if options.get("has_person", True) else "\u5426",
        "\u6700\u7ec8\u662f\u5426\u9700\u8981\u8bd7\u8bcd\u6587\u5b57": "\u662f" if options.get("has_poem", True) else "\u5426",
        "\u8865\u5145\u8981\u6c42": extra,
    }


def create_item_from_task(task: dict, user_text: str, web_options: dict | None = None, poem_content: str | None = None) -> dict:
    row = build_task_row(task, user_text, web_options=web_options, poem_content=poem_content)
    face = find_face_fast(row.get("\u8bd7\u4eba") or "", (web_options or {}).get("grade_band") or "")
    face_value = str(face) if face and face.exists() else ""
    return {"row": row, "face": face_value, "person_info": {}, "web_options": web_options or {}}


def create_item_from_poem(poem: dict, web_options: dict, user_text: str, index: int) -> dict:
    title = compact(poem.get("title") or "")
    author = compact(poem.get("author") or "")
    content = str(poem.get("content") or "").strip()
    task = {"poet": author, "title": title, "mode": "complete", "index": index}
    merged_text = f"\u8bf7\u4e3a{author}\u300a{title}\u300b\u751f\u6210\u6559\u5b66\u914d\u56fe\u3002"
    extra = web_options.get("extra_requirement") or ""
    if extra:
        merged_text += f"\u8865\u5145\u8981\u6c42\uff1a{extra}"
    if user_text and user_text not in merged_text:
        merged_text += f"\u539f\u59cb\u8865\u5145\u8bf4\u660e\uff1a{user_text}"
    return create_item_from_task(task, merged_text, web_options=web_options, poem_content=content or merged_text)


def source_payload_from_item(item: dict) -> dict:
    row = item["row"]
    poet = row.get("\u8bd7\u4eba")
    return {
        "source_index": row.get("\u5e8f\u53f7"),
        "\u8bd7\u4eba": poet,
        "\u671d\u4ee3": "\u5510" if poet not in {"\u9676\u6e0a\u660e"} else "\u4e1c\u664b",
        "\u8bd7\u540d": row.get("\u8bd7\u540d"),
        "\u8bd7\u6b4c\u539f\u6587": row.get("\u8bd7\u6b4c\u5185\u5bb9"),
        "\u8bd7\u6b4c\u91ca\u4e49": row.get("\u8bd7\u6b4c\u91ca\u4e49"),
        "person_should_appear": str(row.get("\u8bd7\u4eba\u662f\u5426\u51fa\u73b0") or "").strip() in {"\u662f", "true", "True", "1"},
        "\u573a\u666f\u7c7b\u578b": row.get("\u573a\u666f\u7c7b\u578b"),
        "\u73b0\u5b9e\u5c42": row.get("\u73b0\u5b9e\u5c42"),
        "\u60f3\u8c61/\u5386\u53f2\u5c42": row.get("\u60f3\u8c61/\u5386\u53f2\u5c42"),
        "\u4e3b\u89c6\u89c9": row.get("\u4e3b\u89c6\u89c9"),
        "\u80cc\u666f\u56fe\u751f\u6210\u8bf4\u660e": row.get("\u80cc\u666f\u56fe\u751f\u6210\u8bf4\u660e"),
        "\u4eba\u7269\u59ff\u6001": row.get("\u4eba\u7269\u59ff\u6001"),
        "\u4eba\u7269\u671d\u5411": row.get("\u4eba\u7269\u671d\u5411"),
        "\u4eba\u7269\u52a8\u4f5c": row.get("\u4eba\u7269\u52a8\u4f5c"),
        "center_x": row.get("center_x"),
        "foot_y": row.get("foot_y"),
        "height_ratio": row.get("height_ratio"),
        "pose": row.get("pose"),
        "facing": row.get("facing"),
        "\u906e\u6321\u5173\u7cfb": row.get("\u906e\u6321\u5173\u7cfb"),
        "\u9634\u5f71\u65b9\u5411": row.get("\u9634\u5f71\u65b9\u5411"),
        "\u878d\u5408\u6ce8\u610f": row.get("\u878d\u5408\u6ce8\u610f"),
        "\u8bd7\u8bcd\u533a\u4f4d\u7f6e": row.get("\u8bd7\u8bcd\u533a\u4f4d\u7f6e"),
        "\u8bd7\u8bcd\u533a\u65b9\u5f0f": row.get("\u8bd7\u8bcd\u533a\u65b9\u5f0f"),
        "\u5b57\u53f7": 22,
        "\u5efa\u7b51\u7c7b\u522b": row.get("\u5efa\u7b51\u7c7b\u522b"),
        "\u5efa\u7b51\u5f62\u5236\u6ce8\u610f": row.get("\u5efa\u7b51\u5f62\u5236\u6ce8\u610f"),
        "\u82b1\u5349": row.get("\u82b1\u5349"),
        "\u82b1\u5349\u5f62\u6001\u6ce8\u610f": row.get("\u82b1\u5349\u5f62\u6001\u6ce8\u610f"),
        "\u907f\u514d\u8bef\u753b": row.get("\u907f\u514d\u8bef\u753b"),
        "\u5b66\u6bb5": row.get("\u5b66\u6bb5"),
        "\u56fe\u7247\u6bd4\u4f8b": row.get("\u56fe\u7247\u6bd4\u4f8b"),
        "\u7528\u6237\u9700\u6c42": row.get("\u7528\u6237\u9700\u6c42"),
        "\u7edf\u4e00\u98ce\u683c": row.get("\u7edf\u4e00\u98ce\u683c"),
        "generation_rules": row.get("\u7ed3\u6784\u5316\u751f\u6210\u89c4\u5219") or {},
        "final_has_person": row.get("\u6700\u7ec8\u662f\u5426\u9700\u8981\u4eba\u7269"),
        "final_has_poem_text": row.get("\u6700\u7ec8\u662f\u5426\u9700\u8981\u8bd7\u8bcd\u6587\u5b57"),
        "extra_requirement": row.get("\u8865\u5145\u8981\u6c42") or "",
    }


def prompt_key(row: dict, variant: int) -> str:
    return f"{base.prompt_key(row)}_web_v{variant}"



def strip_json_code_fence(text: str) -> str:
    text = str(text or "").strip()
    fence = re.search(r"```(?:json)?\s*(.*?)```", text, re.S | re.I)
    return fence.group(1).strip() if fence else text


def remove_trailing_json_commas(text: str) -> str:
    return re.sub(r",\s*([}\]])", r"\1", text)


def balanced_json_candidates(text: str, open_char: str, close_char: str) -> list[str]:
    candidates = []
    for start, ch in enumerate(text):
        if ch != open_char:
            continue
        depth = 0
        in_string = False
        escape = False
        for idx in range(start, len(text)):
            current = text[idx]
            if in_string:
                if escape:
                    escape = False
                elif current == "\\":
                    escape = True
                elif current == '"':
                    in_string = False
                continue
            if current == '"':
                in_string = True
            elif current == open_char:
                depth += 1
            elif current == close_char:
                depth -= 1
                if depth == 0:
                    candidates.append(text[start:idx + 1])
                    break
    return candidates


def coerce_prompt_items(value) -> list[dict]:
    if isinstance(value, list):
        return [x for x in value if isinstance(x, dict)]
    if isinstance(value, dict):
        for key in ["items", "prompts", "variants", "results", "data"]:
            nested = value.get(key)
            if isinstance(nested, list):
                return [x for x in nested if isinstance(x, dict)]
        if value.get("complete_page_prompt") or value.get("background_prompt") or value.get("prompt"):
            return [value]
    return []


def parse_prompt_items_flexible(content: str) -> list[dict]:
    raw = str(content or "").strip()
    if not raw:
        raise ValueError("DeepSeek output is empty")
    texts = []
    fenced = strip_json_code_fence(raw)
    for text in [raw, fenced]:
        cleaned = remove_trailing_json_commas(text.strip())
        if cleaned and cleaned not in texts:
            texts.append(cleaned)
        for candidate in balanced_json_candidates(cleaned, "[", "]"):
            candidate = remove_trailing_json_commas(candidate)
            if candidate not in texts:
                texts.append(candidate)
        for candidate in balanced_json_candidates(cleaned, "{", "}"):
            candidate = remove_trailing_json_commas(candidate)
            if candidate not in texts:
                texts.append(candidate)
    for text in texts:
        try:
            items = coerce_prompt_items(json.loads(text))
            if items:
                return items
        except Exception:
            continue
    # 最后一层兜底：格式完全不标准时，把正文当作单条完整图提示词，避免整批任务直接失败。
    return [{"variant": 1, "plan": {"format_recovered": True}, "complete_page_prompt": raw}]


def ask_json(model: str, system: str, user: str, max_tokens: int = 16000) -> list[dict]:
    app_id, app_key = base.credentials()
    payload = {"model": model, "messages": [{"role": "system", "content": system}, {"role": "user", "content": user}]}
    if model.startswith("gpt-5"):
        payload["max_completion_tokens"] = max_tokens
    else:
        payload["temperature"] = 0.25
        payload["max_tokens"] = max_tokens
    resp = base.post_json(
        "/openai-compatible/v1/chat/completions",
        payload,
        {"Authorization": f"Bearer {app_id}:{app_key}"},
        timeout=600,
    )
    return parse_prompt_items_flexible(resp["choices"][0]["message"].get("content") or "")



def read_text_compatible(path: Path) -> str:
    for encoding in ["utf-8", "utf-8-sig", "gb18030"]:
        try:
            return path.read_text(encoding=encoding)
        except UnicodeDecodeError:
            continue
    return path.read_text(encoding="utf-8", errors="replace")


def load_reference_index() -> list[dict]:
    if not REFERENCE_INDEX_PATH.exists():
        return []
    try:
        data = json.loads(REFERENCE_INDEX_PATH.read_text(encoding="utf-8-sig"))
    except Exception:
        return []
    entries = data.get("entries") if isinstance(data, dict) else []
    return entries if isinstance(entries, list) else []


def reference_need_for_kind(prompt_item: dict, kind: str) -> dict:
    needs = prompt_item.get("reference_needs") or {}
    if not isinstance(needs, dict):
        return {}
    value = needs.get(kind) or needs.get(f"{kind}_reference") or needs.get(f"{kind}_need")
    if value is True:
        return {"needed": True}
    if not isinstance(value, dict):
        return {}
    needed_value = value.get("needed")
    if needed_value is None:
        needed_value = value.get("need")
    if needed_value is None:
        needed_value = value.get("has_reference")
    if needed_value is None:
        needed_value = bool(value.get("category"))
    if not bool(needed_value):
        return {}
    category = compact(value.get("category") or value.get("type") or value.get("name") or "")
    return {"needed": True, "category": category}


def reference_entries_for_kind(prompt_item: dict, kind: str) -> list[dict]:
    need = reference_need_for_kind(prompt_item, kind)
    if not need.get("needed") or not need.get("category"):
        return []
    category = need["category"]
    entries = []
    for entry in load_reference_index():
        if entry.get("kind") == kind and compact(entry.get("category") or "") == category:
            entries.append(entry)
    return entries


def reference_prompt_note(row: dict, prompt_item: dict) -> str:
    parts = []
    labels = {
        "architecture": "\u5efa\u7b51",
        "flower": "\u82b1\u5349/\u690d\u7269",
    }
    for kind in ["architecture", "flower"]:
        entries = reference_entries_for_kind(prompt_item, kind)
        if not entries:
            continue
        entry = entries[0]
        label = labels.get(kind, kind)
        lines = [
            f"\u53c2\u8003{label}\u56fe\u6765\u751f\u6210\uff0c\u53c2\u8003\u56fe\u53ea\u7528\u4e8e\u63a7\u5236{label}\u5f62\u5236\u3001\u7ed3\u6784\u3001\u6750\u8d28\u3001\u989c\u8272\u548c\u65f6\u4ee3\u611f\uff0c\u4e0d\u590d\u5236\u53c2\u8003\u56fe\u6784\u56fe\u3002",
            f"\u53c2\u8003\u7c7b\u522b\uff1a{entry.get('category') or ''}",
        ]
        if entry.get("prompt_advice"):
            lines.append(f"\u5f62\u5236\u5efa\u8bae\uff1a{entry['prompt_advice']}")
        if entry.get("negative_constraints"):
            lines.append(f"\u907f\u514d\u8bef\u751f\u6210\uff1a{entry['negative_constraints']}")
        parts.append("\n".join(lines))
    return "\n\n".join(parts)


def reference_paths_for_prompt(row: dict, prompt_item: dict) -> list[Path]:
    refs: list[Path] = []
    limits = {"architecture": 1, "flower": 2}
    for kind in ["architecture", "flower"]:
        added = 0
        for entry in reference_entries_for_kind(prompt_item, kind):
            image_path = compact(entry.get("image_path") or "")
            if not image_path:
                continue
            path = APP_ROOT / image_path
            if path.exists():
                refs.append(path)
                added += 1
            if added >= limits[kind]:
                break
    return refs

def total_prompt_system(grade_band: str = "") -> str:
    parts = []
    if TOTAL_PROMPT_PATH.exists():
        parts.append(read_text_compatible(TOTAL_PROMPT_PATH))
    else:
        parts.append(CONTRACT_TEXT + "\n\n\u4f60\u662f\u8bd7\u6b4c\u5206\u5c42\u751f\u56fe\u603b agent\u3002\u4e00\u6b21\u6027\u8f93\u51fa\u80cc\u666f\u56fe\u3001\u4eba\u7269\u56fe\u3001\u5408\u6210\u56fe\u63d0\u793a\u8bcd\uff0c\u53ea\u8f93\u51fa JSON \u6570\u7ec4\u3002")
    grade_file = GRADE_PROMPT_FILES.get(normalize_grade_band(grade_band))
    if grade_file and grade_file.exists():
        parts.append(read_text_compatible(grade_file))
    return "\n\n".join(part.strip() for part in parts if part.strip())



GRADE_STYLE_REQUIRED = {
    "lower_primary": "\u753b\u98ce\uff1a3D\u4e2d\u56fd\u513f\u7ae5\u53e4\u98ce\uff0c\u8272\u5f69\u660e\u4e3d\u3001\u5e72\u51c0\uff0c\u9971\u548c\u5ea6\u9002\u4e2d\u504f\u9ad8\uff0c\u5bf9\u6bd4\u5ea6\u7565\u9ad8\u3002",
    "middle_upper_primary": "\u753b\u98ce\uff1a3D\u4e2d\u56fd\u5361\u901a\u53e4\u98ce\u7535\u5f71\u98ce\uff0c\u8272\u5f69\u660e\u4e3d\u3001\u5e72\u51c0\uff0c\u9971\u548c\u5ea6\u9002\u4e2d\u504f\u9ad8\uff0c\u5bf9\u6bd4\u5ea6\u7565\u9ad8\u3002",
    "junior_high": "\u753b\u98ce\uff1a3D\u4e2d\u56fd\u56fd\u98ce\u624b\u6e38\u98ce\u683c\uff0c\u8272\u5f69\u660e\u4e3d\u3001\u5e72\u51c0\uff0c\u9971\u548c\u5ea6\u9002\u4e2d\u504f\u9ad8\uff0c\u5bf9\u6bd4\u5ea6\u7565\u9ad8\u3002",
}

BRIGHTNESS_HARD_RULE = "\u6574\u4f53\u8272\u8c03\u4e0d\u8981\u6697\uff0c\u4e0d\u8981\u7070\u8499\u8499\uff0c\u4e0d\u8981\u538b\u4f4e\u9971\u548c\u5ea6\uff1b\u5373\u4f7f\u8868\u73b0\u75c5\u5f31\u3001\u79cb\u591c\u3001\u60b2\u58ee\u6216\u5b64\u5bc2\u60c5\u7eea\uff0c\u4e5f\u5fc5\u987b\u4fdd\u6301\u753b\u9762\u6e05\u900f\u3001\u660e\u4eae\u3001\u8272\u5f69\u5e72\u51c0\uff0c\u907f\u514d\u5927\u9762\u79ef\u6df1\u7070\u3001\u6df1\u8910\u3001\u9ed1\u84dd\u6216\u6697\u8272\u524d\u666f\u3002"


def enforce_complete_prompt_style(prompt: str, grade_band: str) -> str:
    prompt = str(prompt or "").strip()
    grade_key = normalize_grade_band(grade_band)
    required = GRADE_STYLE_REQUIRED.get(grade_key) or GRADE_STYLE_REQUIRED["junior_high"]
    style_lines = required + "\n" + BRIGHTNESS_HARD_RULE
    if required in prompt and BRIGHTNESS_HARD_RULE in prompt:
        return prompt
    marker = "\u3010\u753b\u98ce\u4e0e\u6784\u56fe\u3011"
    if marker in prompt:
        return prompt.replace(marker, marker + "\n" + style_lines, 1)
    return style_lines + "\n" + prompt


def normalize_total_prompt_items(items: list[dict], count: int, grade_band: str = "") -> list[dict]:
    by_variant = {int(x.get("variant") or i + 1): x for i, x in enumerate(items)}
    normalized = []
    for variant in range(1, count + 1):
        item = dict(by_variant.get(variant, {}))
        complete_prompt = item.get("complete_page_prompt") or item.get("complete_prompt") or ""
        if not complete_prompt:
            complete_prompt = item.get("background_prompt") or item.get("composite_instruction") or ""
        normalized.append({
            **{k: v for k, v in item.items() if k not in {"background_prompt", "person_prompt", "composite_instruction", "target_person_type", "complete_prompt"}},
            "variant": variant,
            "complete_page_prompt": enforce_complete_prompt_style(complete_prompt, grade_band),
            "remove_person_prompt": item.get("remove_person_prompt") or "",
            "remove_poem_text_prompt": item.get("remove_poem_text_prompt") or "",
            "reference_needs": item.get("reference_needs") or (item.get("plan") or {}).get("reference_needs") or {},
        })
    return normalized


def generate_total_prompt_items(item: dict, count: int, job_id: str) -> list[dict]:
    ensure_not_cancelled(job_id)
    base_user_prompt = "\u8bf7\u6839\u636e\u8f93\u5165\u8bd7\u6b4c\u4e00\u6b21\u6027\u751f\u6210\u6307\u5b9a\u6570\u91cf\u7684\u5b8c\u6574\u5206\u5c42\u751f\u56fe\u65b9\u6848\uff0cvariant \u4ece 1 \u5f00\u59cb\u8fde\u7eed\u7f16\u53f7\u3002\n" + build_variant_payload(item, count)
    last_error = None
    for attempt in range(1, 4):
        ensure_not_cancelled(job_id)
        retry_note = "" if attempt == 1 else "\n\n\u4e0a\u4e00\u6b21\u8f93\u51fa\u4e0d\u662f\u5408\u6cd5 JSON \u6570\u7ec4\uff0c\u8bf7\u4e25\u683c\u53ea\u8f93\u51fa JSON \u6570\u7ec4\uff0c\u4e0d\u8981 Markdown\uff0c\u4e0d\u8981\u89e3\u91ca\uff0c\u5b57\u7b26\u4e32\u5185\u7684\u6362\u884c\u7528 \\n \u8f6c\u4e49\u3002"
        try:
            grade_band = (item.get("web_options") or {}).get("grade_band") or ""
            result = ask_json(TOTAL_PROMPT_MODEL, total_prompt_system(grade_band), base_user_prompt + retry_note)
            return normalize_total_prompt_items(result, count, grade_band)
        except Exception as exc:
            last_error = exc
            emit(job_id, "progress", f"{item_label(item)} \u603b\u63d0\u793a\u8bcd\u89e3\u6790\u5931\u8d25\uff0c\u91cd\u8bd5 {attempt}/3\uff1a{exc}", ok=False, slot="prompts")
            time.sleep(min(10, attempt * 3))
    raise RuntimeError(f"{item_label(item)} \u603b\u63d0\u793a\u8bcd\u8fde\u7eed\u5931\u8d25\uff1a{last_error}")

def background_system() -> str:
    return CONTRACT_TEXT + "\n\n你现在只生成背景图提示词。只输出 JSON 数组。每个元素格式：{\"variant\":1,\"background_prompt\":\"\"}"


def person_composite_system() -> str:
    return CONTRACT_TEXT + "\n\n你现在只生成人物图提示词和最终合成提示词。只输出 JSON 数组。每个元素格式：{\"variant\":1,\"target_person_type\":\"poet|non_poet_character|no_person\",\"person_prompt\":\"\",\"composite_instruction\":\"\"}"


def build_variant_payload(item: dict, count: int) -> str:
    payload = source_payload_from_item(item)
    return json.dumps({"variant_count": count, "generation_rules": payload.get("generation_rules") or {}, "poem": payload}, ensure_ascii=False, indent=2)


def merge_prompt_items(bg_items: list[dict], pc_items: list[dict], count: int) -> list[dict]:
    by_bg = {int(x.get("variant") or i + 1): x for i, x in enumerate(bg_items)}
    by_pc = {int(x.get("variant") or i + 1): x for i, x in enumerate(pc_items)}
    prompts = []
    for variant in range(1, count + 1):
        prompts.append({**by_bg.get(variant, {}), **by_pc.get(variant, {}), "variant": variant})
    return prompts


def item_label(item: dict) -> str:
    row = item["row"]
    poet = compact(row.get("\u8bd7\u4eba"))
    title = compact(row.get("\u8bd7\u540d"))
    return f"{poet}\u300a{title}\u300b"


def generate_background_prompt_items(item: dict, count: int, job_id: str) -> list[dict]:
    ensure_not_cancelled(job_id)
    user_bg = "\u8bf7\u6839\u636e\u8f93\u5165\u8bd7\u6b4c\u751f\u6210\u6307\u5b9a\u6570\u91cf\u7684\u80cc\u666f\u56fe\u63d0\u793a\u8bcd\uff0cvariant \u4ece 1 \u5f00\u59cb\u8fde\u7eed\u7f16\u53f7\u3002\n" + build_variant_payload(item, count)
    return ask_json(DEEPSEEK_MODEL, background_system(), user_bg)


def generate_person_composite_prompt_items(item: dict, count: int, job_id: str) -> list[dict]:
    ensure_not_cancelled(job_id)
    user_pc = "\u8bf7\u6839\u636e\u8f93\u5165\u8bd7\u6b4c\u751f\u6210\u6307\u5b9a\u6570\u91cf\u7684\u4eba\u7269\u56fe\u63d0\u793a\u8bcd\u548c\u5408\u6210\u63d0\u793a\u8bcd\uff0cvariant \u4ece 1 \u5f00\u59cb\u8fde\u7eed\u7f16\u53f7\u3002\n" + build_variant_payload(item, count)
    try:
        return ask_json(GPT_MODEL, person_composite_system(), user_pc)
    except Exception as exc:
        emit(job_id, "progress", f"{GPT_MODEL} \u751f\u6210 {item_label(item)} \u4eba\u7269/\u5408\u6210\u63d0\u793a\u8bcd\u5931\u8d25\uff0c\u6539\u7528 DeepSeek\uff1a{exc}")
        ensure_not_cancelled(job_id)
        return ask_json(DEEPSEEK_MODEL, person_composite_system(), user_pc)


def generate_prompts(item: dict, count: int, job_id: str) -> list[dict]:
    emit(job_id, "progress", f"\u5f00\u59cb\u4f7f\u7528\u5355 agent \u603b\u63d0\u793a\u8bcd\u751f\u6210\uff1a\u5171 {count} \u4e2a\u53d8\u4f53\u3002")
    prompts = generate_total_prompt_items(item, count, job_id)
    emit(job_id, "progress", f"\u603b\u63d0\u793a\u8bcd agent \u5df2\u5b8c\u6210\uff1a{len(prompts)}/{count}\u3002")
    return prompts


def generate_prompts_for_items(items: list[dict], count: int, job_id: str, mode: str) -> dict[int, list[dict]]:
    emit(job_id, "progress", f"\u5f00\u59cb\u5e76\u53d1\u8c03\u7528\u5355 agent \u603b\u63d0\u793a\u8bcd\uff1a{len(items)} \u9996\u8bd7\uff0c\u6bcf\u9996 {count} \u4e2a\u53d8\u4f53\uff0c\u5e76\u53d1 {DEEPSEEK_PROMPT_WORKERS}\uff0c\u901f\u7387 {API_RATE_PER_MINUTE}/min\u3002")
    results: dict[int, list[dict]] = {}
    workers = min(DEEPSEEK_PROMPT_WORKERS, len(items))
    with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as ex:
        futs = {ex.submit(generate_total_prompt_items, item, count, job_id): (idx, item) for idx, item in enumerate(items)}
        for done, fut in enumerate(concurrent.futures.as_completed(futs), 1):
            ensure_not_cancelled(job_id)
            idx, item = futs[fut]
            try:
                prompt_items = fut.result()
            except Exception as exc:
                emit(job_id, "progress", f"\u603b\u63d0\u793a\u8bcd\u5931\u8d25 {done}/{len(items)}\uff1a{item_label(item)}\uff1a{exc}\uff0c\u672c\u8bd7\u8df3\u8fc7\u3002", ok=False, slot="prompts")
                results[idx] = []
                continue
            results[idx] = prompt_items
            emit(job_id, "progress", f"\u603b\u63d0\u793a\u8bcd\u5b8c\u6210 {done}/{len(items)}\uff1a{item_label(item)}\uff0c{len(prompt_items)}/{count}\u3002", slot="prompts")
    return results


def enforce_face_reference_prompt(prompt: str) -> str:
    guard = (
        "\u3010\u53c2\u8003\u8138\u56fe\u786c\u6027\u8981\u6c42\u3011\n"
        "\u672c\u6b21\u4eba\u7269\u56fe\u5fc5\u987b\u57fa\u4e8e\u8f93\u5165\u7684\u53c2\u8003\u8138\u56fe\u8fdb\u884c\u56fe\u751f\u56fe\u3002"
        "\u4e25\u683c\u4fdd\u7559\u53c2\u8003\u8138\u56fe\u7684\u4eba\u8138\u957f\u76f8\u3001\u8138\u578b\u8f6e\u5ed3\u3001\u4e94\u5b98\u4f4d\u7f6e\u3001\u7709\u773c\u9f3b\u5634\u6bd4\u4f8b\u3001\u5e74\u9f84\u611f\u548c\u4eba\u7269\u8eab\u4efd\uff1b"
        "\u4e0d\u8981\u91cd\u65b0\u8bbe\u8ba1\u8138\uff0c\u4e0d\u8981\u66ff\u6362\u6210\u964c\u751f\u4eba\uff0c\u4e0d\u8981\u6539\u53d8\u8138\u90e8\u7279\u5f81\u3002"
        "\u53ea\u5141\u8bb8\u6539\u53d8\u670d\u9970\u3001\u52a8\u4f5c\u3001\u671d\u5411\u3001\u59ff\u6001\u3001\u5149\u7ebf\u548c\u53ef\u5206\u79bb\u9053\u5177\u3002\n\n"
    )
    text = str(prompt or "")
    if "\u53c2\u8003\u8138\u56fe\u786c\u6027\u8981\u6c42" in text:
        return text
    return guard + text

def run_image(prompt: str, out: Path, refs: list[Path]) -> tuple[bool, str]:
    out.parent.mkdir(parents=True, exist_ok=True)
    if out.exists() and out.stat().st_size > 10000:
        return True, f"exists {out.name}"
    try:
        if refs:
            resp = base.post_multipart("/openai-compatible/v1/images/edits", {"model": base.IMAGE_MODEL, "prompt": prompt}, [("image[]", p) for p in refs])
        else:
            app_id, app_key = base.credentials()
            resp = base.post_json("/openai-compatible/v1/images/generations", {"model": base.IMAGE_MODEL, "prompt": prompt}, {"api-key": f"{app_id}:{app_key}"}, timeout=900)
        base.save_image_response(resp, out)
        return True, f"generated {out.name}"
    except Exception as exc:
        return False, f"failed-no-retry {out.name}: {exc}"


def image_content_type(path: Path) -> str:
    suffix = path.suffix.lower()
    if suffix in {".jpg", ".jpeg"}:
        return "image/jpeg"
    if suffix == ".webp":
        return "image/webp"
    return "image/png"


def public_image(job_id: str, path: Path) -> dict:
    job = jobs.get(job_id, {})
    if not job.get("save_outputs", True):
        image_id = uuid.uuid4().hex
        body = path.read_bytes()
        job.setdefault("image_store", {})[image_id] = {
            "name": path.name,
            "download_name": short_image_filename(path.name, len(job.setdefault("image_store", {})) + 1),
            "body": body,
            "content_type": image_content_type(path),
        }
        return {
            "name": path.name,
            "url": f"/api/image?job={job_id}&id={image_id}",
            "download_url": f"/api/image?job={job_id}&id={image_id}&download=1",
            "temporary": True,
        }
    encoded = quote_path(path)
    return {"name": path.name, "url": f"/api/file?job={job_id}&path={encoded}", "download_url": f"/api/file?job={job_id}&path={encoded}&download=1"}

def quote_path(path: Path) -> str:
    return base64.urlsafe_b64encode(str(path).encode("utf-8")).decode("ascii")


def unquote_path(value: str) -> Path:
    return Path(base64.urlsafe_b64decode(value.encode("ascii")).decode("utf-8"))


def process_one_item(job_id: str, item: dict, count: int, mode: str, job_dir: Path, images: list[dict], label_prefix: str = "", prompts: list[dict] | None = None) -> None:
    row = item["row"]
    poet = compact(row.get("\u8bd7\u4eba"))
    title = compact(row.get("\u8bd7\u540d"))
    bg_dir = job_dir / "\u80cc\u666f\u56fe"
    person_dir = job_dir / "\u4eba\u7269\u56fe"
    comp_dir = job_dir / "\u5408\u6210\u56fe"
    prompt_dir = job_dir / "\u63d0\u793a\u8bcd"
    for folder in [bg_dir, person_dir, comp_dir, prompt_dir]:
        folder.mkdir(parents=True, exist_ok=True)

    emit(job_id, "progress", f"{label_prefix}\u5df2\u5339\u914d\u8bd7\u6b4c\uff1a{poet}\u300a{title}\u300b\u3002\u5f53\u524d\u6a21\u5f0f\uff1a{mode}\u3002", out_dir=str(job_dir))
    if prompts is None:
        prompts = generate_prompts_for_items([item], count, job_id, mode)[0]
    (prompt_dir / f"{safe_name(poet)}_{safe_name(title)}_prompts.json").write_text(json.dumps(prompts, ensure_ascii=False, indent=2), encoding="utf-8-sig")
    emit(job_id, "progress", f"{label_prefix}\u63d0\u793a\u8bcd\u5df2\u4fdd\u5b58\uff0c\u5f00\u59cb\u8c03\u7528 image2 \u751f\u56fe\u3002")

    face = Path(item.get("face") or "") if item.get("face") else None

    for prompt_item in prompts:
        ensure_not_cancelled(job_id)
        v = int(prompt_item["variant"])
        key = prompt_key(row, v)
        bg_path = bg_dir / f"{key}_background.png"
        person_path = person_dir / f"{key}_person.png"
        comp_path = comp_dir / f"{key}_composite.png"

        ok, msg = run_image(prompt_item.get("background_prompt") or "", bg_path, [])
        emit(job_id, "progress", f"{label_prefix}\u80cc\u666f\u56fe {v}/{count}\uff1a{msg}", ok=ok)
        if not ok:
            continue
        if mode == "background":
            img = public_image(job_id, bg_path)
            images.append(img)
            emit(job_id, "image", f"{label_prefix}\u80cc\u666f\u56fe {v} \u5b8c\u6210", image=img)
            continue

        ensure_not_cancelled(job_id)
        target_type = prompt_item.get("target_person_type") or "poet"
        person_prompt = prompt_item.get("person_prompt") or ""
        if target_type == "no_person" or not person_prompt:
            shutil.copyfile(bg_path, comp_path)
            img = public_image(job_id, comp_path)
            images.append(img)
            emit(job_id, "image", f"{label_prefix}\u5b8c\u6574\u9875\u9762 {v} \u5b8c\u6210\uff1a\u76f4\u63a5\u751f\u6210\u5b8c\u6574\u9875\u9762\u56fe\u3002", image=img)
            continue

        refs = []
        if target_type == "poet":
            if not face or not face.exists():
                emit(job_id, "progress", f"{label_prefix}\u7f3a\u5c11 {poet} \u7684\u4eba\u7269\u8138\u56fe\uff0c\u53d8\u4f53 {v} \u6682\u53ea\u8f93\u51fa\u80cc\u666f\u56fe\u3002", ok=False)
                img = public_image(job_id, bg_path)
                images.append(img)
                emit(job_id, "image", f"{label_prefix}\u80cc\u666f\u56fe {v} \u5b8c\u6210", image=img)
                continue
            refs = [face]
            person_prompt = enforce_face_reference_prompt(person_prompt)

        ok, msg = run_image(person_prompt, person_path, refs)
        emit(job_id, "progress", f"{label_prefix}\u4eba\u7269\u56fe {v}/{count}\uff1a{msg}", ok=ok)
        if not ok:
            continue
        if mode == "person":
            img = public_image(job_id, person_path)
            images.append(img)
            emit(job_id, "image", f"{label_prefix}\u4eba\u7269\u56fe {v} \u5b8c\u6210", image=img)
            continue

        ensure_not_cancelled(job_id)
        ok, msg = run_image(prompt_item.get("composite_instruction") or "", comp_path, [bg_path, person_path])
        emit(job_id, "progress", f"{label_prefix}\u5408\u6210\u56fe {v}/{count}\uff1a{msg}", ok=ok)
        if ok:
            img = public_image(job_id, comp_path)
            images.append(img)
            emit(job_id, "image", f"{label_prefix}\u5b8c\u6574\u9875\u9762 {v} \u5b8c\u6210", image=img)



def item_output_paths(job_dir: Path, item: dict, variant: int) -> dict[str, Path]:
    row = item["row"]
    key = prompt_key(row, variant)
    return {
        "bg": job_dir / "历史中间图" / f"{key}_background.png",
        "person": job_dir / "历史中间图" / f"{key}_person.png",
        "comp": job_dir / "完整图" / f"{key}_complete.png",
        "final": job_dir / "最终图" / f"{key}.png",
    }


def ensure_job_dirs(job_dir: Path) -> None:
    for name in ["完整图", "最终图", "提示词"]:
        (job_dir / name).mkdir(parents=True, exist_ok=True)
def save_item_prompts(job_dir: Path, item: dict, prompts: list[dict]) -> None:
    row = item["row"]
    poet = compact(row.get("\u8bd7\u4eba"))
    title = compact(row.get("\u8bd7\u540d"))
    prompt_dir = job_dir / "\u63d0\u793a\u8bcd"
    prompt_dir.mkdir(parents=True, exist_ok=True)
    (prompt_dir / f"{safe_name(poet)}_{safe_name(title)}_prompts.json").write_text(json.dumps(prompts, ensure_ascii=False, indent=2), encoding="utf-8-sig")


def iter_prompt_entries(items: list[dict], prompt_map: dict[int, list[dict]]):
    for item_idx, item in enumerate(items):
        row = item["row"]
        for prompt_item in prompt_map[item_idx]:
            yield item_idx, item, row, prompt_item, int(prompt_item["variant"])


def append_image_metric(job_id: str, metric_type: str, **data) -> None:
    job = jobs.get(job_id)
    if not job:
        return
    metrics = job.setdefault("image_metrics", {"items": [], "batches": []})
    if metric_type == "batch":
        metrics.setdefault("batches", []).append(data)
    else:
        metrics.setdefault("items", []).append(data)


def summarize_image_metrics(job_id: str) -> dict:
    metrics = jobs.get(job_id, {}).get("image_metrics") or {}
    items = metrics.get("items") or []
    batches = metrics.get("batches") or []
    attempt_items = [x for x in items if not x.get("is_final")]
    final_items = [x for x in items if x.get("is_final")]
    retry_items = [x for x in attempt_items if int(x.get("attempt_round") or 0) > 0]
    first_round_retryable = [x for x in attempt_items if int(x.get("attempt_round") or 0) == 0 and is_retryable_image_error(x.get("message", ""))]
    final_failures = [x for x in final_items if not x.get("ok")]
    success_attempts = [x for x in attempt_items if x.get("ok")]
    durations = [float(x.get("duration_seconds") or 0) for x in success_attempts]
    batch_durations = [float(x.get("duration_seconds") or 0) for x in batches if x.get("duration_seconds") is not None]
    batches_with_retryable = [x for x in batches if int(x.get("retryable_failure_count") or 0) > 0]
    batch_429_details = []
    for batch in batches_with_retryable:
        batch_429_details.append({
            "stage": batch.get("stage") or "",
            "attempt_round": int(batch.get("attempt_round") or 0),
            "batch_index": int(batch.get("batch_index") or 0),
            "total_batches": int(batch.get("total_batches") or 0),
            "image_count": int(batch.get("image_count") or 0),
            "retryable_failure_count": int(batch.get("retryable_failure_count") or 0),
            "duration_seconds": batch.get("duration_seconds"),
        })
    item_details = []
    for item in attempt_items:
        item_details.append({
            "stage": item.get("stage") or "",
            "label": item.get("label") or "",
            "attempt_round": int(item.get("attempt_round") or 0),
            "batch_index": int(item.get("batch_index") or 0),
            "ok": bool(item.get("ok")),
            "duration_seconds": item.get("duration_seconds"),
            "retryable": bool(item.get("retryable")),
            "message": str(item.get("message") or ""),
        })
    batch_details = []
    for batch in batches:
        batch_details.append({
            "stage": batch.get("stage") or "",
            "attempt_round": int(batch.get("attempt_round") or 0),
            "batch_index": int(batch.get("batch_index") or 0),
            "total_batches": int(batch.get("total_batches") or 0),
            "image_count": int(batch.get("image_count") or 0),
            "success_count": int(batch.get("success_count") or 0),
            "retryable_failure_count": int(batch.get("retryable_failure_count") or 0),
            "duration_seconds": batch.get("duration_seconds"),
        })
    batch_count = len(batches)
    return {
        "planned_image_count": len(final_items),
        "attempted_image_count": len(attempt_items),
        "successful_attempt_count": len(success_attempts),
        "final_stage_success_count": len([x for x in final_items if x.get("ok")]),
        "final_stage_failure_count": len(final_failures),
        "item_event_count": len(items),
        "batch_count": batch_count,
        "first_round_429_count": len(first_round_retryable),
        "batch_429_count": len(batches_with_retryable),
        "batch_429_ratio": round(len(batches_with_retryable) / batch_count, 4) if batch_count else None,
        "batch_429_details": batch_429_details,
        "retry_attempt_count": len(retry_items),
        "final_retryable_failure_count": len([x for x in final_failures if is_retryable_image_error(x.get("message", ""))]),
        "max_attempt_round": max([int(x.get("attempt_round") or 0) for x in attempt_items], default=0),
        "avg_success_image_seconds": round(sum(durations) / len(durations), 1) if durations else None,
        "max_success_image_seconds": round(max(durations), 1) if durations else None,
        "avg_batch_seconds": round(sum(batch_durations) / len(batch_durations), 1) if batch_durations else None,
        "max_batch_seconds": round(max(batch_durations), 1) if batch_durations else None,
        "item_details": item_details,
        "batch_details": batch_details,
    }

def is_retryable_image_error(message: str) -> bool:
    text = str(message or "").lower()
    return any(token in text for token in ["http 429", "too many requests", "winerror 10060", "timed out", "timeout"])


def image_job_key(job: dict) -> str:
    return str(job.get("out") or job.get("label") or uuid.uuid4().hex)


def run_parallel_image_stage(job_id: str, stage: str, image_jobs: list[dict], workers: int) -> list[dict]:
    if not image_jobs:
        return []
    batch_size = min(IMAGE_BATCH_SIZE, max(1, workers), len(image_jobs))
    emit(
        job_id,
        "progress",
        f"{stage}：开始统一生图，共 {len(image_jobs)} 张；每批最多 {batch_size} 张，批次间隔 {IMAGE_BATCH_INTERVAL_SECONDS} 秒。遇到 429 会先跑完其他图片，再集中补跑。",
    )
    ordered_keys = [image_job_key(job) for job in image_jobs]
    latest_results: dict[str, dict] = {}

    def run_attempt_round(round_jobs: list[dict], round_index: int) -> list[dict]:
        round_results = []
        done = 0
        total_batches = (len(round_jobs) + batch_size - 1) // batch_size
        round_name = "首轮" if round_index == 0 else f"429 补跑第 {round_index}/{IMAGE_RETRY_ROUNDS} 轮"
        for batch_index in range(total_batches):
            ensure_not_cancelled(job_id)
            start = batch_index * batch_size
            batch_jobs = round_jobs[start:start + batch_size]
            batch_started = time.time()
            emit(job_id, "progress", f"{stage}：{round_name}开始第 {batch_index + 1}/{total_batches} 批，本批 {len(batch_jobs)} 张。", slot=f"image_{stage}")
            last_heartbeat = time.time()
            stage_started = time.time()
            stall_hint_sent = False
            with concurrent.futures.ThreadPoolExecutor(max_workers=len(batch_jobs)) as ex:
                pending = {}
                for job in batch_jobs:
                    job_started = time.time()
                    fut = ex.submit(run_image, job["prompt"], job["out"], job.get("refs", []))
                    pending[fut] = {**job, "_metric_started": job_started}
                while pending:
                    ensure_not_cancelled(job_id)
                    finished, _ = concurrent.futures.wait(pending, timeout=15, return_when=concurrent.futures.FIRST_COMPLETED)
                    if not finished:
                        now = time.time()
                        if now - last_heartbeat >= 15:
                            elapsed = int(now - batch_started)
                            if not stall_hint_sent and now - stage_started >= IMAGE_STALL_HINT_SECONDS:
                                msg = f"{stage} {round_name} {done}/{len(round_jobs)}：本批还有 {len(pending)} 张较慢，已等待 {elapsed} 秒，可能在排队/429 限流。"
                                stall_hint_sent = True
                            else:
                                msg = f"{stage} {round_name} {done}/{len(round_jobs)}：本批还有 {len(pending)} 张在等接口返回，已等待 {elapsed} 秒。"
                            emit(job_id, "progress", msg, slot=f"image_{stage}")
                            last_heartbeat = now
                        continue
                    for fut in finished:
                        job = pending.pop(fut)
                        done += 1
                        finished_at = time.time()
                        ok, msg = fut.result()
                        elapsed = round(finished_at - float(job.get("_metric_started") or finished_at), 1)
                        res = {k: v for k, v in job.items() if not str(k).startswith("_metric_")}
                        res = {**res, "ok": ok, "message": msg, "attempt_round": round_index, "duration_seconds": elapsed}
                        round_results.append(res)
                        latest_results[image_job_key(job)] = res
                        append_image_metric(
                            job_id,
                            "item",
                            stage=stage,
                            label=job.get("label"),
                            out=str(job.get("out") or ""),
                            attempt_round=round_index,
                            batch_index=batch_index + 1,
                            ok=bool(ok),
                            message=str(msg),
                            duration_seconds=elapsed,
                            retryable=is_retryable_image_error(msg),
                            is_final=False,
                        )
                        retry_note = "；已加入 429 补跑队列" if (not ok and is_retryable_image_error(msg) and round_index < IMAGE_RETRY_ROUNDS) else ""
                        emit(job_id, "progress", f"{stage} {round_name} {done}/{len(round_jobs)}：{job['label']}：{msg}，耗时 {elapsed} 秒{retry_note}", ok=ok, slot=f"image_{stage}")
                        last_heartbeat = time.time()
            batch_elapsed = round(time.time() - batch_started, 1)
            batch_round_results = round_results[-len(batch_jobs):]
            append_image_metric(
                job_id,
                "batch",
                stage=stage,
                attempt_round=round_index,
                batch_index=batch_index + 1,
                total_batches=total_batches,
                image_count=len(batch_jobs),
                success_count=sum(1 for x in batch_round_results if x.get("ok")),
                retryable_failure_count=sum(1 for x in batch_round_results if (not x.get("ok") and is_retryable_image_error(x.get("message", "")))),
                duration_seconds=batch_elapsed,
            )
            emit(job_id, "progress", f"{stage}：{round_name}第 {batch_index + 1}/{total_batches} 批结束，耗时 {batch_elapsed} 秒；成功 {sum(1 for x in batch_round_results if x.get('ok'))}/{len(batch_jobs)}。", slot=f"image_{stage}")
            if batch_index < total_batches - 1 and IMAGE_BATCH_INTERVAL_SECONDS:
                emit(job_id, "progress", f"{stage}：{round_name}第 {batch_index + 1}/{total_batches} 批完成，等待 {IMAGE_BATCH_INTERVAL_SECONDS} 秒后继续下一批。", slot=f"image_{stage}")
                time.sleep(IMAGE_BATCH_INTERVAL_SECONDS)
        return round_results

    current_jobs = list(image_jobs)
    for round_index in range(0, IMAGE_RETRY_ROUNDS + 1):
        if round_index > 0:
            if not current_jobs:
                break
            if IMAGE_RETRY_DELAY_SECONDS:
                emit(job_id, "progress", f"{stage}：上一轮有 {len(current_jobs)} 张遇到限流/超时，等待 {IMAGE_RETRY_DELAY_SECONDS} 秒后统一补跑。", slot=f"image_{stage}")
                time.sleep(IMAGE_RETRY_DELAY_SECONDS)
            else:
                emit(job_id, "progress", f"{stage}：上一轮有 {len(current_jobs)} 张遇到限流/超时，开始统一补跑。", slot=f"image_{stage}")
        round_results = run_attempt_round(current_jobs, round_index)
        retry_jobs = []
        for res in round_results:
            if res.get("ok"):
                continue
            if round_index < IMAGE_RETRY_ROUNDS and is_retryable_image_error(res.get("message", "")):
                retry_jobs.append(res)
        current_jobs = retry_jobs

    final_results = [latest_results[key] for key in ordered_keys if key in latest_results]
    for res in final_results:
        append_image_metric(
            job_id,
            "item",
            stage=stage,
            label=res.get("label"),
            out=str(res.get("out") or ""),
            attempt_round=int(res.get("attempt_round") or 0),
            ok=bool(res.get("ok")),
            message=str(res.get("message") or ""),
            duration_seconds=float(res.get("duration_seconds") or 0),
            retryable=is_retryable_image_error(res.get("message", "")),
            is_final=True,
        )
    remaining_retryable = [res for res in final_results if not res.get("ok") and is_retryable_image_error(res.get("message", ""))]
    if remaining_retryable:
        emit(job_id, "progress", f"{stage}：补跑结束，仍有 {len(remaining_retryable)} 张因限流/超时失败，本轮先返回其他成功图片。", ok=False, slot=f"image_{stage}")
    return final_results

def process_items_pipeline(job_id: str, items: list[dict], count: int, mode: str, job_dir: Path, images: list[dict]) -> None:
    count = BATCH_COUNT
    ensure_job_dirs(job_dir)
    prompt_map = generate_prompts_for_items(items, count, job_id, mode)
    for item in items:
        row = item["row"]
        emit(job_id, "progress", f"\u5df2\u5339\u914d\u8bd7\u6b4c\uff1a{compact(row.get('\u8bd7\u4eba'))}\u300a{compact(row.get('\u8bd7\u540d'))}\u300b\u3002\u5f53\u524d\u6a21\u5f0f\uff1a{mode}\u3002", out_dir=str(job_dir))
    for idx, item in enumerate(items):
        save_item_prompts(job_dir, item, prompt_map[idx])
    emit(job_id, "progress", "\u6240\u6709\u63d0\u793a\u8bcd\u5df2\u4fdd\u5b58\uff0c\u5f00\u59cb\u6309\u7c7b\u522b\u7edf\u4e00\u751f\u56fe\u3002")

    bg_jobs = []
    for _, item, row, prompt_item, variant in iter_prompt_entries(items, prompt_map):
        paths = item_output_paths(job_dir, item, variant)
        bg_jobs.append({
            "item": item,
            "row": row,
            "variant": variant,
            "prompt_item": prompt_item,
            "prompt": prompt_item.get("background_prompt") or "",
            "out": paths["bg"],
            "label": f"{item_label(item)} \u5b8c\u6574\u9875\u9762\u56fe {variant}/{count}",
        })
    bg_results = run_parallel_image_stage(job_id, "\u5b8c\u6574\u9875\u9762\u56fe", bg_jobs, IMAGE_WORKERS)
    bg_ok = {(id(res["item"]), res["variant"]): res for res in bg_results if res["ok"]}
    if mode == "background":
        preview_count = 0
        for res in bg_results:
            if res["ok"]:
                img = public_image(job_id, res["out"])
                images.append(img)
                if preview_count < 6:
                    emit(job_id, "image", f"{res['label']} \u5b8c\u6210", image=img)
                    preview_count += 1
        return

    person_jobs = []
    pending_no_person = []
    for _, item, row, prompt_item, variant in iter_prompt_entries(items, prompt_map):
        if (id(item), variant) not in bg_ok:
            continue
        paths = item_output_paths(job_dir, item, variant)
        target_type = prompt_item.get("target_person_type") or "poet"
        person_prompt = prompt_item.get("person_prompt") or ""
        if target_type == "no_person" or not person_prompt:
            pending_no_person.append((item, variant, paths))
            continue
        refs = []
        if target_type == "poet":
            face = Path(item.get("face") or "") if item.get("face") else None
            if not face or not face.exists():
                img = public_image(job_id, paths["bg"])
                images.append(img)
                emit(job_id, "image", f"{item_label(item)} \u80cc\u666f\u56fe {variant} \u5b8c\u6210\uff1a\u7f3a\u5c11\u4eba\u7269\u8138\u56fe", image=img)
                continue
            refs = [face]
            person_prompt = enforce_face_reference_prompt(person_prompt)
        person_jobs.append({
            "item": item,
            "row": row,
            "variant": variant,
            "prompt_item": prompt_item,
            "prompt": person_prompt,
            "out": paths["person"],
            "refs": refs,
            "label": f"{item_label(item)} \u4eba\u7269\u56fe {variant}/{count}",
        })
    person_results = run_parallel_image_stage(job_id, "\u4eba\u7269\u56fe", person_jobs, IMAGE_WORKERS)
    person_ok = {(id(res["item"]), res["variant"]): res for res in person_results if res["ok"]}
    if mode == "person":
        preview_count = 0
        for res in person_results:
            if res["ok"]:
                img = public_image(job_id, res["out"])
                images.append(img)
                if preview_count < 6:
                    emit(job_id, "image", f"{res['label']} \u5b8c\u6210", image=img)
                    preview_count += 1
        return

    for item, variant, paths in pending_no_person:
        shutil.copyfile(paths["bg"], paths["comp"])
        img = public_image(job_id, paths["comp"])
        images.append(img)
        emit(job_id, "image", f"{item_label(item)} \u5b8c\u6574\u9875\u9762 {variant} \u5b8c\u6210\uff1a\u76f4\u63a5\u751f\u6210\u5b8c\u6574\u9875\u9762\u56fe\u3002", image=img)

    comp_jobs = []
    for _, item, row, prompt_item, variant in iter_prompt_entries(items, prompt_map):
        if (id(item), variant) not in bg_ok or (id(item), variant) not in person_ok:
            continue
        paths = item_output_paths(job_dir, item, variant)
        comp_jobs.append({
            "item": item,
            "row": row,
            "variant": variant,
            "prompt_item": prompt_item,
            "prompt": prompt_item.get("composite_instruction") or "",
            "out": paths["comp"],
            "refs": [paths["bg"], paths["person"]],
            "label": f"{item_label(item)} \u5408\u6210\u56fe {variant}/{count}",
        })
    comp_results = run_parallel_image_stage(job_id, "\u5408\u6210\u56fe", comp_jobs, IMAGE_WORKERS)
    preview_count = 0
    for res in comp_results:
        if res["ok"]:
            img = public_image(job_id, res["out"])
            images.append(img)
            if preview_count < 6:
                emit(job_id, "image", f"{item_label(res['item'])} \u5b8c\u6574\u9875\u9762 {res['variant']} \u5b8c\u6210", image=img)
                preview_count += 1





def default_remove_person_prompt() -> str:
    return "\u57fa\u4e8e\u8fd9\u5f20\u5df2\u751f\u6210\u7684\u5b8c\u6574\u56fe\u8fdb\u884c\u4e8c\u6b21\u7f16\u8f91\uff1a\u5220\u9664\u753b\u9762\u4e2d\u7684\u6240\u6709\u4eba\u7269\u548c\u4eba\u7269\u5f71\u5b50\uff0c\u4fdd\u7559\u539f\u6709\u6784\u56fe\u3001\u666f\u522b\u3001\u5149\u5f71\u3001\u8272\u5f69\u548c\u98ce\u683c\uff1b\u7528\u5408\u7406\u7684\u5730\u9762\u3001\u5899\u9762\u3001\u6811\u6728\u3001\u5c71\u77f3\u6216\u5ba4\u5185\u9648\u8bbe\u81ea\u7136\u8865\u5168\u88ab\u5220\u9664\u533a\u57df\uff0c\u4e0d\u8981\u65b0\u589e\u4efb\u4f55\u4eba\u7269\u3002"


def default_remove_poem_text_prompt() -> str:
    return "\u57fa\u4e8e\u8fd9\u5f20\u5df2\u751f\u6210\u7684\u5b8c\u6574\u56fe\u8fdb\u884c\u4e8c\u6b21\u7f16\u8f91\uff1a\u5220\u9664\u753b\u9762\u4e2d\u7684\u6240\u6709\u8bd7\u8bcd\u6587\u5b57\u3001\u9898\u5b57\u3001\u5370\u7ae0\u5f0f\u6587\u5b57\u3001\u6c34\u5370\u548c\u4efb\u4f55\u53ef\u8bfb\u6587\u5b57\uff1b\u4fdd\u7559\u539f\u672c\u7559\u767d\u533a\u7684\u5e72\u51c0\u611f\u548c\u6784\u56fe\u547c\u5438\u611f\uff0c\u4e0d\u8981\u7528\u5927\u5757\u7269\u4ef6\u5835\u6ee1\u7559\u767d\u3002"


def build_edit_prompt(prompt_item: dict, options: dict) -> str:
    parts = []
    if options.get("needs_person_removal"):
        parts.append(prompt_item.get("remove_person_prompt") or default_remove_person_prompt())
    if options.get("needs_poem_text_removal"):
        parts.append(prompt_item.get("remove_poem_text_prompt") or default_remove_poem_text_prompt())
    return "\n\n".join(parts)


def process_items_onepot_pipeline(job_id: str, items: list[dict], count: int, mode: str, job_dir: Path, images: list[dict]) -> None:
    ensure_job_dirs(job_dir)
    prompt_map = generate_prompts_for_items(items, count, job_id, mode)
    for idx, item in enumerate(items):
        save_item_prompts(job_dir, item, prompt_map[idx])
    emit(job_id, "progress", "\u63d0\u793a\u8bcd\u5df2\u4fdd\u5b58\uff0c\u5f00\u59cb\u751f\u6210\u5b8c\u6574\u9875\u9762\u56fe\u3002")
    image_jobs = []
    for _, item, row, prompt_item, variant in iter_prompt_entries(items, prompt_map):
        paths = item_output_paths(job_dir, item, variant)
        complete_prompt = prompt_item.get("complete_page_prompt") or ""
        reference_note = reference_prompt_note(row, prompt_item)
        if reference_note:
            complete_prompt = complete_prompt.rstrip() + "\n\n" + reference_note
        need_poet = bool((prompt_item.get("plan") or {}).get("need_poet"))
        refs = []
        if need_poet:
            face = Path(item.get("face") or "") if item.get("face") else None
            if face and face.exists():
                refs.append(face)
        refs.extend(reference_paths_for_prompt(row, prompt_item))
        image_jobs.append({
            "item": item,
            "row": row,
            "variant": variant,
            "prompt_item": prompt_item,
            "prompt": complete_prompt,
            "out": paths["comp"],
            "refs": refs,
            "label": f"{item_label(item)} \u5b8c\u6574\u9875\u9762\u56fe {variant}/{count}",
        })
    results = run_parallel_image_stage(job_id, "\u5b8c\u6574\u9875\u9762\u56fe", image_jobs, IMAGE_WORKERS)
    edit_jobs = []
    for res in results:
        if not res["ok"]:
            continue
        options = res["item"].get("web_options") or {}
        edit_prompt = build_edit_prompt(res["prompt_item"], options)
        if not edit_prompt:
            img = public_image(job_id, res["out"])
            images.append(img)
            emit(job_id, "image", f"{res['label']} \u5b8c\u6210", image=img)
            continue
        paths = item_output_paths(job_dir, res["item"], res["variant"])
        edit_jobs.append({
            "item": res["item"],
            "row": res["row"],
            "variant": res["variant"],
            "prompt_item": res["prompt_item"],
            "prompt": edit_prompt,
            "out": paths["final"],
            "refs": [res["out"]],
            "label": f"{item_label(res['item'])} \u6700\u7ec8\u56fe\u4e8c\u6b21\u7f16\u8f91 {res['variant']}/{count}",
        })
    if edit_jobs:
        emit(job_id, "progress", f"\u9700\u8981\u53bb\u4eba\u7269/\u53bb\u8bd7\u8bcd\u6587\u5b57\u7684\u56fe\u5df2\u8fdb\u5165\u4e8c\u6b21\u7f16\u8f91\uff0c\u5171 {len(edit_jobs)} \u5f20\u3002")
        edit_results = run_parallel_image_stage(job_id, "\u6700\u7ec8\u56fe\u4e8c\u6b21\u7f16\u8f91", edit_jobs, max(1, min(IMAGE_WORKERS, len(edit_jobs))))
        for res in edit_results:
            if res["ok"]:
                img = public_image(job_id, res["out"])
                images.append(img)
                emit(job_id, "image", f"{res['label']} \u5b8c\u6210", image=img)
            else:
                fallback = public_image(job_id, res["refs"][0]) if res.get("refs") else None
                if fallback:
                    images.append(fallback)
                    emit(job_id, "image", f"{res['label']} \u5931\u8d25\uff0c\u5df2\u6682\u65f6\u8fd4\u56de\u672a\u4e8c\u6b21\u7f16\u8f91\u7684\u5b8c\u6574\u56fe", image=fallback, ok=False)


def run_job(job_id: str, user_text: str, request_data: dict | None = None) -> None:
    images = []
    expected_images = 0
    try:
        request_data = request_data or {}
        selected_poems = selected_poems_from_request(request_data)
        web_options = web_options_from_request(request_data) if request_data else {}

        if selected_poems:
            ensure_not_cancelled(job_id)
            count = int(web_options.get("image_count") or BATCH_COUNT)
            mode = "complete"
            items = [create_item_from_poem(poem, web_options, user_text, index=i + 1) for i, poem in enumerate(selected_poems)]
            expected_images = len(items) * count
            jobs[job_id]["expected_images"] = expected_images
            title_summary = f"{len(items)}\u9996\u8bd7"
            job_dir = TASK_OUTPUTS_DIR / f"{time.strftime('%Y%m%d_%H%M%S')}_{safe_name(title_summary)}_{job_id[:6]}"
            emit(job_id, "progress", f"\u5df2\u8bfb\u53d6\u7f51\u9875\u9009\u62e9\uff1a{web_options.get('grade_label')}\uff0c{len(items)} \u9996\u8bd7\uff0c\u6bcf\u9996 {count} \u5f20\uff0c\u603b\u8ba1 {len(items) * count} \u5f20\u3002\u4e0d\u518d\u8ba9 Codex \u731c\u6d4b\u5b66\u6bb5/\u6570\u91cf/\u6bd4\u4f8b/\u4eba\u7269/\u8bd7\u8bcd\u5b57\u6bb5\u3002", out_dir=str(job_dir))
            process_items_onepot_pipeline(job_id, items, count, mode, job_dir, images)
        else:
            emit(job_id, "progress", "\u672a\u6536\u5230\u7f51\u9875\u9009\u8bd7\uff0c\u8fdb\u5165\u65e7\u7684\u81ea\u7531\u6587\u672c\u515c\u5e95\u6d41\u7a0b\u3002")
            task, parse_note = parse_with_agent(user_text)
            emit(job_id, "progress", parse_note, task=task)
            ensure_not_cancelled(job_id)
            count = max(1, min(int(task.get("count") or BATCH_COUNT), 12))
            mode = "complete"
            expected_images = count
            jobs[job_id]["expected_images"] = expected_images
            all_poems = bool(task.get("all_poems")) or ("\u6279\u91cf" in user_text)
            if all_poems:
                raise RuntimeError("\u5f53\u524d\u4e3b\u6d41\u6d41\u7a0b\u5df2\u5207\u6362\u4e3a\u7f51\u9875\u52fe\u9009\u8bd7\u6b4c\u3002\u8bf7\u5728\u524d\u7aef\u9009\u62e9\u8981\u751f\u56fe\u7684\u8bd7\u6b4c\u540e\u518d\u5f00\u59cb\u751f\u6210\u3002")
            if not task.get("title"):
                raise RuntimeError("\u6ca1\u6709\u8bc6\u522b\u5230\u8bd7\u540d\u3002\u8bf7\u5728\u524d\u7aef\u9009\u62e9\u8bd7\u6b4c\uff0c\u6216\u5728\u6587\u672c\u91cc\u5e26\u4e0a\u300a\u8bd7\u540d\u300b\u3002")
            item = create_item_from_task(task, user_text)
            row = item["row"]
            poet = compact(row.get("\u8bd7\u4eba")) or "\u672a\u7f72\u540d"
            title = compact(row.get("\u8bd7\u540d")) or "\u672a\u547d\u540d\u4efb\u52a1"
            job_dir = TASK_OUTPUTS_DIR / f"{time.strftime('%Y%m%d_%H%M%S')}_{safe_name(poet)}_{safe_name(title)}_{job_id[:6]}"
            emit(job_id, "progress", f"\u65e7\u515c\u5e95\u6d41\u7a0b\uff1a{poet}\u300a{title}\u300b\uff0c\u751f\u6210\u6570\u91cf\uff1a{count}\u3002", out_dir=str(job_dir))
            process_items_onepot_pipeline(job_id, [item], count, mode, job_dir, images)

        ensure_not_cancelled(job_id)
        jobs[job_id]["download"] = f"/api/download?job={job_id}"
        jobs[job_id]["save_url"] = f"/api/save-results"
        jobs[job_id]["job_dir"] = str(job_dir) if jobs[job_id].get("save_outputs", True) else ""
        if jobs[job_id].get("save_outputs", True):
            zip_base = PACKAGE_OUTPUTS_DIR / job_dir.name
            zip_path = shutil.make_archive(str(zip_base), "zip", job_dir)
            jobs[job_id]["zip"] = zip_path
            out_dir = str(job_dir)
        else:
            jobs[job_id]["zip"] = ""
            shutil.rmtree(job_dir, ignore_errors=True)
            out_dir = "临时预览，尚未保存到本地文件夹"
        duration_seconds = round(time.time() - float(jobs[job_id].get("created") or time.time()), 1)
        success_count = len(images)
        target_count = int(jobs[job_id].get("expected_images") or expected_images or success_count)
        failed_count = max(0, target_count - success_count)
        if target_count and success_count >= target_count:
            done_title = "全部成功"
            done_message = f"任务完成：目标 {target_count} 张，成功输出 {success_count} 张。"
            partial = False
        elif success_count > 0:
            done_title = "部分完成"
            done_message = f"任务部分完成：目标 {target_count} 张，成功输出 {success_count} 张，失败或未返回 {failed_count} 张。"
            partial = True
        else:
            done_title = "生成失败"
            done_message = f"任务结束：目标 {target_count} 张，成功输出 0 张，请查看失败日志。"
            partial = True
        write_job_summary(
            job_id,
            "done",
            images=images,
            target_count=target_count,
            success_count=success_count,
            failed_count=failed_count,
            partial=partial,
            title=done_title,
            out_dir=out_dir,
        )
        emit(
            job_id,
            "done",
            done_message,
            download=jobs[job_id]["download"],
            save_url=jobs[job_id]["save_url"],
            out_dir=out_dir,
            success_count=success_count,
            target_count=target_count,
            failed_count=failed_count,
            partial=partial,
            duration_seconds=duration_seconds,
            title=done_title,
        )
    except InterruptedError:
        write_job_summary(job_id, "cancelled", images=images, failed_count=max(0, int(jobs[job_id].get("expected_images") or expected_images or 0) - len(images)))
        emit(job_id, "cancelled", "\u5df2\u505c\u6b62\uff1a\u540e\u7eed\u6b65\u9aa4\u4e0d\u4f1a\u7ee7\u7eed\u6267\u884c\u3002")
    except Exception as exc:
        write_job_summary(job_id, "error", images=images, error=str(exc), failed_count=max(0, int(jobs[job_id].get("expected_images") or expected_images or 0) - len(images)))
        emit(job_id, "error", f"\u4efb\u52a1\u5931\u8d25\uff1a{exc}", ok=False)


def build_web_request_text(data: dict) -> str:
    selected_poems = selected_poems_from_request(data)
    options = web_options_from_request(data)
    extra = options.get("extra_requirement") or ""
    if selected_poems:
        poem_names = "\u3001".join([f"{p.get('author', '')}\u300a{p.get('title', '')}\u300b" for p in selected_poems[:8]])
        if len(selected_poems) > 8:
            poem_names += f"\u7b49{len(selected_poems)}\u9996"
        return (
            f"\u7f51\u9875\u7ed3\u6784\u5316\u4efb\u52a1\uff1a{options.get('grade_label')}\u5b66\u6bb5\uff0c"
            f"\u6bd4\u4f8b{options.get('aspect_ratio')}\uff0c\u6bcf\u9996{options.get('image_count')}\u5f20\uff0c"
            f"\u8bd7\u6b4c\uff1a{poem_names}\u3002\u8865\u5145\u8981\u6c42\uff1a{extra or '\u65e0'}"
        )
    user_description = str(data.get("user_description") or data.get("extra_requirement") or "").strip()
    if not user_description:
        raise ValueError("\u8bf7\u5148\u9009\u62e9\u8981\u751f\u6210\u7684\u8bd7\u6b4c\uff0c\u6216\u586b\u5199\u8865\u5145\u8981\u6c42")
    return user_description
class Handler(BaseHTTPRequestHandler):
    def log_message(self, fmt, *args):
        return

    def send_json(self, data: dict, status: int = 200) -> None:
        body = json.dumps(data, ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self) -> None:
        parsed = urlparse(self.path)
        if parsed.path == "/":
            self.serve_static("index.html")
        elif parsed.path.startswith("/static/"):
            self.serve_static(parsed.path[len("/static/") :])
        elif parsed.path == "/api/events":
            self.serve_events(parse_qs(parsed.query).get("job", [""])[0])
        elif parsed.path == "/api/file":
            self.serve_file(parse_qs(parsed.query))
        elif parsed.path == "/api/image":
            self.serve_memory_image(parse_qs(parsed.query))
        elif parsed.path == "/api/download":
            self.serve_download(parse_qs(parsed.query).get("job", [""])[0])
        else:
            self.send_error(404)

    def do_POST(self) -> None:
        parsed = urlparse(self.path)
        length = int(self.headers.get("Content-Length", "0"))
        data = json.loads(self.rfile.read(length).decode("utf-8") or "{}")
        if parsed.path in {"/api/jobs", "/api/generate"}:
            try:
                text = data.get("message") or build_web_request_text(data)
            except ValueError as exc:
                self.send_json({"error": str(exc)}, 400)
                return
            job_id = uuid.uuid4().hex
            jobs[job_id] = {
                "id": job_id,
                "message": text,
                "request": data,
                "events": [],
                "created": time.time(),
                "cancelled": False,
                "save_outputs": bool(data.get("save_outputs", False)),
                "image_store": {},
            }
            events[job_id] = queue.Queue()
            threading.Thread(target=run_job, args=(job_id, text, data), daemon=True).start()
            self.send_json({"job_id": job_id, "normalized_message": text})
        elif parsed.path == "/api/save-results":
            self.save_results(data.get("job_id") or "")
        elif parsed.path == "/api/cancel":
            job_id = data.get("job_id") or ""
            if job_id in jobs:
                jobs[job_id]["cancelled"] = True
                emit(job_id, "progress", "已收到停止请求，正在结束当前步骤。")
                self.send_json({"ok": True})
            else:
                self.send_json({"ok": False, "error": "job not found"}, 404)
        else:
            self.send_error(404)
    def serve_static(self, name: str) -> None:
        path = (STATIC / name).resolve()
        if not str(path).startswith(str(STATIC.resolve())) or not path.exists():
            self.send_error(404)
            return
        suffix = path.suffix.lower()
        ctype = {
            ".html": "text/html; charset=utf-8",
            ".css": "text/css; charset=utf-8",
            ".js": "application/javascript; charset=utf-8",
            ".png": "image/png",
            ".jpg": "image/jpeg",
            ".jpeg": "image/jpeg",
            ".webp": "image/webp",
        }.get(suffix, "application/octet-stream")
        body = path.read_bytes()
        self.send_response(200)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def serve_events(self, job_id: str) -> None:
        if job_id not in events:
            self.send_error(404)
            return
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream; charset=utf-8")
        self.send_header("Cache-Control", "no-cache")
        self.end_headers()
        q = events[job_id]
        while True:
            event = q.get()
            payload = f"data: {json.dumps(event, ensure_ascii=False)}\n\n".encode("utf-8")
            self.wfile.write(payload)
            self.wfile.flush()
            if event.get("kind") == "close":
                break

    def serve_file(self, query: dict) -> None:
        if not query.get("path"):
            self.send_error(404)
            return
        path = unquote_path(query["path"][0])
        if not path.exists() or TASK_OUTPUTS_DIR not in path.parents:
            self.send_error(404)
            return
        body = path.read_bytes()
        self.send_response(200)
        self.send_header("Content-Type", image_content_type(path))
        if query.get("download"):
            safe_download_name = quote(path.name.encode("utf-8"))
            self.send_header("Content-Disposition", f"attachment; filename=ai-image.png; filename*=UTF-8''{safe_download_name}")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def serve_memory_image(self, query: dict) -> None:
        job_id = query.get("job", [""])[0]
        image_id = query.get("id", [""])[0]
        image = jobs.get(job_id, {}).get("image_store", {}).get(image_id)
        if not image:
            self.send_error(404)
            return
        body = image["body"]
        self.send_response(200)
        self.send_header("Content-Type", image.get("content_type") or "image/png")
        if query.get("download"):
            name = image.get("download_name") or short_image_filename(image.get("name") or "image.png")
            safe_download_name = quote(name.encode("utf-8"))
            self.send_header("Content-Disposition", f"attachment; filename=ai-image.png; filename*=UTF-8''{safe_download_name}")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def memory_zip_bytes(self, job_id: str) -> bytes:
        store = jobs.get(job_id, {}).get("image_store", {})
        if not store:
            return b""
        buffer = io.BytesIO()
        with zipfile.ZipFile(buffer, "w", compression=zipfile.ZIP_DEFLATED) as zf:
            used = set()
            for idx, image in enumerate(store.values(), start=1):
                name = image.get("download_name") or short_image_filename(image.get("name") or "image.png", idx)
                if name in used:
                    stem = Path(name).stem
                    suffix = Path(name).suffix or ".png"
                    name = f"{stem}_{idx}{suffix}"
                used.add(name)
                zf.writestr(name, image["body"])
        return buffer.getvalue()

    def serve_download(self, job_id: str) -> None:
        job = jobs.get(job_id, {})
        raw_path = job.get("zip")
        if raw_path:
            zip_path = Path(raw_path)
            if not zip_path.exists() or not zip_path.is_file():
                self.send_error(404)
                return
            body = zip_path.read_bytes()
        else:
            body = self.memory_zip_bytes(job_id)
            if not body:
                self.send_error(404)
                return
        self.send_response(200)
        self.send_header("Content-Type", "application/zip")
        self.send_header("Content-Disposition", 'attachment; filename="ai-book-agent-result.zip"')
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def save_results(self, job_id: str) -> None:
        job = jobs.get(job_id, {})
        store = job.get("image_store", {})
        if not store:
            self.send_json({"ok": False, "error": "没有可保存的临时结果"}, 404)
            return
        save_dir = SAVED_OUTPUTS_DIR / f"{time.strftime('%Y%m%d_%H%M%S')}_{job_id[:6]}"
        save_dir.mkdir(parents=True, exist_ok=True)
        used = set()
        count = 0
        for image in store.values():
            name = image.get("download_name") or short_image_filename(image.get("name") or f"image_{count + 1}.png", count + 1)
            if name in used:
                stem = Path(name).stem
                suffix = Path(name).suffix or ".png"
                name = f"{stem}_{count + 1}{suffix}"
            used.add(name)
            (save_dir / name).write_bytes(image["body"])
            count += 1
        job["saved_dir"] = str(save_dir)
        self.send_json({"ok": True, "count": count, "saved_dir": str(save_dir)})

def main() -> None:
    port = int(os.getenv("PORT", "8765"))
    server = ThreadingHTTPServer(("127.0.0.1", port), Handler)
    print(f"AI Book Agent Demo: http://127.0.0.1:{port}", flush=True)
    server.serve_forever()


if __name__ == "__main__":
    main()






