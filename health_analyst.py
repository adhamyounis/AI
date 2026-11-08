"""
Health Analyst (Ollama) -- TREND mode
==================================================================
Sends each instance's recent readings (oldest -> newest) plus the previous
verdict to the local Ollama model acting as an experienced SRE, one request per
instance, in a small parallel pool. Per-instance calls succeed or fail
INDEPENDENTLY, so a slow or timed-out instance does not take the whole batch
down -- the fast ones still land.

Every emitted verdict carries the full ISO `timestamp` of its newest reading.
The HH:MM `time` alone is NOT sortable across midnight ("23:40" > "03:30" as a
string), so any consumer ordering or date-filtering these records must use
`timestamp`.

(Batch mode was tried but a single slow generation on CPU-without-AMX timed out
as a unit and lost all 16 rows. Per-instance is more resilient until AMX is
enabled on the host.)

Designed for a downstream portal with two views:
  - "current"  -> health_latest.json        (overwritten each run, the 16 rows)
  - "history"  -> health_assessments.jsonl  (appended each run, 4+ day trail)

Public API
----------
run(records, ollama_conf, run_time=None) -> dict
"""

from __future__ import annotations
import json
import shlex
import urllib.request
import urllib.error
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any

# paramiko carries the SSH transport, used when the inference host is not
# reachable on its HTTP port. It is optional: a direct-HTTP deployment does not
# need it installed.
try:
    import paramiko
except ImportError:
    paramiko = None

# ---------------------------------------------------------------------------
# Defaults
# ---------------------------------------------------------------------------

DEFAULT_OLLAMA = {
    "url":         "http://10.230.67.48:11434/api/generate",
    "model":       "qwen-sre",
    "timeout":     600,       # CPU-only without AMX is slow; generous per-call
    "temperature": 0.3,
    "keep_alive":  "30m",     # stay warm across 5-min cron runs (avoid cold load)
    "num_predict": 512,       # cap output tokens per instance
    "num_ctx":     8192,      # context window (system prompt + one record)
    "max_workers": 4,         # parallel instances; keeps CPU sane on the box

    # --- transport ---------------------------------------------------------
    # "http" talks to `url` directly. "ssh" opens an SSH session to the
    # inference host and runs curl against localhost there, for when the HTTP
    # port is closed between the two machines but port 22 is open. The request
    # body is piped through STDIN rather than embedded in the command, so a
    # large prompt cannot hit the shell argument limit or be mangled by
    # quoting.
    "transport":    "ssh",
    "ssh_host":     "10.230.67.48",
    "ssh_user":     "vmpop",
    "ssh_password": "",        # <-- set this, or leave blank to use an SSH key
    "ssh_url":      "http://localhost:11434/api/generate",
}

# The SRE system prompt (holistic judgement, no fixed thresholds).    ###need to be optimized######
SYSTEM_PROMPT = (
    "You are an experienced SRE assessing one instance holistically using "
    "judgment, no fixed thresholds. instance_pool_connection_pct is HikariCP DB "
    "pool usage (near 100% = request queuing and latency); CONNECT_TIMEOUT and "
    "JDBC are DB exception counts (nonzero = warning sign); cpu/memory fields are "
    "percentages; db_active_sessions_pct is Oracle session load vs a 19500 "
    "ceiling. Correlate signals. Return ONLY JSON: "
    '{"instance":string,"health":"healthy"|"degraded"|"unhealthy",'
    '"assessment":string,"likely_cause":string,"recommended_action":string}'
)

# Context metrics carried into the assessment output so the portal can show the
# verdict AND the numbers behind it without joining two files.
#
# Failure modes are NOT listed here. They used to be the ten fixed buckets, but
# the modes are now named by what actually failed ("Timeout with livechecks")
# and the set differs per window, so they are carried dynamically instead --
# see _carry_failure_modes. Hardcoding the old names here would have made every
# one of them None.
_CARRY_FIELDS = [
    "instance_cpu_pct", "instance_memory_pct", "machine_cpu_pct",
    "machine_memory_pct", "instance_pool_connection_pct", "traffic_last_5min",
    "success_pct", "avg_elapsed_time_ms",
    "db_active_sessions_pct", "db_inactive_sessions_pct",
]

# Everything in a reading that is NOT one of these is a failure mode.
_NON_MODE_FIELDS = set(_CARRY_FIELDS) | {
    "timestamp", "time", "times", "timestamps", "component", "instance",
}


def _post(body: dict, conf: dict) -> dict:
    """
    Send one request to Ollama and return the parsed envelope.

    Two transports, chosen by conf["transport"]:
      "http" -- a direct POST to conf["url"].
      "ssh"  -- an SSH session to the inference host running curl against
                localhost. Used when the HTTP port is closed between the hosts
                but port 22 is open.

    Raises on failure; the caller turns that into health "unknown" so one bad
    instance never takes the batch down.
    """
    data = json.dumps(body).encode("utf-8")
    timeout = conf.get("timeout", 600)

    if conf.get("transport", "http") != "ssh":
        req = urllib.request.Request(
            conf.get("url", DEFAULT_OLLAMA["url"]), data=data, method="POST")
        req.add_header("Content-Type", "application/json")
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return json.loads(resp.read().decode("utf-8"))

    # --- SSH transport -----------------------------------------------------
    if paramiko is None:
        raise RuntimeError("paramiko is not installed; needed for transport=ssh")

    url = conf.get("ssh_url", "http://localhost:11434/api/generate")
    # --data-binary @- reads the body from stdin, so the prompt never touches
    # the command line.
    remote = (f"curl -s --max-time {timeout} -X POST "
              f"-H 'Content-Type: application/json' --data-binary @- "
              f"{shlex.quote(url)}")

    client = paramiko.SSHClient()
    client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    try:
        client.connect(
            conf.get("ssh_host", "10.230.67.48"),
            username=conf.get("ssh_user"),
            password=conf.get("ssh_password") or None,
            timeout=20,
            # force password auth when one is given; fall back to keys if not
            allow_agent=not conf.get("ssh_password"),
            look_for_keys=not conf.get("ssh_password"),
        )
        stdin, stdout, stderr = client.exec_command(remote, timeout=timeout)
        stdin.write(data)
        stdin.flush()
        stdin.channel.shutdown_write()      # tell curl the body is complete
        out = stdout.read().decode("utf-8", "replace")
        err = stderr.read().decode("utf-8", "replace")
    finally:
        try:
            client.close()
        except Exception:
            pass

    if not out.strip():
        raise RuntimeError("empty response from Ollama over SSH"
                           + (f" | stderr: {err.strip()[:160]}" if err.strip() else ""))
    return json.loads(out)


def _series_for(readings: list[dict], field: str) -> list | None:
    """
    The value of one field across every reading, oldest to newest.

    Emitted ALONGSIDE the scalar (as "<field>_series") rather than replacing
    it: the history file already holds days of records whose fields are plain
    numbers, and changing the type would break every chart built on them. A
    consumer uses the series when present and falls back to the scalar when
    not, so old records still render.

    Returns None when no reading carries the field, so an absent metric stays
    absent instead of becoming a list of nulls.
    """
    vals = [r.get(field) for r in readings]
    return vals if any(v is not None for v in vals) else None


def _carry_failure_modes(reading: dict) -> dict:
    """
    The non-zero failure modes from a reading, for the portal to display.

    Analyze.py already strips the zeros before the prompt, so whatever mode
    fields remain are ones that actually occurred -- including any the rules do
    not yet name, which must reach the portal rather than being filtered out by
    a hardcoded list.
    """
    return {k: v for k, v in reading.items()
            if k not in _NON_MODE_FIELDS and not k.startswith("_") and v}

_VERDICT_KEYS = ("health", "assessment", "likely_cause", "recommended_action")


# ---------------------------------------------------------------------------
# Trend-aware assessment (3 timepoints + previous verdict)
# ---------------------------------------------------------------------------

# System prompt for the trend view: the model sees an instance's last N readings
# (oldest -> newest) plus what it concluded last time, and judges the trend.
SYSTEM_PROMPT_TREND = (
    "You are an experienced SRE assessing ONE instance over its last few readings "
    "(oldest to newest) to judge the TREND, not just the latest snapshot.\n"
    "Metrics: instance_pool_connection_pct is HikariCP DB pool usage (near 100% "
    "= request queuing and latency); success_pct is the percent of traffic that "
    "succeeded; cpu/memory fields are percentages; db_active_sessions_pct is "
    "Oracle session load against a 19500 ceiling; traffic_last_5min is the "
    "request count for the window.\n"
    "Any other field is a FAILURE MODE named by what went wrong, valued as a "
    "percentage of this instance's traffic. Only modes that actually occurred "
    "are listed. Read them literally and weigh where the fault lies:\n"
    "- 'Timeout with livechecks', 'Timeout from bigdata side', 'Timeout with "
    "business handlers', 'Failed to connect to business handlers', 'Livechecks "
    "returned HTTP 500' point at a DOWNSTREAM dependency, not at this instance. "
    "If they dominate while cpu, memory and pool are normal, the instance is "
    "healthy and the dependency is at fault.\n"
    "- 'JDBC' is the DB connection pool failing to hand out a transaction. "
    "Treat it together with instance_pool_connection_pct: both high means pool "
    "exhaustion.\n"
    "- 'Customer not in dump (extraction side)', 'Invalid MSISDN parameter', "
    "'MSISDN missing from request' are upstream data or caller problems, not "
    "faults of this instance.\n"
    "- 'Missing cohort configuration', 'Missing communication type in scripts', "
    "'Script variable missing', 'No handlers configured' are configuration "
    "problems needing a change, not a restart.\n"
    "- '400 BAD REQUEST', 'Fulfillment failed: ...', 'Script error', 'Null "
    "pointer exception' are application-level failures in request handling.\n"
    "A low percentage of a business failure (no balance, duplicate renewal) is "
    "normal operation, not a fault.\n"
    "You are also given your PREVIOUS verdict for this instance - use it to note "
    "whether things are improving, worsening, or stable. Correlate signals "
    "across the readings. Return ONLY JSON: "
    '{"instance":string,"health":"healthy"|"degraded"|"unhealthy",'
    '"trend":"improving"|"stable"|"worsening","assessment":string,'
    '"likely_cause":string,"recommended_action":string}'
)

_TREND_VERDICT_KEYS = ("health", "trend", "assessment", "likely_cause", "recommended_action")


def _assess_trend(instance: str, readings: list[dict], prev: dict | None,
                  conf: dict) -> dict:
    """
    Assess one instance from its ordered list of readings (oldest -> newest),
    plus the previous verdict/reason. Never raises; failure -> health 'unknown'.

    readings : list of the SAME instance's records at consecutive timepoints
    prev     : {"health": ..., "reason": ...} from the last analysis, or None
    """
    latest = readings[-1] if readings else {}
    times = [r.get("time") for r in readings]

    # Compact previous-verdict line for the prompt
    if prev and prev.get("health"):
        prev_line = f'{prev.get("health")} ({prev.get("reason") or "no reason recorded"})'
    else:
        prev_line = "none (no prior assessment)"

    prompt = (
        f"Assess this instance's trend over {len(readings)} readings "
        f"(oldest to newest): {json.dumps(readings, ensure_ascii=False)}\n"
        f"Previous verdict: {prev_line}"
    )

    body = {
        "model":   conf.get("model", "qwen-sre"),
        "stream":  False,
        "format":  "json",
        "think":   False,
        "keep_alive": conf.get("keep_alive", "30m"),
        "options": {
            "temperature": conf.get("temperature", 0.3),
            "num_predict": conf.get("num_predict", 512),
            "num_ctx":     conf.get("num_ctx", 8192),
        },
        "system":  SYSTEM_PROMPT_TREND,
        "prompt":  prompt,
    }

    # Skeleton carries the LATEST reading's metrics for the portal
    out: dict[str, Any] = {
        # Full ISO timestamp WITH offset, taken from the newest reading. The
        # HH:MM "time" alone is not sortable across midnight ("23:40" > "03:30"
        # as a string), so anything ordering these records must use this.
        "timestamp": latest.get("timestamp"),
        "time": latest.get("time"),
        "times": times,                 # the 3 timepoint labels this verdict covers
        "timestamps": [r.get("timestamp") for r in readings],
        "instance": instance,
        "health": "unknown",
        "trend": None,
        "assessment": None,
        "likely_cause": None,
        "recommended_action": None,
        "prev_health": (prev or {}).get("health"),
    }
    for f in _CARRY_FIELDS:
        out[f] = latest.get(f)
        series = _series_for(readings, f)
        if series is not None:
            out[f + "_series"] = series
    # the failure modes that actually occurred, so the portal shows what failed
    modes = _carry_failure_modes(latest)
    out.update(modes)
    # ...and their trend across the window, so a mode that is climbing is
    # distinguishable from one that is steady at the same latest value
    for f in modes:
        series = _series_for(readings, f)
        if series is not None:
            # Analyze.py strips zero-valued modes before the prompt, so a
            # reading where this mode did not occur simply lacks the field.
            # That absence MEANS zero, and recording None would read as
            # "unknown" -- hiding how sharply a mode appeared mid-window.
            out[f + "_series"] = [0 if v is None else v for v in series]

    try:
        envelope = _post(body, conf)
        raw = envelope.get("response", "")
        verdict = json.loads(raw) if isinstance(raw, str) else (raw or {})
        for k in _TREND_VERDICT_KEYS:
            if verdict.get(k) is not None:
                out[k] = verdict[k]
        out["instance"] = instance
    except urllib.error.URLError as exc:
        out["assessment"] = f"Ollama unreachable: {exc}"
    except json.JSONDecodeError as exc:
        out["assessment"] = f"Model returned non-JSON: {exc}"
    except Exception as exc:
        out["assessment"] = f"Assessment failed: {exc}"

    return out


def run_trend(grouped: dict[str, list[dict]], prev_state: dict | None = None,
              ollama_conf: dict | None = None) -> dict[str, Any]:
    """
    Assess many instances, each from its own list of timepoint readings, in a
    small parallel pool. Per-instance independent (fast ones land).

    Parameters
    ----------
    grouped    : {instance_id: [reading_oldest, ..., reading_newest]}
    prev_state : {instance_id: {"health":..., "reason":...}} from last run
    ollama_conf: {url, model, timeout, max_workers, ...}

    Returns
    -------
    {status, script, mode:"trend", assessments:[...], error}
    """
    conf = {**DEFAULT_OLLAMA, **(ollama_conf or {})}
    prev_state = prev_state or {}
    result: dict[str, Any] = {
        "status": "ok",
        "script": "health_analyst",
        "mode": "trend",
        "assessments": [],
        "error": None,
    }

    if not grouped:
        result["status"] = "error"
        result["error"] = "no instances to assess"
        return result

    max_workers = min(conf.get("max_workers", 4), len(grouped))
    with ThreadPoolExecutor(max_workers=max_workers) as ex:
        futures = {
            ex.submit(_assess_trend, inst, readings, prev_state.get(inst), conf): inst
            for inst, readings in grouped.items()
        }
        for fut in as_completed(futures):
            result["assessments"].append(fut.result())

    result["assessments"].sort(key=lambda x: x.get("instance", ""))

    ok = sum(1 for a in result["assessments"] if a.get("health") != "unknown")
    total = len(result["assessments"])
    if ok == 0:
        result["status"] = "error"
        result["error"] = "all assessments failed (is Ollama reachable?)"
    elif ok < total:
        result["status"] = "partial"

    return result


# ---------------------------------------------------------------------------
# Single-record call (legacy snapshot mode - kept for compatibility)
# ---------------------------------------------------------------------------

def _assess_one(record: dict, conf: dict, run_time: str | None) -> dict:
    """
    Send one feature-vector record to the model and return a portal-ready
    assessment dict. Never raises -- any failure yields health='unknown'
    with the error in 'assessment', so the run always returns every row.
    """
    instance = record.get("instance", "unknown")
    rec_time = record.get("time", run_time)

    body = {
        "model":   conf.get("model", "qwen-sre"),
        "stream":  False,
        "format":  "json",
        "think":   False,
        "keep_alive": conf.get("keep_alive", "30m"),
        "options": {
            "temperature": conf.get("temperature", 0.3),
            "num_predict": conf.get("num_predict", 512),
            "num_ctx":     conf.get("num_ctx", 8192),
        },
        "system":  SYSTEM_PROMPT,
        "prompt":  "Assess: " + json.dumps(record, ensure_ascii=False),
    }

    # Base output skeleton (filled on success, kept on failure)
    out: dict[str, Any] = {
        "time": rec_time,
        "instance": instance,
        "health": "unknown",
        "assessment": None,
        "likely_cause": None,
        "recommended_action": None,
    }
    for f in _CARRY_FIELDS:
        out[f] = record.get(f)

    try:
        # Ollama returns the model text under "response"; with format=json it
        # is itself a JSON string we parse into the health verdict.
        envelope = _post(body, conf)
        raw = envelope.get("response", "")
        verdict = json.loads(raw) if isinstance(raw, str) else (raw or {})

        for k in _VERDICT_KEYS:
            if verdict.get(k) is not None:
                out[k] = verdict[k]
        # Trust the record's instance id over the model's echo
        out["instance"] = instance

    except urllib.error.URLError as exc:
        out["assessment"] = f"Ollama unreachable: {exc}"
    except json.JSONDecodeError as exc:
        out["assessment"] = f"Model returned non-JSON: {exc}"
    except Exception as exc:
        out["assessment"] = f"Assessment failed: {exc}"

    return out


# ---------------------------------------------------------------------------
# Main callable
# ---------------------------------------------------------------------------

def run(records: list[dict], ollama_conf: dict | None = None,
        run_time: str | None = None) -> dict[str, Any]:
    """
    Assess every record with one request per instance, in a small parallel pool.

    Parameters
    ----------
    records     : the current run's feature-vector rows (e.g. 16)
    ollama_conf : {url, model, timeout, max_workers, temperature, ...}
    run_time    : HH:MM stamp for this batch (falls back to each record's "time")

    Returns
    -------
    {
        "status": "ok" | "partial" | "error",
        "script": "health_analyst",
        "mode":   "per_instance",
        "assessments": [ {time, instance, health, assessment, ...}, ... ],
        "error": null | "<msg>"
    }
    """
    conf = {**DEFAULT_OLLAMA, **(ollama_conf or {})}
    result: dict[str, Any] = {
        "status": "ok",
        "script": "health_analyst",
        "mode": "per_instance",
        "assessments": [],
        "error": None,
    }

    if not records:
        result["status"] = "error"
        result["error"] = "no records to assess"
        return result

    max_workers = min(conf.get("max_workers", 4), len(records))
    with ThreadPoolExecutor(max_workers=max_workers) as ex:
        futures = {
            ex.submit(_assess_one, rec, conf, run_time): rec.get("instance")
            for rec in records
        }
        for fut in as_completed(futures):
            result["assessments"].append(fut.result())

    # Stable order for the portal
    result["assessments"].sort(key=lambda x: x.get("instance", ""))

    # Status rollup: how many produced a real verdict (not 'unknown')
    ok = sum(1 for a in result["assessments"] if a.get("health") != "unknown")
    total = len(result["assessments"])
    if ok == 0:
        result["status"] = "error"
        result["error"] = "all assessments failed (is Ollama running on :11434?)"
    elif ok < total:
        result["status"] = "partial"

    return result