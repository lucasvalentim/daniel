# VM-side agent for the bressay CLI. Not executed directly: bressay.py ships
# this file's source to the VM on every call (with a dispatch(...) line
# appended), so the VM always runs the version checked out locally.
#
# Protocol: dispatch(cmd, cfg) prints exactly one line starting with
# "@@BRESSAY@@ " followed by a JSON object; bressay.py parses that line.
# Long-running work is always launched detached (nohup via intermediate .sh,
# the pattern that survives on Colab VMs) and observed through log markers.
import json
import os
import subprocess

REPO = "/content/daniel"
VENV_PY = "/content/denv/bin/python"
FORK_URL = "https://github.com/lucasvalentim/daniel.git"
WEIGHTS_URL = "https://zenodo.org/records/15846534/files/daniel_pretrained_weights.zip?download=1"
TOKENIZER_URL = "https://zenodo.org/api/records/15846599/files/subwords.zip/content"
IAM_CKPT = REPO + "/outputs/daniel_iam_ner_strategy_A_custom_split/checkpoints/best-IAM_NER_165.pt"
DATASET_DIR = REPO + "/Datasets/formatted/bressay_page/train"
TOKENIZER_DIR = REPO + "/basic/subwords/tokenizer-daniel"
DRIVE = "/content/drive/MyDrive"
ASSETS = DRIVE + "/daniel_assets"


def sh(cmd, timeout=120):
    try:
        r = subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=timeout)
        return r.returncode, (r.stdout or "") + (r.stderr or "")
    except subprocess.TimeoutExpired:
        return 124, "(timeout after {}s)".format(timeout)


def detach(name, body):
    """Write a bash script and launch it detached (nohup + intermediate launcher)."""
    script = "/content/{}.sh".format(name)
    with open(script, "w") as f:
        f.write("#!/bin/bash\n" + body)
    launcher = "/content/launch_{}.sh".format(name)
    with open(launcher, "w") as f:
        f.write("#!/bin/bash\nnohup bash {} >/dev/null 2>&1 &\necho $!\n".format(script))
    _, out = sh("bash " + launcher)
    return out.strip()


def log_has(path, marker):
    return os.path.exists(path) and marker in open(path, errors="replace").read()


def tail(path, n=4):
    if not os.path.exists(path):
        return ""
    with open(path, errors="replace") as f:
        return "\n".join(f.read().splitlines()[-n:])


# ---------------------------------------------------------------- commands

def cmd_check_drive(cfg):
    return {"drive": os.path.isdir(DRIVE), "assets": os.path.isdir(ASSETS)}


def cmd_sync_repo(cfg):
    """Clone the fork's main, or hard-reset an existing clone to origin/main
    (an existing clone is NEVER trusted to be current)."""
    if not os.path.isdir(REPO + "/.git"):
        rc, out = sh("cd /content && rm -rf daniel && git clone --depth 1 -b main {} 2>&1".format(FORK_URL), timeout=300)
        action = "clone"
    else:
        rc, out = sh("cd {} && git fetch --depth 1 origin main && git reset --hard origin/main 2>&1".format(REPO), timeout=300)
        action = "reset"
    _, commit = sh("cd {} && git log --format='%h %s' -1".format(REPO))
    return {"ok": rc == 0, "action": action, "commit": commit.strip(), "detail": out.strip()[-200:]}


def cmd_setup(cfg):
    """Launch (detached, idempotent) whatever is still missing: assets from
    Drive when available, external fallbacks for weights/tokenizer, venv."""
    launched = []
    have_drive = os.path.isdir(ASSETS)

    if not os.path.isdir(DATASET_DIR):
        if have_drive:
            detach("pull_dataset",
                   'LOG=/content/pull_dataset.log\necho "[ds] start" > $LOG\n'
                   'mkdir -p {r}/Datasets/formatted\n'
                   'cp -rf {a}/bressay_page {r}/Datasets/formatted/ >> $LOG 2>&1\n'
                   'echo "[ds] DATASET_DONE $(ls {d} | wc -l) imgs" >> $LOG\n'.format(r=REPO, a=ASSETS, d=DATASET_DIR))
            launched.append("dataset(drive)")
        else:
            launched.append("dataset(SKIPPED: no Drive — mount Drive or upload manually)")

    if not os.path.isfile(IAM_CKPT):
        if have_drive and os.path.isfile(ASSETS + "/iam_weights/checkpoints/best-IAM_NER_165.pt"):
            detach("pull_weights",
                   'LOG=/content/pull_weights.log\necho "[w] start" > $LOG\n'
                   'mkdir -p $(dirname {c})\ncp -f {a}/iam_weights/checkpoints/best-IAM_NER_165.pt {c} >> $LOG 2>&1\n'
                   'echo "[w] WEIGHTS_DONE" >> $LOG\n'.format(c=IAM_CKPT, a=ASSETS))
            launched.append("weights(drive)")
        else:
            with open("/content/extract_w.py", "w") as f:
                f.write("import zipfile\n"
                        "z = zipfile.ZipFile('/content/weights.zip')\n"
                        "pref = 'daniel_iam_ner_strategy_A_custom_split/'\n"
                        "z.extractall('{}/outputs', [m for m in z.namelist() if m.startswith(pref)])\n"
                        "print('EXTRACTED')\n".format(REPO))
            detach("pull_weights",
                   'LOG=/content/pull_weights.log\necho "[w] zenodo start" > $LOG\n'
                   'which aria2c >/dev/null || (apt-get update -q && apt-get install -y -q aria2) >> $LOG 2>&1\n'
                   'cd /content && rm -f weights.zip weights.zip.aria2\n'
                   'aria2c -x16 -s16 -k1M --console-log-level=warn --summary-interval=30 '
                   '--file-allocation=none -o weights.zip "{u}" >> $LOG 2>&1\n'
                   'python3 /content/extract_w.py >> $LOG 2>&1\nrm -f /content/weights.zip\n'
                   'echo "[w] WEIGHTS_DONE" >> $LOG\n'.format(u=WEIGHTS_URL))
            launched.append("weights(zenodo)")

    if not os.path.isdir(TOKENIZER_DIR):
        src = ('cp -rf {a}/subwords/tokenizer-daniel {r}/basic/subwords/ >> $LOG 2>&1\n'
               'cp -f {a}/subwords/replace_dict.pkl {r}/basic/subwords/ >> $LOG 2>&1\n'.format(a=ASSETS, r=REPO)
               if have_drive and os.path.isdir(ASSETS + "/subwords/tokenizer-daniel") else
               'cd {r} && wget -q -O subwords.zip "{u}" >> $LOG 2>&1\n'
               'cd {r} && python3 -c "import zipfile; zipfile.ZipFile(\'subwords.zip\').extractall(\'basic\')" >> $LOG 2>&1\n'
               'rm -f {r}/subwords.zip\n'.format(r=REPO, u=TOKENIZER_URL))
        detach("pull_tokenizer",
               'LOG=/content/pull_tokenizer.log\necho "[tk] start" > $LOG\nmkdir -p {r}/basic/subwords\n'.format(r=REPO)
               + src + 'echo "[tk] TOKENIZER_DONE" >> $LOG\n')
        launched.append("tokenizer")

    if not log_has("/content/venv.log", "VENV_DONE"):
        with open("/content/mk_nerval.py", "w") as f:
            f.write("import glob, os\n"
                    "p = '/content/denv/lib/python3.9/site-packages/nerval'\n"
                    "os.makedirs(p, exist_ok=True)\n"
                    "[os.remove(x) for x in glob.glob(p + '/*.py')]\n"
                    "open(p+'/__init__.py','w').write('# stub\\n')\n"
                    "open(p+'/evaluate.py','w').write('def _u(*a,**k):\\n raise NotImplementedError(\\'nerval stub\\')\\ncompute_matches=get_labels_aligned=compute_scores=_u\\n')\n"
                    "open(p+'/parse.py','w').write('def get_type_label(l):\\n raise NotImplementedError(0)\\ndef get_position_label(l):\\n raise NotImplementedError(0)\\n')\n")
        detach("setup_venv",
               'export PATH="$HOME/.local/bin:$PATH"\nLOG=/content/venv.log\necho "[venv] start $(date)" > $LOG\n'
               'pip install -q uv >> $LOG 2>&1 || true\n'
               '[ -d /content/denv ] || uv venv --python 3.9 /content/denv >> $LOG 2>&1\n'
               'uv pip install --python {p} -r {r}/requirements.txt >> $LOG 2>&1\n'
               'uv pip install --python {p} pyarrow==12.0.1 >> $LOG 2>&1\n'
               '{p} /content/mk_nerval.py >> $LOG 2>&1\n'
               'echo "[venv] VENV_DONE" >> $LOG\n'.format(p=VENV_PY, r=REPO))
        launched.append("venv")

    return {"launched": launched or ["nothing (all present)"], "drive_assets": have_drive}


def cmd_setup_status(cfg):
    st = {
        "dataset": os.path.isdir(DATASET_DIR),
        "weights": os.path.isfile(IAM_CKPT),
        "tokenizer": os.path.isdir(TOKENIZER_DIR),
        "venv": log_has("/content/venv.log", "VENV_DONE") and os.path.isfile(VENV_PY),
    }
    st["ready"] = all(st.values())
    st["logs"] = {n: tail("/content/{}.log".format(n), 1) for n in
                  ("pull_dataset", "pull_weights", "pull_tokenizer", "venv") if os.path.exists("/content/{}.log".format(n))}
    return st


def cmd_preflight(cfg):
    if not cfg.get("preflight"):
        return {"ok": True, "detail": "no preflight for this experiment"}
    rc, out = sh("cd {} && {} {}".format(REPO, VENV_PY, cfg["preflight"]), timeout=240)
    return {"ok": rc == 0, "output": out.strip()[-800:]}


def _paths(cfg):
    out = REPO + "/outputs/" + cfg["out"]
    return {"ck": out + "/checkpoints", "res": out + "/results",
            "drive_ck": DRIVE + "/" + cfg["drive"], "log": "/content/train_{}.log".format(cfg["name"])}


def _pat(s):
    """pgrep/pkill -f pattern that can't match its own sh -c wrapper."""
    return "[{}]{}".format(s[0], s[1:])


def _train_alive(cfg):
    rc, out = sh("pgrep -f '{}'".format(_pat(cfg["script"])))
    return bool(out.strip()) and rc == 0


def cmd_train(cfg):
    if _train_alive(cfg):
        return {"ok": False, "error": "training for '{}' is ALREADY RUNNING (use pause first)".format(cfg["name"])}
    p = _paths(cfg)
    if not os.path.isdir(DRIVE):
        return {"ok": False, "error": "Drive not mounted — checkpoints would not be persisted"}
    # sync daemon with heartbeat (parameterized version of the proven loop)
    detach("drive_sync_" + cfg["name"],
           'DEST="{dck}"\nCKDIR="{ck}"\nRES="{res}"\nmkdir -p "$DEST" "$DEST/results"\n'
           'while true; do\n'
           '  if [ -d "$CKDIR" ]; then\n'
           '    for f in "$CKDIR"/*.pt; do\n'
           '      [ -e "$f" ] || continue\n'
           '      b=$(basename "$f"); now=$(date +%s); mt=$(stat -c %Y "$f")\n'
           '      if [ $((now-mt)) -gt 20 ] && [ "$f" -nt "$DEST/$b" ]; then\n'
           '        cp -f "$f" "$DEST/$b.tmp" && mv -f "$DEST/$b.tmp" "$DEST/$b"\n'
           '      fi\n'
           '    done\n'
           '  fi\n'
           '  cp -f "$RES"/events.out.tfevents.* "$DEST/results/" 2>/dev/null\n'
           '  ALIVE=$(pgrep -f {script} >/dev/null && echo true || echo false)\n'
           '  LINE=$(grep -aoE "EPOCH [0-9]+/[0-9]+:[^|]*cer[^,]*" {log} 2>/dev/null | tail -1 | tr -d \'"\')\n'
           '  printf \'{{"ts": "%s", "train_alive": %s, "last": "%s"}}\\n\' "$(date -u +%FT%TZ)" "$ALIVE" "$LINE" > "$DEST/status.json"\n'
           '  sleep 60\ndone\n'.format(dck=p["drive_ck"], ck=p["ck"], res=p["res"], script=cfg["script"], log=p["log"]))
    # restore newest checkpoints from Drive, wait for IAM weights, launch training
    detach("train_" + cfg["name"],
           'LOG=/content/resume_{name}.log\necho "[resume] start $(date)" > $LOG\n'
           'mkdir -p "{ck}" "{res}"\n'
           'LAST=$(ls -t "{dck}"/last_*.pt 2>/dev/null | head -1)\n'
           'BEST=$(ls -t "{dck}"/best_*.pt 2>/dev/null | head -1)\n'
           '[ -n "$LAST" ] && cp -f "$LAST" "{ck}"/ && echo "[resume] restored $LAST" >> $LOG\n'
           '[ -n "$BEST" ] && cp -f "$BEST" "{ck}"/ && echo "[resume] restored $BEST" >> $LOG\n'
           'cp -f "{dck}"/results/events.out.tfevents.* "{res}"/ 2>/dev/null\n'
           'for i in $(seq 1 120); do [ -f "{iam}" ] && break; sleep 10; done\n'
           'cd {repo}\n'
           'export PYTHONUNBUFFERED=1 TOKENIZERS_PARALLELISM=false HF_HUB_DISABLE_TELEMETRY=1\n'
           'export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True\n'
           'nohup {py} OCR/document_OCR/daniel/custom_dataset/{script} --mode train > {log} 2>&1 &\n'
           'echo "[resume] training pid $!" >> $LOG\necho "[resume] DONE" >> $LOG\n'.format(
               name=cfg["name"], ck=p["ck"], res=p["res"], dck=p["drive_ck"],
               iam=IAM_CKPT, repo=REPO, py=VENV_PY, script=cfg["script"], log=p["log"]))
    return {"ok": True, "launched": True, "log": p["log"], "drive_ckpts": p["drive_ck"]}


def cmd_pause(cfg):
    sh("pkill -f '{}'".format(_pat(cfg["script"])))
    sh("pkill -f '{}'".format(_pat("drive_sync_" + cfg["name"])))
    return {"ok": True, "stopped": [cfg["script"], "drive_sync_" + cfg["name"]]}


def cmd_status(cfg):
    p = _paths(cfg)
    st = {"exp": cfg["name"], "train_alive": _train_alive(cfg)}
    _, sync = sh("pgrep -f '{}'".format(_pat("drive_sync_" + cfg["name"])))
    st["sync_alive"] = bool(sync.strip())
    _, gpu = sh("nvidia-smi --query-gpu=memory.used,memory.total --format=csv,noheader 2>/dev/null")
    st["gpu"] = gpu.strip() or "n/a"
    st["setup"] = cmd_setup_status(cfg)
    for label, d in (("ckpts_vm", p["ck"]), ("ckpts_drive", p["drive_ck"])):
        st[label] = sorted(os.listdir(d)) if os.path.isdir(d) else []
    _, line = sh("grep -aoE 'EPOCH [0-9]+/[0-9]+:[^|]*cer[^,]*' {} 2>/dev/null | tail -1".format(p["log"]))
    st["last_train_line"] = line.strip()
    _, err = sh("grep -aE 'Traceback|OutOfMemory|RuntimeError|Killed' {} 2>/dev/null | tail -1".format(p["log"]))
    st["error"] = err.strip()
    # validation trend from TensorBoard events (via venv python)
    if os.path.isfile(VENV_PY) and os.path.isdir(p["res"]):
        with open("/content/_tb.py", "w") as f:
            f.write("import glob, os, json\n"
                    "from tensorboard.backend.event_processing.event_accumulator import EventAccumulator\n"
                    "fs = sorted(glob.glob('{}/events.out.tfevents.*'), key=os.path.getmtime)\n"
                    "def merged(tag):\n"
                    "    d = {{}}\n"
                    "    for f in fs:\n"
                    "        try:\n"
                    "            a = EventAccumulator(f, size_guidance={{'scalars': 0}}); a.Reload()\n"
                    "            for x in a.Scalars(tag): d[x.step] = round(x.value, 4)\n"
                    "        except KeyError: pass\n"
                    "    return [(k, d[k]) for k in sorted(d)]\n"
                    "print(json.dumps({{'valid_cer': merged('bressay-valid_cer'), 'valid_wer': merged('bressay-valid_wer')}}))\n".format(p["res"]))
        rc, out = sh(VENV_PY + " /content/_tb.py 2>/dev/null", timeout=90)
        try:
            st["trend"] = json.loads(out.strip().splitlines()[-1])
        except Exception:
            st["trend"] = None
    return st


def cmd_logs(cfg):
    return {"tail": tail(_paths(cfg)["log"], int(cfg.get("n", 40))),
            "resume": tail("/content/resume_{}.log".format(cfg["name"]), 4)}


def cmd_doctor(cfg):
    d = cmd_check_drive(cfg)
    d.update(cmd_setup_status(cfg))
    _, commit = sh("cd {} && git log --format='%h %s' -1 2>/dev/null".format(REPO))
    d["repo_commit"] = commit.strip()
    _, disk = sh("df -h /content | tail -1")
    d["disk"] = " ".join(disk.split())
    d["gpu"] = sh("nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null")[1].strip() or "none"
    return d


COMMANDS = {n[4:]: f for n, f in list(globals().items()) if n.startswith("cmd_")}


def dispatch(cmd, cfg):
    try:
        result = COMMANDS[cmd](cfg or {})
    except Exception as e:
        result = {"ok": False, "error": "{}: {}".format(type(e).__name__, e)}
    print("@@BRESSAY@@ " + json.dumps(result))
