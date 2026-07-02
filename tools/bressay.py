#!/usr/bin/env python3
# bressay — operate the DANIEL/BRESSAY fine-tuning on Google Colab.
#
# Run FROM YOUR OWN TERMINAL (the colab keep-alive daemon lives under it):
#   python3 tools/bressay.py up   --exp wtconv          # full bootstrap + train
#   python3 tools/bressay.py status --exp wtconv        # anytime, one call
#   python3 tools/bressay.py train/pause/logs/doctor/stop
#
# Design: every VM interaction ships tools/vmctl.py (this checkout's version)
# to the kernel with a dispatch(...) line appended, and parses the single
# "@@BRESSAY@@ {json}" line it prints. No VM state is trusted between calls;
# long work runs detached on the VM and is polled with short calls.
import argparse
import json
import os
import re
import subprocess
import sys
import tempfile
import time

TOOLS_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_DIR = os.path.dirname(TOOLS_DIR)
VMCTL = os.path.join(TOOLS_DIR, "vmctl.py")
IAM = "outputs/daniel_iam_ner_strategy_A_custom_split/checkpoints/best-IAM_NER_165.pt"

EXPERIMENTS = {
    "baseline": {
        "name": "baseline",
        "script": "daniel_bressay_fine_tuning.py",
        "out": "daniel_bressay",
        "drive": "daniel_bressay_ckpts",
        "preflight": None,
    },
    "wtconv": {
        "name": "wtconv",
        "script": "daniel_bressay_wtconv_fine_tuning.py",
        "out": "daniel_bressay_wtconv",
        "drive": "daniel_bressay_wtconv_ckpts",
        # gate: proves transfer learning is intact before any GPU-hour is spent
        "preflight": "tests/test_wtconv_equivalence.py --checkpoint " + IAM,
    },
}


def run(cmd, timeout=300):
    r = subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=timeout)
    return r.returncode, (r.stdout or ""), (r.stderr or "")


def vm(session, command, cfg, timeout=300, retries=3):
    """Ship vmctl.py + dispatch(command, cfg) to the VM; parse the JSON line."""
    src = open(VMCTL).read()
    payload = src + "\n\ndispatch({!r}, json.loads({!r}))\n".format(command, json.dumps(cfg))
    with tempfile.NamedTemporaryFile("w", suffix=".py", delete=False) as f:
        f.write(payload)
        tmp = f.name
    try:
        for attempt in range(1, retries + 1):
            rc, out, err = run("colab exec -s {} -f {}".format(session, tmp), timeout=timeout)
            m = re.search(r"@@BRESSAY@@ (\{.*\})", out)
            if m:
                return json.loads(m.group(1))
            if attempt < retries:
                print("   (vm call '{}' tentativa {} falhou; retry em 8s)".format(command, attempt))
                time.sleep(8)
        sys.exit("ERRO: vm call '{}' falhou apos {} tentativas:\n{}".format(
            command, retries, (out + err).strip()[-600:]))
    finally:
        os.unlink(tmp)


def session_exists(session):
    _, out, _ = run("colab sessions 2>/dev/null")
    return "[{}]".format(session) in out or " {} ".format(session) in out or session in out


def header(msg):
    print("==> " + msg, flush=True)


# ---------------------------------------------------------------- commands

def cmd_up(args, exp):
    header("[1/6] Sessão '{}'".format(args.session))
    if session_exists(args.session):
        print("   já existe — reutilizando (keep-alive do terminal original)")
    else:
        gpu = "" if args.gpu.lower() in ("cpu", "none") else " --gpu " + args.gpu
        rc, out, err = run("colab new -s {}{}".format(args.session, gpu), timeout=300)
        print((out + err).strip()[-200:])
        if rc != 0:
            sys.exit("ERRO: não consegui criar a sessão")
        print("   MANTENHA ESTE TERMINAL ABERTO (keep-alive da sessão).")

    header("[2/6] Google Drive")
    d = vm(args.session, "check_drive", {})
    if d["drive"]:
        print("   montado (assets: {})".format("OK" if d["assets"] else "daniel_assets AUSENTE"))
    elif args.skip_drive:
        print("   NÃO montado — pulando por --skip-drive (checkpoints NÃO serão persistidos)")
    else:
        print("   AUTORIZE o Drive no navegador quando pedir:")
        subprocess.run("colab drivemount -s {}".format(args.session), shell=True)
        d = vm(args.session, "check_drive", {})
        if not d["drive"]:
            sys.exit("ERRO: Drive segue desmontado")

    header("[3/6] Código (fork main) na VM")
    r = vm(args.session, "sync_repo", {})
    print("   {}: {}".format(r["action"], r.get("commit", "?")))
    if not r["ok"]:
        sys.exit("ERRO no clone/reset: " + r.get("detail", ""))

    header("[4/6] Assets + venv (detached)")
    s = vm(args.session, "setup", {})
    print("   lançado: " + ", ".join(s["launched"]))
    while True:
        st = vm(args.session, "setup_status", {})
        flags = " ".join("{}={}".format(k, "OK" if v else "...")
                         for k, v in st.items() if k not in ("ready", "logs"))
        print("   " + flags, flush=True)
        if st["ready"] or (args.skip_drive and st["venv"] and st["weights"] and st["tokenizer"]):
            break
        time.sleep(25)
    print("   ambiente pronto." if st["ready"] else "   pronto (sem dataset — só possível com Drive).")

    header("[5/6] Preflight do experimento '{}'".format(exp["name"]))
    if exp["preflight"]:
        p = vm(args.session, "preflight", exp, timeout=420)
        print("   " + p.get("output", "").strip().splitlines()[-1] if p.get("output") else "")
        if not p["ok"]:
            print(p.get("output", ""))
            sys.exit("ERRO: preflight FALHOU — treino abortado")
        print("   preflight OK")
    else:
        print("   (sem preflight)")

    header("[6/6] Treino")
    if args.no_train:
        print("   pulado (--no-train). Para lançar: bressay.py train --exp " + exp["name"])
    else:
        t = vm(args.session, "train", exp)
        if not t["ok"]:
            sys.exit("ERRO: " + t.get("error", "?"))
        print("   lançado (detached). Log: {} | ckpts no Drive: {}".format(t["log"], t["drive_ckpts"]))
    print("\nPRONTO. Monitorar: python3 tools/bressay.py status --exp {} -s {}".format(exp["name"], args.session))


def cmd_status(args, exp):
    if not session_exists(args.session):
        print("Sessão '{}' NÃO existe (VM morta ou nunca criada).".format(args.session))
        print("Último heartbeat: MyDrive/{}/status.json (visível no app do Drive).".format(exp["drive"]))
        print("Para religar: python3 tools/bressay.py up --exp " + exp["name"])
        return
    st = vm(args.session, "status", exp)
    print("experimento : {}".format(st["exp"]))
    print("treino vivo : {} | sync vivo: {}".format(st["train_alive"], st["sync_alive"]))
    print("GPU         : {}".format(st["gpu"]))
    print("setup       : " + " ".join("{}={}".format(k, "OK" if v else "FALTA")
                                      for k, v in st["setup"].items() if k not in ("ready", "logs")))
    print("ckpts VM    : {}".format(", ".join(st["ckpts_vm"]) or "(nenhum)"))
    print("ckpts Drive : {}".format(", ".join(st["ckpts_drive"]) or "(nenhum)"))
    if st.get("last_train_line"):
        print("último passo: " + st["last_train_line"])
    if st.get("error"):
        print("!! ERRO no train.log: " + st["error"])
    tr = st.get("trend") or {}
    if tr.get("valid_cer"):
        print("\nvalid CER por época:")
        for step, v in tr["valid_cer"]:
            wer = dict(tr.get("valid_wer") or []).get(step, "")
            print("  época {:>4}: cer {:.4f}   wer {}".format(step, v, wer))
    else:
        print("(sem dados de validação ainda)")


def cmd_train(args, exp):
    t = vm(args.session, "train", exp)
    print(json.dumps(t, indent=2) if not t["ok"] else
          "treino lançado. Log: {} | Drive: {}".format(t["log"], t["drive_ckpts"]))


def cmd_pause(args, exp):
    print(json.dumps(vm(args.session, "pause", exp)))


def cmd_logs(args, exp):
    cfg = dict(exp, n=args.n)
    r = vm(args.session, "logs", cfg)
    print(r.get("resume", ""))
    print(r.get("tail", "(vazio)"))


def cmd_stop(args, exp):
    subprocess.run("colab stop -s {}".format(args.session), shell=True)


def cmd_doctor(args, exp):
    print("== local ==")
    rc, out, _ = run("colab sessions 2>&1", timeout=60)
    print("colab auth  : {}".format("OK" if rc == 0 else "FALHOU (gcloud auth application-default login ...)"))
    print("sessão      : {}".format("viva" if session_exists(args.session) else "inexistente"))
    run("git -C {} fetch fork main 2>/dev/null".format(REPO_DIR), timeout=60)
    _, local, _ = run("git -C {} rev-parse main".format(REPO_DIR))
    _, remote, _ = run("git -C {} rev-parse fork/main".format(REPO_DIR))
    print("main==fork  : {}".format("OK" if local.strip() == remote.strip()
                                    else "DIVERGENTE — a VM clona fork/main; faça push!"))
    _, dirty, _ = run("git -C {} status --porcelain".format(REPO_DIR))
    if dirty.strip():
        print("working tree: SUJO ({} arquivos) — mudanças não commitadas não chegam à VM".format(
            len(dirty.strip().splitlines())))
    if session_exists(args.session):
        print("== VM ==")
        d = vm(args.session, "doctor", exp)
        for k in ("drive", "assets", "dataset", "weights", "tokenizer", "venv", "repo_commit", "gpu", "disk"):
            print("{:<12}: {}".format(k, d.get(k)))


def main():
    ap = argparse.ArgumentParser(description="Opera o fine-tuning DANIEL/BRESSAY no Colab")
    ap.add_argument("command", choices=["up", "status", "train", "pause", "logs", "stop", "doctor"])
    ap.add_argument("--exp", default="baseline", choices=sorted(EXPERIMENTS))
    ap.add_argument("-s", "--session", default="bressay")
    ap.add_argument("--gpu", default="A100", help="A100|T4|...|cpu (cpu = sem acelerador)")
    ap.add_argument("--skip-drive", action="store_true", help="segue sem Drive (teste de ambiente; sem persistência)")
    ap.add_argument("--no-train", action="store_true", help="up: prepara tudo mas não lança o treino")
    ap.add_argument("-n", type=int, default=40, help="logs: linhas do tail")
    args = ap.parse_args()
    globals()["cmd_" + args.command](args, EXPERIMENTS[args.exp])


if __name__ == "__main__":
    main()
