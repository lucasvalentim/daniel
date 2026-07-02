# tools/bressay.py — operação do fine-tuning no Colab

CLI único que substitui os scripts soltos de `colab_scripts/` (bootstrap_full,
bootstrap_drive, resume, status): configura a VM de ponta a ponta (Drive,
código, assets, venv), lança/pausa o treino com guard anti-duplicação e
responde o status a qualquer momento em uma chamada.

**Rode do seu próprio terminal** (o keep-alive da sessão vive sob ele):

```bash
# bootstrap completo + treino (idempotente; re-rode após qualquer queda)
python3 tools/bressay.py up --exp wtconv

# status a qualquer momento (setup, processos, GPU, ckpts, tendência de CER)
python3 tools/bressay.py status --exp wtconv

# demais
python3 tools/bressay.py train  --exp wtconv   # (re)lança só o treino
python3 tools/bressay.py pause  --exp wtconv   # para treino+sync, mantém a VM
python3 tools/bressay.py logs   --exp wtconv -n 60
python3 tools/bressay.py doctor --exp wtconv   # diagnóstico local+VM
python3 tools/bressay.py stop                  # encerra a VM (billable!)
```

## Experimentos

Definidos no manifesto `EXPERIMENTS` em `bressay.py` — script de treino, pasta
de outputs, pasta de checkpoints no Drive e preflight opcional:

| exp | script | ckpts no Drive | preflight |
|---|---|---|---|
| `baseline` | `daniel_bressay_fine_tuning.py` | `MyDrive/daniel_bressay_ckpts` | — |
| `wtconv` | `daniel_bressay_wtconv_fine_tuning.py` | `MyDrive/daniel_bressay_wtconv_ckpts` | teste de equivalência (aborta o treino se o transfer learning quebrar) |

Experimento novo = uma entrada nova no dicionário.

## Como funciona

- `bressay.py` (local) envia `vmctl.py` **desta checkout** à VM em cada chamada
  (com uma linha `dispatch(...)` no fim) e lê a única linha `@@BRESSAY@@ {json}`
  que ele imprime — nada de grep em logs espalhados, nenhum estado confiado
  entre chamadas.
- Trabalho pesado (assets, venv, treino) roda **detached** na VM (nohup via
  launcher intermediário, o padrão que sobrevive no Colab) e é observado por
  chamadas curtas — nenhuma chamada `colab exec` fica aberta tempo suficiente
  para estourar o timeout do websocket.
- `up` é **idempotente e retomável**: reusa sessão viva, só monta o Drive se
  preciso (único passo interativo), e sempre faz `git fetch + reset --hard
  origin/main` num clone existente — VM nunca roda código desatualizado.
- `train` **recusa duplicar**: se já há um processo do experimento vivo, erro
  claro (use `pause` antes). Restaura o `last_*.pt`/`best_*.pt` mais novos do
  Drive antes de lançar (retomada automática).
- O daemon de sync copia checkpoints/eventos para o Drive e escreve um
  **heartbeat** `MyDrive/<exp>_ckpts/status.json` (timestamp, treino vivo?,
  última linha de época) a cada 60s — se a VM morrer, o último estado fica
  legível até pelo app do Drive no celular.
- Assets: Drive (`MyDrive/daniel_assets`) quando disponível; fallback Zenodo
  para pesos IAM e tokenizer. O **dataset** requer Drive (sem fallback).
- `doctor` confere de uma vez: auth do colab, `main` local == `fork/main`
  (a VM clona o fork!), working tree sujo, sessão, Drive, assets, venv, GPU.

## Sessões e custos

- Sessão padrão: `bressay` (`-s` para mudar). A100 por padrão (`--gpu`).
- `--skip-drive --no-train --gpu cpu` = teste de ambiente barato sem
  persistência (o mesmo fluxo usado para validar esta ferramenta).
- **Sempre `stop` ao terminar** — A100 parada também é cobrada.
