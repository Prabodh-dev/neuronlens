# Demo start-up

How to start NeuronLens on the demo machine in real mode. This is not the demo script; Anand owns that.

## Back end

WSL, venv active, from the repo root, one line:

```
NEURONLENS_MODE=real NEURONLENS_DB=data/neuronlens.db NEURONLENS_CKPT=$HOME/runs/gpt2-l8-k32/final.pt uvicorn app.main:app --app-dir server --port 8000
```

Wait for "Application startup complete" before opening the page.

## Front end

From the repo root:

```
git checkout main
git pull origin main
cd web
npm install
npm run gen:types
echo "NEXT_PUBLIC_MODEL_ONLY=1" > .env.local
npm run dev
```

Then open http://localhost:3000.

## Optional second model (experimental)

A second server on port 8001 with its own database and checkpoint, started the same way in another WSL terminal, from the repo root with the venv active, one line:

```
NEURONLENS_MODE=real NEURONLENS_DB=data/neuronlens-neo.db NEURONLENS_CKPT=$HOME/runs/gpt-neo-125m-l8-k32/final.pt NEURONLENS_MODEL_NAME=EleutherAI/gpt-neo-125M uvicorn app.main:app --app-dir server --port 8001
```

Wait for "Application startup complete". Both servers share the GPU, so each one needs its own memory; do not start it if the GPU is short. When both answer, the app opens on a model selector and the top bar has a Models link. With only the GPT-2 server running, nothing changes: no selector, no Models link.

## Before the judges arrive

- Close other GPU programs, and do not train at the same time.
- Do one full warm-up run: go through every page once so the model is already loaded when the visitors start.
- If the page shows "Not running on the model", fix that before anything else. The server is not in real mode, or it is not reachable at the address the web app uses. Check the back end window, then reload.
