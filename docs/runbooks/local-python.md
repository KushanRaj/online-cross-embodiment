# Local Python Runbook

## Rule

Do not run project Python through bare system Python.

Use:

```bash
uv venv .venv
.venv/bin/python <script.py>
```

If dependencies are missing, install into the `uv` environment or use the remote EC2 environment for GPU/Cosmos jobs.

## Local Work That Is Safe

Typical local tasks:

- inspect CSV/JSON summaries,
- render or rebuild static HTML indexes,
- run lightweight plot scripts if dependencies are present,
- validate documentation links,
- inspect HDF5 metadata if local dependencies exist.

## Local Work That Is Usually Not Safe

Usually remote:

- Cosmos inference,
- simulator rollouts,
- IDM training on GPU,
- large feature caching,
- heavy video generation if dependencies are remote-only.

## Before Running

Check:

```bash
git status --short
which uv
test -x .venv/bin/python || uv venv .venv
```
