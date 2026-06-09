# Agent Entry Point

Start with [docs/INDEX.md](docs/INDEX.md). The documentation is now a nested
Obsidian-style system: core docs, code maps, plan bundles, research sessions,
experiment records, and runbooks.

Minimum read order before experiment work:

1. [docs/INDEX.md](docs/INDEX.md)
2. [docs/reflections.md](docs/reflections.md)
3. [docs/code/README.md](docs/code/README.md)
4. The relevant plan bundle under [docs/plans/](docs/plans/README.md)
5. The relevant runbook under [docs/runbooks/](docs/runbooks/README.md)
6. The exact source files named by those docs

Hard rules:

- Reuse existing rollout, evaluation, diagnostic, and video-rendering code
  unless the existing HDF5 contract cannot represent the task.
- Do not silently change experiment semantics, metrics, horizons, reference
  action datasets, preprocessing, model architecture, optimizer, or policy
  execution cadence.
- Local Python work must use the repo `uv` environment, not bare system Python.
- Remote-heavy data, model, and simulator jobs run on EC2; code should be edited
  locally and then synced or pulled remotely.
- If EC2 is started, stop it when the requested remote work is complete.
- Keep code/docs committed after each coherent unit of work.
