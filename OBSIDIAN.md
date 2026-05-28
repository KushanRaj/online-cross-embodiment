# Obsidian Documentation Workflow

This repository mirrors the relevant Obsidian structure for distilled research notes, paper summaries, experiment plans, and synthesis.

Do not rely on the Obsidian MCP connector by default. In this environment it may not be running or reachable. Use filesystem search and direct Markdown file edits instead.

## Local Vault Location

The original local Obsidian vault is outside this repo. On a new machine, rediscover it with:

```bash
sed -n '1,240p' "$HOME/Library/Application Support/obsidian/obsidian.json"
find "$HOME/Documents" -maxdepth 4 -type d -name .obsidian 2>/dev/null
```

Avoid broad home-directory `find` scans unless needed; they are slow.

## How To Edit The Vault

Use normal filesystem tools against the vault path:

```bash
rg --files "$HOME/Documents/Obsidian Vault" | sed -n '1,160p'
find "$HOME/Documents/Obsidian Vault" -maxdepth 2 -type d | sort
```

For file edits, prefer `apply_patch`, the same way repository files are edited.

## Repo Layout

Use the repo structure directly:

- `Robotics World Models.md`: top-level map of content.
- `papers/`: paper notes and `paper-catalog.md`.
- `pdfs/`: tracked local PDFs.
- `experiment-design/`: experiment proposals, synthesis reports, and compute/benchmark notes.
- `docs/`: planning artifacts and Slack-facing exports.

For paper notes, use:

`papers/<paper-title>.md`

For active research maps, use the root MOC:

`Robotics World Models.md`

## What To Put In Obsidian

Put distilled notes in Obsidian:

- core claims,
- personal interpretation,
- experiment hooks,
- open questions,
- links between concepts,
- decisions and research direction.

Do not dump full paper text into Obsidian. Keep PDFs in `pdfs/` and link to them from the note when useful.

## Current Robotics Notes

Current MOC:

`Robotics World Models.md`

Current paper notes:

`papers/Reconstruction or Semantics - Semantic Latents for Robotic World Models.md`

`papers/Dreamer 4 - Training Agents Inside Scalable World Models.md`

`papers/Cosmos Policy - Video Diffusion Planning and Value Models.md`

`papers/Fast-WAM - Video Co-Training Without Test-Time Future Imagination.md`

`papers/LingBot-VA - Causal Autoregressive Video-Action World Model.md`

`papers/Joint Embedding vs Reconstruction - SSL Inductive Bias.md`

`papers/VLA-JEPA - Latent World Model for VLA.md`

Common PDF paths:

`pdfs/Reconstruction-or-Semantics-Semantic-WM.pdf`

`pdfs/Dreamer-4.pdf`

`pdfs/Cosmos-Policy.pdf`

`pdfs/Fast-VAM.pdf`

`pdfs/LingBot-VA.pdf`

`pdfs/Joint-Embedding-vs-Reconstruction-SSL.pdf`

`pdfs/VLA-JEPA.pdf`
