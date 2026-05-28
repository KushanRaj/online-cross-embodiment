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

## Where Notes Go

Use the existing vault structure:

- `00_INBOX`: raw notes, quick thoughts, unresolved dumps.
- `10_SOURCES`: papers, talks, repositories, datasets, and external references.
- `20_CONCEPTS`: evergreen concept notes.
- `30_MOCs`: maps of content for active research areas.
- `40_DAILIES`: daily notes and short session logs.
- `90_MEDIA`: media attachments when they should live inside the vault.

For paper notes, use:

`10_SOURCES/Papers/<area>/<paper-title>.md`

For active research maps, use:

`30_MOCs/<topic>.md`

## What To Put In Obsidian

Put distilled notes in Obsidian:

- core claims,
- personal interpretation,
- experiment hooks,
- open questions,
- links between concepts,
- decisions and research direction.

Do not dump full paper text into Obsidian. Keep PDFs outside git or in another explicit paper directory, and link to them from the note when useful.

## Current Robotics Notes

Current MOC:

`30_MOCs/Robotics World Models.md`

Current paper notes:

`10_SOURCES/Papers/Robotics World Models/Reconstruction or Semantics - Semantic Latents for Robotic World Models.md`

`10_SOURCES/Papers/Robotics World Models/Dreamer 4 - Training Agents Inside Scalable World Models.md`

`10_SOURCES/Papers/Robotics World Models/Cosmos Policy - Video Diffusion Planning and Value Models.md`

`10_SOURCES/Papers/Robotics World Models/Fast-WAM - Video Co-Training Without Test-Time Future Imagination.md`

`10_SOURCES/Papers/Robotics World Models/LingBot-VA - Causal Autoregressive Video-Action World Model.md`

`10_SOURCES/Papers/Robotics World Models/Joint Embedding vs Reconstruction - SSL Inductive Bias.md`

`10_SOURCES/Papers/Robotics World Models/VLA-JEPA - Latent World Model for VLA.md`

Local PDFs are intentionally not tracked in the public repo. Common local filenames:

`Reconstruction-or-Semantics-Semantic-WM.pdf`

`Dreamer-4.pdf`

`Cosmos-Policy.pdf`

`Fast-VAM.pdf`

`LingBot-VA.pdf`

`Joint-Embedding-vs-Reconstruction-SSL.pdf`

`VLA-JEPA.pdf`
