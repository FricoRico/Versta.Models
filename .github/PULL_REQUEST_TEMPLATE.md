## Summary

<!-- What changes and why, in which module(s). Link the issue. -->

## Test plan

<!-- How this was verified: pytest run, or a smoke run of the tool
(`uv run python -m versta.<tool> ...`) into the gitignored `output/` tree.
Note any bundle/metadata produced. -->

## Checklist

- [ ] `uvx ruff check .` and `uvx ruff format --check .` pass in every touched module
- [ ] `translation/` changes: `uv run mypy` and `uv run pytest` pass
- [ ] Changes to `metadata.json` / `models.json` schemas verified against how Versta.Android parses them
- [ ] `version.txt` bumped when the produced bundle format changes
- [ ] Module `README.md` updated with any capability, CLI, or bundle-format changes
- [ ] Models fetch only from official upstream sources; no new third-party redirects or telemetry
- [ ] No generated artifacts committed (weights, tarballs, checksums, `output/`/`export/`/`results/` trees)
