# Versta.Models verification

Runs belong on the arm64 Paseo host. Follow the root and affected module
`AGENTS.md` files. Models has independent uv projects, not one workspace:
identify the affected modules and run their relevant `uv run` lint, format,
type checks and pytest or small-model smoke checks. Read the CI workflow for
the required gates. Record every command and its result; do not claim success
for unrun or unavailable checks. For metadata or bundle contract changes,
check the consuming Versta.Android code and record the exact commits used.
