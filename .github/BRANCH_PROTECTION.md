# Branch protection proposal

Applies to `main` (this repo has no `develop`; feature branches merge straight
to `main` via PR).

| Setting | Proposed value | Why |
| --- | --- | --- |
| Require a pull request before merging | on | The maintainer reviews every PR; no direct pushes. |
| Required approvals | 1 | Small team; keep it cheap. |
| Require status checks to pass | on | The three CI jobs below. |
| Required checks | `lint`, `typecheck`, `test` | These are the job names in `.github/workflows/ci.yml`. |
| Require branches to be up to date before merging | on | Guarantees the merged result was actually tested by CI. |
| Require conversation resolution | on | Review comments don't get lost. |
| Require signed commits | off | Not worth the setup cost at this stage. |
| Allow force pushes / deletions | off | Protect history. |
| Bypass for admins | off | The gate only works if it also gates its owner. |

Apply via: repo **Settings → Branches → Add rule** for `main`.

Notes for the maintainer (one-time actions):
1. GitHub Actions is enabled for the repo (default for public repos).
2. Add the three check names above as *required* only after the first workflow
   run has completed on `main`, otherwise GitHub cannot find them in the
   picker. All three jobs run on every PR and every push — the workflow has
   no `paths:` filters on purpose, because a path-skipped required check never
   posts a status and would block merging forever.
3. No secrets are needed: the suite is fully offline (mocked registry) and
   publishes nothing.
