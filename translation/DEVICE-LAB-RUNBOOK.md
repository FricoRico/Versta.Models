# Device-lab runbook

Operational reference for the shared redroid device lab: how to reach it,
how to treat it, and its standing weekly pack-acceptance pass. Read this
before touching `android-a` or `android-b` for any reason.

Lab etiquette and the weekly slot are QA-owned (Priya Nair). Changes to
either go through QA and are recorded here.

## Lab topology & canonical endpoints

The lab is two long-lived redroid 13 arm64 containers running in the same
Kubernetes cluster as the CI runners, exposed as ClusterIP Services in
namespace `android` (source: NEU-129 verdict):

| Endpoint | ClusterIP | Role |
|---|---|---|
| `android-a.android.svc.cluster.local:5555` | 10.43.248.69 | primary |
| `android-b.android.svc.cluster.local:5555` | 10.43.181.108 | spare / split-load |

Device identity (both): redroid 13, `arm64_only` build
(`TQ3A.230805.001.S1/eng.frank.20240527`), SDK 33 / Android 13 /
`arm64-v8a`, no camera HAL. `android-a` holds the 1.3.1 control app; any
other tenant state on either device is hands-off.

Reach path from any runner pod (namespace `paperclip-neurora`), verified
from a cold runner 2026-10-25 23:29 UTC (NEU-129):

```bash
adb connect android-a.android.svc.cluster.local:5555   # → connected → device
adb connect android-b.android.svc.cluster.local:5555   # → connected → device
```

Runner pods resolve cluster DNS natively (service CIDR `10.43.0.0/16`,
pod CIDR `10.42.0.0/16`). The Tailscale egress proxy is **not** involved
(`NO_PROXY` covers `10.0.0.0/8`). No VPN, no port-forward, no keys. adbd
on these redroid images accepts connections without adb key auth (stock
redroid test-keys default); access control is network scope — the
Services are ClusterIP, reachable only from pods inside this cluster (our
runners, nothing external).

**Retired endpoint assumption — do not resurrect:** the `10.0.70.0/24`
sweep target is dead. The lab never lived on that subnet from the runner
network; NEU-77/NEU-127-era "lab is down" conclusions came from probing
the wrong address, while both containers ran uninterrupted through those
failures. If the lab appears unreachable, check endpoints and DNS from a
runner first and escalate to QA — do not start probing subnets.

**No physical reference phone exists.** "Pixel-6a-class mid-range" was a
target spec, never procured hardware (NEU-129). Absolute on-device
latency/RSS budgets are deferred to real silicon and gated on procurement
approval `6a201a44`; until then device passes report functional +
relative metrics only.

## Lab etiquette (all users)

Baseline rules (NEU-61 spec, kept):

- **Exclusive use while sweeping** — one pass per device at a time.
- **No reboots.**
- **Clean up `/data/local/tmp`** after your pass; leave nothing behind.
- **Redroid containers stay up** — never stop/start containers or
  services as part of a pass.

The weekly pack-acceptance pass additionally follows the QA-required
rules in its section below; ad-hoc passes should treat them as the norm
too.

## Weekly pack-acceptance pass (NEU-61)

### Purpose

Weekly functional verification of all 32 published bundles (63
directions) on the redroid device lab (Android arm64), running the pinned
`leanmt-probe` against the traveler-v1 probe corpus. Output: per-pair
device latency/RSS table next to the CI scorecards (deltas called out),
plus load/chrF/digit gates. Absolute budgets wait on real silicon
(procurement approval `6a201a44`); this pass reports functional +
relative metrics only.

### Schedule (canonical slot)

| Item | Value |
|---|---|
| Slot | **Monday 08:15–08:45 UTC** (30-minute exclusivity reservation) |
| Actual sweep wall time | ~5 min measured (pass #1, 2026-10-26; 16+16 bundle split) |
| Anchor | Same-day-after CI weekly runner sweep (Monday 06:15 UTC, `pack-sweep-weekly.yml`), so device-vs-CI scorecard deltas compare same-day data |
| Retry policy | If the pass cannot run Monday (lab incident), one retry Tuesday 08:15 UTC; further slips escalate to QA — cadence stability matters for trend comparability |
| Owner of the slot | QA (Priya Nair). Slot changes go through QA and are recorded here |

### Why no collision

- The standing NEU-40 connected-test job (NEU-137) runs on GitHub-hosted
  KVM runners, **not** the redroid lab — it consumes no lab time, so no
  double-booking is possible with it.
- The redroid lab has no other standing weekly tenant. Remaining
  consumers are ad-hoc one-off passes (app-side instrumentation,
  benches); the Monday slot is reserved in the shared lab calendar sense
  by this entry, and the etiquette rules make contention self-limiting
  given the ≤30-min reservation.

### QA-required etiquette for this pass

1. **Pre-flight gate.** `adb connect` both endpoints; require
   `adb get-state == device` and `sys.boot_completed=1` before starting.
   If either check fails, **abort and escalate to QA — do not attempt
   remediation.**
2. **Concurrency token.** Before pushing, check
   `/data/local/tmp/versta-probe/` for another run dir with mtime < 12h.
   If present, abort and notify QA — never two sweeps on one device.
3. **Path discipline.** Touch only `/data/local/tmp/versta-probe/<run-id>/`.
   Never enumerate or uninstall packages; `android-a` holds the 1.3.1
   control app and any other tenant state is hands-off.
4. **No installs, no lingering tunnels.** The pass is push/run/pull/delete
   only: no `adb install`, and remove any `adb forward` you create;
   `adb disconnect` from both endpoints at the end so connections never
   linger between passes.
5. **Pinned artifacts.** The `leanmt-probe` pin (currently leanmt
   `07015949`) and the traveler-v1 corpus are the sweep contract. Pin
   bumps happen through a separate reviewed change, never ad-hoc from
   inside the weekly pass.
6. **Guaranteed cleanup.** Cleanup of the run dir executes even on
   mid-pass failure (trap/finally in the harness). Verify the probe root
   is empty afterwards and report any residue in the sweep summary.
7. **Container distress.** If a device stops responding mid-pass (state
   != device, `sys.boot_completed` 0, repeated adb errors): stop
   immediately, capture evidence, escalate to QA. Do not restart
   containers or services.
8. **Attribution.** Every sweep records run-id, start/end UTC, devices
   used, and outcome in the published sweep summary, so lab usage is
   auditable.
9. **Wall-time bound.** The reservation is ≤30 min. If sweep growth
   pushes actual use toward that bound, flag QA to re-negotiate the slot
   *before* it becomes a collision source.
10. **Real-silicon future.** When physical reference hardware lands
    (procurement `6a201a44`), the same etiquette applies per device; this
    runbook will be amended and absolute budget gates stay as they are.

## Provenance

- Canonical endpoints + topology: NEU-129 verdict comment `3fad6c63`
  (device lab verified live 2026-10-25; `10.0.70.0/24` retired).
- Weekly slot, etiquette additions, and sign-off agreed on NEU-143
  (document `device-lab-weekly-cadence`, rev `181e5de8`).
- First-pass measurements: NEU-61 document `device-sweep-2026-10-26`.
