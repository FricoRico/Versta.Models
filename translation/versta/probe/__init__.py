"""
Pack acceptance probing for translation bundles (NEU-60, epic NEU-59).

A bundle is unproven until it loads and translates. This package drives the
headless acceptance runner:

1. ``install`` — take a bundle tarball, verify its checksum sidecar against
   ``.sha256`` and install it through the same metadata/parse path the app
   uses: the ``versta.schema`` validators (which mirror the app's Kotlin
   loader shapes ``LanguageModelMetadata``/``LanguageBundleMetadata``) plus
   the on-disk file-existence rule of the app's ``files.isValid(root)``.
2. ``bridge`` — for every direction declared in the bundle manifest, invoke
   the pinned ``leanmt-probe`` binary with the exact field mapping the app's
   JNI bridge applies (``LeanmtPackage``/``LeanmtModelConfig`` +
   app-default decode: beam 4, max sequence length 256, translation cache
   off for measurement).
3. ``metrics`` — chrF against the probe corpus references (direction sanity;
   the NEU-19 mislabeled-direction class collapses this score) and language-
   agnostic digit preservation (the NEU-30 harm class).
4. ``scorecard`` — assemble one JSON scorecard per bundle with load ok,
   load ms, chrF + per-direction floor verdict, digit preservation rates and
   failing ids, p50/p95 latency and peak RSS, and gate the run by the
   acceptance rules plus the caller's expectation (``pass`` or the
   adversarial negative control ``direction-fail``).

The probe corpus itself ships as package data under ``corpora/``; the CI
workflow that runs this on the ``ubuntu-24.04-arm`` runner is
``.github/workflows/pack-acceptance.yml``.
"""
