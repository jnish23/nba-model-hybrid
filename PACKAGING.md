# Public models package contract

`nba-pregame-models==0.1.0` provides the import namespaces `core`, `team` and `props`.
The distribution includes `core/live_feed_overrides.csv`. It excludes model binaries,
databases, calibration/research datasets, dashboards, launch scripts and private trading
packages. Those remain explicit deployment artifacts, not wheel payloads.

Build offline with the existing setuptools/wheel environment:

```powershell
python -m pip wheel --no-index --no-deps --no-build-isolation --wheel-dir <output> .
python -m pip install --no-index --no-deps <output>/nba_pregame_models-0.1.0-py3-none-any.whl
```

These commands must target an isolated build/install environment. No dependency downloads
or production environment installation are authorized. Optional `training`, `collection`
and `test` groups describe module-specific dependencies. Production's existing pinned
requirements remain the reproducibility baseline; metadata ranges are compatibility
bounds, not a replacement lock. Python 3.12 is exercised; other supported versions need CI.

Consumers import public utilities normally; they do not add this checkout to `sys.path`.
Runtime data/model/config paths are deployment inputs. Installing this wheel alone does
not populate production artifacts or make checkout-relative default data paths portable.
Pass explicit artifact paths where the API supports them; storage cutover remains frozen.
Legacy `models.market_models` pickle paths are registered by
`team.model_registry.alias_legacy_pickle_modules()` before loading historical models.

Version 0.1.0 is the initial separation contract. Keep a released version immutable;
bump it when publishing changed public code. No package has been published externally.
