# Public model comparison contract

Baseline public revision: `0d11f8344e65274b9490b28b674166aba9c30569`.
Expected check results are in [models_smoke.json](models_smoke.json); exact frozen
input/output identities are in [model_fixture_manifest.json](model_fixture_manifest.json).
The external recovery set supplies the frozen bytes without reading another checkout.

| Component | Required exact comparison |
|---|---|
| Team | Two historical games, fixed illustrative opening lines and existing missing-injury encoding: feature columns/NaN masks/order, selected model identities and predictions. |
| Props | Six stored feature rows, active r1 bundle, seed 21 and 20,000 simulations: ordered features, minutes and PMF arrays. |
| Decisions/settlement | Selected player/market/side/line, eligibility reasons, void/DNP status, payout and P&L. |
| Exports | Stable row keys, column names/order, values and exported records. |
| Deployment | Generate a complete standalone bundle and run the app with only bundle modules available. Module source/import paths may change; the resulting tables must remain equal. |

No blanket numerical tolerance is allowed. Only PMF normalization permits absolute
error 2e-6 for float32 storage; forecast comparisons remain exact under this recorded
environment. For npz files compare array keys/dtypes/shapes/bytes: ZIP metadata need
not be identical. Expected metadata differences are inspection timestamps, output
paths, code revision and generated source-module layout. Preserve artifact hashes,
model class lookup paths, input row order and decision boundaries.

Use offline subprocesses, no production writable state, and fixed restored inputs.
No retraining, API call, publication or model promotion is part of this verification.
