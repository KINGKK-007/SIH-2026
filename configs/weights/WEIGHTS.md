# Pretrained weights provenance

Checkpoints live in this folder and are git-ignored. For each file record: source URL, repo commit,
licence, SHA-256, and the input preprocessing it expects (README 4.3, 6.3). Filled in Phase 8, T8.1.

D-022 (`docs/DECISIONS.md`): the production model is LSK3DNet, not a range-view network; see
`docs/MODEL_CARD.md` for the full model card and environment setup.

| File | Model | Source URL | Commit | Licence | SHA-256 |
|---|---|---|---|---|---|
| `opensource_9ks_s030_w64_0.pt` | LSK3DNet (SemanticKITTI val) | Model Zoo link in `LSK3DNet-main/README.md` (OneDrive, `[HUMAN]` download required) | n/a — vendored as a zip snapshot, no git history | MIT | TBD — fill after download: `python -c "from foveamap.models.lsk3dnet import checkpoint_sha256; print(checkpoint_sha256('configs/weights/opensource_9ks_s030_w64_0.pt'))"` |
