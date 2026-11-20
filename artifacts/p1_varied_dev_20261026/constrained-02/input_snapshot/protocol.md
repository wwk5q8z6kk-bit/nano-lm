# Exposed varied development set, 2026-10-26

24 invented dialogues; no private data. Six strata, four cases each: direct
request/reply variation, unresolved medication conflict in both orders,
coexisting medicines, field-local uncertainty, missing fields with unrelated
context, and qualified/other-person medication language.

The coding agent authored and labelled these cases after implementing the
bounded repair. Labels are literal source-derived expectations reviewed by that
same agent before model inference; they are NOT blinded or independent human
annotations. This tests software/model integration and known-language variation,
not broad transfer, clinical safety or a confirmatory hypothesis. Five scalar
fields only. Explicitly unresolved use plus blanket denial is C; multiple
compatible medicine names are U under the scalar schema; scoped denials are U.

Freeze input hashes before inference; preserve all outputs and attempts. Run the
pinned cached base Qwen2.5-1.5B offline in constrained and free modes. Both use
the same sources and gold; each saved generation supplies raw, checked and final
note scores. Keep parse/transport failures in denominators. Report assertion
retention, false assertions, abstentions and constant-U/M controls. Zero paid
compute. Do not modify historical cohorts, H6/H7, or protected artifacts.
