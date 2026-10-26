Interrupted development attempt, preserved with exact source snapshots.

Confirmed tokenizer boundary defect: Qwen tokenizes ':S' as one token, whereas
prefilling 'CC:' and forcing a standalone 'S' shifts the continuation distribution.
The model repeatedly chose U with wrong-field replies. The next attempt includes
the colon inside the token trie and prefills only the field label. The question
guard regression from run-04 is fixed in this attempt. No confirmatory inference.
