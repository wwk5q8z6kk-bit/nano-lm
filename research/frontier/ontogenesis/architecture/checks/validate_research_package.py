"""Validate the portable research package, without writing by default.

Requires jsonschema>=4.18,<5 (see requirements-checks.txt). This checks document
links, the event contract, history continuity and explicit local artifact hashes.
It does not authenticate a reviewer, fetch remote commits, grant permissions or
establish the scientific truth of a claim. Markdown links use the documented
inline/reference-link subset, not a complete CommonMark parser.
"""
from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime
import hashlib
import json
from pathlib import Path
import re
import sys
from urllib.parse import unquote, urlsplit

ROOT = Path(__file__).resolve().parents[1]
DOCS = ("README.md", "01_ARCHITECTURE_AND_THEOREMS.md", "02_DECISION_LEDGER.md",
        "03_EXPERIMENTS_AND_EXECUTION.md", "04_EVIDENCE_AND_REFERENCES.md",
        "05_GAP_CLOSURE_AND_FORMAL_REVIEW.md", "06_VALIDATION_AND_UNRESOLVED_WORK.md")
REQUIRED = DOCS + ("ledger_events.jsonl", "ledger_event.schema.json",
                   "ledger_history.json", "architecture_scheme.mmd")
LINK = re.compile(r'!?\[[^\]\n]*\]\(([^)\n]+)\)')
REFERENCE = re.compile(r'^ {0,3}\[[^\]\n]+\]:\s*(?:<([^>]+)>|(\S+))', re.M)
DECISION_ROW = re.compile(r'^\|\s*(D-\d{3,6})\s*\|', re.M)
FENCE = re.compile(r'^ {0,3}(`{3,}|~{3,})(.*)$')


def _prose(markdown: str):
    """Exclude fenced code and report an unclosed fence; retain heading text."""
    lines, opened = [], None
    for line in markdown.splitlines():
        match = FENCE.match(line)
        if match:
            marker, trailing = match.groups()
            if opened is None:
                opened = (marker[0], len(marker))
                continue
            if marker[0] == opened[0] and len(marker) >= opened[1] and not trailing.strip():
                opened = None
                continue
        if opened is None:
            lines.append(line)
    return "\n".join(lines), opened is not None


def _anchors(markdown: str):
    """GitHub-style anchors for the ATX headings used in this package."""
    prose, _ = _prose(markdown)
    seen, anchors = Counter(), set()
    for heading in re.findall(r'^ {0,3}#{1,6}\s+(.+?)\s*#*$', prose, re.M):
        heading = re.sub(r'<[^>]+>', '', heading).lower()
        heading = re.sub(r'[^\w\-\s]', '', heading, flags=re.UNICODE)
        slug = re.sub(r'\s', '-', heading)
        suffix = f'-{seen[slug]}' if seen[slug] else ''
        seen[slug] += 1
        anchors.add(slug + suffix)
    return anchors


def _unique_keys(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f'duplicate JSON object key: {key}')
        result[key] = value
    return result


def _json_file(path: Path, issues: list[str]):
    try:
        return json.loads(path.read_text(encoding='utf-8'), object_pairs_hook=_unique_keys)
    except (OSError, UnicodeError, ValueError) as exc:
        issues.append(f'{path.name}: cannot read JSON: {exc}')
        return None


def _local_path(root: Path, origin: Path, reference: str):
    decoded = unquote(reference)
    split = urlsplit(decoded)
    if split.scheme or split.netloc:
        return None
    candidate = (origin.parent / split.path).resolve() if split.path else origin.resolve()
    try:
        candidate.relative_to(root)
    except ValueError:
        raise ValueError('path escapes the portable package')
    return candidate


def _check_reference(root: Path, origin: Path, reference: str, issues: list[str], *, source=False):
    reference = reference.strip()
    # Markdown allows an optional title after a URL; sources must be exact paths.
    if not source:
        reference = reference[1:reference.index('>')] if reference.startswith('<') and '>' in reference else reference.split()[0]
    try:
        split = urlsplit(reference)
    except ValueError as exc:
        issues.append(f'{origin.name}: malformed reference {reference}: {exc}')
        return
    if split.scheme in ('http', 'https'):
        if not split.netloc:
            issues.append(f'{origin.name}: malformed URL {reference}')
        return
    if not source and split.scheme in ('mailto', 'data', 'sandbox'):
        return
    if split.scheme or split.netloc:
        issues.append(f'{origin.name}: unsupported reference {reference}')
        return
    try:
        target = _local_path(root, origin, reference)
    except ValueError as exc:
        issues.append(f'{origin.name}: {reference}: {exc}')
        return
    if target is None or not target.exists() or (source and not target.is_file()):
        issues.append(f'{origin.name}: unresolved file {reference}')
        return
    if split.fragment and target.suffix.lower() == '.md':
        try:
            if unquote(split.fragment) not in _anchors(target.read_text(encoding='utf-8')):
                issues.append(f'{origin.name}: unresolved heading {reference}')
        except (OSError, UnicodeError) as exc:
            issues.append(f'{origin.name}: unreadable link target {reference}: {exc}')


def validate(root=ROOT, baseline_ledger=None):
    root = Path(root).resolve()
    issues, latest, all_events = [], {}, {}
    result = {'status': 'FAIL', 'errors': issues, 'events_checked': 0,
              'decisions_checked': 0, 'documents_checked': 0,
              'history_prefix_preserved': False, 'local_evidence_hashes_checked': 0,
              'full_baseline_append_only': 'NOT_CHECKED',
              'scope': 'Structural contract and local hash checks only; no scientific or remote-source authentication.'}
    for name in REQUIRED:
        if not (root / name).is_file():
            issues.append('missing ' + name)
    documents = {}
    for path in sorted(root.rglob('*.md')):
        relative_name = path.relative_to(root).as_posix()
        try:
            documents[relative_name] = path.read_text(encoding='utf-8')
        except (OSError, UnicodeError) as exc:
            issues.append(f'{path.name}: unreadable document: {exc}')
            continue
        prose, unclosed = _prose(documents[relative_name])
        if unclosed:
            issues.append(path.name + ': unclosed code fence')
        for uri in LINK.findall(prose):
            if uri.strip():
                _check_reference(root, path, uri, issues)
        for first, second in REFERENCE.findall(prose):
            _check_reference(root, path, first or second, issues)
    result['documents_checked'] = len(documents)
    human_ids = DECISION_ROW.findall(documents.get('02_DECISION_LEDGER.md', ''))
    duplicated = sorted(d for d, count in Counter(human_ids).items() if count > 1)
    if duplicated:
        issues.append('duplicate human decision rows: ' + ', '.join(duplicated))

    history = _json_file(root / 'ledger_history.json', issues)
    schema = _json_file(root / 'ledger_event.schema.json', issues)
    history_valid = isinstance(history, dict)
    history_fields = {'schema_version', 'source_commit', 'ledger_path', 'prefix_line_count',
                      'prefix_sha256', 'legacy_human_decision_ids', 'legacy_event_ids', 'scope'}
    if history_valid:
        history_valid = (set(history) == history_fields and history.get('schema_version') == 1
                         and history.get('ledger_path') == 'ledger_events.jsonl'
                         and type(history.get('prefix_line_count')) is int
                         and history['prefix_line_count'] > 0
                         and isinstance(history.get('source_commit'), str)
                         and bool(re.fullmatch(r'[0-9a-f]{40}', history['source_commit']))
                         and isinstance(history.get('prefix_sha256'), str)
                         and bool(re.fullmatch(r'[0-9a-f]{64}', history['prefix_sha256']))
                         and isinstance(history.get('legacy_human_decision_ids'), list)
                         and all(isinstance(d, str) and re.fullmatch(r'D-\d{3,6}', d) for d in history['legacy_human_decision_ids'])
                         and isinstance(history.get('legacy_event_ids'), list)
                         and len(history['legacy_event_ids']) == history['prefix_line_count']
                         and all(isinstance(e, str) for e in history['legacy_event_ids'])
                         and isinstance(history.get('scope'), str))
    if not history_valid:
        issues.append('invalid legacy history manifest')
    try:
        from jsonschema import Draft202012Validator, FormatChecker
        if schema is None:
            validator = None
        else:
            Draft202012Validator.check_schema(schema)
            formats = FormatChecker()
            # jsonschema's optional format extras may be absent. Register the
            # UTC calendar check explicitly so invalid dates cannot pass silently.
            @formats.checks('date-time')
            def utc_calendar_timestamp(value):
                if not isinstance(value, str) or not value.endswith('Z'):
                    return False
                try:
                    datetime.fromisoformat(value[:-1] + '+00:00')
                    return True
                except ValueError:
                    return False
            validator = Draft202012Validator(schema, format_checker=formats)
    except ImportError:
        issues.append('jsonschema is required; install checks/requirements-checks.txt')
        validator = None
    except Exception as exc:
        issues.append(f'invalid event JSON Schema: {exc}')
        validator = None
    try:
        raw = (root / 'ledger_events.jsonl').read_bytes()
        raw_lines = raw.splitlines(keepends=True)
    except OSError as exc:
        issues.append(f'cannot read event log: {exc}')
        raw, raw_lines = b'', []
    result['ledger_sha256'] = hashlib.sha256(raw).hexdigest()
    if baseline_ledger is not None:
        try:
            previous_bytes = Path(baseline_ledger).read_bytes()
            result['baseline_ledger_sha256'] = hashlib.sha256(previous_bytes).hexdigest()
            preserved = bool(previous_bytes) and raw.startswith(previous_bytes)
            result['full_baseline_append_only'] = 'PASS' if preserved else 'FAIL'
            if not preserved:
                issues.append('event log is not a byte-exact extension of the supplied prior ledger')
        except OSError as exc:
            result['full_baseline_append_only'] = 'FAIL'
            issues.append(f'cannot read supplied prior ledger: {exc}')
    prefix_count = history['prefix_line_count'] if history_valid else 0
    legacy_ids = history['legacy_human_decision_ids'] if history_valid else []
    if history_valid:
        prefix = b''.join(raw_lines[:prefix_count])
        result['history_prefix_preserved'] = (len(raw_lines) >= prefix_count and
            hashlib.sha256(prefix).hexdigest() == history['prefix_sha256'])
        if not result['history_prefix_preserved']:
            issues.append('legacy event prefix differs from the pinned historical bytes')
        if human_ids[:len(legacy_ids)] != legacy_ids:
            issues.append('historical human decision IDs are missing, duplicated or reordered')
    known_decisions = set(legacy_ids)
    last_recorded = None
    for number, raw_line in enumerate(raw_lines, 1):
        label = f'event line {number}'
        try:
            event = json.loads(raw_line, object_pairs_hook=_unique_keys)
        except (ValueError, UnicodeError) as exc:
            issues.append(f'{label}: invalid JSON: {exc}')
            continue
        result['events_checked'] += 1
        if validator is None:
            continue
        errors = sorted(validator.iter_errors(event), key=lambda e: str(list(e.absolute_path)))
        if errors:
            for error in errors:
                issues.append(f'{label}: schema {list(error.absolute_path)}: {error.message}')
            continue
        decision, event_id = event['decision_id'], event['event_id']
        if event_id in all_events:
            issues.append(f'{label}: duplicate event_id {event_id}')
            continue
        if number <= prefix_count:
            if history_valid and event_id != history['legacy_event_ids'][number - 1]:
                issues.append(f'{label}: legacy event ID order changed')
        else:
            if event.get('schema_version') != 2:
                issues.append(f'{label}: new events require schema_version 2 and provenance')
            elif event['previous_event_id'] != latest.get(decision):
                issues.append(f'{label}: previous_event_id must equal the latest prior event for {decision} ({latest.get(decision)!r})')
        try:
            recorded = datetime.fromisoformat(event['recorded_at_utc'].replace('Z', '+00:00'))
        except ValueError as exc:
            issues.append(f'{label}: invalid recorded timestamp: {exc}')
            continue
        if last_recorded is not None and recorded < last_recorded:
            issues.append(f'{label}: recorded_at_utc is earlier than the preceding append')
        last_recorded = recorded
        for superseded in event['supersedes_ids']:
            if superseded == decision or superseded not in known_decisions:
                issues.append(f'{label}: supersedes_ids must name a different earlier decision: {superseded}')
        for reference in event['source_refs']:
            _check_reference(root, root / 'ledger_events.jsonl', reference, issues, source=True)
        for artifact in event.get('provenance', {}).get('evidence_artifacts', []):
            try:
                path = _local_path(root, root / 'ledger_events.jsonl', artifact['path'])
                if path is None or '#' in artifact['path'] or '?' in artifact['path']:
                    raise ValueError('evidence artifact must be a local file path without a fragment/query')
                actual = hashlib.sha256(path.read_bytes()).hexdigest()
                result['local_evidence_hashes_checked'] += 1
                if actual != artifact['sha256']:
                    issues.append(f'{label}: evidence SHA-256 mismatch for {artifact["path"]}')
            except (ValueError, OSError) as exc:
                issues.append(f'{label}: unreadable evidence artifact {artifact["path"]}: {exc}')
        latest[decision] = event_id
        all_events[event_id] = event
        known_decisions.add(decision)
    for decision in sorted(set(latest) - set(human_ids)):
        issues.append('missing human decision row ' + decision)
    for decision in sorted(set(human_ids) - set(legacy_ids) - set(latest)):
        issues.append('missing machine event for human decision ' + decision)
    result['decisions_checked'] = len(latest)
    result['latest_events'] = dict(sorted(latest.items()))
    result['status'] = 'PASS' if not issues else 'FAIL'
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=ROOT, help='Portable package root')
    parser.add_argument('--baseline-ledger', type=Path,
                        help='Optional exact prior Git ledger; require its complete bytes to remain the current prefix')
    parser.add_argument('--out', type=Path, help='Optional NEW report; exclusive creation prevents overwrites')
    args = parser.parse_args()
    report = validate(args.root, args.baseline_ledger)
    if args.out is not None:
        with args.out.open('x', encoding='utf-8') as target:
            json.dump(report, target, indent=2)
            target.write('\n')
    print(json.dumps(report, indent=2))
    return 1 if report['errors'] else 0


if __name__ == '__main__':
    sys.exit(main())
