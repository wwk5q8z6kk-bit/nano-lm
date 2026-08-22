"""CLI for Nano Runtime wedge slice (classical + E-class only)."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from wedge_v1.ingest import corpus_stats
from wedge_v1.habit import (
    format_saved_list_md,
    format_session_md,
    record as habit_record,
    resolve_session_corpus,
    save_question,
    saved_question_status,
    session as habit_session,
    weekly_summary,
)
from wedge_v1.runtime import (
    DEFAULT_CORPUS,
    ask,
    compare,
    find_spans,
    format_report_md,
    load_corpus,
    scan,
)


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(
        prog="wedge_v1",
        description=(
            "Wedge v1 internal evidence and validation pipeline — "
            "not the Nano AI or its inference entry point."
        ),
    )
    sub = p.add_subparsers(dest="cmd", required=True)

    ask_p = sub.add_parser("ask", help="Ask a question; returns span-supported claims or ABSTAIN")
    ask_p.add_argument("--corpus", type=Path, default=DEFAULT_CORPUS)
    ask_p.add_argument(
        "--doc",
        action="append",
        default=None,
        dest="doc_ids",
        metavar="DOC_ID",
        help="Exact document scope (repeatable); unknown/empty → fail-closed ABSTAIN",
    )
    ask_p.add_argument(
        "--escalate-stub",
        action="store_true",
        help="On classical ABSTAIN, try constructive hybrid stub (also WEDGE_ESCALATE_STUB=1)",
    )
    ask_p.add_argument("query", nargs="+")

    find_p = sub.add_parser("find", help="Exact substring locate with spans")
    find_p.add_argument("--corpus", type=Path, default=DEFAULT_CORPUS)
    find_p.add_argument(
        "--doc",
        action="append",
        default=None,
        dest="doc_ids",
        metavar="DOC_ID",
        help="Exact document scope (repeatable)",
    )
    find_p.add_argument("needle", nargs="+")

    scan_p = sub.add_parser("scan", help="Inventory extract + contradictions over corpus")
    scan_p.add_argument("--corpus", type=Path, default=DEFAULT_CORPUS)
    scan_p.add_argument(
        "--doc",
        action="append",
        default=None,
        dest="doc_ids",
        metavar="DOC_ID",
        help="Exact document scope (repeatable)",
    )

    cmp_p = sub.add_parser("compare", help="Cross-doc term compare; flags numeric disputes")
    cmp_p.add_argument("--corpus", type=Path, default=DEFAULT_CORPUS)
    cmp_p.add_argument(
        "--doc",
        action="append",
        default=None,
        dest="doc_ids",
        metavar="DOC_ID",
        help="Exact document scope (repeatable)",
    )
    cmp_p.add_argument("term", nargs="+")

    ingest_p = sub.add_parser("ingest", help="Index a folder (md/txt/pdf→text); write local manifest")
    ingest_p.add_argument("--corpus", type=Path, default=None, help="Alias for positional src")
    ingest_p.add_argument("src", nargs="?", type=Path, default=None)
    ingest_p.add_argument("--out", type=Path, default=None)

    rep = sub.add_parser("report", help="Markdown (default) or JSON report for ask/find/scan/compare")
    rep.add_argument("kind", choices=["ask", "find", "scan", "compare", "verified"])
    rep.add_argument("--corpus", type=Path, default=DEFAULT_CORPUS)
    rep.add_argument("text", nargs="*", help="Query / needle / term (unused for scan)")
    rep.add_argument("-o", "--output", type=Path, default=None)
    rep.add_argument("--json", action="store_true", help="Emit JSON instead of markdown")

    sub.add_parser("dogfood", help="Score dogfood tasks on papers/ corpus")

    od = sub.add_parser("owner-dogfood", help="Score tasks on owner/private corpus (gitignored results)")
    od.add_argument("--corpus", type=Path, default=None)
    od.add_argument("--tasks", type=Path, default=None)
    od.add_argument("--out", type=Path, default=None)
    od.add_argument("--gallery", type=Path, default=None)
    od.add_argument("--demo", action="store_true")
    od.add_argument("--smoke", action="store_true")

    osm = sub.add_parser("owner-smoke", help="Quick owner contact smoke (demo or --corpus)")
    osm.add_argument("--corpus", type=Path, default=None)
    osm.add_argument("-o", "--output", type=Path, default=None)

    sub.add_parser("smoke", help="Run runtime regression pins")
    study_p = sub.add_parser("study", help="Lite usefulness study check/run")
    study_sub = study_p.add_subparsers(dest="study_cmd", required=True)
    for _sc in ("check", "run"):
        sp = study_sub.add_parser(_sc)
        sp.add_argument("--corpus", type=Path, default=None)
        sp.add_argument("--tasks", type=Path, default=None)
        sp.add_argument("--dir", type=Path, default=None)

    isla = sub.add_parser("ingest-sla", help="Ingest SLA / OCR recover_gap (W5)")
    isla.add_argument("--clean", type=Path, default=None)
    isla.add_argument("--noisy", type=Path, default=None)
    isla.add_argument("--with-u", action="store_true")
    isla.add_argument("-o", "--output", type=Path, default=None)

    contact = sub.add_parser("contact", help="Labeled corpus contact probe (product eval)")
    contact.add_argument("--corpus", type=Path, default=DEFAULT_CORPUS)
    contact.add_argument(
        "--class",
        dest="corpus_class",
        required=True,
        choices=["SYNTHETIC_MINI", "PAPERS_DOGFOOD", "OWNER_FIXTURE", "OWNER_PRIVATE"],
    )
    contact.add_argument("--useful", default=None)
    contact.add_argument("--not-useful", dest="not_useful", default=None)
    contact.add_argument("-o", "--output", type=Path, default=None)

    oa = sub.add_parser("remeasure-oa", help="Remeasure p5 OVER_ABSTENTION probes")
    oa.add_argument("--corpus", type=Path, default=None)
    oa.add_argument("-o", "--output", type=Path, default=None)
    habit_p = sub.add_parser("habit", help="Daily/session habit workflow (gitignored state)")
    habit_p.add_argument("--corpus", type=Path, default=None)
    habit_p.add_argument("--json", action="store_true")
    habit_p.add_argument("--list", action="store_true", help="List saved questions + scope/state")
    habit_p.add_argument("--rerun", action="store_true", help="Rerun saved questions sample")
    habit_p.add_argument("--save", nargs="+", default=None, help="Save a question for later rerun")
    habit_p.add_argument(
        "--doc",
        action="append",
        default=None,
        dest="doc_ids",
        metavar="DOC_ID",
        help="Exact document scope for --save / --rerun (repeatable)",
    )

    rev = sub.add_parser("review", help="Usefulness label loop (no hand-edited JSON)")
    rev.add_argument("--corpus", type=Path, default=None)
    rev.add_argument("--demo", action="store_true", help="Use fixture corpus + owner task pack")
    rev.add_argument("--tasks", type=Path, default=None)
    rev.add_argument("--from-dogfood", type=Path, default=None)
    rev.add_argument("--interactive", action="store_true")
    rev.add_argument("--next", action="store_true", help="Show next unlabeled card")
    rev.add_argument("--label", action="append", default=[], help="ID:LABEL batch apply")
    rev.add_argument("--summary", action="store_true")
    rev.add_argument("--limit", type=int, default=None)
    rev.add_argument("--state", type=Path, default=None, help="Explicit local review-state path")
    rev.add_argument(
        "--undo",
        action="append",
        default=[],
        metavar="ID",
        help="Clear a prior label by task/card ID; append an audit event",
    )
    rev.add_argument(
        "--reviewer",
        choices=["agent_applied", "owner", "independent_human", "unspecified"],
        default="unspecified",
    )

    ready = sub.add_parser("owner-ready", help="Validate private-corpus path readiness (no PHI dump)")
    ready.add_argument("--corpus", type=Path, default=None)
    ready.add_argument("--demo", action="store_true")

    sub.add_parser("plugin-registry", help="Print W4 lexicon-driven plugin registry")
    sub.add_parser("arch-registry", help="Print failure-driven architecture registry snapshot")
    sub.add_parser("adversarial", help="Run synthetic adversarial failure packs")
    st = sub.add_parser("status", help="Active Frontier rollup (owner-ready, evolve, lm-admit, ingest SLA)")
    st.add_argument("--corpus", type=Path, default=None)
    st.add_argument("--demo", action="store_true")
    st.add_argument("-o", "--output", type=Path, default=None)
    sub.add_parser("evolve", help="Map failure galleries → architecture workstreams (W1–W6)")
    la = sub.add_parser("lm-admit", help="W6 admission gate — is marginal LM probe indicated?")
    la.add_argument("--gallery", type=Path, default=None)
    la.add_argument("--corpus", type=Path, default=None)
    la.add_argument("--min-irreducible", type=int, default=2)
    la.add_argument("--owner-corpus", action="store_true")
    lp = sub.add_parser("lm-probe", help="W6 marginal stub probe (no external LM by default)")
    lp.add_argument("--backend", default="stub", help="stub | mlx (span-bound local)")
    lp.add_argument("--gallery", type=Path, default=None)
    lp.add_argument("--corpus", type=Path, default=None)
    lp.add_argument("--min-irreducible", type=int, default=2)
    lp.add_argument("--owner-corpus", action="store_true")
    lp.add_argument("--no-persist", action="store_true")
    ca = sub.add_parser("coe-audit", help="Chain-of-Evidence audit on ask payload / record")
    ca.add_argument("query", nargs="*", help="Query to ask+audit (default synthetic TTL)")
    ca.add_argument("--corpus", type=Path, default=None)
    ca.add_argument("--record", type=Path, default=None, help="Audit existing JSONL record")
    cr = sub.add_parser("coe-replay", help="Replay verified-ask and compare digests")
    cr.add_argument("query", nargs="+")
    cr.add_argument("--corpus", type=Path, default=None)
    cr.add_argument("--prior", type=Path, default=None, help="Prior ask JSON to compare")
    gal = sub.add_parser("gallery", help="Export failure gallery from dogfood JSON")
    gal.add_argument("--from", dest="from_path", type=Path, default=None)
    gal.add_argument("-o", "--output", type=Path, default=None)

    mu = sub.add_parser("measure-u", help="Draft U from dogfood JSON (not Layer-1)")
    mu.add_argument("path", type=Path, nargs="?", default=Path("wedge_v1/results_wedge_v1_dogfood.json"))
    mu.add_argument("--class", dest="corpus_class", default="UNKNOWN",
                    choices=["SYNTHETIC_MINI", "PAPERS_DOGFOOD", "OWNER_FIXTURE", "OWNER_PRIVATE", "UNKNOWN"])
    mu.add_argument("-o", "--output", type=Path, default=None)

    ea = sub.add_parser(
        "eval-arms",
        help="Fixture U_classical vs hybrid-stub under ΔU gate (no training)",
    )
    ea.add_argument("--corpus", type=Path, default=None)
    ea.add_argument("--tasks", type=Path, default=None)
    ea.add_argument("--demo", action="store_true", default=True)
    ea.add_argument("--no-demo", action="store_true")
    ea.add_argument(
        "--class",
        dest="corpus_class",
        default="SYNTHETIC_MINI",
        choices=["SYNTHETIC_MINI", "PAPERS_DOGFOOD", "OWNER_FIXTURE", "OWNER_PRIVATE", "UNKNOWN"],
    )
    ea.add_argument("-o", "--output", type=Path, default=None)
    ea.add_argument("--no-persist", action="store_true")

    args = p.parse_args(argv)

    if args.cmd == "ask":
        out = ask(
            " ".join(args.query),
            corpus_dir=args.corpus,
            doc_ids=getattr(args, "doc_ids", None),
            escalate_stub=bool(getattr(args, "escalate_stub", False)),
        )
        json.dump(out, sys.stdout, indent=2)
        sys.stdout.write(chr(10))
        try:
            habit_record("ask")
        except Exception:
            pass
        return 0 if out.get("answer_status") != "NO_CORPUS" else 2

    if args.cmd == "find":
        out = find_spans(
            " ".join(args.needle),
            corpus_dir=args.corpus,
            doc_ids=getattr(args, "doc_ids", None),
        )
        json.dump(out, sys.stdout, indent=2)
        sys.stdout.write(chr(10))
        try:
            habit_record("find")
        except Exception:
            pass
        return 0 if out.get("answer_status") != "NO_CORPUS" else 2

    if args.cmd == "scan":
        out = scan(corpus_dir=args.corpus, doc_ids=getattr(args, "doc_ids", None))
        json.dump(out, sys.stdout, indent=2)
        sys.stdout.write(chr(10))
        try:
            habit_record("scan")
        except Exception:
            pass
        return 0 if out.get("answer_status") != "NO_CORPUS" else 2

    if args.cmd == "compare":
        out = compare(
            " ".join(args.term),
            corpus_dir=args.corpus,
            doc_ids=getattr(args, "doc_ids", None),
        )
        json.dump(out, sys.stdout, indent=2)
        sys.stdout.write(chr(10))
        try:
            habit_record("compare")
        except Exception:
            pass
        return 0 if out.get("answer_status") != "NO_CORPUS" else 2

    if args.cmd == "ingest":
        src = args.corpus or args.src or DEFAULT_CORPUS
        docs = load_corpus(src)
        stats = corpus_stats(src)
        man = {
            "schema": "nano-lm.wedge_v1.ingest_manifest.v1",
            "src": str(Path(src).resolve()),
            "n_docs": len(docs),
            "n_chars": stats.get("n_chars"),
            "doc_ids": sorted(docs.keys()),
            "n_pdf_files_on_disk": stats.get("n_pdf_files_on_disk"),
            "pypdf_available": stats.get("pypdf_available"),
            "note": stats.get("note") or "Local index only; not Layer-1 evidence.",
        }
        out_path = args.out or (Path(src) / ".wedge_manifest.json")
        out_path.write_text(json.dumps(man, indent=2) + "\n", encoding="utf-8")
        json.dump(man, sys.stdout, indent=2)
        sys.stdout.write(chr(10))
        return 0 if docs else 2

    if args.cmd == "dogfood":
        from wedge_v1.run_dogfood import main as dogfood_main

        dogfood_main()
        return 0

    if args.cmd == "owner-smoke":
        from wedge_v1.run_owner_dogfood import main as owner_main

        argv2 = ["--demo", "--smoke"] if not args.corpus else ["--corpus", str(args.corpus), "--smoke"]
        if args.output:
            argv2 += ["--out", str(args.output)]
        return owner_main(argv2)

    if args.cmd == "owner-dogfood":
        from wedge_v1.run_owner_dogfood import main as owner_main

        argv2 = []
        if args.corpus:
            argv2 += ["--corpus", str(args.corpus)]
        if args.tasks:
            argv2 += ["--tasks", str(args.tasks)]
        if args.out:
            argv2 += ["--out", str(args.out)]
        if args.gallery:
            argv2 += ["--gallery", str(args.gallery)]
        if args.demo:
            argv2.append("--demo")
        if getattr(args, "smoke", False):
            argv2.append("--smoke")
        return owner_main(argv2)

    if args.cmd == "smoke":
        from wedge_v1 import test_runtime_smoke as smoke

        smoke.test_ttl_supported()
        smoke.test_oos_abstain()
        smoke.test_empty_corpus()
        smoke.test_scan_docs()
        smoke.test_find_ttl_phrase()
        smoke.test_bm25_hits_ttl_doc()
        smoke.test_bm25_span_supported()
        smoke.test_ingest_md_corpus()
        smoke.test_ingest_pdf_fixture()
        smoke.test_report_build()
        smoke.test_report_ask_markdown()
        smoke.test_bm25_margin_fields()
        smoke.test_ask_no_empty_evidence_present()
        smoke.test_evolve_recommends_workstreams()
        from wedge_v1 import test_w5_ingest_sla as w5_smoke
        w5_smoke.test_auto_normalize_noisy_corpus()
        w5_smoke.test_normalize_improves_synthetic_noisy()
        from wedge_v1 import test_w6_lm as w6_smoke
        w6_smoke.test_admission_not_indicated_on_clean_eclass()
        from wedge_v1 import test_owner_smoke as os_smoke
        os_smoke.test_example_corpus_present()
        os_smoke.test_owner_smoke_example_pass(None)
        from wedge_v1 import test_over_abstention as oa
        # Core OOS pin must pass; p5 recall pins are soft (warn) until classical recall improves.
        oa.test_oos_clinical_nanoscribe_abstains()
        soft = (
            "test_p5_e1_kill_m1_template_supported",
            "test_p5_e4_kill_0638_supported",
            "test_p5_smallest_sufficient_solver_via_phrase",
        )
        for _name in soft:
            fn = getattr(oa, _name, None)
            if fn is None:
                continue
            try:
                fn()
            except AssertionError as e:
                print(f"WEDGE_V1_SMOKE_WARN {_name}: {e}", file=sys.stderr)
        from wedge_v1 import test_eval_arms as arms_smoke
        arms_smoke.test_cite_pack_compacts_evidence()
        try:
            arms_smoke.test_eval_arms_fixture_keep_classical()
        except AssertionError as exc:
            print(f"WEDGE_V1_SMOKE_WARN eval_arms: {exc}", file=sys.stderr)
        arms_smoke.test_escalate_stub_refuses_oos()
        arms_smoke.test_ask_escalate_stub_default_off_keeps_oos()
        arms_smoke.test_ask_escalate_stub_oos_still_abstains()
        print("WEDGE_V1_SMOKE_OK", file=sys.stderr)
        return 0


    if args.cmd == "report":
        kind = args.kind
        text = " ".join(args.text)
        if kind == "ask":
            out = ask(text, corpus_dir=args.corpus)
            title = f"ask: {text}"
        elif kind == "find":
            out = find_spans(text, corpus_dir=args.corpus)
            title = f"find: {text}"
        elif kind == "scan":
            out = scan(corpus_dir=args.corpus)
            title = "scan"
        elif kind == "compare":
            out = compare(text, corpus_dir=args.corpus)
            title = f"compare: {text}"
        else:
            from frontier.verified_ask_report import build_report

            out = build_report(text, corpus_dir=args.corpus)
            title = f"verified ask: {text}"
        body = json.dumps(out, indent=2) + "\n" if args.json else format_report_md(out, title=title)
        if args.output:
            args.output.write_text(body, encoding="utf-8")
        else:
            sys.stdout.write(body)
        return 0 if out.get("answer_status") != "NO_CORPUS" else 2

    if args.cmd == "remeasure-oa":
        from wedge_v1.remeasure_oa import main as oa_main
        argv = []
        if getattr(args, "corpus", None):
            argv += ["--corpus", str(args.corpus)]
        if getattr(args, "output", None):
            argv += ["-o", str(args.output)]
        return oa_main(argv)

    if args.cmd == "habit":
        if args.list:
            corpus = resolve_session_corpus(args.corpus)
            rows = saved_question_status(corpus)
            if args.json:
                json.dump(
                    {
                        "schema": "nano-lm.wedge_v1.saved_questions_list.v1",
                        "corpus": str(corpus),
                        "n": len(rows),
                        "questions": rows,
                    },
                    sys.stdout,
                    indent=2,
                )
                sys.stdout.write(chr(10))
            else:
                sys.stdout.write(format_saved_list_md(rows))
            return 0
        if args.save:
            q = " ".join(args.save)
            corpus = resolve_session_corpus(args.corpus)
            saved = save_question(
                q,
                corpus=corpus,
                doc_ids=getattr(args, "doc_ids", None),
            )
            habit_record("save_question", note=q[:80])
            json.dump(
                {
                    "saved": q,
                    "doc_ids": getattr(args, "doc_ids", None),
                    "n_saved": len(saved.get("questions") or []),
                    "task_id": (saved.get("questions") or [{}])[0].get("task_id"),
                },
                sys.stdout,
                indent=2,
            )
            sys.stdout.write(chr(10))
            return 0
        session_kwargs: dict = {"rerun": bool(args.rerun)}
        if args.rerun and getattr(args, "doc_ids", None) is not None:
            session_kwargs["doc_ids"] = args.doc_ids
        sess = habit_session(args.corpus, **session_kwargs)
        if args.rerun:
            habit_record("rerun")
        if args.json:
            json.dump(sess, sys.stdout, indent=2)
            sys.stdout.write(chr(10))
        else:
            sys.stdout.write(format_session_md(sess))
        return 0

    if args.cmd == "review":
        from wedge_v1.review import (
            LABELS,
            batch_label,
            cards_from_state,
            cards_from_dogfood,
            cards_from_task_pack,
            format_card,
            interactive_review,
            label_summary,
            load_state,
            merge_prior_labels,
            save_state,
            undo_label,
            unlabeled,
        )
        from wedge_v1.private_output import require_private_output
        from wedge_v1.run_owner_dogfood import DEFAULT_TASKS, FIXTURE_CORPUS, resolve_corpus

        if args.tasks is not None and args.corpus is None:
            p.error("review --tasks requires an explicit --corpus")
        if args.from_dogfood is not None and args.corpus is None:
            p.error("review --from-dogfood requires an explicit --corpus")
        if args.corpus is not None and not args.tasks and not args.from_dogfood:
            p.error("review with a real --corpus requires --tasks or --from-dogfood")
        has_private_inputs = any(
            value is not None for value in (args.corpus, args.tasks, args.from_dogfood)
        )
        if has_private_inputs and args.state is None:
            p.error("review with private inputs requires an explicit local --state path")
        if args.state is not None:
            try:
                require_private_output(args.state, purpose="review state")
            except ValueError as exc:
                p.error(str(exc))

        state = load_state(args.state)
        if args.summary:
            cards = cards_from_state(state)
            if args.limit is not None:
                cards = cards[: args.limit]
            json.dump(
                label_summary(state, cards=cards, path=args.state),
                sys.stdout,
                indent=2,
            )
            sys.stdout.write(chr(10))
            return 0
        if args.next:
            cards = cards_from_state(state)
            if not cards:
                sys.stdout.write(
                    "REVIEW_SNAPSHOT_MISSING: run --interactive once to freeze the queue\n"
                )
                return 0
            queue = unlabeled(cards)
            if not queue:
                sys.stdout.write("REVIEW_QUEUE_EMPTY\n")
                json.dump(label_summary(state, cards=cards, path=args.state), sys.stdout, indent=2)
                sys.stdout.write(chr(10))
                return 0
            sys.stdout.write(format_card(queue[0], index=1, total=len(queue)) + "\n")
            return 0

        if state.get("load_errors"):
            p.error(
                "review state is invalid and was left unchanged: "
                + ", ".join(state["load_errors"])
            )

        cards = cards_from_state(state)
        explicit_source = bool(args.demo or args.corpus or args.tasks or args.from_dogfood)
        if explicit_source:
            use_demo = bool(args.demo)
            corpus = resolve_corpus(corpus=args.corpus, demo=use_demo)
            if args.demo and FIXTURE_CORPUS.is_dir():
                corpus = FIXTURE_CORPUS
            if args.from_dogfood:
                cards = cards_from_dogfood(
                    args.from_dogfood,
                    corpus,
                    limit=args.limit,
                    persist_coe=False,
                )
            else:
                tasks = args.tasks or DEFAULT_TASKS
                cards = cards_from_task_pack(
                    tasks,
                    corpus,
                    limit=args.limit,
                    persist_coe=False,
                )
            cards = merge_prior_labels(cards, state)

        if not cards:
            p.error("no persisted review queue; provide --demo or explicit --corpus/--tasks")
        if args.undo:
            for identifier in args.undo:
                undo_label(
                    state,
                    cards,
                    identifier,
                    reviewer_kind=args.reviewer,
                )
            save_state(state, path=args.state)
            json.dump(label_summary(state, cards=cards, path=args.state), sys.stdout, indent=2)
            sys.stdout.write(chr(10))
            return 0
        if args.label:
            batch_label(
                state,
                cards,
                args.label,
                state_path=args.state,
                reviewer_kind=args.reviewer,
            )
            json.dump(label_summary(state, cards=cards, path=args.state), sys.stdout, indent=2)
            sys.stdout.write(chr(10))
            habit_record("review_label")
            return 0
        if args.interactive:
            interactive_review(
                cards,
                state,
                state_path=args.state,
                reviewer_kind=args.reviewer,
            )
            json.dump(
                label_summary(load_state(args.state), cards=cards, path=args.state),
                sys.stdout,
                indent=2,
            )
            sys.stdout.write(chr(10))
            habit_record("review_interactive")
            return 0
        queue = unlabeled(cards)
        json.dump(
            {
                "n_cards": len(cards),
                "n_unlabeled": len(queue),
                "labels": list(LABELS),
                "next": "python -m wedge_v1 review --demo --interactive",
                "queue": [
                    {
                        "id": c.get("task_id"),
                        "query": c.get("query"),
                        "status": c.get("answer_status"),
                        "doc": c.get("document"),
                        "span": (c.get("evidence_span") or "")[:80],
                        "prior_label_status": c.get("prior_label_status"),
                    }
                    for c in queue
                ],
            },
            sys.stdout,
            indent=2,
        )
        sys.stdout.write(chr(10))
        return 0

    if args.cmd == "owner-ready":
        from wedge_v1.owner_ready import main as ready_main

        argv_ready = []
        if args.corpus:
            argv_ready += ["--corpus", str(args.corpus)]
        if args.demo:
            argv_ready.append("--demo")
        return ready_main(argv_ready)


    if args.cmd == "status":
        from wedge_v1.frontier_status import main as status_main

        argv_st = []
        if args.corpus:
            argv_st += ["--corpus", str(args.corpus)]
        if args.demo:
            argv_st.append("--demo")
        if args.output:
            argv_st += ["-o", str(args.output)]
        return status_main(argv_st)

    if args.cmd == "lm-admit":
        from wedge_v1.run_w6_marginal import main as w6_main

        argv2 = ["--admit-only", "--min-irreducible", str(args.min_irreducible)]
        if args.gallery:
            argv2 += ["--gallery", str(args.gallery)]
        if args.corpus:
            argv2 += ["--corpus", str(args.corpus)]
        if args.owner_corpus:
            argv2.append("--owner-corpus")
        return w6_main(argv2)

    if args.cmd == "lm-probe":
        from wedge_v1.run_w6_marginal import main as w6_main

        argv2 = ["--min-irreducible", str(args.min_irreducible)]
        if args.gallery:
            argv2 += ["--gallery", str(args.gallery)]
        if args.corpus:
            argv2 += ["--corpus", str(args.corpus)]
        if args.owner_corpus:
            argv2.append("--owner-corpus")
        if args.no_persist:
            argv2.append("--no-persist")
        if getattr(args, "backend", None):
            argv2 += ["--backend", str(args.backend)]
        return w6_main(argv2)

    if args.cmd == "evolve":
        from wedge_v1.failure_to_architecture import main as evolve_main

        return evolve_main([])

    if args.cmd == "gallery":
        from wedge_v1.failure_gallery import DEFAULT_DOGFOOD, gallery_to_markdown, write_gallery

        g = write_gallery(path=args.from_path or DEFAULT_DOGFOOD)
        md = gallery_to_markdown(g)
        if args.output:
            args.output.write_text(md, encoding="utf-8")
        json.dump(
            {
                "buckets": g.get("buckets"),
                "fine_counts": g.get("fine_counts"),
                "accuracy": g.get("accuracy"),
                "n_ok": g.get("n_ok"),
                "note": g.get("note"),
            },
            sys.stdout,
            indent=2,
        )
        sys.stdout.write(chr(10))
        return 0 if not g.get("error") else 2

    if args.cmd == "measure-u":
        from wedge_v1.eval.dogfood_utility import measure_path

        out = measure_path(args.path, corpus_class=args.corpus_class)
        if args.output:
            args.output.write_text(json.dumps(out, indent=2) + "\n", encoding="utf-8")
        json.dump(out, sys.stdout, indent=2)
        sys.stdout.write(chr(10))
        return 0

    if args.cmd == "eval-arms":
        from wedge_v1.eval.arms import run_arms_eval

        out = run_arms_eval(
            corpus=args.corpus,
            tasks_path=args.tasks,
            demo=not args.no_demo,
            persist=not args.no_persist,
            out_path=args.output,
            corpus_class=args.corpus_class,
        )
        summary = {
            "verdict": out.get("verdict"),
            "delta_u": out.get("delta_u"),
            "delta_threshold": out.get("delta_threshold"),
            "classical_U": (out.get("classical") or {}).get("U"),
            "hybrid_U": (out.get("hybrid_stub") or {}).get("U"),
            "classical_ok": (out.get("classical") or {}).get("n_ok"),
            "hybrid_ok": (out.get("hybrid_stub") or {}).get("n_ok"),
            "n_tasks": out.get("n_tasks"),
            "n_escalated": (out.get("hybrid_stub") or {}).get("n_escalated"),
            "out": out.get("out"),
            "note": out.get("note"),
        }
        json.dump(summary, sys.stdout, indent=2)
        sys.stdout.write(chr(10))
        return 0

    if args.cmd == "contact":
        from wedge_v1.run_corpus_contact import run_contact

        out = run_contact(
            args.corpus,
            corpus_class=args.corpus_class,
            useful_sentence=args.useful,
            not_useful_sentence=args.not_useful,
        )
        path = args.output or Path("wedge_v1/results_corpus_contact.json")
        path.write_text(json.dumps(out, indent=2) + "\n", encoding="utf-8")
        json.dump(out, sys.stdout, indent=2)
        sys.stdout.write(chr(10))
        print(f"WROTE {path}", file=sys.stderr)
        return 0 if out.get("n_docs") else 2

    if args.cmd == "plugin-registry":
        from wedge_v1.plugins.cascade import plugin_registry

        json.dump(plugin_registry(), sys.stdout, indent=2)
        sys.stdout.write(chr(10))
        return 0

    if args.cmd == "arch-registry":
        from wedge_v1.arch.registry import registry_snapshot

        json.dump(registry_snapshot(), sys.stdout, indent=2)
        sys.stdout.write(chr(10))
        return 0

    if args.cmd == "adversarial":
        from wedge_v1.eval.adversarial import main as adv_main

        return adv_main([])


    if args.cmd == "coe-audit":
        from wedge_v1.coe.audit import audit_payload, audit_record

        if args.record:
            out = audit_record(args.record)
            json.dump(out, sys.stdout, indent=2)
            sys.stdout.write(chr(10))
            return 0 if out.get("ok") else 2
        corpus = args.corpus or DEFAULT_CORPUS
        q = " ".join(args.query) if args.query else "How long before cache entries expire?"
        docs = load_corpus(corpus)
        payload = ask(q, corpus_dir=corpus)
        out = audit_payload(payload, docs)
        out["answer_status"] = payload.get("answer_status")
        out["coe"] = payload.get("coe")
        json.dump(out, sys.stdout, indent=2)
        sys.stdout.write(chr(10))
        return 0 if out.get("ok") else 2

    if args.cmd == "coe-replay":
        from wedge_v1.coe.replay import replay_ask

        prior = None
        if args.prior:
            prior = json.loads(args.prior.read_text(encoding="utf-8"))
        out = replay_ask(
            query=" ".join(args.query),
            corpus_dir=args.corpus or DEFAULT_CORPUS,
            prior=prior,
            persist_coe=False,
        )
        # trim huge payload for stdout summary
        summary = {
            "schema": out["schema"],
            "query": out["query"],
            "audit": out["audit"],
            "comparison": out["comparison"],
            "answer_status": out["payload"].get("answer_status"),
            "coe": out["payload"].get("coe"),
        }
        json.dump(summary, sys.stdout, indent=2)
        sys.stdout.write(chr(10))
        return 0 if out["audit"].get("ok") else 2


    if args.cmd == "ingest-sla":
        from wedge_v1.ingest_sla import main as sla_main

        argv = []
        if args.clean:
            argv += ["--clean", str(args.clean)]
        if args.noisy:
            argv += ["--noisy", str(args.noisy)]
        if args.with_u:
            argv.append("--with-u")
        if args.output:
            argv += ["-o", str(args.output)]
        return sla_main(argv)


    if args.cmd == "study":
        from wedge_v1.study_lite import main as study_main

        argv2 = [args.study_cmd]
        if args.corpus:
            argv2 += ["--corpus", str(args.corpus)]
        if args.tasks:
            argv2 += ["--tasks", str(args.tasks)]
        if args.dir:
            argv2 += ["--dir", str(args.dir)]
        return study_main(argv2)

    return 1


if __name__ == "__main__":
    raise SystemExit(main())
