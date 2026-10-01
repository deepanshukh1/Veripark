"""Single-file HTML report for a non-technical Product Owner. No JavaScript, no external assets.

Everything that comes from the chatbot or the data is HTML-escaped: chatbot output is untrusted.
"""
from __future__ import annotations

import html
from collections import defaultdict

from .models import CaseResult, Verdict
from .pipeline import summarize

ISSUE_LABELS = {
    "wrong_fact": "Wrong fact (number, fee, rule)",
    "hallucination": "Invented information",
    "outdated": "Outdated information",
    "refusal": "Refused or unhelpful",
    "missing_info": "Incomplete answer",
    "low_similarity": "Does not match the approved answer",
    "other": "Other problem",
}
REVIEW_LABELS = {
    "evaluator_error": "An automated check failed to run",
    "unstable_judge": "The AI judge gave inconsistent scores",
    "judge_pass_but_hallucination_flagged": "Judge approved it, but the fact-check found claims the approved answer does not support",
    "judge_fail_but_other_checks_pass": "Judge rejected it, but the other checks found no problem",
    "borderline_judge_score": "Score is close to the pass line",
    "high_risk_topic": "High-risk topic: policy requires a human to sign off",
    "target_error": "The chatbot did not return an answer",
}

CSS = """
:root{--bg:#f3f5f8;--panel:#fff;--ink:#1b2430;--muted:#5b6675;--rule:#d5dae1;--pass:#1f7a4d;--fail:#b3261e;--review:#a35f00}
@media (prefers-color-scheme:dark){:root{--bg:#10151c;--panel:#18202a;--ink:#e6ebf1;--muted:#9aa6b5;--rule:#2b3644;--pass:#4cc38a;--fail:#ff7b72;--review:#e3a23b}}
*{box-sizing:border-box}
body{margin:0;background:var(--bg);color:var(--ink);font:16px/1.55 system-ui,-apple-system,"Segoe UI",Roboto,sans-serif}
main{max-width:1040px;margin:0 auto;padding:32px 20px 64px}
h1{font-size:1.7rem;margin:0 0 4px}
h2{font-size:1.2rem;margin:40px 0 8px}
h3{font-size:1rem;margin:20px 0 6px}
p{margin:6px 0}.muted{color:var(--muted)}.small{font-size:.86rem}
.meta{color:var(--muted);font-size:.86rem;margin-bottom:20px}
.headline{background:var(--panel);border:1px solid var(--rule);border-left:6px solid var(--c);padding:18px 22px;border-radius:6px}
.headline .big{font-size:2.4rem;font-weight:700;color:var(--c);line-height:1.1}
.bar{display:flex;height:14px;border-radius:7px;overflow:hidden;margin:16px 0 6px;background:var(--rule)}
.bar span{display:block}
.legend{display:flex;gap:20px;flex-wrap:wrap;font-size:.9rem}
.dot{display:inline-block;width:10px;height:10px;border-radius:50%;margin-right:6px}
.pass{--c:var(--pass)}.fail{--c:var(--fail)}.review{--c:var(--review)}
.bg-pass{background:var(--pass)}.bg-fail{background:var(--fail)}.bg-review{background:var(--review)}
.tablewrap{overflow-x:auto;background:var(--panel);border:1px solid var(--rule);border-radius:6px}
table{border-collapse:collapse;width:100%;font-size:.9rem}
th,td{text-align:left;vertical-align:top;padding:9px 12px;border-bottom:1px solid var(--rule)}
th{font-weight:600;color:var(--muted);white-space:nowrap}
tr:last-child td{border-bottom:0}
td.q{min-width:200px}td.a{min-width:240px}
.tag{display:inline-block;padding:1px 8px;border-radius:10px;font-size:.78rem;font-weight:600;border:1px solid var(--c);color:var(--c)}
details{background:var(--panel);border:1px solid var(--rule);border-radius:6px;padding:10px 14px;margin-top:12px}
summary{cursor:pointer;font-weight:600}
footer{margin-top:48px;color:var(--muted);font-size:.84rem;border-top:1px solid var(--rule);padding-top:14px}
"""


def _e(text) -> str:
    return html.escape("" if text is None else str(text))


def _table(headers: list[str], rows: list[list[str]], classes: list[str] | None = None) -> str:
    classes = classes or [""] * len(headers)
    head = "".join(f"<th>{_e(h)}</th>" for h in headers)
    body = "".join("<tr>" + "".join(f'<td class="{c}">{cell}</td>' for cell, c in zip(row, classes)) + "</tr>" for row in rows)
    return f'<div class="tablewrap"><table><thead><tr>{head}</tr></thead><tbody>{body}</tbody></table></div>'


def _fail_group(r: CaseResult) -> str:
    return r.decision.reason if r.decision.reason in ISSUE_LABELS else "other"


def _scores(r: CaseResult) -> str:
    ev = r.evaluation
    if ev.judge is None or ev.similarity is None:
        return "n/a"
    return f"judge {ev.judge.score:.2f}<br>similarity {ev.similarity:.2f}"


def render_report(results: list[CaseResult], meta: dict) -> str:
    s = summarize(results)
    n = max(1, s["total"])
    fails = [r for r in results if r.decision.verdict is Verdict.FAIL]
    reviews = [r for r in results if r.decision.verdict is Verdict.REVIEW]
    passes = [r for r in results if r.decision.verdict is Verdict.PASS]
    tone = "fail" if fails else ("review" if reviews else "pass")
    avg_latency = sum(r.reply.latency_s for r in results) / n

    out = [f"<!doctype html><html lang='en'><head><meta charset='utf-8'><meta name='viewport' content='width=device-width,initial-scale=1'>"
           f"<title>ChatEval report</title><style>{CSS}</style></head><body><main>",
           "<h1>Chatbot regression report</h1>",
           f"<p class='meta'>{_e(meta['run_at'])} &middot; bot: {_e(meta['target'])} &middot; judge: {_e(meta['provider'])}/{_e(meta['chat_model'])}"
           f" &middot; prompts {_e(meta['prompt_version'])} &middot; {s['total']} questions &middot; average bot latency {avg_latency:.2f}s</p>"]

    out.append(f"<div class='headline {tone}'><div class='big'>{s['pass_rate']:.0%} passed</div>"
               f"<p>{s['pass']} of {s['total']} answers were judged correct. <b>{s['fail']}</b> failed. "
               f"<b>{s['review']}</b> need a person to decide.</p>"
               f"<p class='muted small'>This is a screening result, not a release sign-off. Every item in the review queue, "
               f"and a sample of the passes, still needs a human check.</p></div>")
    out.append("<div class='bar'>" + "".join(
        f"<span class='bg-{c}' style='width:{100 * k / n:.2f}%'></span>"
        for c, k in (("pass", s["pass"]), ("review", s["review"]), ("fail", s["fail"]))) + "</div>")
    out.append("<div class='legend'>" + "".join(
        f"<span><span class='dot bg-{c}'></span>{label} {k}</span>"
        for c, label, k in (("pass", "Pass", s["pass"]), ("review", "Needs human review", s["review"]), ("fail", "Fail", s["fail"]))) + "</div>")

    # Failures grouped by reason
    out.append("<h2>Failures by reason</h2>")
    if not fails:
        out.append("<p class='muted'>No answers failed.</p>")
    groups: dict[str, list[CaseResult]] = defaultdict(list)
    for r in fails:
        groups[_fail_group(r)].append(r)
    for key, items in sorted(groups.items(), key=lambda kv: -len(kv[1])):
        out.append(f"<h3>{_e(ISSUE_LABELS[key])} ({len(items)})</h3>")
        out.append(_table(["ID", "Customer question", "Bot said", "Approved answer", "Why it failed"],
                          [[_e(r.case.id), _e(r.case.question), _e(r.reply.answer), _e(r.case.groundtruth),
                            _e(r.evaluation.judge.reasoning if r.evaluation.judge else r.decision.detail)] for r in items],
                          ["", "q", "a", "a", ""]))

    # Top hallucinations
    claims = [(r, c) for r in results if r.evaluation.hallucination for c in r.evaluation.hallucination.flagged]
    claims.sort(key=lambda rc: rc[1].status != "contradicted")
    out.append("<h2>Top hallucinations</h2>")
    out.append("<p class='muted small'>Statements in the bot's answer that the approved answer does not support (unsupported) or that it contradicts.</p>")
    if claims:
        out.append(_table(["ID", "Claim made by the bot", "Finding"],
                          [[_e(r.case.id), _e(c.text), f"<span class='tag {'fail' if c.status == 'contradicted' else 'review'}'>{_e(c.status)}</span>"]
                           for r, c in claims[:10]], ["", "a", ""]))
    else:
        out.append("<p class='muted'>None found.</p>")

    # Review queue
    out.append(f"<h2>Review queue ({len(reviews)})</h2>")
    out.append("<p class='muted small'>The tool is not confident about these. A person should decide.</p>")
    if reviews:
        out.append(_table(["ID", "Customer question", "Bot said", "Approved answer", "Why it is here", "Scores"],
                          [[_e(r.case.id), _e(r.case.question), _e(r.reply.answer), _e(r.case.groundtruth),
                            _e(REVIEW_LABELS.get(r.decision.reason, r.decision.reason)) + (f"<br><span class='muted small'>{_e(r.decision.detail)}</span>" if r.decision.detail else ""),
                            _scores(r)] for r in reviews], ["", "q", "a", "a", "", ""]))
    else:
        out.append("<p class='muted'>Nothing to review.</p>")

    # High-risk passes
    hr = [r for r in passes if r.case.high_risk]
    out.append(f"<h2>High-risk passes to spot-check ({len(hr)})</h2>")
    out.append("<p class='muted small'>Topics such as fees, limits, eligibility and compliance. An automated pass is not enough for these.</p>")
    if hr:
        out.append(_table(["ID", "Customer question", "Bot said"],
                          [[_e(r.case.id), _e(r.case.question), _e(r.reply.answer)] for r in hr], ["", "q", "a"]))
    else:
        out.append("<p class='muted'>None.</p>")

    # Everything
    rows = [[_e(r.case.id), f"<span class='tag {'pass' if r.decision.verdict is Verdict.PASS else 'fail' if r.decision.verdict is Verdict.FAIL else 'review'}'>{_e(r.decision.verdict.value)}</span>",
             _e(r.case.question), _scores(r)] for r in results]
    out.append("<details><summary>All results</summary>" + _table(["ID", "Verdict", "Customer question", "Scores"], rows, ["", "", "q", ""]) + "</details>")

    t = meta["thresholds"]
    out.append(f"<footer>Judge pass threshold {t['judge_pass']} (human review within &plusmn;{t['review_margin']}), "
               f"similarity threshold {t['similarity_pass']}. Automated judgement can be wrong: it can miss fluent wrong answers and "
               f"flag correct ones. Full per-answer detail is in results.csv and audit.jsonl.</footer></main></body></html>")
    return "".join(out)
