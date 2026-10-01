# Assumptions

Where the brief was open, I made these calls. Change any of them; none is load-bearing for the design.

1. **Fictional bank.** "Marlow & Finch Bank" and every fee, limit, phone number and product are invented. Currency is £ for readability only. Any resemblance to a real bank's terms is coincidental.
2. **Judge default model is `gpt-4o-mini`** (cheap, so a reviewer can afford to run it). The brief mentions GPT-4 as judge; set `llm.openai.chat_model` in `config.yaml` to the strongest model you can justify. Model names change; check your provider's current list.
3. **No OpenAI SDK.** The OpenAI client talks to the REST API with `httpx`, which keeps dependencies minimal and works with any OpenAI-compatible gateway via `OPENAI_BASE_URL`.
4. **The offline `--mock-llm` is a lexical heuristic, not a model.** It exists so a reviewer with no key can run everything. It is crude (it over-flags some correct paraphrases) and its benchmark numbers are *not* evidence. The report, console and benchmark file all say so.
5. **Thresholds are uncalibrated defaults.** `judge_pass=0.7` comes from the brief. `similarity_pass=0.75` is a guess for `text-embedding-3-small`; the mock uses 0.5. Real values must come from calibration on a labelled set.
6. **Verdict rules are my design, not a standard.** The judge is the primary signal; similarity and the hallucination check are cross-checks. Disagreement and borderline scores go to a human. The rules and their rationale are in `chateval/verdict.py` and unit-tested.
7. **A refusal counts as a failure unless the groundtruth says the bot should decline.** Groundtruth for out-of-scope questions is written as "The bot should politely decline...".
8. **"Hallucination" = a claim not supported by the groundtruth/source**, including true-in-the-world facts the reference omits. That is deliberately strict: for a bank bot, an unsourced claim is a risk even if plausible.
9. **Every answer is judged against the groundtruth alone** (single-answer grading, not pairwise), which sidesteps position bias but not verbosity bias.
10. **Benchmark answers are frozen** in `benchmark/benchmark_set.csv` so results are reproducible and independent of any bot.
11. **The benchmark set was drafted by the AI assistant, labels included.** Every row starts as `label_status = DRAFT`. The owner must read every row, correct labels where they disagree, and set `VERIFIED`. `benchmark.py` prints a warning on all output until every row is verified. The 56 rows (27 defective) are synthetic and cover correct, paraphrased, wrong, hallucinated, outdated, refused and one prompt-injection answer.
12. **`expected_keywords` for Baseline A** are the key fact(s) a tester would plausibly assert (e.g. `£500`, `no monthly fee`). A different choice would change Baseline A's numbers.
13. **Cost is never hard-coded.** Provider prices change, so `pricing:` in `config.yaml` is blank and the cost line stays a placeholder until you fill it.
14. **"High-risk" topics** are detected by keyword (`config.yaml: policy.high_risk_keywords`) or an explicit `risk=high` column. This is a crude router, not a classifier.
15. **XLSX** input reads the first sheet; the header row must contain `question` and `groundtruth`.
16. **English only.** Not tested on other languages.
17. **The HTML screenshot is not committed by the assistant.** The sandbox could not download a browser; `make screenshot` produces `docs/report_screenshot.png` on your machine.
