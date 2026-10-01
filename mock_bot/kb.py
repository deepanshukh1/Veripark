"""FAQ knowledge base for a FICTIONAL retail bank ("Marlow & Finch Bank"). All data is synthetic.

Five answers are deliberately defective so ChatEval has something to find (see PLANTED_ERRORS).
"""
from __future__ import annotations

import re

REFUSAL = "I'm sorry, I can't help with that. Please contact our support team."

PLANTED_ERRORS = {
    "faq_overdraft": "wrong fee (says £10/day cap £50; truth is £5/day cap £25)",
    "faq_crypto": "hallucinated product (invents a 'Crypto Saver' account)",
    "faq_atm": "outdated policy (still quotes the old £300 limit)",
    "faq_dormant": "refusal (declines a question it should answer)",
    "faq_documents": "hallucinated extra requirement (employer reference letter)",
    # Correct-but-reworded answers (should still PASS): faq_lost_card, faq_hours
}

FAQS = [
    {"id": "faq_overdraft", "keywords": "unarranged overdraft fee monthly daily charge",
     "answer": "Unarranged overdrafts are charged a flat fee of £10 per day, capped at £50 per month."},
    {"id": "faq_everyday", "keywords": "everyday current account cost monthly fee",
     "answer": "The Everyday Current Account has no monthly fee."},
    {"id": "faq_lost_card", "keywords": "report lost stolen debit card",
     "answer": "You can freeze your card instantly in the mobile app, or phone our 24-hour line on 0800-555-0142. "
               "We'll post a replacement, which normally arrives within 3 working days."},
    {"id": "faq_crypto", "keywords": "crypto cryptocurrency bitcoin trading account wallet",
     "answer": "Yes! Our Crypto Saver account lets you trade bitcoin and earn 4% monthly rewards."},
    {"id": "faq_junior", "keywords": "minimum age junior saver account open child",
     "answer": "Junior Saver accounts are for children under 16 and must be opened by a parent or guardian."},
    {"id": "faq_atm", "keywords": "daily atm cash withdrawal limit",
     "answer": "The daily ATM withdrawal limit is £300."},
    {"id": "faq_faster", "keywords": "faster payment long take transfer arrive",
     "answer": "Faster Payments usually arrive within seconds and always within 2 hours."},
    {"id": "faq_close", "keywords": "close closing account online",
     "answer": "Accounts can be closed in branch or by secure message in the app. Closure takes up to 5 working days."},
    {"id": "faq_fx", "keywords": "foreign transaction fee platinum credit card abroad",
     "answer": "The Platinum Credit Card charges a 2.5% foreign transaction fee."},
    {"id": "faq_dormant", "keywords": "dormant account inactive happens",
     "answer": REFUSAL},
    {"id": "faq_hours", "keywords": "branch opening hours open saturday",
     "answer": "Our branches open Monday to Friday from 9:00 to 17:00, and on Saturdays from 9:00 to 13:00."},
    {"id": "faq_saver", "keywords": "easy saver interest rate pay",
     "answer": "Easy Saver pays 3.1% AER variable."},
    {"id": "faq_protection", "keywords": "money protected safe deposit guarantee",
     "answer": "Eligible deposits are protected up to £85,000 per person under the national deposit guarantee scheme."},
    {"id": "faq_dd", "keywords": "set up direct debit",
     "answer": "Set up the direct debit with the company you are paying, using your sort code and account number; "
               "you can view and cancel it in the app."},
    {"id": "faq_invest", "keywords": "recommend good stock shares buy invest",
     "answer": "I'm not able to give investment advice. Please speak to our Investments team."},
    {"id": "faq_documents", "keywords": "documents need open account id proof address",
     "answer": "You need photo ID (passport or driving licence), proof of address dated within 3 months, "
               "and a reference letter from your employer."},
]


def _words(text: str) -> set[str]:
    return set(re.findall(r"[a-z]+", text.lower()))


def answer(question: str) -> tuple[str, str]:
    """Return (answer, faq_id). Crude keyword overlap - good enough for a demo target."""
    q = _words(question)
    best, best_score = None, 0
    for faq in FAQS:
        score = len(q & _words(faq["keywords"]))
        if score > best_score:
            best, best_score = faq, score
    if best is None or best_score < 2:
        return REFUSAL, "none"
    return best["answer"], best["id"]
