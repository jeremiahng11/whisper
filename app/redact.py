"""Hide identity and payment numbers in transcripts and notes (REDACT=1, the default).

- Singapore NRIC / FIN:  S1234567D, T 1234567 J, F/G/M ...           -> [NRIC]
- card numbers: 13-19 digits (spaces/dashes allowed) passing the Luhn check -> [card number]
- IBAN                                                              -> [IBAN]
- bank account numbers: 6-17 digits shortly after "account", "acct", "a/c" -> [account number]
"""
import re

NRIC = re.compile(r"\b[STFGM][\s-]?\d{3}[\s-]?\d{4}[\s-]?[A-Z]\b", re.I)
CARD = re.compile(r"(?<![\d-])(?:\d[ -]?){12,18}\d(?![\d-])")
IBAN = re.compile(r"\b[A-Z]{2}\d{2}(?:[ ]?[A-Z0-9]{4}){2,7}(?:[ ]?[A-Z0-9]{1,4})?\b")
ACCOUNT = re.compile(r"(\b(?:account|acct|a/c|bank account)(?:\s+(?:number|no\.?|num))?\s*(?:is|:|#)?\s*)((?:\d[ -]?){5,16}\d)", re.I)


def _luhn(digits: str) -> bool:
    total, alt = 0, False
    for ch in reversed(digits):
        d = int(ch)
        if alt:
            d *= 2
            if d > 9:
                d -= 9
        total += d
        alt = not alt
    return total % 10 == 0


def _card(m: re.Match) -> str:
    digits = re.sub(r"\D", "", m.group(0))
    return "[card number]" if 13 <= len(digits) <= 19 and _luhn(digits) else m.group(0)


def redact(text: str) -> str:
    if not text:
        return text
    text = NRIC.sub("[NRIC]", text)
    text = CARD.sub(_card, text)
    text = IBAN.sub(lambda m: "[IBAN]" if re.search(r"\d{4}", m.group(0)[4:]) else m.group(0), text)
    text = ACCOUNT.sub(lambda m: m.group(1) + "[account number]", text)
    return text
