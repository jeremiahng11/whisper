"""Built-in vocabularies: payments / fintech, tech, finance & compliance terms that come up in almost every meeting.

They're used like your own Names & terms - a hint to the speech model, the right spelling in transcripts ("k y c",
"K.Y.C." -> KYC, "pay now" -> PayNow, "s g q r" -> SGQR) and a glossary for the minutes - but with less weight: your
own Names & terms and corrections always win. Terms that are also everyday English words ("fast", "bin", "pin",
"mid") are only hints, never "corrected" in the text. Turn packs on/off with VOCAB_PACKS (default all) or on the
Names page."""
import re
import threading
import time

from . import db, worker
from .config import settings

PACKS = {
    "payments": ("Payments & fintech", """KYC AML CFT KYB eKYC CDD EDD PEP STR SAR sanctions screening travel rule VASP DPT
        stablecoin USDT USDC on-ramp off-ramp custody wallet e-wallet top-up payout remittance cross-border acquirer issuer
        PSP PSPs payment gateway merchant onboarding MDR interchange scheme fee chargeback chargebacks dispute refund
        settlement reconciliation float nostro vostro FX spread BIN MCC MID TID POS soft POS tokenisation 3DS 3D Secure
        OTP PIN PayNow PayNow Corporate SGQR FAST GIRO NETS MEPS+ SWIFT IBAN SEPA ACH RTGS UPI DuitNow PromptPay QRIS
        Alipay Alipay+ WeChat Pay GrabPay ShopeePay UnionPay Visa Mastercard Amex JCB Stripe Adyen Airwallex Wise Nium
        BaaS embedded finance virtual account virtual card prepaid card debit card credit card card issuing
        open banking API banking ledger e-money MPI SPI Payment Services Act PSA"""),
    "tech": ("Tech", """API APIs REST GraphQL webhook webhooks SDK SDKs JSON JWT OAuth OAuth2 OpenID SSO SAML MFA 2FA
        RBAC IAM VPN WAF CDN DNS TLS SSL HSM KMS PKI UAT SIT QA CI/CD DevOps DevSecOps GitHub GitLab Docker Kubernetes
        K8s Helm Terraform Coolify AWS Azure GCP Cloudflare Lambda S3 EC2 RDS PostgreSQL MySQL MongoDB Redis Kafka
        RabbitMQ Elasticsearch microservices monolith backend frontend Flutter React Node.js Django Python Java Go
        iOS Android SaaS PaaS IaaS SLA SLO SLI uptime latency throughput load balancer failover DR BCP RTO RPO
        go-live hotfix rollback deployment staging production sprint backlog Jira Confluence Slack Teams LLM GPT
        OCR ESP32"""),
    "finance": ("Finance & compliance", """MAS TRM MAS TRM guidelines outsourcing guidelines PDPA PDPC ACRA IRAS GST
        SOC 2 ISO 27001 PCI DSS PCI-DSS GDPR audit internal audit external audit audit finding remediation risk register
        risk appetite compliance officer MLRO board MANCO P&L EBITDA CAPEX OPEX COGS revenue gross margin net margin
        ARR MRR GMV TPV take rate unit economics CAC LTV burn rate runway cash flow working capital balance sheet
        budget forecast variance accrual invoice AR AP KPI KPIs OKR OKRs ROI NPV IRR term sheet cap table due diligence
        MOU NDA SOW MSA RFP RFQ PO vendor partner"""),
}
# also everyday English words: only hints for the speech model, never changed in the text
NOT_IN_TEXT = {"fast", "bin", "pin", "mid", "pos", "dr", "sit", "nets", "go", "float", "board", "budget", "revenue",
               "refund", "dispute", "custody", "wallet", "ledger", "invoice", "vendor", "partner", "audit", "backlog",
               "sprint", "staging", "production", "deployment", "rollback", "hotfix", "uptime", "latency", "throughput",
               "monolith", "backend", "frontend", "teams", "slack", "lambda", "visa", "wise", "ach", "spi", "po", "ap",
               "ar", "qa", "go-live", "remittance", "settlement", "reconciliation", "spread", "scheme fee", "payout",
               "top-up", "forecast", "variance", "accrual", "remediation", "sanctions screening", "payment gateway",
               "merchant onboarding", "virtual account", "virtual card", "prepaid card", "debit card", "credit card",
               "card issuing", "open banking", "embedded finance", "cash flow", "working capital", "balance sheet",
               "gross margin", "net margin", "take rate", "unit economics", "burn rate", "runway", "term sheet",
               "cap table", "due diligence", "risk register", "risk appetite", "compliance officer", "internal audit",
               "external audit", "audit finding", "travel rule", "load balancer", "failover", "microservices",
               "outsourcing guidelines", "on-ramp", "off-ramp", "cross-border", "stablecoin", "chargeback", "chargebacks",
               "interchange", "acquirer", "issuer", "webhook", "webhooks", "e-wallet", "e-money", "soft pos"}


def _split(blob: str) -> list[str]:
    """Multi-word terms are written with single spaces; keep known two/three-word terms together."""
    multi = ["sanctions screening", "travel rule", "payment gateway", "merchant onboarding", "scheme fee", "soft POS",
             "3D Secure", "PayNow Corporate", "WeChat Pay", "virtual account", "virtual card", "prepaid card",
             "debit card", "credit card", "card issuing", "open banking", "API banking", "embedded finance",
             "Payment Services Act", "load balancer", "MAS TRM guidelines", "outsourcing guidelines", "SOC 2",
             "ISO 27001", "PCI DSS", "internal audit", "external audit", "audit finding", "risk register",
             "risk appetite", "compliance officer", "gross margin", "net margin", "take rate", "unit economics",
             "burn rate", "cash flow", "working capital", "balance sheet", "term sheet", "cap table", "due diligence",
             "MAS TRM", "on-ramp", "off-ramp"]
    text = " ".join(blob.split())
    out = []
    for m in sorted(multi, key=len, reverse=True):
        if m in text:
            out.append(m)
            text = text.replace(m, " ")
    out += text.split()
    seen, res = set(), []
    for t in out:
        if t.lower() not in seen:
            seen.add(t.lower())
            res.append(t)
    return res


def enabled_packs() -> list[str]:
    raw = settings.vocab_packs
    if raw.strip().lower() in ("", "all"):
        names = list(PACKS)
    elif raw.strip().lower() in ("none", "off", "0"):
        names = []
    else:
        names = [p.strip() for p in raw.split(",") if p.strip() in PACKS]
    p = settings.data_dir / "vocab_off.txt"                   # turned off on the Names page
    off = set(p.read_text().split()) if p.exists() else set()
    return [n for n in names if n not in off]


def set_off(off: list[str]) -> None:
    p = settings.data_dir / "vocab_off.txt"
    p.write_text(" ".join(x for x in off if x in PACKS))


def terms(for_text: bool = False) -> list[str]:
    out = []
    for name in enabled_packs():
        out += _split(PACKS[name][1])
    if for_text:
        out = [t for t in out if t.lower() not in NOT_IN_TEXT]
    return out


def acronyms() -> list[str]:
    """All-caps short terms that are spoken letter by letter (K Y C, S G Q R)."""
    return [t for t in terms(True) if re.fullmatch(r"[A-Z0-9]{2,6}\+?", t) and not t.isdigit()]


_recent = {"at": 0.0, "terms": []}
_lock = threading.Lock()


def recent(limit: int = 40) -> list[str]:
    """Pack terms your meetings actually use, most used first - the speech model's hint is short, so these go in."""
    with _lock:
        if time.time() - _recent["at"] < 1800:
            return _recent["terms"][:limit]
    counts = {}
    pack = terms()
    low = {t.lower(): t for t in pack}
    for j in db.list_jobs(20, "done"):
        tr = worker.load_transcript(j["id"]) or {}
        text = " " + " ".join(s.get("text", "") for s in tr.get("segments", [])).lower() + " "
        for k, t in low.items():
            n = text.count(" " + k + " ") + text.count(" " + k + ",") + text.count(" " + k + ".")
            if n:
                counts[t] = counts.get(t, 0) + n
    core = [t for t in ("KYC", "AML", "API", "PayNow", "SGQR", "UAT", "MAS", "TRM", "PSP", "chargeback") if t in pack]
    ranked = sorted(counts, key=lambda t: -counts[t])
    out = list(dict.fromkeys(ranked + core))
    with _lock:
        _recent.update(at=time.time(), terms=out)
    return out[:limit]


def spelled_acronyms(text: str, acr: list[str]) -> str:
    """"k y c", "K.Y.C.", "K-Y-C" -> "KYC" (only for acronyms in the packs or your terms)."""
    if not text or not acr:
        return text
    want = {a.replace("+", "").lower(): a for a in acr}

    def rep(m):
        letters = re.sub(r"[^A-Za-z0-9]", "", m.group(0)).lower()
        return want.get(letters, m.group(0))
    return re.sub(r"(?<![A-Za-z0-9])(?:[A-Za-z0-9][ .\-]){1,5}[A-Za-z0-9]\.?(?![A-Za-z0-9])", rep, text)
