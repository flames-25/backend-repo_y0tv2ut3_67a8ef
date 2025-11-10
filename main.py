import os
import io
import re
import csv
from datetime import datetime
from typing import List, Optional, Dict, Any

from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

app = FastAPI(title="Bank Fee & Charges Detector API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ---------- Utility: Keywords and helpers ----------
FEE_KEYWORDS = {
    "ATM Withdraw Charges": ["atm fee", "atm charge", "atm withdraw", "atm withdrawal charge"],
    "Service Charges": ["service charge", "service tax", "bank charge", "convenience fee"],
    "SMS Alerts Fee": ["sms alert", "sms charge", "sms alerts", "alert charge"],
    "Debit Card Annual Fee": ["debit card annual", "annual fee", "card annual fee", "debit card fee"],
    "Minimum Balance Penalty": ["min balance", "minimum balance", "non-maintenance", "amb charge"],
    "Account Maintenance Fee": ["account maintenance", "maintenance charge", "amc"],
    "NEFT/IMPS Charges": ["neft charge", "imps charge", "rtgs charge", "transfer charge"],
}

KEYWORD_REGEX = {
    cat: re.compile(r"|".join([re.escape(k) for k in words]), re.IGNORECASE)
    for cat, words in FEE_KEYWORDS.items()
}

AMOUNT_REGEX = re.compile(r"[₹Rs\.\s]*([+-]?\d{1,3}(?:,\d{3})*(?:\.\d+)?|[+-]?\d+(?:\.\d+)?)")
DATE_FORMATS = [
    "%d-%m-%Y", "%d/%m/%Y", "%Y-%m-%d", "%d-%b-%Y", "%d %b %Y", "%m/%d/%Y",
]


def try_parse_date(raw: str) -> Optional[str]:
    raw = raw.strip()
    for fmt in DATE_FORMATS:
        try:
            dt = datetime.strptime(raw, fmt)
            return dt.date().isoformat()
        except Exception:
            continue
    # try extracting date-like substring
    m = re.search(r"(\d{1,2}[-/ ][A-Za-z]{3}[-/ ]\d{2,4}|\d{4}-\d{2}-\d{2}|\d{1,2}[-/]\d{1,2}[-/]\d{2,4})", raw)
    if m:
        return try_parse_date(m.group(0))
    return None


def parse_amount(raw: str) -> Optional[float]:
    if raw is None:
        return None
    raw = str(raw)
    m = AMOUNT_REGEX.search(raw)
    if not m:
        return None
    num = m.group(1).replace(",", "")
    try:
        return float(num)
    except Exception:
        return None


def detect_category(text: str) -> Optional[str]:
    if not text:
        return None
    for cat, rx in KEYWORD_REGEX.items():
        if rx.search(text):
            return cat
    return None


# ---------- CSV Parsing ----------
CSV_DATE_FIELDS = ["date", "txn date", "transaction date", "value date"]
CSV_DESC_FIELDS = [
    "description", "narration", "details", "particulars", "remarks", "txn details",
]
CSV_AMOUNT_FIELDS = [
    "amount", "debit", "withdrawal amt", "withdrawal", "fee", "charges", "dr amt",
]
CSV_CREDIT_FIELDS = ["credit", "deposit", "cr amt"]
CSV_TYPE_FIELDS = ["type", "dr/cr", "debit/credit"]


def parse_csv(content: bytes) -> List[Dict[str, Any]]:
    text = content.decode("utf-8", errors="ignore")
    reader = csv.DictReader(io.StringIO(text))
    rows = []
    headers = [h.strip().lower() for h in (reader.fieldnames or [])]

    def pick(fields: List[str], row: Dict[str, Any]) -> Optional[str]:
        for f in fields:
            if f in row:
                return row.get(f)
        # try loose match
        for f in fields:
            for h in row.keys():
                if f in h:
                    return row.get(h)
        return None

    for raw in reader:
        row = {k.strip().lower(): v for k, v in raw.items()}
        date_raw = pick(CSV_DATE_FIELDS, row)
        desc_raw = pick(CSV_DESC_FIELDS, row)
        amt_raw = pick(CSV_AMOUNT_FIELDS, row)
        credit_raw = pick(CSV_CREDIT_FIELDS, row)
        type_raw = pick(CSV_TYPE_FIELDS, row)

        date_iso = try_parse_date(str(date_raw)) if date_raw else None

        amount: Optional[float] = None
        if amt_raw is not None:
            amount = parse_amount(str(amt_raw))
        elif credit_raw is not None:
            # If only credit column exists, treat positive as credit
            amount = -abs(parse_amount(str(credit_raw)) or 0.0)

        # Determine sign using type when available
        if type_raw:
            t = str(type_raw).strip().lower()
            if "d" in t:  # debit
                amount = abs(amount or 0.0)
            elif "c" in t:
                amount = -abs(amount or 0.0)

        rows.append({
            "date": date_iso,
            "description": str(desc_raw or "").strip(),
            "amount": amount,
            "raw": row,
        })
    return rows


# ---------- PDF Parsing (best-effort line parser) ----------

def parse_pdf(content: bytes) -> List[Dict[str, Any]]:
    try:
        from pdfminer.high_level import extract_text
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"PDF parsing not available: {e}")

    text = extract_text(io.BytesIO(content)) or ""
    lines = [l.strip() for l in text.splitlines() if l.strip()]
    results: List[Dict[str, Any]] = []
    for line in lines:
        cat = detect_category(line)
        if not cat:
            continue
        amt = parse_amount(line)
        date_iso = try_parse_date(line)
        results.append({
            "date": date_iso,
            "description": line,
            "amount": amt,
            "raw": {"line": line},
        })
    return results


# ---------- Core analyzer ----------

def analyze_rows(rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    matches = []
    currency_symbol = "₹"  # default for India; will try to infer
    start_date: Optional[str] = None
    end_date: Optional[str] = None

    by_category: Dict[str, Dict[str, Any]] = {}

    for r in rows:
        desc = (r.get("description") or "").strip()
        amt = r.get("amount")
        cat = detect_category(desc)
        if not cat:
            # Some statements put fee word in other columns; check raw
            if isinstance(r.get("raw"), dict):
                joined = " ".join(map(str, r["raw"].values()))
                cat = detect_category(joined)
        if not cat:
            continue

        # consider only debits (fees): positive amounts; if sign unknown, infer by context
        if amt is None:
            # try to fish from description
            amt = parse_amount(desc)
        if amt is None:
            continue

        # Infer currency
        if "₹" in desc or "INR" in desc.upper() or "Rs" in desc or "RS" in desc:
            currency_symbol = "₹"

        # Normalize: treat positive as debit (fee). If negative, flip sign just for total fee.
        fee_amt = abs(amt)

        date_iso = r.get("date")
        if date_iso:
            if not start_date or date_iso < start_date:
                start_date = date_iso
            if not end_date or date_iso > end_date:
                end_date = date_iso

        matches.append({
            "date": date_iso,
            "description": desc,
            "amount": round(fee_amt, 2),
            "category": cat,
        })

        if cat not in by_category:
            by_category[cat] = {"count": 0, "amount": 0.0}
        by_category[cat]["count"] += 1
        by_category[cat]["amount"] += fee_amt

    total_amount = sum(v["amount"] for v in by_category.values())

    return {
        "summary": {
            "total_fee": round(total_amount, 2),
            "currency": currency_symbol,
            "start_date": start_date,
            "end_date": end_date,
            "total_count": len(matches),
        },
        "by_category": [
            {"category": k, "count": v["count"], "amount": round(v["amount"], 2)}
            for k, v in sorted(by_category.items(), key=lambda x: -x[1]["amount"])
        ],
        "matches": matches,
    }


# ---------- Routes ----------

@app.get("/")
def read_root():
    return {"message": "Hello from FastAPI Backend!"}


@app.post("/api/fees/analyze")
async def analyze_statement(file: UploadFile = File(...)):
    filename = (file.filename or "").lower()
    content = await file.read()

    if not content:
        raise HTTPException(status_code=400, detail="Empty file uploaded")

    rows: List[Dict[str, Any]]
    if filename.endswith(".csv"):
        rows = parse_csv(content)
    elif filename.endswith(".pdf"):
        rows = parse_pdf(content)
    else:
        # try to sniff CSV
        head = content[:1024].decode("utf-8", errors="ignore")
        if "," in head and "\n" in head:
            rows = parse_csv(content)
        else:
            raise HTTPException(status_code=400, detail="Unsupported file type. Upload CSV or PDF.")

    analysis = analyze_rows(rows)
    return JSONResponse(content=analysis)


@app.get("/api/hello")
def hello():
    return {"message": "Hello from the backend API!"}


@app.get("/test")
def test_database():
    """Test endpoint to check if database is available and accessible"""
    response = {
        "backend": "✅ Running",
        "database": "❌ Not Available",
        "database_url": None,
        "database_name": None,
        "connection_status": "Not Connected",
        "collections": []
    }

    try:
        # Try to import database module
        from database import db

        if db is not None:
            response["database"] = "✅ Available"
            response["database_url"] = "✅ Configured"
            response["database_name"] = db.name if hasattr(db, 'name') else "✅ Connected"
            response["connection_status"] = "Connected"

            # Try to list collections to verify connectivity
            try:
                collections = db.list_collection_names()
                response["collections"] = collections[:10]  # Show first 10 collections
                response["database"] = "✅ Connected & Working"
            except Exception as e:
                response["database"] = f"⚠️  Connected but Error: {str(e)[:50]}"
        else:
            response["database"] = "⚠️  Available but not initialized"

    except ImportError:
        response["database"] = "❌ Database module not found (run enable-database first)"
    except Exception as e:
        response["database"] = f"❌ Error: {str(e)[:50]}"

    # Check environment variables
    import os
    response["database_url"] = "✅ Set" if os.getenv("DATABASE_URL") else "❌ Not Set"
    response["database_name"] = "✅ Set" if os.getenv("DATABASE_NAME") else "❌ Not Set"

    return response


if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)
