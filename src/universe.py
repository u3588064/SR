"""
universe.py — Global Systemically Important Banks (G-SIBs) Registry

Source: FSB 2023 G-SIB list (updated annually each November).
Each bank entry contains:
    - id         : short key used in filenames and API responses
    - name       : full name
    - region     : one of US / CN / GB / EU / JP
    - yf_ticker  : Yahoo Finance ticker (primary price source)
    - ak_ticker  : AkShare ticker (A-share fallback for CN banks, None otherwise)
    - index_yf   : Yahoo Finance ticker of the regional market index used as
                   the "system" return in CoVaR / ΔCoVaR calculations.
                   Documented here for full transparency.

CoVaR Index Rationale (documented per region):
    US  → ^GSPC  (S&P 500)        — primary domestic systemic benchmark
    CN  → 000300.SS (CSI 300)     — for A-share pricing / domestic risk
          ^HSI    (Hang Seng)     — for H-share pricing
          (each CN bank gets the index matching its primary listing)
    GB  → ^FTSE  (FTSE 100)       — post-Brexit separate benchmark from EU
    EU  → ^STOXX50E (EURO STOXX 50) — eurozone systemic reference
    JP  → ^N225  (Nikkei 225)     — domestic systemic reference
"""

from typing import Optional
from dataclasses import dataclass, field


@dataclass
class Bank:
    id: str
    name: str
    region: str          # US | CN | GB | EU | JP
    yf_ticker: str       # Yahoo Finance ticker
    index_yf: str        # Regional index ticker for CoVaR
    ak_ticker: Optional[str] = None  # AkShare code (CN A-share only)
    listing: str = "equity"          # equity | adr


# ---------------------------------------------------------------------------
# G-SIBs — FSB 2023 list (29 institutions; Credit Suisse absorbed by UBS)
# ---------------------------------------------------------------------------
BANKS: list[Bank] = [
    # ── United States ────────────────────────────────────────────────────
    Bank("JPM",   "JPMorgan Chase",          "US", "JPM",      "^GSPC"),
    Bank("BAC",   "Bank of America",          "US", "BAC",      "^GSPC"),
    Bank("C",     "Citigroup",                "US", "C",        "^GSPC"),
    Bank("WFC",   "Wells Fargo",              "US", "WFC",      "^GSPC"),
    Bank("GS",    "Goldman Sachs",            "US", "GS",       "^GSPC"),
    Bank("MS",    "Morgan Stanley",           "US", "MS",       "^GSPC"),
    Bank("BK",    "Bank of New York Mellon",  "US", "BK",       "^GSPC"),
    Bank("STT",   "State Street",             "US", "STT",      "^GSPC"),

    # ── China ────────────────────────────────────────────────────────────
    # A-share tickers used for price; AkShare code supplied for fallback.
    # CoVaR index = CSI 300 for A-share, HSI for H-share primary listing.
    Bank("ICBC",  "Industrial & Commercial Bank of China", "CN",
         "1398.HK", "^HSI", ak_ticker="601398"),
    Bank("CCB",   "China Construction Bank",               "CN",
         "0939.HK", "^HSI", ak_ticker="601939"),
    Bank("ABC",   "Agricultural Bank of China",            "CN",
         "1288.HK", "^HSI", ak_ticker="601288"),
    Bank("BOC",   "Bank of China",                         "CN",
         "3988.HK", "^HSI", ak_ticker="601988"),
    Bank("BOCOM", "Bank of Communications",                "CN",
         "3328.HK", "^HSI", ak_ticker="601328"),

    # ── United Kingdom ───────────────────────────────────────────────────
    Bank("HSBC",  "HSBC Holdings",            "GB", "HSBA.L",   "^FTSE"),
    Bank("BARC",  "Barclays",                 "GB", "BARC.L",   "^FTSE"),
    Bank("STAN",  "Standard Chartered",       "GB", "STAN.L",   "^FTSE"),

    # ── France ───────────────────────────────────────────────────────────
    Bank("BNP",   "BNP Paribas",              "EU", "BNP.PA",   "^STOXX50E"),
    Bank("ACA",   "Crédit Agricole",          "EU", "ACA.PA",   "^STOXX50E"),
    Bank("GLE",   "Société Générale",         "EU", "GLE.PA",   "^STOXX50E"),
    Bank("BPCE",  "Groupe BPCE",              "EU", "GLE.PA",   "^STOXX50E"),  # BPCE not listed; proxy via SocGen sector
    # ── Germany ──────────────────────────────────────────────────────────
    Bank("DBK",   "Deutsche Bank",            "EU", "DBK.DE",   "^STOXX50E"),
    # ── Switzerland ──────────────────────────────────────────────────────
    Bank("UBS",   "UBS Group",                "EU", "UBSG.SW",  "^STOXX50E"),
    # ── Netherlands ──────────────────────────────────────────────────────
    Bank("ING",   "ING Groep",                "EU", "INGA.AS",  "^STOXX50E"),
    # ── Spain ────────────────────────────────────────────────────────────
    Bank("SAN",   "Banco Santander",          "EU", "SAN.MC",   "^STOXX50E"),
    Bank("BBVA",  "BBVA",                     "EU", "BBVA.MC",  "^STOXX50E"),
    # ── Italy ────────────────────────────────────────────────────────────
    Bank("UCG",   "UniCredit",                "EU", "UCG.MI",   "^STOXX50E"),

    # ── Japan ────────────────────────────────────────────────────────────
    Bank("MUFG",  "Mitsubishi UFJ Financial", "JP", "8306.T",   "^N225"),
    Bank("SMFG",  "Sumitomo Mitsui Financial","JP", "8316.T",   "^N225"),
    Bank("MFG",   "Mizuho Financial Group",   "JP", "8411.T",   "^N225"),
]

# Quick-lookup helpers
BANK_BY_ID: dict[str, Bank] = {b.id: b for b in BANKS}
REGIONS: list[str] = sorted(set(b.region for b in BANKS))

# All unique index tickers needed (deduplicated)
ALL_INDICES: list[str] = sorted(set(b.index_yf for b in BANKS))


def get_bank(bank_id: str) -> Bank:
    """Return Bank by id (case-insensitive). Raises KeyError if not found."""
    return BANK_BY_ID[bank_id.upper()]


def banks_by_region(region: str) -> list[Bank]:
    """Return all banks in a given region."""
    return [b for b in BANKS if b.region == region.upper()]
