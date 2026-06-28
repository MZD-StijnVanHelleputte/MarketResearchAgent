import logging
import re
from dataclasses import dataclass
from io import BytesIO

from config import settings
from clients.edgar_client import EdgarClient, get_edgar_client
from clients.base_http_client import ClientError

logger = logging.getLogger(__name__)

# S-K 1300 only requires a Technical Report Summary for fiscal years after
# 2021-01-01, so scanning the most recent annual reports is sufficient — no
# need to paginate into a filer's older filing history.
_ANNUAL_FORMS = {"10-K", "20-F"}
_MAX_FILINGS_SCANNED = 25

# The filing index page's "Document Format Files" table is the only place EDGAR
# exposes exhibit *type* (e.g. "EX-96.3") — the machine-readable index.json
# directory listing only has filenames, with no exhibit-type metadata at all.
_INDEX_ROW_RE = re.compile(r"<tr[^>]*>(.*?)</tr>", re.S | re.I)
_INDEX_CELL_RE = re.compile(r"<td[^>]*>(.*?)</td>", re.S | re.I)
_TAG_RE = re.compile(r"<[^>]+>")


class ServiceError(Exception):
    pass


@dataclass
class TechnicalReportSummary:
    ticker: str
    cik: str
    company_name: str
    form_type: str
    filing_date: str
    exhibit_name: str
    exhibit_url: str
    excerpt: str
    mine_name_matched: str | None


class EdgarFilingService:
    """Resolves a mining company ticker to its most recent S-K 1300 Technical
    Report Summary (Exhibit 96 of a 10-K/20-F) and returns the document text."""

    def __init__(self, edgar_client: EdgarClient | None = None) -> None:
        self._edgar = edgar_client or get_edgar_client()

    async def get_technical_report_summary(
        self, ticker: str, mine_name: str | None = None
    ) -> TechnicalReportSummary:
        try:
            cik = await self._edgar.get_cik_for_ticker(ticker)
        except ClientError as exc:
            raise ServiceError(f"EDGAR ticker lookup failed for {ticker}: {exc}") from exc
        if not cik:
            raise ServiceError(f"No SEC CIK found for ticker '{ticker}'")

        try:
            submissions = await self._edgar.get_submissions(cik)
        except ClientError as exc:
            raise ServiceError(f"EDGAR submissions request failed for CIK {cik}: {exc}") from exc

        company_name = submissions.get("name", "") or ticker
        annual_filings = self._recent_annual_filings(submissions)
        if not annual_filings:
            raise ServiceError(f"No 10-K/20-F filings found for {ticker} (CIK {cik})")

        cik_int = int(cik)
        for filing in annual_filings[:_MAX_FILINGS_SCANNED]:
            accession = filing["accessionNumber"]
            accession_nodash = accession.replace("-", "")
            index_url = (
                f"https://www.sec.gov/Archives/edgar/data/{cik_int}/"
                f"{accession_nodash}/{accession}-index.htm"
            )
            try:
                index_html = (await self._edgar.get_document_bytes(index_url)).decode(
                    "utf-8", errors="ignore"
                )
            except ClientError as exc:
                logger.info("EDGAR filing index unavailable at %s: %s", index_url, exc)
                continue

            exhibit_name, matched_mine = self._find_exhibit_96(index_html, mine_name)
            if exhibit_name is None:
                continue

            exhibit_url = (
                f"https://www.sec.gov/Archives/edgar/data/{cik_int}/"
                f"{accession_nodash}/{exhibit_name}"
            )
            try:
                raw = await self._edgar.get_document_bytes(exhibit_url)
            except ClientError as exc:
                raise ServiceError(f"EDGAR exhibit fetch failed for {exhibit_url}: {exc}") from exc

            excerpt = self._extract_text(raw, exhibit_name)
            return TechnicalReportSummary(
                ticker=ticker.upper(),
                cik=cik,
                company_name=company_name,
                form_type=filing["form"],
                filing_date=filing["filingDate"],
                exhibit_name=exhibit_name,
                exhibit_url=exhibit_url,
                excerpt=excerpt[: settings.edgar_exhibit_excerpt_max_chars],
                mine_name_matched=matched_mine,
            )

        raise ServiceError(
            f"No S-K 1300 Technical Report Summary (Exhibit 96) found in the "
            f"{len(annual_filings[:_MAX_FILINGS_SCANNED])} most recent 10-K/20-F filings "
            f"for {ticker} (CIK {cik})"
        )

    @staticmethod
    def _recent_annual_filings(submissions: dict) -> list[dict]:
        recent = submissions.get("filings", {}).get("recent", {})
        forms = recent.get("form", [])
        accessions = recent.get("accessionNumber", [])
        dates = recent.get("filingDate", [])
        filings = [
            {"form": f, "accessionNumber": a, "filingDate": d}
            for f, a, d in zip(forms, accessions, dates)
            if f in _ANNUAL_FORMS
        ]
        return sorted(filings, key=lambda x: x["filingDate"], reverse=True)

    @classmethod
    def _find_exhibit_96(
        cls, index_html: str, mine_name: str | None
    ) -> tuple[str | None, str | None]:
        """Parse the filing index page's document table for an EX-96.* exhibit.

        Table columns are [Seq, Description, Document, Type, Size]; Type is the
        only field that reliably carries the exhibit number, and Description is
        often blank, so disambiguation by mine_name falls back to matching the
        document filename (miners commonly name files after the project, e.g.
        "a2025trsmorenci-finalxpubl.pdf").
        """
        candidates: list[str] = []
        for row_match in _INDEX_ROW_RE.finditer(index_html):
            cells_raw = _INDEX_CELL_RE.findall(row_match.group(1))
            if len(cells_raw) < 4:
                continue
            cells = [_TAG_RE.sub("", c).strip() for c in cells_raw]
            doc_type = cells[3]
            if doc_type.upper().startswith("EX-96"):
                candidates.append(cells[2])

        if not candidates:
            return None, None
        if mine_name:
            needle = mine_name.lower().replace(" ", "")
            for name in candidates:
                if needle in name.lower().replace(" ", ""):
                    return name, mine_name
        return candidates[0], None

    @staticmethod
    def _extract_text(raw: bytes, document_name: str) -> str:
        from markitdown import MarkItDown

        extension = "." + document_name.rsplit(".", 1)[-1] if "." in document_name else ".htm"
        result = MarkItDown().convert_stream(BytesIO(raw), file_extension=extension)
        return result.text_content
