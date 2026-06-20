import logging
from dataclasses import dataclass, field

from clients.alpha_vantage_client import AlphaVantageClient
from clients.base_http_client import ClientError

logger = logging.getLogger(__name__)

_AV_NOTICE_KEYS = ("Information", "Note", "Error Message")
_VALID_SORT = {"LATEST", "RELEVANCE", "SENTIMENT"}
_VALID_HORIZONS = {"3month", "6month", "12month"}


class ServiceError(Exception):
    pass


@dataclass
class SentimentArticle:
    title: str
    url: str
    source: str
    published: str
    overall_sentiment_score: float | None
    overall_sentiment_label: str
    ticker_sentiments: list[dict] = field(default_factory=list)


@dataclass
class NewsSentimentResult:
    tickers: str
    topics: str
    sort: str
    items_fetched: int
    articles: list[SentimentArticle] = field(default_factory=list)
    source: str = "alpha_vantage"


@dataclass
class TranscriptSegment:
    speaker: str
    text: str


@dataclass
class EarningsTranscriptResult:
    symbol: str
    quarter: str
    segments: list[TranscriptSegment] = field(default_factory=list)
    raw_text: str = ""
    source: str = "alpha_vantage"


@dataclass
class InsiderTransaction:
    executive: str
    shares: float | None
    transaction_type: str
    date: str
    price: float | None


@dataclass
class InsiderTransactionsResult:
    symbol: str
    transactions: list[InsiderTransaction] = field(default_factory=list)
    source: str = "alpha_vantage"


@dataclass
class EarningsEvent:
    symbol: str
    name: str
    report_date: str
    fiscal_date_ending: str
    estimate: float | None
    currency: str


@dataclass
class EarningsCalendarResult:
    symbol: str
    horizon: str
    events: list[EarningsEvent] = field(default_factory=list)
    source: str = "alpha_vantage"


class EquityIntelligenceService:
    """Alpha Vantage Alpha Intelligence endpoints for competitive and market intel.

    NEWS_SENTIMENT, EARNINGS_CALL_TRANSCRIPT, and INSIDER_TRANSACTIONS require
    a premium Alpha Vantage subscription. The free-tier notice response is caught
    and re-raised as ServiceError so the ReAct loop can fall back gracefully.
    """

    def __init__(self, client: AlphaVantageClient | None = None) -> None:
        self._client = client or AlphaVantageClient()

    async def get_news_sentiment(
        self,
        tickers: str = "",
        topics: str = "",
        sort: str = "LATEST",
        limit: int = 25,
    ) -> NewsSentimentResult:
        sort = sort.strip().upper()
        if sort not in _VALID_SORT:
            raise ServiceError(
                f"Invalid sort '{sort}'. Use one of: {', '.join(sorted(_VALID_SORT))}."
            )
        limit = max(1, min(limit, 50))
        try:
            raw = await self._client.get_news_sentiment(tickers, topics, sort, limit)
        except ClientError as exc:
            raise ServiceError(f"Alpha Vantage news sentiment request failed: {exc}") from exc
        self._raise_on_notice(raw, "NEWS_SENTIMENT")
        return self._parse_news_sentiment(raw, tickers, topics, sort)

    async def get_earnings_transcript(
        self, symbol: str, quarter: str
    ) -> EarningsTranscriptResult:
        symbol = symbol.strip().upper()
        quarter = quarter.strip().upper()
        try:
            raw = await self._client.get_earnings_transcript(symbol, quarter)
        except ClientError as exc:
            raise ServiceError(
                f"Alpha Vantage earnings transcript request failed for {symbol}: {exc}"
            ) from exc
        self._raise_on_notice(raw, f"EARNINGS_CALL_TRANSCRIPT:{symbol}")
        return self._parse_transcript(raw, symbol, quarter)

    async def get_insider_transactions(self, symbol: str) -> InsiderTransactionsResult:
        symbol = symbol.strip().upper()
        try:
            raw = await self._client.get_insider_transactions(symbol)
        except ClientError as exc:
            raise ServiceError(
                f"Alpha Vantage insider transactions request failed for {symbol}: {exc}"
            ) from exc
        self._raise_on_notice(raw, f"INSIDER_TRANSACTIONS:{symbol}")
        return self._parse_insider_transactions(raw, symbol)

    async def get_earnings_calendar(
        self, symbol: str = "", horizon: str = "3month"
    ) -> EarningsCalendarResult:
        horizon = horizon.strip().lower()
        if horizon not in _VALID_HORIZONS:
            raise ServiceError(
                f"Invalid horizon '{horizon}'. Use one of: {', '.join(sorted(_VALID_HORIZONS))}."
            )
        symbol_upper = symbol.strip().upper()
        try:
            raw = await self._client.get_earnings_calendar(symbol_upper, horizon)
        except ClientError as exc:
            raise ServiceError(
                f"Alpha Vantage earnings calendar request failed: {exc}"
            ) from exc
        self._raise_on_notice(raw, "EARNINGS_CALENDAR")
        return self._parse_earnings_calendar(raw, symbol_upper, horizon)

    @staticmethod
    def _raise_on_notice(raw: dict | str, label: str) -> None:
        if not isinstance(raw, dict):
            return  # CSV responses (earnings calendar) come as text; handled downstream
        notices = {k: raw[k] for k in _AV_NOTICE_KEYS if k in raw}
        if notices:
            logger.warning("Alpha Vantage notice for %s: %s", label, notices)
            first = next(iter(notices.values()))
            raise ServiceError(f"Alpha Vantage did not return data for '{label}': {first}")

    @staticmethod
    def _parse_news_sentiment(
        raw: dict, tickers: str, topics: str, sort: str
    ) -> NewsSentimentResult:
        feed = raw.get("feed", [])
        articles: list[SentimentArticle] = []
        for item in feed if isinstance(feed, list) else []:
            score_raw = item.get("overall_sentiment_score")
            try:
                score = float(score_raw) if score_raw is not None else None
            except (TypeError, ValueError):
                score = None
            articles.append(
                SentimentArticle(
                    title=item.get("title", ""),
                    url=item.get("url", ""),
                    source=item.get("source", ""),
                    published=item.get("time_published", ""),
                    overall_sentiment_score=score,
                    overall_sentiment_label=item.get("overall_sentiment_label", ""),
                    ticker_sentiments=item.get("ticker_sentiment", []),
                )
            )
        return NewsSentimentResult(
            tickers=tickers,
            topics=topics,
            sort=sort,
            items_fetched=len(articles),
            articles=articles,
        )

    @staticmethod
    def _parse_transcript(
        raw: dict, symbol: str, quarter: str
    ) -> EarningsTranscriptResult:
        transcript_list = raw.get("transcript", [])
        segments: list[TranscriptSegment] = []
        raw_parts: list[str] = []
        if isinstance(transcript_list, list):
            for entry in transcript_list:
                if isinstance(entry, dict):
                    speaker = entry.get("speaker", "")
                    text = entry.get("speech", entry.get("text", ""))
                    segments.append(TranscriptSegment(speaker=speaker, text=text))
                    raw_parts.append(f"{speaker}: {text}")
        elif isinstance(raw.get("transcript"), str):
            raw_parts.append(raw["transcript"])
        return EarningsTranscriptResult(
            symbol=symbol,
            quarter=quarter,
            segments=segments,
            raw_text="\n\n".join(raw_parts),
        )

    @staticmethod
    def _parse_insider_transactions(
        raw: dict, symbol: str
    ) -> InsiderTransactionsResult:
        data = raw.get("data", [])
        transactions: list[InsiderTransaction] = []
        for item in data if isinstance(data, list) else []:
            if not isinstance(item, dict):
                continue

            def _float(key: str) -> float | None:
                v = item.get(key)
                try:
                    return float(str(v).replace(",", "")) if v else None
                except (TypeError, ValueError):
                    return None

            transactions.append(
                InsiderTransaction(
                    executive=item.get("executive", item.get("name", "")),
                    shares=_float("shares"),
                    transaction_type=item.get("acquisition_or_disposal", item.get("transaction", "")),
                    date=item.get("transaction_date", item.get("date", "")),
                    price=_float("share_price"),
                )
            )
        return InsiderTransactionsResult(symbol=symbol, transactions=transactions)

    @staticmethod
    def _parse_earnings_calendar(
        raw: dict | str, symbol: str, horizon: str
    ) -> EarningsCalendarResult:
        # AV returns EARNINGS_CALENDAR as CSV text (HTTP 200 with text/csv body).
        # When the client parses it as a dict, fall back to empty events gracefully.
        events: list[EarningsEvent] = []
        if isinstance(raw, str):
            lines = raw.strip().splitlines()
            if len(lines) > 1:
                for line in lines[1:]:
                    parts = line.split(",")
                    if len(parts) < 6:
                        continue

                    def _maybe_float(v: str) -> float | None:
                        v = v.strip()
                        try:
                            return float(v) if v else None
                        except ValueError:
                            return None

                    events.append(
                        EarningsEvent(
                            symbol=parts[0].strip(),
                            name=parts[1].strip(),
                            report_date=parts[2].strip(),
                            fiscal_date_ending=parts[3].strip(),
                            estimate=_maybe_float(parts[4]),
                            currency=parts[5].strip() if len(parts) > 5 else "",
                        )
                    )
        elif isinstance(raw, dict):
            # Some AV client wrappers may decode the CSV into a list under a key
            rows = raw.get("earnings", raw.get("data", []))
            for item in rows if isinstance(rows, list) else []:
                if not isinstance(item, dict):
                    continue
                events.append(
                    EarningsEvent(
                        symbol=item.get("symbol", ""),
                        name=item.get("name", ""),
                        report_date=item.get("reportDate", item.get("report_date", "")),
                        fiscal_date_ending=item.get("fiscalDateEnding", ""),
                        estimate=None,
                        currency=item.get("currency", "USD"),
                    )
                )
        return EarningsCalendarResult(symbol=symbol, horizon=horizon, events=events)
