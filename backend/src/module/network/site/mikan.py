import logging
from datetime import datetime, timezone
from email.utils import parsedate_to_datetime

logger = logging.getLogger(__name__)


def _parse_pub_date(item) -> datetime | None:
    """Parse an RSS <pubDate> (RFC 2822) into a UTC-aware datetime.

    Used to tell a genuinely new release from an old one showing up in the
    feed for the first time (e.g. a rerun, or the feed being subscribed to
    for the first time) -- something the file's own download time can
    never tell us. Malformed/missing dates degrade to None rather than
    failing the whole item; callers already treat a missing pub_date as
    "no evidence" and fall back to a weaker signal.
    """
    node = item.find("pubDate")
    if node is None or not node.text:
        return None
    try:
        parsed = parsedate_to_datetime(node.text.strip())
    except (TypeError, ValueError):
        return None
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc)


def rss_parser(soup):
    results = []
    for item in soup.findall("./channel/item"):
        try:
            title = item.find("title").text
            enclosure = item.find("enclosure")
            if enclosure is not None:
                homepage = item.find("link").text
                url = enclosure.attrib.get("url")
            else:
                url = item.find("link").text
                homepage = ""
            pub_date = _parse_pub_date(item)
            results.append((title, url, homepage, pub_date))
        except Exception as e:
            logger.warning("Failed to parse RSS item: %s", e)
            continue
    return results


def mikan_title(soup):
    return soup.find("title").text
