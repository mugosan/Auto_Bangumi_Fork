"""NexusPHP PT 站点（如 Audiences.me）torrentrss.php 输出的解析回归测试。

PT 支持依赖通用 RSS 解析路径：条目带 <enclosure>（.torrent 下载地址含
passkey）时取 enclosure 为下载链接、<link>（details.php）为主页；
linktype=dl 且无 enclosure 时 <link> 本身就是下载地址。
"""

import xml.etree.ElementTree as ET
from datetime import datetime, timezone

from module.network.site import rss_parser

NEXUSPHP_RSS_WITH_ENCLOSURE = """<?xml version="1.0" encoding="UTF-8"?>
<rss version="2.0">
<channel>
  <title>Audiences :: torrents</title>
  <link>https://audiences.me</link>
  <item>
    <title>[Sakurato] Kusuriya no Hitorigoto [21][AVC-8bit 1080p AAC][CHS]</title>
    <link>https://audiences.me/details.php?id=12345</link>
    <enclosure url="https://audiences.me/download.php?id=12345&amp;passkey=abc123" type="application/x-bittorrent" length="694157244"/>
    <guid>https://audiences.me/details.php?id=12345</guid>
    <pubDate>Sat, 04 Jul 2026 12:00:00 +0800</pubDate>
  </item>
</channel>
</rss>"""

NEXUSPHP_RSS_LINKTYPE_DL = """<?xml version="1.0" encoding="UTF-8"?>
<rss version="2.0">
<channel>
  <title>Audiences :: torrents</title>
  <link>https://audiences.me</link>
  <item>
    <title>[ANi] Boku no Hero Academia S07 - 08 [1080P][WEB-DL][AAC AVC][CHT]</title>
    <link>https://audiences.me/download.php?id=67890&amp;passkey=abc123</link>
    <guid>https://audiences.me/details.php?id=67890</guid>
    <pubDate>Sat, 04 Jul 2026 12:00:00 +0800</pubDate>
  </item>
</channel>
</rss>"""


def test_rss_parser_nexusphp_enclosure_yields_download_url() -> None:
    soup = ET.fromstring(NEXUSPHP_RSS_WITH_ENCLOSURE)
    results = rss_parser(soup)
    assert results == [
        (
            "[Sakurato] Kusuriya no Hitorigoto [21][AVC-8bit 1080p AAC][CHS]",
            "https://audiences.me/download.php?id=12345&passkey=abc123",
            "https://audiences.me/details.php?id=12345",
            datetime(2026, 7, 4, 4, 0, 0, tzinfo=timezone.utc),
        )
    ]


def test_rss_parser_nexusphp_linktype_dl_uses_link_as_download_url() -> None:
    soup = ET.fromstring(NEXUSPHP_RSS_LINKTYPE_DL)
    results = rss_parser(soup)
    assert results == [
        (
            "[ANi] Boku no Hero Academia S07 - 08 [1080P][WEB-DL][AAC AVC][CHT]",
            "https://audiences.me/download.php?id=67890&passkey=abc123",
            "",
            datetime(2026, 7, 4, 4, 0, 0, tzinfo=timezone.utc),
        )
    ]


RSS_MISSING_PUBDATE = """<?xml version="1.0" encoding="UTF-8"?>
<rss version="2.0">
<channel>
  <item>
    <title>[Sub] Anime - 01</title>
    <link>https://example.com/1</link>
  </item>
</channel>
</rss>"""

RSS_MALFORMED_PUBDATE = """<?xml version="1.0" encoding="UTF-8"?>
<rss version="2.0">
<channel>
  <item>
    <title>[Sub] Anime - 01</title>
    <link>https://example.com/1</link>
    <pubDate>not a date</pubDate>
  </item>
</channel>
</rss>"""

RSS_NAIVE_PUBDATE = """<?xml version="1.0" encoding="UTF-8"?>
<rss version="2.0">
<channel>
  <item>
    <title>[Sub] Anime - 01</title>
    <link>https://example.com/1</link>
    <pubDate>Sat, 04 Jul 2026 12:00:00</pubDate>
  </item>
</channel>
</rss>"""


def test_rss_parser_missing_pub_date_is_none() -> None:
    """A feed item with no <pubDate> at all degrades to None rather than
    failing the whole item -- callers already treat that as "no evidence"."""
    soup = ET.fromstring(RSS_MISSING_PUBDATE)
    results = rss_parser(soup)
    assert results[0][3] is None


def test_rss_parser_malformed_pub_date_is_none() -> None:
    soup = ET.fromstring(RSS_MALFORMED_PUBDATE)
    results = rss_parser(soup)
    assert results[0][3] is None


def test_rss_parser_naive_pub_date_assumed_utc() -> None:
    """A non-compliant feed omitting the timezone entirely parses as a naive
    datetime under parsedate_to_datetime; assumed UTC rather than left
    ambiguous (comparisons against other UTC-aware dates would otherwise
    raise)."""
    soup = ET.fromstring(RSS_NAIVE_PUBDATE)
    results = rss_parser(soup)
    assert results[0][3] == datetime(2026, 7, 4, 12, 0, 0, tzinfo=timezone.utc)
