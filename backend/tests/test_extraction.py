"""Tests for the text extraction stage.

Every fixture is built on disk in a tmp_path, so these exercise the real
extractors with no network and no committed binary files.
"""

from __future__ import annotations

import zipfile
from pathlib import Path

import pytest

from app.core.exceptions import DocumentProcessingError, UnsupportedDocumentError
from app.ingestion.extraction import (
    ArchiveTextExtractor,
    CsvTextExtractor,
    ExtractorRegistry,
    HtmlTextExtractor,
    MarkdownTextExtractor,
    PlainTextExtractor,
    read_text_file,
)


def write(path: Path, text: str, encoding: str = "utf-8") -> Path:
    """Write `text` with no newline translation.

    `Path.write_text` would rewrite "\\n" as "\\r\\n" on Windows, which makes
    byte-exact assertions fail for reasons that have nothing to do with the
    extractor under test.
    """
    with path.open("w", encoding=encoding, newline="") as handle:
        handle.write(text)
    return path


class TestRegistry:
    def test_ships_every_advertised_format(self):
        extensions = ExtractorRegistry().supported_extensions

        for expected in (".pdf", ".docx", ".html", ".md", ".csv", ".txt", ".zip"):
            assert expected in extensions

    def test_unsupported_extension_names_what_is_supported(self, tmp_path):
        with pytest.raises(UnsupportedDocumentError, match="\\.pdf"):
            ExtractorRegistry().extract(write(tmp_path / "a.xyz", "hi"))

    def test_dispatch_is_case_insensitive(self, tmp_path):
        assert ExtractorRegistry().supports(tmp_path / "REPORT.PDF")


class TestPlainText:
    def test_reads_verbatim(self, tmp_path):
        result = PlainTextExtractor().extract(write(tmp_path / "a.txt", "line one\nline two"))

        assert result.text == "line one\nline two"
        assert result.page_count == 1

    def test_falls_back_when_the_file_is_not_utf8(self, tmp_path):
        path = tmp_path / "a.txt"
        path.write_bytes("café costs £5".encode("cp1252"))

        # cp1252 bytes are invalid utf-8; the reader must not raise.
        assert "costs" in read_text_file(path)

    def test_strips_a_utf8_bom(self, tmp_path):
        path = tmp_path / "a.txt"
        path.write_bytes(b"\xef\xbb\xbfHello")

        assert read_text_file(path) == "Hello"


class TestMarkdown:
    def test_keeps_headings_for_structural_chunking(self, tmp_path):
        result = MarkdownTextExtractor().extract(
            write(tmp_path / "a.md", "# Title\n\n## Section\n\ntext")
        )

        assert "# Title" in result.text
        assert "## Section" in result.text

    def test_strips_emphasis_and_link_syntax(self, tmp_path):
        source = "Read the [handbook](https://x.test/h) and **note** the _rules_ in `code`."

        text = MarkdownTextExtractor().extract(write(tmp_path / "a.md", source)).text

        assert "Read the handbook and note the rules in code." in text
        for syntax in ("](", "**", "`"):
            assert syntax not in text

    def test_keeps_image_alt_text(self, tmp_path):
        text = MarkdownTextExtractor().extract(
            write(tmp_path / "a.md", "![org chart](chart.png)")
        ).text

        assert text.strip() == "org chart"

    def test_keeps_fenced_code_but_drops_the_fence(self, tmp_path):
        text = MarkdownTextExtractor().extract(
            write(tmp_path / "a.md", "```python\nx = 1\n```")
        ).text

        assert "x = 1" in text
        assert "```" not in text


class TestHtml:
    HTML = """<html><head><title>T</title><style>p{color:red}</style></head>
    <body><nav>navigation</nav><script>evil()</script>
    <h1>Security</h1><p>Encrypt laptops.</p>
    <h2>Passwords</h2><ul><li>14 characters</li></ul>
    <table><tr><th>Control</th><th>Owner</th></tr><tr><td>MFA</td><td>IT</td></tr></table>
    </body></html>"""

    @pytest.fixture
    def text(self, tmp_path) -> str:
        return HtmlTextExtractor().extract(write(tmp_path / "a.html", self.HTML)).text

    def test_drops_scripts_styles_and_navigation(self, text):
        for noise in ("evil()", "color:red", "navigation"):
            assert noise not in text

    def test_headings_become_markdown(self, text):
        assert "# Security" in text
        assert "## Passwords" in text

    def test_list_items_become_bullets(self, text):
        assert "- 14 characters" in text

    def test_table_rows_become_pipe_separated(self, text):
        assert "Control | Owner" in text
        assert "MFA | IT" in text

    def test_malformed_html_still_yields_text(self, tmp_path):
        result = HtmlTextExtractor().extract(
            write(tmp_path / "a.html", "<p>unclosed<div>tags")
        )

        assert "unclosed" in result.text


class TestCsv:
    def test_each_row_keeps_its_column_names(self, tmp_path):
        source = "Vendor,Spend\nAcme,120000\nBolt,45000\n"

        text = CsvTextExtractor().extract(write(tmp_path / "v.csv", source)).text

        assert "Vendor: Acme | Spend: 120000" in text
        assert "Vendor: Bolt | Spend: 45000" in text

    def test_emits_no_filename_heading(self, tmp_path):
        """The owner of the file supplies its name, not the extractor."""
        text = CsvTextExtractor().extract(write(tmp_path / "v.csv", "A,B\n1,2\n")).text

        assert not text.startswith("#")

    def test_falls_back_to_positional_rows_without_a_header(self, tmp_path):
        text = CsvTextExtractor().extract(write(tmp_path / "v.csv", "a,,c\n1,2,3\n")).text

        assert "a | | c" in text or "a |  | c" in text

    def test_reads_semicolon_delimited_files(self, tmp_path):
        text = CsvTextExtractor().extract(
            write(tmp_path / "v.csv", "Vendor;Spend\nAcme;12\n")
        ).text

        assert "Vendor: Acme | Spend: 12" in text

    def test_reads_tsv(self, tmp_path):
        text = CsvTextExtractor().extract(
            write(tmp_path / "v.tsv", "Vendor\tSpend\nAcme\t12\n")
        ).text

        assert "Vendor: Acme | Spend: 12" in text

    def test_empty_file_is_not_an_error(self, tmp_path):
        assert CsvTextExtractor().extract(write(tmp_path / "v.csv", "")).text == ""


class TestArchive:
    """Notion and Confluence exports are both ZIPs of formats already supported."""

    def notion(self, tmp_path: Path) -> Path:
        path = tmp_path / "notion.zip"
        with zipfile.ZipFile(path, "w") as archive:
            archive.writestr(
                "Export/Engineering Handbook a1b2c3d4e5f6789012345678901234ab.md",
                "# Engineering Handbook\n\nDeploys happen on Tuesdays.\n",
            )
            archive.writestr(
                "Export/Team Roster 0011223344556677889900aabbccddee.csv",
                "Name,Team\nAda,Platform\n",
            )
            archive.writestr("Export/logo.png", b"\x89PNG binary")
            archive.writestr("__MACOSX/._junk", b"junk")
        return path

    def confluence(self, tmp_path: Path) -> Path:
        path = tmp_path / "confluence.zip"
        with zipfile.ZipFile(path, "w") as archive:
            archive.writestr(
                "space/Incident Response.html",
                "<html><h1>Incident Response</h1><p>Page on-call in five minutes.</p></html>",
            )
            archive.writestr("space/styles/site.css", "body{}")
        return path

    def test_notion_export_reads_markdown_and_csv_members(self, tmp_path):
        result = ExtractorRegistry().extract(self.notion(tmp_path))

        assert "Deploys happen on Tuesdays." in result.text
        assert "Name: Ada | Team: Platform" in result.text

    def test_notion_page_ids_are_stripped_from_titles(self, tmp_path):
        text = ExtractorRegistry().extract(self.notion(tmp_path)).text

        assert "# Team Roster" in text
        assert "0011223344556677889900aabbccddee" not in text

    def test_a_member_heading_is_not_duplicated(self, tmp_path):
        """Markdown carries its own H1; the archive must not add a second."""
        text = ExtractorRegistry().extract(self.notion(tmp_path)).text

        assert text.count("# Engineering Handbook") == 1

    def test_confluence_export_reads_html_members(self, tmp_path):
        text = ExtractorRegistry().extract(self.confluence(tmp_path)).text

        assert "Page on-call in five minutes." in text

    def test_unsupported_members_are_skipped_not_fatal(self, tmp_path):
        """Images, attachments and stylesheets must not fail the upload."""
        result = ExtractorRegistry().extract(self.notion(tmp_path))

        assert result.page_count == 2  # the .md and the .csv, not the .png

    def test_page_count_is_the_number_of_members_read(self, tmp_path):
        assert ExtractorRegistry().extract(self.confluence(tmp_path)).page_count == 1

    def test_an_archive_of_only_unsupported_files_is_rejected(self, tmp_path):
        path = tmp_path / "images.zip"
        with zipfile.ZipFile(path, "w") as archive:
            archive.writestr("a.png", b"binary")

        with pytest.raises(DocumentProcessingError, match="No supported files"):
            ExtractorRegistry().extract(path)

    def test_a_corrupt_archive_is_reported_clearly(self, tmp_path):
        path = tmp_path / "broken.zip"
        path.write_bytes(b"this is not a zip")

        with pytest.raises(DocumentProcessingError, match="not a readable ZIP"):
            ExtractorRegistry().extract(path)

    def test_a_zip_bomb_is_refused(self, tmp_path, monkeypatch):
        from app.core.config import settings

        monkeypatch.setattr(settings, "archive_max_uncompressed_bytes", 10)

        with pytest.raises(DocumentProcessingError, match="beyond the"):
            ExtractorRegistry().extract(self.notion(tmp_path))

    def test_nested_archives_are_not_followed(self, tmp_path):
        """Refusing to recurse keeps a zip-of-zips from expanding unboundedly."""
        inner = tmp_path / "inner.zip"
        with zipfile.ZipFile(inner, "w") as archive:
            archive.writestr("a.txt", "hidden")

        outer = tmp_path / "outer.zip"
        with zipfile.ZipFile(outer, "w") as archive:
            archive.write(inner, "inner.zip")
            archive.writestr("b.txt", "visible")

        text = ExtractorRegistry().extract(outer).text

        assert "visible" in text
        assert "hidden" not in text

    def test_registry_wires_itself_into_the_archive_extractor(self):
        """The archive dispatches members back through the same registry."""
        registry = ExtractorRegistry()

        assert isinstance(registry._extractors[".zip"], ArchiveTextExtractor)
