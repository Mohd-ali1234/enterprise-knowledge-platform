"""Stage: Text Extraction.

File formats are handled by per-format extractors registered in
`ExtractorRegistry`; supporting a new format means adding one extractor and
registering it, with no change to the ingestion pipeline.

Every extractor normalises its input to **markdown-flavoured plain text**:
headings become `#` lines, tables become pipe rows. That is deliberate, not
cosmetic - `StructuralChunker` splits on markdown headings, ALL-CAPS lines and
numbered clauses, so emitting `#` headings is what lets a DOCX or an HTML page
be sectioned as well as a PDF is.

Paged formats report a real page count; the rest report 1 (or, for archives,
the number of member files that yielded text), because pagination is a
rendering property those formats do not have.
"""

from __future__ import annotations

import csv
import io
import re
import zipfile
from pathlib import Path

from app.core.config import settings
from app.core.exceptions import DocumentProcessingError, UnsupportedDocumentError
from app.core.logging import get_logger
from app.domain.documents import ExtractedText
from app.ports.text_extractor import TextExtractor

logger = get_logger(__name__)

# Encodings tried in order when a text-ish file has no declared encoding.
_TEXT_ENCODINGS = ("utf-8-sig", "utf-8", "cp1252", "latin-1")

# Notion suffixes exported filenames with a 32-character page id.
_NOTION_ID_SUFFIX = re.compile(r"[ _-]+[0-9a-f]{32}$")


def read_text_file(file_path: Path) -> str:
    """Read a text file, trying a few encodings before giving up.

    Exports arrive from many systems; a hard utf-8 assumption rejects perfectly
    readable files. `latin-1` maps every byte, so the last attempt cannot fail.
    """
    raw = file_path.read_bytes()
    for encoding in _TEXT_ENCODINGS:
        try:
            return raw.decode(encoding)
        except UnicodeDecodeError:
            continue
    return raw.decode("latin-1", errors="replace")  # pragma: no cover - unreachable


# ── Paged formats ────────────────────────────────────────────────


class PdfTextExtractor:
    """Extracts text from PDFs using PyMuPDF."""

    supported_extensions = frozenset({".pdf"})

    def extract(self, file_path: Path) -> ExtractedText:
        try:
            import fitz  # PyMuPDF
        except ImportError as exc:  # pragma: no cover - dependency is declared
            raise DocumentProcessingError(f"PyMuPDF is not installed: {exc}") from exc

        try:
            with fitz.open(file_path) as document:
                pages = [text for page in document if (text := page.get_text())]
                page_count = document.page_count
        except Exception as exc:  # noqa: BLE001 - PyMuPDF raises many error types
            raise DocumentProcessingError(f"Could not read PDF '{file_path.name}': {exc}") from exc

        logger.debug("Extracted %d pages from %s", page_count, file_path.name)
        return ExtractedText(text="\n".join(pages), page_count=page_count)


class DocxTextExtractor:
    """Extracts text from Word documents using python-docx.

    Word's built-in `Heading N` styles are mapped to markdown headings so the
    document's own outline survives into structural chunking. Tables are
    flattened to pipe-separated rows.
    """

    supported_extensions = frozenset({".docx"})

    def extract(self, file_path: Path) -> ExtractedText:
        try:
            import docx
        except ImportError as exc:  # pragma: no cover - dependency is declared
            raise DocumentProcessingError(f"python-docx is not installed: {exc}") from exc

        try:
            document = docx.Document(str(file_path))
        except Exception as exc:  # noqa: BLE001 - python-docx raises many error types
            raise DocumentProcessingError(
                f"Could not read DOCX '{file_path.name}': {exc}. "
                "Legacy .doc files are not supported; save as .docx first."
            ) from exc

        blocks = [text for p in document.paragraphs if (text := self._paragraph(p))]
        blocks.extend(self._table(table) for table in document.tables)

        return ExtractedText(text="\n\n".join(b for b in blocks if b), page_count=1)

    @staticmethod
    def _paragraph(paragraph) -> str:  # noqa: ANN001 - docx types are untyped
        text = paragraph.text.strip()
        if not text:
            return ""

        style = (paragraph.style.name or "") if paragraph.style else ""
        match = re.match(r"Heading (\d)", style)
        if match:
            return f"{'#' * min(int(match.group(1)), 6)} {text}"
        if style == "Title":
            return f"# {text}"
        return text

    @staticmethod
    def _table(table) -> str:  # noqa: ANN001 - docx types are untyped
        rows = [
            " | ".join(cell.text.strip() for cell in row.cells)
            for row in table.rows
        ]
        return "\n".join(row for row in rows if row.replace("|", "").strip())


# ── Markup formats ───────────────────────────────────────────────


class HtmlTextExtractor:
    """Extracts readable text from HTML using BeautifulSoup.

    Script, style, navigation and boilerplate elements are dropped; headings
    and list items are re-emitted as markdown so the page's outline survives.
    Confluence space exports are HTML, so this extractor is what makes them
    work once unpacked from their archive.
    """

    supported_extensions = frozenset({".html", ".htm", ".xhtml"})

    # Elements that never carry article content.
    _DROP = ("script", "style", "noscript", "template", "svg", "nav", "footer", "form")

    def extract(self, file_path: Path) -> ExtractedText:
        return ExtractedText(text=self.to_text(read_text_file(file_path)), page_count=1)

    def to_text(self, html: str) -> str:
        """Convert an HTML string to markdown-flavoured plain text."""
        try:
            from bs4 import BeautifulSoup
        except ImportError as exc:  # pragma: no cover - dependency is declared
            raise DocumentProcessingError(f"beautifulsoup4 is not installed: {exc}") from exc

        soup = BeautifulSoup(html, "html.parser")
        for element in soup(self._DROP):
            element.decompose()

        # Rewrite structural tags in place, so get_text() emits markdown.
        for level in range(1, 7):
            for heading in soup.find_all(f"h{level}"):
                heading.string = f"{'#' * level} {heading.get_text(' ', strip=True)}"
        for item in soup.find_all("li"):
            item.string = f"- {item.get_text(' ', strip=True)}"
        for row in soup.find_all("tr"):
            cells = row.find_all(["td", "th"])
            if cells:
                row.string = " | ".join(c.get_text(" ", strip=True) for c in cells)

        text = soup.get_text("\n", strip=True)
        # get_text leaves a blank line between every block; keep paragraphs apart
        # but collapse the runs so cleaning has less to do.
        return re.sub(r"\n{3,}", "\n\n", text)


class MarkdownTextExtractor:
    """Reads Markdown as text, stripping only the syntax that hurts retrieval.

    Headings are left as-is - `StructuralChunker` wants them. Emphasis markers,
    link/image syntax and code fences are removed so that embeddings see prose
    rather than punctuation. No markdown library is needed for this.
    """

    supported_extensions = frozenset({".md", ".markdown", ".mdx"})

    _SUBSTITUTIONS: tuple[tuple[re.Pattern[str], str], ...] = (
        (re.compile(r"^```.*$", re.MULTILINE), ""),          # fence markers, keep the code
        (re.compile(r"!\[([^\]]*)\]\([^)]*\)"), r"\1"),      # images -> alt text
        (re.compile(r"\[([^\]]+)\]\([^)]*\)"), r"\1"),       # links -> label
        (re.compile(r"^\s*[-*_]{3,}\s*$", re.MULTILINE), ""),  # horizontal rules
        (re.compile(r"^\s{0,3}>\s?", re.MULTILINE), ""),     # block quotes
        (re.compile(r"(\*\*|__)(.+?)\1", re.DOTALL), r"\2"),  # bold
        (re.compile(r"(?<!\w)([*_])(?!\s)(.+?)(?<!\s)\1(?!\w)"), r"\2"),  # italic
        (re.compile(r"`{1,3}([^`]+)`{1,3}"), r"\1"),         # inline code
    )

    def extract(self, file_path: Path) -> ExtractedText:
        return ExtractedText(text=self.to_text(read_text_file(file_path)), page_count=1)

    def to_text(self, markdown: str) -> str:
        for pattern, replacement in self._SUBSTITUTIONS:
            markdown = pattern.sub(replacement, markdown)
        return markdown


# ── Plain and tabular formats ────────────────────────────────────


class PlainTextExtractor:
    """Reads a plain text file verbatim."""

    supported_extensions = frozenset({".txt", ".text", ".log", ".rst"})

    def extract(self, file_path: Path) -> ExtractedText:
        return ExtractedText(text=read_text_file(file_path), page_count=1)


class CsvTextExtractor:
    """Flattens a CSV into one labelled line per row.

    `Name: Ada | Role: Engineer` retrieves far better than a bare `Ada,Engineer`
    row, because each value keeps its column name next to it in the embedding.
    Files with no usable header fall back to positional rows.

    No title line is emitted: a CSV has no heading of its own, and the name it
    should carry is supplied by whoever owns it - `DocumentMetadata.filename`
    for a direct upload, or the member name for a file inside an archive.
    """

    supported_extensions = frozenset({".csv", ".tsv"})

    def extract(self, file_path: Path) -> ExtractedText:
        raw = read_text_file(file_path)
        delimiter = "\t" if file_path.suffix.lower() == ".tsv" else self._sniff(raw)

        try:
            rows = list(csv.reader(io.StringIO(raw), delimiter=delimiter))
        except csv.Error as exc:
            raise DocumentProcessingError(
                f"Could not parse '{file_path.name}' as CSV: {exc}"
            ) from exc

        if not rows:
            return ExtractedText(text="", page_count=1)

        header, *body = rows
        labelled = all(cell.strip() for cell in header) and bool(body)

        lines: list[str] = []
        if labelled:
            lines += [
                " | ".join(
                    f"{name.strip()}: {value.strip()}"
                    for name, value in zip(header, row)
                    if value.strip()
                )
                for row in body
            ]
        else:
            lines += [" | ".join(cell.strip() for cell in row) for row in rows]

        return ExtractedText(text="\n".join(line for line in lines if line), page_count=1)

    @staticmethod
    def _sniff(sample: str) -> str:
        try:
            return csv.Sniffer().sniff(sample[:4096], delimiters=",;\t|").delimiter
        except csv.Error:
            return ","


# ── Archives (Notion / Confluence exports) ───────────────────────


class ArchiveTextExtractor:
    """Extracts a whole export archive as a single document.

    Notion exports a `.zip` of Markdown (plus CSV for databases); Confluence
    exports a `.zip` of HTML. Neither needs bespoke handling: unpack the
    archive and run every member back through the registry, which already knows
    those formats. Members with no registered extractor - images, attachments,
    stylesheets - are skipped rather than failing the upload.

    Each member contributes a `# <name>` heading, so the export's page titles
    become section titles downstream.
    """

    supported_extensions = frozenset({".zip"})

    def __init__(self, registry: ExtractorRegistry):
        self._registry = registry

    def extract(self, file_path: Path) -> ExtractedText:
        try:
            with zipfile.ZipFile(file_path) as archive:
                members = self._selectable(archive, file_path.name)
                blocks = self._extract_members(archive, members)
        except zipfile.BadZipFile as exc:
            raise DocumentProcessingError(
                f"'{file_path.name}' is not a readable ZIP archive: {exc}"
            ) from exc

        if not blocks:
            raise DocumentProcessingError(
                f"No supported files inside '{file_path.name}'. Supported types: "
                f"{', '.join(sorted(self._registry.supported_extensions - {'.zip'}))}."
            )

        logger.info("Extracted %d member(s) from archive %s", len(blocks), file_path.name)
        return ExtractedText(text="\n\n".join(blocks), page_count=len(blocks))

    def _selectable(self, archive: zipfile.ZipFile, archive_name: str) -> list[zipfile.ZipInfo]:
        """Members worth extracting, in a deterministic order and within limits."""
        members: list[zipfile.ZipInfo] = []
        budget = settings.archive_max_uncompressed_bytes

        for info in sorted(archive.infolist(), key=lambda i: i.filename):
            if info.is_dir() or self._is_noise(info.filename):
                continue
            # A member with no extractor (image, attachment, CSS) is not an error.
            if not self._registry.supports(info.filename) or info.filename.endswith(".zip"):
                continue

            budget -= info.file_size
            if budget < 0:
                raise DocumentProcessingError(
                    f"'{archive_name}' expands beyond the "
                    f"{settings.archive_max_uncompressed_bytes} byte limit."
                )

            members.append(info)
            if len(members) >= settings.archive_max_members:
                logger.warning(
                    "Archive %s truncated at %d members", archive_name, settings.archive_max_members
                )
                break

        return members

    def _extract_members(
        self, archive: zipfile.ZipFile, members: list[zipfile.ZipInfo]
    ) -> list[str]:
        import tempfile

        blocks: list[str] = []
        with tempfile.TemporaryDirectory(prefix="ekip-archive-") as workspace:
            root = Path(workspace)
            for info in members:
                name = Path(info.filename)
                # Never trust a member's path: write it under a flat, safe name.
                target = root / f"{len(blocks)}{name.suffix.lower()}"
                try:
                    with archive.open(info) as source:
                        target.write_bytes(source.read())
                    extracted = self._registry.extract(target)
                except (DocumentProcessingError, UnsupportedDocumentError, OSError) as exc:
                    # One bad page must not sink a 400-page export.
                    logger.warning("Skipping '%s' in archive: %s", info.filename, exc)
                    continue

                if text := extracted.text.strip():
                    blocks.append(self._titled(text, self._title_for(name)))

        return blocks

    @staticmethod
    def _titled(text: str, title: str) -> str:
        """Ensure the member opens with a heading, without duplicating one.

        Markdown and HTML members usually carry their own `#` heading already;
        prepending the member name on top of it would repeat the title in the
        embedding for no gain. CSV and plain text have none, so they get one.
        """
        if text.lstrip().startswith("#"):
            return text
        return f"# {title}\n\n{text}"

    @staticmethod
    def _title_for(member: Path) -> str:
        """A readable page title: Notion's trailing 32-char id is stripped."""
        return _NOTION_ID_SUFFIX.sub("", member.stem).replace("_", " ").strip() or member.stem

    @staticmethod
    def _is_noise(filename: str) -> bool:
        parts = Path(filename).parts
        return (
            filename.startswith("__MACOSX/")
            or any(part.startswith(".") for part in parts)
            or ".." in parts
        )


# ── Registry ─────────────────────────────────────────────────────


def default_extractors(registry: ExtractorRegistry) -> list[TextExtractor]:
    """Every extractor shipped with the platform.

    `registry` is passed in because the archive extractor dispatches its
    members back through it.
    """
    return [
        PdfTextExtractor(),
        DocxTextExtractor(),
        HtmlTextExtractor(),
        MarkdownTextExtractor(),
        PlainTextExtractor(),
        CsvTextExtractor(),
        ArchiveTextExtractor(registry),
    ]


class ExtractorRegistry:
    """Dispatches a file to the extractor that handles its extension."""

    def __init__(self, extractors: list[TextExtractor] | None = None):
        self._extractors: dict[str, TextExtractor] = {}
        for extractor in extractors if extractors is not None else default_extractors(self):
            self.register(extractor)

    def register(self, extractor: TextExtractor) -> None:
        for extension in extractor.supported_extensions:
            self._extractors[extension.lower()] = extractor

    @property
    def supported_extensions(self) -> frozenset[str]:
        return frozenset(self._extractors)

    def supports(self, file_path: Path | str) -> bool:
        return Path(file_path).suffix.lower() in self._extractors

    def extract(self, file_path: Path | str) -> ExtractedText:
        path = Path(file_path)
        extractor = self._extractors.get(path.suffix.lower())

        if extractor is None:
            supported = ", ".join(sorted(self._extractors)) or "none"
            raise UnsupportedDocumentError(
                f"'{path.suffix or path.name}' is not supported. Supported types: {supported}."
            )

        return extractor.extract(path)
