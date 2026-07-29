"""Core OCR orchestration service."""

from __future__ import annotations

import csv
import json
import logging
import re
import zipfile
from datetime import datetime
from html import unescape
from html.parser import HTMLParser
from pathlib import Path
from typing import Any
from xml.sax.saxutils import escape

import httpx

from src.config import settings
from src.schemas.ocr import (
    OCRMetadata,
    OCRPageResult,
    OCRResponse,
    OCRType,
    OutputFormat,
)
from src.utils.image_processing import encode_image_bytes_to_base64, preprocess_image
from src.utils.pdf_processing import pdf_to_images

logger = logging.getLogger(__name__)

OUTPUT_DIR = Path(__file__).resolve().parents[2] / "ocr_outputs"
OUTPUT_EXTENSIONS = {
    OutputFormat.MARKDOWN: ".md",
    OutputFormat.JSON: ".json",
    OutputFormat.PLAIN_TEXT: ".txt",
    OutputFormat.EXCEL: ".xlsx",
    OutputFormat.CSV: ".csv",
}

_WRAPPING_FENCE_RE = re.compile(
    r"^\s*```(?:json|markdown|md|text|plain_text|plaintext)?\s*\n?(.*?)\n?```\s*$",
    re.IGNORECASE | re.DOTALL,
)
_FENCE_LINE_RE = re.compile(r"^\s*```[A-Za-z0-9_-]*\s*$", re.MULTILINE)
_HTML_TAG_RE = re.compile(r"<[^>]+>")


class _HTMLTableParser(HTMLParser):
    """Small stdlib HTML table parser for model outputs."""

    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.tables: list[list[list[dict[str, Any]]]] = []
        self._table_stack: list[list[list[dict[str, Any]]]] = []
        self._current_row: list[dict[str, Any]] | None = None
        self._current_cell: dict[str, Any] | None = None
        self._cell_chunks: list[str] = []

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        tag = tag.lower()
        attrs_dict = dict(attrs)

        if tag == "table":
            self._table_stack.append([])
        elif tag == "tr" and self._table_stack:
            self._current_row = []
        elif tag in {"td", "th"} and self._current_row is not None:
            self._cell_chunks = []
            self._current_cell = {
                "text": "",
                "header": tag == "th",
                "colspan": self._positive_int(attrs_dict.get("colspan"), 1),
                "rowspan": self._positive_int(attrs_dict.get("rowspan"), 1),
            }
        elif tag == "br" and self._current_cell is not None:
            self._cell_chunks.append("\n")

    def handle_endtag(self, tag: str) -> None:
        tag = tag.lower()

        if tag == "table" and self._table_stack:
            table = self._table_stack.pop()
            if table:
                self.tables.append(table)
        elif tag == "tr" and self._table_stack and self._current_row is not None:
            self._table_stack[-1].append(self._current_row)
            self._current_row = None
        elif tag in {"td", "th"} and self._current_row is not None and self._current_cell is not None:
            text = " ".join("".join(self._cell_chunks).split())
            self._current_cell["text"] = unescape(text)
            self._current_row.append(self._current_cell)
            self._current_cell = None
            self._cell_chunks = []

    def handle_data(self, data: str) -> None:
        if self._current_cell is not None:
            self._cell_chunks.append(data)

    def close(self) -> None:
        super().close()

        if self._current_row is not None and self._table_stack:
            self._table_stack[-1].append(self._current_row)
            self._current_row = None

        while self._table_stack:
            table = self._table_stack.pop()
            if table:
                self.tables.append(table)

    @staticmethod
    def _positive_int(value: str | None, default: int) -> int:
        try:
            parsed = int(value or default)
        except (TypeError, ValueError):
            return default
        return max(1, parsed)


class OCRService:
    """Stateless service; instantiate once and reuse across requests."""

    def __init__(self) -> None:
        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    @staticmethod
    def _format_instructions(output_format: OutputFormat) -> str:
        if output_format in (OutputFormat.EXCEL, OutputFormat.CSV):
            return """
Return ONLY valid JSON that can be converted into Excel:
- No Markdown fences, comments, prose before JSON, prose after JSON, or HTML/XML tags.
- Top-level value must be an object: {"tables":[{"title":"","rows":[["Column 1","Column 2"],["Value","Value"]]}]}.
- Use one array per row and one string/number/null per cell.
- Keep blank cells as "" so columns do not shift.
- Preserve the visible table order and column order."""

        if output_format == OutputFormat.JSON:
            return """
Return ONLY valid JSON:
- No Markdown fences, comments, prose before JSON, or prose after JSON.
- Do not output HTML/XML tags such as <table>, <tr>, <td>, or <br>.
- Use double quotes for all strings and property names.
- Do not use trailing commas.
- Use null for empty values, not invented text.
- Preserve line breaks inside string values with \\n.
- Top-level value must be a JSON object.
- For tables, use {"tables":[{"rows":[[{"text":"...","colspan":1,"rowspan":1}]]}]}."""

        if output_format == OutputFormat.PLAIN_TEXT:
            return """
Return ONLY plain text:
- No Markdown headings, bullets added by you, JSON, HTML/XML tags, or code fences.
- For tables, use tab-separated columns and one row per line.
- Preserve natural line breaks from the document.
- Do not add explanations or confidence commentary unless text is unclear."""

        return """
Return ONLY Markdown:
- Do not output HTML/XML tags such as <table>, <tr>, <td>, or <br>.
- No enclosing code fence.
- Use headings only when the document has clear sections.
- Use Markdown tables for real tables and keep column counts consistent.
- Preserve original wording, numbers, punctuation, and meaningful line breaks."""

    @classmethod
    def _build_prompt(cls, ocr_type: OCRType, output_format: OutputFormat) -> str:
        base = """You are an expert OCR engine.

Extract every visible text element from the image with maximum fidelity.
Preserve the original spelling, punctuation, capitalization, numbers, dates,
codes, symbols, and reading order. Do not summarize. Do not translate.
If a character or word is unclear, mark it as [uncertain] and keep going.
If no text is visible, return an empty result in the requested format."""

        type_instructions: dict[OCRType, str] = {
            OCRType.GENERAL: """
Document handling:
- Read headers, body text, captions, stamps, labels, signatures, and footers.
- Keep related lines together and separate unrelated text blocks.
- Preserve useful spacing where it affects meaning.""",
            OCRType.TABLE: """
Table handling:
- Detect every table and preserve row/column structure.
- Include table titles, headers, merged-cell text, totals, notes, and footnotes.
- Keep blank cells empty instead of shifting values across columns.
- For Markdown output, every table row must have the same number of cells.""",
            OCRType.FORM: """
Form handling:
- Extract labels with their corresponding values.
- Capture checkboxes as [X] or [ ].
- Keep grouped fields together.
- Preserve printed instructions and handwritten entries.""",
            OCRType.HANDWRITTEN: """
Handwriting handling:
- Read cursive and printed handwriting carefully.
- Mark illegible words as [illegible].
- Mark ambiguous words as [uncertain: best_guess].
- Do not normalize spelling or grammar.""",
            OCRType.DENSE: """
Dense document handling:
- Process the page systematically by visible reading order.
- Preserve columns as separate blocks.
- Capture small print, footnotes, page numbers, stamps, and annotations.
- Do not merge unrelated sections.""",
        }

        return "\n\n".join(
            [
                base,
                type_instructions.get(ocr_type, type_instructions[OCRType.GENERAL]),
                cls._format_instructions(output_format),
            ]
        )

    @staticmethod
    def _strip_wrapping_fence(text: str) -> str:
        cleaned = text.strip()
        match = _WRAPPING_FENCE_RE.match(cleaned)
        if match:
            return match.group(1).strip()
        return cleaned

    @staticmethod
    def _strip_fence_lines(text: str) -> str:
        return _FENCE_LINE_RE.sub("", text).strip()

    @staticmethod
    def _extract_html_tables(text: str) -> list[list[list[dict[str, Any]]]]:
        if "<table" not in text.lower():
            return []

        parser = _HTMLTableParser()
        parser.feed(text)
        parser.close()
        return parser.tables

    @staticmethod
    def _expanded_table_rows(table: list[list[dict[str, Any]]]) -> list[list[str]]:
        rows: list[list[str]] = []
        for row in table:
            expanded_row: list[str] = []
            for cell in row:
                colspan = max(1, int(cell.get("colspan", 1)))
                expanded_row.append(str(cell.get("text", "")))
                expanded_row.extend([""] * (colspan - 1))
            rows.append(expanded_row)

        width = max((len(row) for row in rows), default=0)
        return [row + [""] * (width - len(row)) for row in rows]

    @classmethod
    def _html_tables_to_json(cls, tables: list[list[list[dict[str, Any]]]]) -> dict[str, Any]:
        return {
            "tables": [
                {
                    "rows": [
                        [
                            {
                                "text": cell.get("text", ""),
                                "colspan": int(cell.get("colspan", 1)),
                                "rowspan": int(cell.get("rowspan", 1)),
                            }
                            for cell in row
                        ]
                        for row in table
                    ]
                }
                for table in tables
            ]
        }

    @classmethod
    def _html_tables_to_markdown(cls, tables: list[list[list[dict[str, Any]]]]) -> str:
        markdown_tables: list[str] = []
        for table in tables:
            rows = cls._expanded_table_rows(table)
            if not rows:
                continue

            width = len(rows[0])
            header = rows[0]
            separator = ["---"] * width
            body = rows[1:] or [[""] * width]
            markdown_rows = [header, separator, *body]
            markdown_tables.append(
                "\n".join(
                    "| " + " | ".join(cell.replace("|", "\\|") for cell in row) + " |"
                    for row in markdown_rows
                )
            )

        return "\n\n".join(markdown_tables)

    @classmethod
    def _html_tables_to_plain_text(cls, tables: list[list[list[dict[str, Any]]]]) -> str:
        plain_tables: list[str] = []
        for table in tables:
            rows = cls._expanded_table_rows(table)
            if rows:
                plain_tables.append("\n".join("\t".join(row).rstrip() for row in rows))
        return "\n\n".join(plain_tables)

    @staticmethod
    def _strip_html_tags(text: str) -> str:
        return unescape(_HTML_TAG_RE.sub("", text)).strip()

    @staticmethod
    def _cell_text(value: Any) -> str:
        if value is None:
            return ""
        if isinstance(value, dict):
            return str(value.get("text", ""))
        return str(value)

    @classmethod
    def _json_tables_to_rows(cls, parsed: Any) -> list[dict[str, Any]]:
        if isinstance(parsed, list):
            if all(isinstance(row, list) for row in parsed):
                return [{"title": "Table 1", "rows": parsed}]
            rows = [["Value"]]
            rows.extend([[json.dumps(item, ensure_ascii=False)] for item in parsed])
            return [{"title": "Items", "rows": rows}]

        if not isinstance(parsed, dict):
            return []

        raw_tables = parsed.get("tables")
        if isinstance(raw_tables, list):
            tables: list[dict[str, Any]] = []
            for idx, raw_table in enumerate(raw_tables, start=1):
                title = f"Table {idx}"
                rows: Any = []
                if isinstance(raw_table, dict):
                    title = str(raw_table.get("title") or title)
                    rows = raw_table.get("rows") or raw_table.get("data") or []
                elif isinstance(raw_table, list):
                    rows = raw_table

                normalized_rows: list[list[str]] = []
                for row in rows:
                    if isinstance(row, dict):
                        normalized_rows.append([cls._cell_text(value) for value in row.values()])
                    elif isinstance(row, list):
                        normalized_rows.append([cls._cell_text(value) for value in row])

                if normalized_rows:
                    tables.append({"title": title, "rows": normalized_rows})
            return tables

        if "rows" in parsed and isinstance(parsed["rows"], list):
            rows = [
                [cls._cell_text(value) for value in row]
                for row in parsed["rows"]
                if isinstance(row, list)
            ]
            return [{"title": str(parsed.get("title") or "Table 1"), "rows": rows}] if rows else []

        rows = [["Field", "Value"]]
        for key, value in parsed.items():
            if isinstance(value, (dict, list)):
                value = json.dumps(value, ensure_ascii=False)
            rows.append([str(key), cls._cell_text(value)])
        return [{"title": "Data", "rows": rows}]

    @classmethod
    def _markdown_tables_to_rows(cls, text: str) -> list[dict[str, Any]]:
        tables: list[dict[str, Any]] = []
        current: list[list[str]] = []

        def flush() -> None:
            nonlocal current
            if current:
                tables.append({"title": f"Table {len(tables) + 1}", "rows": current})
                current = []

        for line in text.splitlines():
            stripped = line.strip()
            if stripped.startswith("|") and stripped.endswith("|"):
                cells = [cell.strip().replace("\\|", "|") for cell in stripped.strip("|").split("|")]
                if cells and all(re.fullmatch(r":?-{3,}:?", cell) for cell in cells):
                    continue
                current.append(cells)
            else:
                flush()
        flush()
        return tables

    @classmethod
    def _plain_text_to_rows(cls, text: str) -> list[dict[str, Any]]:
        rows = []
        for line in text.splitlines():
            if "\t" in line:
                rows.append([cell.strip() for cell in line.split("\t")])
            elif line.strip():
                rows.append([line.strip()])
        return [{"title": "OCR Text", "rows": rows}] if rows else []

    @classmethod
    def _tables_for_export(cls, text: str) -> list[dict[str, Any]]:
        html_tables = cls._extract_html_tables(text)
        if html_tables:
            return [
                {"title": f"Table {idx}", "rows": cls._expanded_table_rows(table)}
                for idx, table in enumerate(html_tables, start=1)
            ]

        try:
            json_tables = cls._json_tables_to_rows(cls._parse_json_from_text(text))
            if json_tables:
                return json_tables
        except ValueError:
            pass

        markdown_tables = cls._markdown_tables_to_rows(text)
        if markdown_tables:
            return markdown_tables

        return cls._plain_text_to_rows(text)

    @staticmethod
    def _column_letter(index: int) -> str:
        letters = ""
        while index:
            index, remainder = divmod(index - 1, 26)
            letters = chr(65 + remainder) + letters
        return letters or "A"

    @staticmethod
    def _sheet_name(name: str, used_names: set[str]) -> str:
        cleaned = re.sub(r"[\[\]\:\*\?\/\\]", " ", name).strip() or "Sheet"
        cleaned = cleaned[:31]
        candidate = cleaned
        suffix = 2
        while candidate in used_names:
            tail = f" {suffix}"
            candidate = f"{cleaned[:31 - len(tail)]}{tail}"
            suffix += 1
        used_names.add(candidate)
        return candidate

    @staticmethod
    def _xml_text(value: Any) -> str:
        return escape(str(value), {'"': "&quot;", "'": "&apos;"})

    @classmethod
    def _sheet_xml(cls, rows: list[list[str]]) -> str:
        width = max((len(row) for row in rows), default=1)
        rows = [row + [""] * (width - len(row)) for row in rows] or [[""]]
        column_xml = "".join(
            f'<col min="{idx}" max="{idx}" width="{min(max_width, 45)}" customWidth="1"/>'
            for idx, max_width in enumerate(
                [
                    max(12, min(45, max(len(str(row[col_idx])) for row in rows) + 2))
                    for col_idx in range(width)
                ],
                start=1,
            )
        )

        row_xml = []
        for row_idx, row in enumerate(rows, start=1):
            cell_xml = []
            for col_idx, value in enumerate(row, start=1):
                cell_ref = f"{cls._column_letter(col_idx)}{row_idx}"
                style = ' s="1"' if row_idx == 1 else ""
                cell_xml.append(
                    f'<c r="{cell_ref}" t="inlineStr"{style}><is><t>{cls._xml_text(value)}</t></is></c>'
                )
            row_xml.append(f'<row r="{row_idx}">{"".join(cell_xml)}</row>')

        last_ref = f"{cls._column_letter(width)}{len(rows)}"
        return f'''<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<worksheet xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main"
 xmlns:r="http://schemas.openxmlformats.org/officeDocument/2006/relationships">
<dimension ref="A1:{last_ref}"/>
<sheetViews><sheetView workbookViewId="0"><pane ySplit="1" topLeftCell="A2" activePane="bottomLeft" state="frozen"/></sheetView></sheetViews>
<cols>{column_xml}</cols>
<sheetData>{"".join(row_xml)}</sheetData>
</worksheet>'''

    @classmethod
    def _write_xlsx(cls, output_path: Path, tables: list[dict[str, Any]]) -> None:
        if not tables:
            tables = [{"title": "OCR Text", "rows": [["No extractable text"]]}]

        used_names: set[str] = set()
        sheets = [
            {
                "name": cls._sheet_name(str(table.get("title") or f"Table {idx}"), used_names),
                "rows": table.get("rows") or [[""]],
            }
            for idx, table in enumerate(tables, start=1)
        ]

        workbook_sheets = "".join(
            f'<sheet name="{cls._xml_text(sheet["name"])}" sheetId="{idx}" r:id="rId{idx}"/>'
            for idx, sheet in enumerate(sheets, start=1)
        )
        workbook_rels = "".join(
            f'<Relationship Id="rId{idx}" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/worksheet" Target="worksheets/sheet{idx}.xml"/>'
            for idx, _ in enumerate(sheets, start=1)
        )
        workbook_rels += f'<Relationship Id="rId{len(sheets) + 1}" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/styles" Target="styles.xml"/>'
        overrides = "".join(
            f'<Override PartName="/xl/worksheets/sheet{idx}.xml" ContentType="application/vnd.openxmlformats-officedocument.spreadsheetml.worksheet+xml"/>'
            for idx, _ in enumerate(sheets, start=1)
        )

        with zipfile.ZipFile(output_path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
            archive.writestr(
                "[Content_Types].xml",
                f'''<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<Types xmlns="http://schemas.openxmlformats.org/package/2006/content-types">
<Default Extension="rels" ContentType="application/vnd.openxmlformats-package.relationships+xml"/>
<Default Extension="xml" ContentType="application/xml"/>
<Override PartName="/xl/workbook.xml" ContentType="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet.main+xml"/>
<Override PartName="/xl/styles.xml" ContentType="application/vnd.openxmlformats-officedocument.spreadsheetml.styles+xml"/>
{overrides}
</Types>''',
            )
            archive.writestr(
                "_rels/.rels",
                '''<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">
<Relationship Id="rId1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/officeDocument" Target="xl/workbook.xml"/>
</Relationships>''',
            )
            archive.writestr(
                "xl/workbook.xml",
                f'''<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<workbook xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main"
 xmlns:r="http://schemas.openxmlformats.org/officeDocument/2006/relationships">
<sheets>{workbook_sheets}</sheets>
</workbook>''',
            )
            archive.writestr(
                "xl/_rels/workbook.xml.rels",
                f'''<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">
{workbook_rels}
</Relationships>''',
            )
            archive.writestr(
                "xl/styles.xml",
                '''<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<styleSheet xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main">
<fonts count="2"><font><sz val="11"/><name val="Calibri"/></font><font><b/><sz val="11"/><name val="Calibri"/></font></fonts>
<fills count="2"><fill><patternFill patternType="none"/></fill><fill><patternFill patternType="gray125"/></fill></fills>
<borders count="1"><border><left/><right/><top/><bottom/><diagonal/></border></borders>
<cellStyleXfs count="1"><xf numFmtId="0" fontId="0" fillId="0" borderId="0"/></cellStyleXfs>
<cellXfs count="2"><xf numFmtId="0" fontId="0" fillId="0" borderId="0" xfId="0"/><xf numFmtId="0" fontId="1" fillId="0" borderId="0" xfId="0"/></cellXfs>
</styleSheet>''',
            )
            for idx, sheet in enumerate(sheets, start=1):
                archive.writestr(f"xl/worksheets/sheet{idx}.xml", cls._sheet_xml(sheet["rows"]))

    @staticmethod
    def _parse_json_from_text(text: str) -> Any:
        cleaned = OCRService._strip_wrapping_fence(text)
        try:
            return json.loads(cleaned)
        except json.JSONDecodeError:
            pass

        decoder = json.JSONDecoder()
        for match in re.finditer(r"[\{\[]", cleaned):
            try:
                parsed, _ = decoder.raw_decode(cleaned[match.start() :])
                return parsed
            except json.JSONDecodeError:
                continue

        raise ValueError("Model output did not contain valid JSON.")

    @classmethod
    def _write_csv(cls, output_path: Path, tables: list[dict[str, Any]]) -> None:
        if not tables:
            tables = [{"title": "OCR Text", "rows": [["No extractable text"]]}]

        with output_path.open("w", encoding="utf-8-sig", newline="") as fh:
            writer = csv.writer(fh)
            multi_table = len(tables) > 1
            for idx, table in enumerate(tables):
                if multi_table:
                    if idx > 0:
                        writer.writerow([])
                    writer.writerow([table.get("title") or f"Table {idx + 1}"])
                for row in table.get("rows") or [[""]]:
                    writer.writerow(row)

    @classmethod
    def _normalize_output(cls, text: str, output_format: OutputFormat) -> str:
        text = text or ""
        tables = cls._extract_html_tables(text)

        if output_format in (OutputFormat.EXCEL, OutputFormat.CSV):
            if export_tables:
                markdown_tables = [
                    cls._html_tables_to_markdown(
                        [
                            [
                                [{"text": cell, "colspan": 1, "rowspan": 1} for cell in row]
                                for row in table["rows"]
                            ]
                        ]
                    )
                    for table in export_tables
                ]
                return "\n\n".join(markdown_tables)
            return cls._strip_html_tags(cls._strip_fence_lines(text))

        if output_format == OutputFormat.JSON:
            if tables:
                parsed = cls._html_tables_to_json(tables)
            else:
                try:
                    parsed = cls._parse_json_from_text(text)
                    if not isinstance(parsed, dict):
                        parsed = {"items": parsed}
                except ValueError:
                    parsed = {
                        "text": cls._strip_html_tags(cls._strip_fence_lines(text)),
                        "warning": "model_output_was_not_valid_json",
                    }
            return json.dumps(parsed, ensure_ascii=False, indent=2)

        if output_format == OutputFormat.PLAIN_TEXT:
            if tables:
                return cls._html_tables_to_plain_text(tables)
            return cls._strip_fence_lines(cls._strip_wrapping_fence(text))

        if tables:
            return cls._html_tables_to_markdown(tables)

        return cls._strip_wrapping_fence(text).strip()

    @staticmethod
    def _safe_output_name(original_filename: str, output_format: OutputFormat) -> str:
        base_name = Path(original_filename).stem or "ocr_result"
        safe_name = "".join(
            char if char.isalnum() or char in ("-", "_", " ") else "_"
            for char in base_name
        ).strip()
        safe_name = safe_name or "ocr_result"
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        return f"{safe_name}_ocr_{timestamp}{OUTPUT_EXTENSIONS[output_format]}"

    @classmethod
    def save_output(
        cls,
        text: str,
        original_filename: str,
        output_format: OutputFormat,
    ) -> str:
        """Save OCR text to the correct output file type."""
        output_path = OUTPUT_DIR / cls._safe_output_name(original_filename, output_format)
        if output_format == OutputFormat.EXCEL:
            cls._write_xlsx(output_path, cls._tables_for_export(text))
        elif output_format == OutputFormat.CSV:
            cls._write_csv(output_path, cls._tables_for_export(text))
        else:
            output_path.write_text(text, encoding="utf-8")
        logger.info("Saved OCR output to %s", output_path)
        return str(output_path)

    @classmethod
    def _tables_to_plain_text(cls, tables: list[dict[str, Any]]) -> str:
        return "\n\n".join(
            "\n".join("\t".join(cell for cell in row).rstrip() for row in table.get("rows") or [])
            for table in tables
        )

    @classmethod
    def _tables_to_markdown(cls, tables: list[dict[str, Any]]) -> str:
        blocks: list[str] = []
        for table in tables:
            rows = table.get("rows") or []
            if not rows:
                continue
            width = len(rows[0])
            header, body = rows[0], rows[1:] or [[""] * width]
            all_rows = [header, ["---"] * width, *body]
            blocks.append(
                "\n".join(
                    "| " + " | ".join(cell.replace("|", "\\|") for cell in row) + " |"
                    for row in all_rows
                )
            )
        return "\n\n".join(blocks)

    @classmethod
    def _tables_to_json(cls, tables: list[dict[str, Any]]) -> str:
        return json.dumps({"tables": tables}, ensure_ascii=False, indent=2)

    @classmethod
    def convert_output(cls, source_filename: str, target_format: OutputFormat) -> str:
        """Convert an already-saved output file (json/md/txt/xlsx source text)
        into any other output format, including Excel.

        Works regardless of the source file's original format because
        `_tables_for_export` already knows how to pull tabular data out of
        HTML, JSON, Markdown, or plain text -- so the format that produced
        the source file doesn't matter, only its content does.
        """
        output_dir = OUTPUT_DIR.resolve()
        source_path = (output_dir / source_filename).resolve()

        if source_path.parent != output_dir or not source_path.exists() or not source_path.is_file():
            raise FileNotFoundError(f"Source output '{source_filename}' not found.")

        if source_path.suffix.lower() in (".xlsx", ".csv"):
            raise ValueError(
                "Converting from an existing .xlsx or .csv file isn't supported. "
                "Convert from the json, md, or txt output instead."
            )

        raw_text = source_path.read_text(encoding="utf-8")
        tables = cls._tables_for_export(raw_text)

        base_stem = re.sub(r"_ocr_\d{8}_\d{6}$", "", source_path.stem) or source_path.stem
        output_path = OUTPUT_DIR / cls._safe_output_name(base_stem, target_format)

        if target_format == OutputFormat.EXCEL:
            cls._write_xlsx(output_path, tables)
        elif target_format == OutputFormat.CSV:
            cls._write_csv(output_path, tables)
        elif target_format == OutputFormat.JSON:
            output_path.write_text(cls._tables_to_json(tables), encoding="utf-8")
        elif target_format == OutputFormat.PLAIN_TEXT:
            output_path.write_text(cls._tables_to_plain_text(tables), encoding="utf-8")
        else:
            output_path.write_text(cls._tables_to_markdown(tables), encoding="utf-8")

        logger.info("Converted %s -> %s", source_path.name, output_path)
        return str(output_path)

    async def _ocr_single_image(
        self,
        image_bytes: bytes,
        ocr_type: OCRType,
        output_format: OutputFormat,
        preprocess: bool,
        *,
        resize_factor: float = 1.0,
        denoise: bool = False,
        enhance_contrast: bool = False,
        sharpen: bool = False,
        binarize: bool = False,
    ) -> dict[str, Any]:
        """Run OCR on a single image and return a raw result dict."""
        try:
            if preprocess:
                logger.info("Preprocessing image")
                image_b64 = preprocess_image(
                    image_bytes,
                    resize_factor=resize_factor,
                    denoise=denoise,
                    enhance_contrast=enhance_contrast,
                    sharpen=sharpen,
                    binarize=binarize,
                )
            else:
                image_b64 = encode_image_bytes_to_base64(image_bytes)
        except Exception as exc:
            logger.exception("Image preparation failed")
            return {
                "success": False,
                "text": None,
                "error": f"Image preparation failed: {exc}",
            }

        generation_params = settings.generation_params
        payload: dict[str, Any] = {
            "model": settings.OCR_MODEL,
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": self._build_prompt(ocr_type, output_format)},
                        {
                            "type": "image_url",
                            "image_url": {"url": f"data:image/png;base64,{image_b64}"},
                        },
                    ],
                }
            ],
            "max_tokens": settings.MAX_TOKENS,
            "temperature": generation_params["temperature"],
            "top_p": generation_params["top_p"],
            "top_k": generation_params["top_k"],
            "repetition_penalty": generation_params["repetition_penalty"],
            "frequency_penalty": generation_params["frequency_penalty"],
            "presence_penalty": generation_params["presence_penalty"],
            "min_p": generation_params["min_p"],
        }

        if "localhost" not in settings.OCR_API_URL and "127.0.0.1" not in settings.OCR_API_URL:
            payload["mm_processor_kwargs"] = {
                "max_soft_tokens": settings.MAX_SOFT_TOKENS,
            }

        try:
            async with httpx.AsyncClient(timeout=httpx.Timeout(settings.OCR_TIMEOUT)) as client:
                response = await client.post(
                    settings.OCR_API_URL,
                    headers={"Content-Type": "application/json"},
                    content=json.dumps(payload),
                )
                response.raise_for_status()

            result = response.json()
            text = result["choices"][0]["message"]["content"]
            normalized_text = self._normalize_output(text, output_format)

            return {
                "success": True,
                "text": normalized_text,
                "tokens_used": result.get("usage", {}),
            }

        except httpx.HTTPStatusError as exc:
            logger.error("VL model API returned %s: %s", exc.response.status_code, exc.response.text)
            return {
                "success": False,
                "text": None,
                "error": f"Model API returned HTTP {exc.response.status_code}",
            }
        except httpx.RequestError as exc:
            logger.error("Request to VL model failed: %s", exc)
            return {
                "success": False,
                "text": None,
                "error": f"Request failed: {exc}",
            }
        except (KeyError, IndexError, json.JSONDecodeError) as exc:
            logger.error("Unexpected VL model response structure: %s", exc)
            return {
                "success": False,
                "text": None,
                "error": f"Unexpected model response: {exc}",
            }

    async def process_file(
        self,
        file_bytes: bytes,
        filename: str,
        is_pdf: bool,
        ocr_type: OCRType = OCRType.GENERAL,
        output_format: OutputFormat = OutputFormat.MARKDOWN,
        preprocess: bool = False,
        *,
        resize_factor: float = 1.0,
        denoise: bool = False,
        enhance_contrast: bool = False,
        sharpen: bool = False,
        binarize: bool = False,
    ) -> OCRResponse | dict[str, Any]:
        """Process an uploaded image or PDF and return a structured response."""
        effective_preprocess = (
            preprocess
            or denoise
            or enhance_contrast
            or sharpen
            or binarize
            or resize_factor != 1.0
        )

        preprocess_kwargs = {
            "resize_factor": resize_factor,
            "denoise": denoise,
            "enhance_contrast": enhance_contrast,
            "sharpen": sharpen,
            "binarize": binarize,
        }

        if is_pdf:
            result = await self._process_pdf(
                file_bytes,
                filename,
                ocr_type,
                output_format,
                effective_preprocess,
                **preprocess_kwargs,
            )
        else:
            result = await self._process_image(
                file_bytes,
                filename,
                ocr_type,
                output_format,
                effective_preprocess,
                **preprocess_kwargs,
            )

        if isinstance(result, OCRResponse):
            output_path = self.save_output(result.text, filename, output_format)
            result.output_path = output_path
            result.output_url = f"/outputs/{Path(output_path).name}"
            if output_format == OutputFormat.MARKDOWN:
                result.md_output_path = output_path

        return result

    async def _process_image(
        self,
        image_bytes: bytes,
        filename: str,
        ocr_type: OCRType,
        output_format: OutputFormat,
        preprocess: bool,
        **kwargs: Any,
    ) -> OCRResponse | dict[str, Any]:
        result = await self._ocr_single_image(
            image_bytes,
            ocr_type,
            output_format,
            preprocess,
            **kwargs,
        )

        if not result["success"]:
            return {"success": False, "error": result["error"]}

        return OCRResponse(
            filename=filename,
            text=result["text"],
            pages=[OCRPageResult(page_number=1, text=result["text"])],
            metadata=OCRMetadata(
                model=settings.OCR_MODEL,
                ocr_type=ocr_type.value,
                output_format=output_format.value,
                preprocessed=preprocess,
                total_pages=1,
                tokens_used=result["tokens_used"],
            ),
        )

    @staticmethod
    def _format_failed_page(idx: int, error: str, output_format: OutputFormat) -> str:
        message = f"OCR failed on page {idx}: {error}"
        if output_format == OutputFormat.JSON:
            return json.dumps({"error": message}, ensure_ascii=False, indent=2)
        return f"[{message}]"

    @staticmethod
    def _combine_pdf_texts(pages: list[OCRPageResult], output_format: OutputFormat) -> str:
        if output_format == OutputFormat.JSON:
            page_payloads = []
            for page in pages:
                try:
                    content = json.loads(page.text)
                except json.JSONDecodeError:
                    content = {"text": page.text}
                page_payloads.append(
                    {
                        "page_number": page.page_number,
                        "content": content,
                    }
                )
            return json.dumps({"pages": page_payloads}, ensure_ascii=False, indent=2)

        if output_format == OutputFormat.PLAIN_TEXT:
            return "\n\n".join(
                f"Page {page.page_number}\n\n{page.text}".strip()
                for page in pages
            )

        return "\n\n---\n\n".join(
            f"## Page {page.page_number}\n\n{page.text}".strip()
            for page in pages
        )

    async def _process_pdf(
        self,
        pdf_bytes: bytes,
        filename: str,
        ocr_type: OCRType,
        output_format: OutputFormat,
        preprocess: bool,
        **kwargs: Any,
    ) -> OCRResponse | dict[str, Any]:
        try:
            page_images = pdf_to_images(pdf_bytes, dpi=settings.PDF_DPI)
        except ValueError as exc:
            return {"success": False, "error": str(exc)}

        if not page_images:
            return {"success": False, "error": "PDF contains no pages."}

        pages: list[OCRPageResult] = []
        total_tokens: dict[str, Any] = {}
        failed_pages = 0

        for idx, img_bytes in enumerate(page_images, start=1):
            logger.info("Processing PDF page %d / %d", idx, len(page_images))

            result = await self._ocr_single_image(
                img_bytes,
                ocr_type,
                output_format,
                preprocess,
                **kwargs,
            )

            if not result["success"]:
                failed_pages += 1
                page_text = self._format_failed_page(idx, result["error"], output_format)
                logger.warning("OCR failed on page %d: %s", idx, result["error"])
            else:
                page_text = result["text"]
                for key, value in result.get("tokens_used", {}).items():
                    if isinstance(value, int):
                        total_tokens[key] = total_tokens.get(key, 0) + value

            pages.append(OCRPageResult(page_number=idx, text=page_text))

        if failed_pages == len(pages):
            return {
                "success": False,
                "error": "OCR failed on all pages. Check model API connectivity and logs.",
            }

        return OCRResponse(
            filename=filename,
            text=self._combine_pdf_texts(pages, output_format),
            pages=pages,
            metadata=OCRMetadata(
                model=settings.OCR_MODEL,
                ocr_type=ocr_type.value,
                output_format=output_format.value,
                preprocessed=preprocess,
                total_pages=len(page_images),
                tokens_used=total_tokens,
            ),
        )
