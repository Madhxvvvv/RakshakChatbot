"""Advanced document loading for the Healthcare AI Chatbot.

Handles different document types with intelligent content extraction:
- PDF: Layout-aware extraction with table detection and multi-column support (PyMuPDF)
- Markdown: Structure-preserving loading
- Text: Plain text loading

Designed to produce high-quality, well-structured text for the RAG pipeline.
"""

import logging
from pathlib import Path

from langchain_core.documents import Document

logger = logging.getLogger(__name__)


# =============================================================================
# Supported file extensions
# =============================================================================

SUPPORTED_EXTENSIONS = {".pdf", ".md", ".txt"}


# =============================================================================
# Main loader function
# =============================================================================


def load_documents(directory: Path) -> list[Document]:
    """Load all supported documents from a directory.

    Auto-detects file type and uses the appropriate loader. Returns
    LangChain Document objects with content and metadata.

    Args:
        directory: Path to the directory containing documents.

    Returns:
        List of LangChain Document objects.
    """
    if not directory.exists():
        logger.warning("Document directory does not exist: %s", directory)
        return []

    files = [
        f for f in directory.iterdir()
        if f.is_file() and f.suffix.lower() in SUPPORTED_EXTENSIONS
    ]

    if not files:
        logger.warning("No supported documents found in %s", directory)
        return []

    logger.info("Found %d documents to load from %s", len(files), directory)

    all_documents: list[Document] = []
    for filepath in sorted(files):
        try:
            docs = _load_file(filepath)
            all_documents.extend(docs)
            logger.info("Loaded %s → %d document(s)", filepath.name, len(docs))
        except Exception as e:
            logger.error("Failed to load %s: %s", filepath.name, e)

    logger.info("Total documents loaded: %d", len(all_documents))
    return all_documents


# =============================================================================
# File type handlers
# =============================================================================

def load_single_document(filepath: Path) -> list[Document]:
    """Load a single document based on its extension.
    
    Args:
        filepath: Path to the file.
        
    Returns:
        List of LangChain Document objects.
    """
    if not filepath.exists() or not filepath.is_file():
        logger.error("File does not exist: %s", filepath)
        return []
    
    if filepath.suffix.lower() not in SUPPORTED_EXTENSIONS:
        logger.error("Unsupported file extension: %s", filepath.suffix)
        return []
        
    try:
        docs = _load_file(filepath)
        logger.info("Loaded %s → %d document(s)", filepath.name, len(docs))
        return docs
    except Exception as e:
        logger.error("Failed to load %s: %s", filepath.name, e)
        return []




def _load_file(filepath: Path) -> list[Document]:
    """Load a single file based on its extension.

    Args:
        filepath: Path to the file.

    Returns:
        List of LangChain Document objects.
    """
    suffix = filepath.suffix.lower()
    source_name = filepath.stem.replace("_", " ").title()

    if suffix == ".pdf":
        return _load_pdf(filepath, source_name)
    elif suffix == ".md":
        return _load_markdown(filepath, source_name)
    elif suffix == ".txt":
        return _load_text(filepath, source_name)
    else:
        logger.warning("Unsupported file type: %s", suffix)
        return []


# =============================================================================
# PDF Loader (Advanced — PyMuPDF)
# =============================================================================


def _load_pdf(filepath: Path, source_name: str) -> list[Document]:
    """Load a PDF with advanced extraction using PyMuPDF.

    Features:
    - Layout-aware text extraction preserving reading order
    - Table detection and conversion to markdown format
    - Multi-column layout handling
    - Page-level metadata (page number, total pages)

    Args:
        filepath: Path to the PDF file.
        source_name: Human-readable source name for metadata.

    Returns:
        List of Document objects (one per page).
    """
    try:
        import pymupdf
    except ImportError:
        logger.error("pymupdf is required for PDF processing. Install: pip install pymupdf")
        return []

    documents: list[Document] = []

    try:
        doc = pymupdf.open(str(filepath))
        total_pages = len(doc)

        for page_num in range(total_pages):
            page = doc[page_num]
            page_content_parts: list[str] = []

            # --- 1. Extract tables first ---
            tables_text = _extract_tables(page)
            if tables_text:
                page_content_parts.append(tables_text)

            # --- 2. Extract text with layout awareness ---
            text_content = _extract_text_with_layout(page)
            if text_content:
                page_content_parts.append(text_content)

            # Combine all content from this page
            page_content = "\n\n".join(page_content_parts).strip()

            if page_content:
                documents.append(
                    Document(
                        page_content=page_content,
                        metadata={
                            "source": source_name,
                            "file_name": filepath.name,
                            "page_number": page_num + 1,
                            "total_pages": total_pages,
                            "file_type": "pdf",
                        },
                    )
                )

        doc.close()
        logger.info(
            "PDF loaded: %s (%d pages, %d non-empty)",
            filepath.name,
            total_pages,
            len(documents),
        )

    except Exception as e:
        logger.error("Error processing PDF %s: %s", filepath.name, e)

    return documents


def _extract_tables(page) -> str:
    """Extract tables from a PDF page and convert to markdown format.

    Uses PyMuPDF's built-in table detection. Each table is converted
    to a markdown table for clean downstream processing.

    Args:
        page: A PyMuPDF page object.

    Returns:
        Markdown-formatted table string, or empty string if no tables found.
    """
    try:
        tables = page.find_tables()
        if not tables or len(tables.tables) == 0:
            return ""

        table_parts: list[str] = []

        for table_idx, table in enumerate(tables):
            try:
                # Extract table data as list of lists
                data = table.extract()
                if not data or len(data) < 2:
                    continue

                # Convert to markdown table
                md_table = _table_to_markdown(data)
                if md_table:
                    table_parts.append(f"**Table {table_idx + 1}:**\n{md_table}")

            except Exception as e:
                logger.debug("Failed to extract table %d: %s", table_idx, e)

        return "\n\n".join(table_parts) if table_parts else ""

    except Exception as e:
        logger.debug("Table detection not available or failed: %s", e)
        return ""


def _table_to_markdown(data: list[list]) -> str:
    """Convert a table (list of rows) to markdown table format.

    Args:
        data: Table data as list of rows, where each row is a list of cell values.

    Returns:
        Markdown-formatted table string.
    """
    if not data:
        return ""

    # Clean cell values
    cleaned = []
    for row in data:
        cleaned_row = []
        for cell in row:
            cell_text = str(cell).strip() if cell else ""
            # Replace pipes and newlines that would break markdown table
            cell_text = cell_text.replace("|", "\\|").replace("\n", " ")
            cleaned_row.append(cell_text)
        cleaned.append(cleaned_row)

    # Ensure all rows have the same number of columns
    max_cols = max(len(row) for row in cleaned)
    for row in cleaned:
        while len(row) < max_cols:
            row.append("")

    # Build markdown table
    lines: list[str] = []

    # Header row
    lines.append("| " + " | ".join(cleaned[0]) + " |")
    # Separator row
    lines.append("| " + " | ".join(["---"] * max_cols) + " |")
    # Data rows
    for row in cleaned[1:]:
        lines.append("| " + " | ".join(row) + " |")

    return "\n".join(lines)


def _extract_text_with_layout(page) -> str:
    """Extract text from a PDF page with layout awareness.

    Handles multi-column layouts by extracting text blocks with their
    positions and sorting them in reading order (top-to-bottom, left-to-right).

    Args:
        page: A PyMuPDF page object.

    Returns:
        Extracted text in proper reading order.
    """
    try:
        # Get text blocks with position information
        # Each block: (x0, y0, x1, y1, "text", block_no, block_type)
        blocks = page.get_text("blocks")

        if not blocks:
            return ""

        # Filter to text blocks only (block_type == 0)
        text_blocks = [b for b in blocks if b[6] == 0 and b[4].strip()]

        if not text_blocks:
            return ""

        # Detect if this is a multi-column layout
        page_width = page.rect.width
        page_center = page_width / 2

        # Check if blocks cluster on left and right sides (multi-column indicator)
        left_blocks = [b for b in text_blocks if b[2] < page_center + 20]
        right_blocks = [b for b in text_blocks if b[0] > page_center - 20]

        is_multi_column = (
            len(left_blocks) > 2
            and len(right_blocks) > 2
            and len(left_blocks) + len(right_blocks) > len(text_blocks) * 0.8
        )

        if is_multi_column:
            # Sort: left column (top to bottom), then right column (top to bottom)
            left = sorted(
                [b for b in text_blocks if (b[0] + b[2]) / 2 < page_center],
                key=lambda b: b[1],
            )
            right = sorted(
                [b for b in text_blocks if (b[0] + b[2]) / 2 >= page_center],
                key=lambda b: b[1],
            )
            sorted_blocks = left + right
        else:
            # Single column: sort top to bottom
            sorted_blocks = sorted(text_blocks, key=lambda b: (b[1], b[0]))

        # Combine text blocks
        paragraphs: list[str] = []
        for block in sorted_blocks:
            text = block[4].strip()
            if text:
                paragraphs.append(text)

        return "\n\n".join(paragraphs)

    except Exception as e:
        logger.debug("Layout-aware extraction failed, falling back to simple: %s", e)
        # Fallback to simple text extraction
        return page.get_text("text").strip()


# =============================================================================
# Markdown Loader
# =============================================================================


def _load_markdown(filepath: Path, source_name: str) -> list[Document]:
    """Load a markdown file as a single document.

    Args:
        filepath: Path to the markdown file.
        source_name: Human-readable source name.

    Returns:
        List containing a single Document.
    """
    content = filepath.read_text(encoding="utf-8").strip()
    if not content:
        return []

    return [
        Document(
            page_content=content,
            metadata={
                "source": source_name,
                "file_name": filepath.name,
                "file_type": "markdown",
            },
        )
    ]


# =============================================================================
# Text Loader
# =============================================================================


def _load_text(filepath: Path, source_name: str) -> list[Document]:
    """Load a plain text file as a single document.

    Args:
        filepath: Path to the text file.
        source_name: Human-readable source name.

    Returns:
        List containing a single Document.
    """
    content = filepath.read_text(encoding="utf-8").strip()
    if not content:
        return []

    return [
        Document(
            page_content=content,
            metadata={
                "source": source_name,
                "file_name": filepath.name,
                "file_type": "text",
            },
        )
    ]
