"""Package the public operator manuals at their repository-relative paths."""

import sys
from pathlib import Path
from zipfile import ZIP_DEFLATED, ZipFile, ZipInfo

ROOT = Path(__file__).resolve().parents[2]
OUTPUT = ROOT / "frontend/static/vision-bill-operator-guides.zip"
PUBLIC_FILES = (
    "README.md",
    "AGENTS.md",
    "HAENDLERVERZEICHNIS.md",
    "MULTI_IMAGE_RECEIPTS.md",
    "OPENAI_COMPATIBLE.md",
    "RELEASES.md",
    "UPGRADE_AND_VERIFY.md",
    ".github/workflows/distribution-inputs.yml",
    ".github/workflows/distribution-release.yml",
    "distribution/README.md",
    "distribution/artifacts.schema.json",
    "frontend/README.md",
    "frontend/patches/README.md",
    "paddleocr-service/README.md",
    "paddleocr-service/first-run-fixtures/README.md",
    "paddleocr-service/profile.json",
    "paddleocr-service/release-bundle.schema.json",
    "scripts/README.md",
    "scripts/distribution.mjs",
    "scripts/ocr_harness/README.md",
    "scripts/ocr_harness/BENCHMARK_RESULTS.md",
    "scripts/receipt_split_experiment.md",
    "scripts/receipt_split_results.md",
    "tests/e2e/data/README.md",
)


def build(output: Path) -> None:
    with ZipFile(output, "w") as archive:
        for name in PUBLIC_FILES:
            # A fixed timestamp makes identical documentation produce identical bytes.
            archive.writestr(ZipInfo(name), (ROOT / name).read_bytes(), ZIP_DEFLATED)


if __name__ == "__main__":
    build(Path(sys.argv[1]) if len(sys.argv) > 1 else OUTPUT)
