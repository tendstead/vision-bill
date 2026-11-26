"""Verify the packaged manuals have their local link targets after extraction."""

import posixpath
import re
import tempfile
import unittest
from pathlib import Path
from urllib.parse import unquote, urlsplit
from zipfile import ZipFile

from operator_guides import build


class OperatorGuideTest(unittest.TestCase):
    def test_archive_contains_linked_files(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "guides.zip"
            build(path)
            with ZipFile(path) as archive:
                names = set(archive.namelist())
                required = {"paddleocr-service/README.md", "distribution/README.md"}
                self.assertTrue(required <= names)
                for name in sorted(names):
                    if not name.endswith(".md"):
                        continue
                    source = archive.read(name).decode()
                    for link in re.findall(r"\]\(([^\s)]+)\)", source):
                        url = urlsplit(link)
                        if url.scheme or url.netloc or not url.path:
                            continue
                        target = posixpath.normpath(
                            posixpath.join(posixpath.dirname(name), unquote(url.path))
                        )
                        self.assertIn(target, names, f"{name} links to {link}")
