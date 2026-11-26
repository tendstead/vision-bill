# First-run OCR fixtures

These project-authored English documents contain fictional transactions and no
personal data. The source launcher packages the tracked PNGs listed in
`fixtures.json` for explicit physical qualification. Enabling Dedicated OCR
verifies model files and leaves the worker parked. Only qualification runs
the fixture corpus. The files are licensed under AGPL-3.0-or-later.

| Input | Document |
| --- | --- |
| `upright.png` | A grocery receipt with right-aligned prices and totals |
| `rotated.png` | A restaurant receipt rotated 90° counterclockwise in the pixels, with no EXIF rotation |
| `warped.png` | A hardware receipt with curved lines and perspective distortion |
| `table.png` | A ruled invoice with quantity, unit price and amount columns |
| `formula.png` | An algebra page with a displayed quadratic equation and a typeset fraction |

The fixtures exercise native inference and the table and formula recognition
paths. The authored images are functional test inputs. Coverage of naturally
photographed receipts and model accuracy require separate evaluation.

The source is `render.py`. Pillow 12.3.0 and DejaVu fonts 2.37 were used to
create the images. The images contain rendered text pixels only. DejaVu fonts
are available under the Bitstream Vera font license with DejaVu changes in the
public domain. [Font license](https://dejavu-fonts.github.io/License.html)

To regenerate locally with the Debian `fonts-dejavu-core` package installed,
run the following from the repository root. PNG byte identity can vary with
the FreeType and PNG encoder versions. The tracked PNGs are the authoritative
inputs for `first-run-ocr-fixtures-v1`.

```bash
uv run --no-project --with Pillow==12.3.0 python \
  paddleocr-service/first-run-fixtures/render.py /tmp/vision-bill-first-run-fixtures
```
