import math


def normalize_result(data):
    if (
        not isinstance(data, dict)
        or not isinstance(data.get("text"), str)
        or not data["text"].strip()
    ):
        raise ValueError("OCR returned no recognized text")
    blocks = data.get("blocks", [])
    if not isinstance(blocks, list):
        raise ValueError("blocks must be a list")  # noqa: TRY004 - invalid serialized contract
    clean = []
    for block in blocks:
        if not isinstance(block, dict) or not isinstance(block.get("text"), str):
            raise ValueError("Invalid OCR block")  # noqa: TRY004 - invalid serialized contract
        points = block.get("bbox", [])
        if not isinstance(points, list) or any(
            not isinstance(p, (list, tuple))
            or len(p) != 2
            or any(not isinstance(v, (float, int)) or not math.isfinite(v) for v in p)
            for p in points
        ):
            raise ValueError("Invalid block coordinates")
        confidence = block.get("confidence")
        if confidence is not None and (
            not isinstance(confidence, (int, float))
            or not math.isfinite(confidence)
            or not 0 <= confidence <= 1
        ):
            raise ValueError("Invalid confidence")
        clean.append({"text": block["text"], "bbox": points, "confidence": confidence})
    return {
        "text": data["text"].strip(),
        "blocks": clean,
        "row_text": row_text(clean) or data["text"].strip(),
        "versions": data.get("versions", {}),
        "warnings": data.get("warnings", []),
    }


def row_text(blocks):
    """Group horizontal receipt boxes by vertical center, then left-to-right."""
    positioned = []
    loose = []
    for block in blocks:
        points = block.get("bbox", [])
        if not points:
            loose.append(block["text"])
            continue
        xs, ys = zip(*points)
        positioned.append(
            (min(xs), (min(ys) + max(ys)) / 2, max(max(ys) - min(ys), 1), block["text"])
        )
    rows = []
    for item in sorted(positioned, key=lambda b: (b[1], b[0])):
        # Keep the first box as a fixed anchor to avoid drifting into another row.
        if rows and abs(item[1] - rows[-1][0][1]) <= 0.45 * min(item[2], rows[-1][0][2]):
            rows[-1].append(item)
        else:
            rows.append([item])
    return "\n".join(["  ".join(b[3] for b in sorted(row)) for row in rows] + loose)
