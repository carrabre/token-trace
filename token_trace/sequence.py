"""Token labels and sentence grouping without model dependencies."""
import re


def sentence_spans(text):
    """Editorial grouping for display. A separator never splits the actual model input."""
    spans = []
    start = 0
    for match in re.finditer(r"(?<=[.!?])\s+|\n+", text):
        if match.start() > start:
            spans.append((start, match.start()))
        start = match.end()
    if start < len(text):
        spans.append((start, len(text)))
    return spans


def tokens_for_sequence(ids, text, offsets=None, prompt_count=None, special_ids=()):
    spans = sentence_spans(text)
    result = []
    sentence = 0
    for position, (token_id, piece) in enumerate(ids):
        record = {"position": position, "id": token_id, "text": piece,
                  "segment": "generated" if prompt_count is not None and position >= prompt_count else "input"}
        if token_id in special_ids:
            record["special"] = True
        offset = offsets[position] if offsets and position < len(offsets) else None
        if offset and offset[1] > offset[0]:
            while sentence + 1 < len(spans) and offset[0] >= spans[sentence][1]:
                sentence += 1
            if spans:
                record["sentence"] = sentence + 1
            record["start"], record["end"] = offset
        result.append(record)
    return result
