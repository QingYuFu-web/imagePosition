"""Verify a complete short label across alternate cuts of the same visible ink."""

from itertools import combinations


def _mask_key(mask):
    return mask.shape, mask.tobytes()


def _pair_reading(a, b, first_mask, second_mask, index):
    if len(a) < 2 or len(b) < 2 or a[0]["text"] != b[0]["text"]:
        return None
    first, second = a[0], b[0]
    first_margin = a[1]["distance"] - first["distance"]
    second_margin = b[1]["distance"] - second["distance"]
    if first_mask == second_mask:
        # Repeating an identical mask cannot rescue an uncertain match.
        if first["distance"] > index.max_distance or first_margin < index.min_margin:
            return None
    elif (min(first["distance"], second["distance"]) > .205 or
          max(first["distance"], second["distance"]) > .238 or
          max(first_margin, second_margin) < .016 or min(first_margin, second_margin) <= 0 or
          (first_margin + second_margin) / 2 < .008):
        return None
    support = min(((first, first_margin), (second, second_margin)),
                  key=lambda row: (row[0]["distance"], -row[1]))
    return {**support[0], "margin": round(support[1], 5)}


def verify_window(window, index):
    """Require one complete reading backed by the same two distinct cuts."""
    middle = window["masks"]
    if not 2 <= len(middle) <= 7 or window["support"] < 4:
        return None
    cuts = (middle, *window["variants"])
    if len(cuts) != 3 or any(len(cut) != len(middle) for cut in cuts):
        return None
    cache = {}
    paths = {(a, b): [] for a, b in combinations(range(3), 2)}
    for alternatives in zip(*cuts, strict=True):
        keys = [_mask_key(mask) for mask in alternatives]
        rankings = []
        for key, mask in zip(keys, alternatives, strict=True):
            if key not in cache:
                cache[key] = index.rank(mask, limit=2)
            rankings.append(cache[key])
        for a, b in tuple(paths):
            if reading := _pair_reading(rankings[a], rankings[b], keys[a], keys[b], index):
                paths[a, b].append(reading)
            else:
                del paths[a, b]
        if not paths:
            return None
    confirmed = [glyphs for (a, b), glyphs in paths.items()
                 if any(_mask_key(first) != _mask_key(second) for first, second in zip(cuts[a], cuts[b], strict=True))]
    if not confirmed or len({tuple(row["text"] for row in glyphs) for glyphs in confirmed}) != 1:
        return None
    return min(confirmed, key=lambda glyphs: (max(row["distance"] for row in glyphs),
                                             sum(row["distance"] for row in glyphs)))
