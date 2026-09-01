import json
from pathlib import Path


def load_history(path):
    if path is None:
        return []

    history_path = Path(path)
    data = json.loads(history_path.read_text(encoding="utf-8"))

    if isinstance(data, dict) and "reports" in data:
        return data["reports"]
    if isinstance(data, dict):
        return [data]
    if isinstance(data, list):
        return data

    raise ValueError(f"Unsupported history format in {history_path}")


def _level_for_count(matched_count, threshold):
    if matched_count >= threshold:
        return "elevated"
    if matched_count >= max(threshold - 2, 1):
        return "borderline"
    if matched_count > 0:
        return "limited"
    return "none"


def _snapshot_from_report(report, index):
    aggregate = report.get("aggregate_evidence") or {}
    return {
        "week_label": report.get("week_label", f"previous_report_{index}"),
        "source": "provided_report_history",
        "screen_positive": aggregate.get("screen_positive"),
        "evidence_level": aggregate.get("evidence_level"),
        "matched_biomarker_count": aggregate.get("matched_biomarker_count"),
        "available_biomarker_count": aggregate.get("available_biomarker_count"),
        "matched_biomarkers": aggregate.get("matched_biomarkers", []),
    }


def simulate_previous_weeks(current_aggregate, weeks):
    if weeks <= 0:
        return []

    current_count = int(current_aggregate.get("matched_biomarker_count") or 0)
    available_count = int(current_aggregate.get("available_biomarker_count") or 0)
    threshold = int(current_aggregate.get("aggregate_match_threshold") or 6)
    current_markers = current_aggregate.get("matched_biomarkers") or []
    snapshots = []

    for offset in range(weeks, 0, -1):
        simulated_count = max(0, current_count - ((offset + 1) // 2))
        simulated_count = min(simulated_count, available_count)
        snapshots.append({
            "week_label": f"simulated_week_minus_{offset}",
            "source": "simulated_previous_week",
            "screen_positive": simulated_count >= threshold,
            "evidence_level": _level_for_count(simulated_count, threshold),
            "matched_biomarker_count": simulated_count,
            "available_biomarker_count": available_count,
            "matched_biomarkers": current_markers[:simulated_count],
        })

    return snapshots


def _summarise_trend(snapshots, current_aggregate):
    current_count = int(current_aggregate.get("matched_biomarker_count") or 0)
    if not snapshots:
        return "No previous-week context was supplied or simulated."

    previous_counts = [
        snapshot.get("matched_biomarker_count")
        for snapshot in snapshots
        if snapshot.get("matched_biomarker_count") is not None
    ]
    if not previous_counts:
        return "Previous-week context exists, but it does not contain aggregate matched-biomarker counts."

    earliest = previous_counts[0]
    latest_previous = previous_counts[-1]

    if current_count > latest_previous:
        direction = "increased compared with the most recent previous week"
    elif current_count < latest_previous:
        direction = "decreased compared with the most recent previous week"
    else:
        direction = "remained stable compared with the most recent previous week"

    if current_count > earliest:
        longer_trend = "higher than the earliest available previous week"
    elif current_count < earliest:
        longer_trend = "lower than the earliest available previous week"
    else:
        longer_trend = "similar to the earliest available previous week"

    return (
        f"The current matched-biomarker count is {current_count}; it has {direction} "
        f"and is {longer_trend}. This trend context is supportive only and must not be treated as diagnosis."
    )


def build_recursive_context(
    current_aggregate,
    history=None,
    simulated_weeks=0
):
    history = history or []
    snapshots = [
        _snapshot_from_report(report, index + 1)
        for index, report in enumerate(history)
    ]
    simulated_snapshots = simulate_previous_weeks(current_aggregate, simulated_weeks)
    snapshots.extend(simulated_snapshots)

    source_types = sorted({snapshot["source"] for snapshot in snapshots})
    simulated = any(
        snapshot["source"] == "simulated_previous_week"
        for snapshot in snapshots
    )

    return {
        "enabled": bool(snapshots),
        "source_types": source_types,
        "simulated": simulated,
        "warning": (
            "Previous-week entries marked simulated_previous_week are synthetic prototype context, not real patient history."
            if simulated
            else None
        ),
        "snapshots": snapshots,
        "summary": _summarise_trend(snapshots, current_aggregate),
        "current_week": {
            "screen_positive": current_aggregate.get("screen_positive"),
            "evidence_level": current_aggregate.get("evidence_level"),
            "matched_biomarker_count": current_aggregate.get("matched_biomarker_count"),
            "available_biomarker_count": current_aggregate.get("available_biomarker_count"),
            "matched_biomarkers": current_aggregate.get("matched_biomarkers", []),
        },
    }
