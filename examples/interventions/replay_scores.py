"""Verify experimental counts from the recorded logits on a CPU."""
import json
from pathlib import Path
from statistics import median


def winner(logits):
    a, b = logits
    if logits[a] == logits[b]:
        return None
    return max(logits, key=logits.get)


def counts(data):
    results = {}
    baselines = {}
    for run, record in data.items():
        baseline = {}
        for split, mesh, order, case, logits in record["baseline"]:
            baseline[mesh, order, case] = logits
        baselines[run] = baseline
        main = {k: v for k, v in baseline.items() if not k[-1].endswith("_color")}
        answer = {"faceted": "true", "smooth": "false"}
        results[run + "_baseline"] = [
            sum(winner(v) == answer.get(k[-1], k[-1]) for k, v in main.items()),
            len(main),
        ]
    initial = baselines["coordinate_initial"]
    results["coordinate_initial_outward"] = [
        sum(winner(v) == "outward" for k, v in initial.items() if k[-1] == "outward"),
        sum(k[-1] == "outward" for k in initial),
    ]

    explicit = data["coordinate_explicit"]
    baseline = baselines["coordinate_explicit"]
    candidates, selections, control_shifts = [], [], {}
    for row in explicit["heldout"]:
        original = baseline[row["mesh"], row["order"], row["receiver"]]
        assert winner(original) == row["receiver"]
        donor, receiver = row["donor"], row["receiver"]
        before = original[donor] - original[receiver]
        effect = row["effects"]["candidate"]
        assert effect[:2] == [explicit["candidate"]["site"], explicit["candidate"]["layer"]]
        candidates.append(winner(effect[-1]) == donor)
        selections.append(winner(row["selected"][-1]) == donor)
        group = control_shifts.setdefault((row["order"], receiver), {})
        for method, observation in row["effects"].items():
            logits = observation[-1]
            group.setdefault(method, []).append(logits[donor] - logits[receiver] - before)
    results["full_vector_flips"] = [sum(candidates), len(candidates)]
    results["selected_coordinate_flips"] = [sum(selections), len(selections)]
    results["selected_coordinates"] = len(set(explicit["selected_coordinates"]))
    results["candidate_beats_controls_by_group"] = sum(
        median(group["candidate"]) > max(abs(median(group[name])) for name in ("self", "color", "other_site"))
        for group in control_shifts.values()
    )
    return results


if __name__ == "__main__":
    data = json.loads(Path(__file__).with_name("recorded_logits.json").read_text())
    results = counts(data)
    expected = {
        "shading_baseline": [11, 22],
        "coordinate_initial_baseline": [21, 44],
        "coordinate_explicit_baseline": [44, 44],
        "coordinate_initial_outward": [4, 22],
        "full_vector_flips": [20, 20],
        "selected_coordinate_flips": [0, 20],
        "selected_coordinates": 128,
        "candidate_beats_controls_by_group": 4,
    }
    if results != expected:
        raise SystemExit(f"Recorded scores differ from the published results: {results}")
    print(json.dumps(results, indent=2))
