import json

for filename in [
        "2022-06-28-cont.json",
        "2022-06-28-finsteps123.json",
        "2022-06-28-infsteps180.json"
        ]:

    print(f"# {filename}")
    with open(filename,"r") as f:
        data = json.load(f)

    for instance in data:
        name = instance["name"] \
        .replace("origA",    "${\cal I}^{AO}_{") \
        .replace("origB",    "${\cal I}^{BO}_{") \
        .replace("trackA",   "${\cal I}^{AT}_{") \
        .replace("trackB",   "${\cal I}^{BT}_{") \
        .replace("stationA", "${\cal I}^{AS}_{") \
        .replace("stationB", "${\cal I}^{BS}_{") + "}$"

        worsttime = max([s["sol_time"] if "sol_time" in s else 9999 for s in instance["solves"]])
        if worsttime < 100.0:
            continue

        bigm = next((s for s in instance["solves"] if s["solver_name"] == "BigMLazy"))
        ddd = next((s for s in instance["solves"] if s["solver_name"] == "MaxSatDdd"))

        cols = [
                name,
                str(bigm["iteration"]) if "iteration" in bigm else "-",
                str(bigm["added_conflict_pairs"]) if "added_conflict_pairs" in bigm else "-",
                f"{bigm['sol_time']:.0f}" if "sol_time" in bigm else "\\timeout",
                # & Iter. & Interv. & Avg. Interv. & Time
                str(ddd["iterations"]) if "iterations" in ddd else "-",
                str(ddd["num_time_points"]) if "num_time_points" in ddd else "-",
                f"{ddd['avg_time_points']:.1f}" if "avg_time_points" in ddd else "-",
                f"{ddd['sol_time']:.0f}" if "sol_time" in ddd else "\\timeout",
                f"{(bigm['sol_time']/ddd['sol_time']):.1f}x" if "sol_time" in bigm and "sol_time" in ddd else "-",
                ]

        print(" & ".join(cols) + " \\\\")
