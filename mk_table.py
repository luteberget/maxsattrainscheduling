# Convert train dispatching JSON results format into Latex table.

import json

def tr_name(x):
        name = x \
        .replace("origA",    "${\cal I}^{AO}_{") \
        .replace("origB",    "${\cal I}^{BO}_{") \
        .replace("trackA",   "${\cal I}^{AT}_{") \
        .replace("trackB",   "${\cal I}^{BT}_{") \
        .replace("stationA", "${\cal I}^{AS}_{") \
        .replace("stationB", "${\cal I}^{BS}_{") + "}$"
        return name

comparison = {}

for filename in [
        "results/2023-05-23-continuous.json",
        "results/2023-05-23-infsteps180.json",
        "results/2023-05-23-finsteps123.json",
        ]:

    print(f"# {filename}")
    with open(filename,"r") as f:
        data = json.load(f)

    for instance in data:

        if not instance["name"] in comparison:
            comparison[instance["name"]] = []


        worsttime = max([s["sol_time"] if "sol_time" in s else 9999 for s in instance["solves"]])

        bigm = next((s for s in instance["solves"] if s["solver_name"] == "BigMLazy"))
        satddd = next((s for s in instance["solves"] if s["solver_name"] == "MaxSatDdd"))
        mipddd = next((s for s in instance["solves"] if s["solver_name"] == "MipDdd"))

        # The "worst instances" comparison in done on the finsteps123 objective.
        if "finsteps123" in filename:
            comparison[instance["name"]].append((filename,"bigm",bigm["sol_time"] if "sol_time" in bigm else None, f"{bigm['sol_time']:.0f}" if "sol_time" in bigm else "\\timeout"))
            comparison[instance["name"]].append((filename,"satddd", satddd["sol_time"] if "sol_time" in satddd else None, f"{satddd['sol_time']:.0f}" if "sol_time" in satddd else "\\timeout"))
            comparison[instance["name"]].append((filename,"mipddd", mipddd["sol_time"] if "sol_time" in mipddd else None, f"{mipddd['sol_time']:.0f}" if "sol_time" in mipddd else "\\timeout"))
        #comparison.append((instance["name"],cmpx))

        #if worsttime < 100.0:
        #    continue
        cols = [
                # Instance name
                tr_name(instance["name"]),

                # BIGM iterations, conflicts, solve time
                str(bigm["iteration"]) if "iteration" in bigm else "-",
                str(bigm["added_conflict_pairs"]) if "added_conflict_pairs" in bigm else "-",
                f"{bigm['sol_time']:.0f}" if "sol_time" in bigm else "\\timeout",


                # MIP DDD solve time
                f"{mipddd['sol_time']:.0f}" if "sol_time" in mipddd else "\\timeout",

                # Iterations, total intervals, avg. intervals (per event), solve time
                str(satddd["iterations"]) if "iterations" in satddd else "-",
                str(satddd["num_time_points"]) if "num_time_points" in satddd else "-",
                f"{satddd['avg_time_points']:.1f}" if "avg_time_points" in satddd else "-",
                f"{satddd['sol_time']:.0f}" if "sol_time" in satddd else "\\timeout",

                # Ratio of solve time between bigm and satddd
                f"{(bigm['sol_time']/satddd['sol_time']):.1f}x" if "sol_time" in bigm and "sol_time" in satddd else "-",
                ]

        print(" & ".join(cols) + " \\\\")

instances = [(k,v) for k,v in comparison.items()]
instances.sort(key= lambda x: sum((-(t or 100000)  for _,_,t,_ in x[1] )))
instances = instances[:10]

print("#  objective comparison")
for (n,xs) in instances:
    #print("% " + "  ;  ".join([f"{f},{a}" for f,a,_,_ in xs]))
    cols = [ tr_name(n) ]
    for f,a,_,t in xs:
        cols.append(t)
    print(" & ".join(cols) + " \\\\")

