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

worst_instances = [
    "stationA1",
    "stationA2",
    "stationA8",
    "stationA11",
    "stationA12",
    "trackA1",
    "trackA2",
    "trackA8",
    "trackA11",
    "trackA12"]

worst_instance_lines = []


alg_time = {"bigm": {"total_time": 0.0, "alg_time": 0.0},
            "satddd": {"total_time": 0.0, "alg_time": 0.0},
            "mipddd": {"total_time": 0.0, "alg_time": 0.0}}

for filename in [
        "results/2024-04-10-cont.json",
        "results/2024-04-10-infsteps180.json",
        "results/2024-04-10-finsteps123.json",
        ]:

    print(f"# {filename}")
    with open(filename,"r") as f:
        data = json.load(f)

    prev_instance_class = None
    for instance in data:

        if not instance["name"] in comparison:
            comparison[instance["name"]] = []



        worsttime = max([s["sol_time"] if "sol_time" in s else 9999 for s in instance["solves"]])

        bigm = next((s for s in instance["solves"] if s["solver_name"] == "BigMLazy"))
        satddd = next((s for s in instance["solves"] if s["solver_name"] == "MaxSatDdd" or s["solver_name"] == "MaxSatDddLadderRC2"))
        mipddd = next((s for s in instance["solves"] if s["solver_name"] == "MipDdd"))

        # The "worst instances" comparison in done on the finsteps123 objective.
        if "finsteps123" in filename:
            comparison[instance["name"]].append((filename,"bigm",bigm["sol_time"] if "sol_time" in bigm else None, f"{bigm['sol_time']:.0f}" if "sol_time" in bigm else "\\timeout"))
            comparison[instance["name"]].append((filename,"satddd", satddd["sol_time"] if "sol_time" in satddd else None, f"{satddd['sol_time']:.0f}" if "sol_time" in satddd else "\\timeout"))
            comparison[instance["name"]].append((filename,"mipddd", mipddd["sol_time"] if "sol_time" in mipddd else None, f"{mipddd['sol_time']:.0f}" if "sol_time" in mipddd else "\\timeout"))
        #comparison.append((instance["name"],cmpx))

        for solver_name,solver in [("bigm",bigm), ("satddd",satddd), ("mipddd", mipddd)]:
            if "total_time" in solver:
                alg_time[solver_name]["total_time"] += solver["total_time"]
            if "algorithm_time" in solver:
                alg_time[solver_name]["alg_time"] += solver["algorithm_time"]

        bigm_gap = 100.0*(bigm["ub"] - bigm["lb"])/float(bigm["ub"])
        sat_gap = 100.0*(satddd["ub"] - satddd["lb"])/float(satddd["ub"])
        mipddd_gap = 100.0*(mipddd["ub"] - mipddd["lb"])/float(mipddd["ub"])

        #if worsttime < 100.0:
        #    continue

        instance_class = instance["name"][:1]
        if prev_instance_class is not None and instance_class != prev_instance_class:
            print("\\midrule")
        prev_instance_class = instance_class

        cols = [
                # Instance name
                tr_name(instance["name"]),
                ]

                # BIGM iterations, conflicts, solve time
        bigm_to = "sol_time" not in bigm

        def gapcols(g):
            cols.append("-")
            cols.append("-")
            cols.append(f"\\multicolumn{{1}}{{c|}}{{\\scriptsize [{g:.0f}\\%]}}")

        if bigm_to:
            gapcols(bigm_gap)
        else:
                cols += [
                     str(bigm["iteration"]) if "iteration" in bigm else "-",
                    str(bigm["added_conflict_pairs"]) if "added_conflict_pairs" in bigm else "-",
                f"{bigm['sol_time']:.0f}" if "sol_time" in bigm else "\\timeout"]

        sat_to = "sol_time" not in satddd
        if sat_to:
            gapcols(sat_gap)
        else:
            cols += [ # Iterations, total intervals, avg. intervals (per event), solve time
                    str(satddd["iterations"]) if "iterations" in satddd else "-",
                    str(satddd["num_time_points"]) if "num_time_points" in satddd else "-",
                    f"{satddd['sol_time']:.0f}" if "sol_time" in satddd else "\\timeout",
                    ]

                # MIP DDD solve time
        mipddd_to = "sol_time" not in mipddd
        if mipddd_to:
            gapcols(mipddd_gap)
        else:
            cols += [f"{mipddd['iteration']:.0f}" if "iteration" in mipddd else "-",
                    f"{mipddd['intervals']:.0f}" if "intervals" in mipddd else "-",
                    f"{mipddd['sol_time']:.0f}" if "sol_time" in mipddd else "\\timeout"]


                # Ratio of solve time between bigm and satddd
        cols += [f"{(bigm['sol_time']/satddd['sol_time']):.1f}x" if "sol_time" in bigm and "sol_time" in satddd else "-",
                f"{(mipddd['sol_time']/satddd['sol_time']):.1f}x" if "sol_time" in mipddd and "sol_time" in satddd else "-",
                ]

        print(" & ".join(cols) + " \\\\")

        if instance["name"] in worst_instances and "finsteps" in filename:
            ws = [tr_name(instance["name"])]
            for slv in [bigm, mipddd, satddd]:
                ws.append( f"{slv['sol_time']:.0f}" if 'sol_time' in slv else "\\timeout ")

            for slv in [bigm,mipddd]:
                if 'sol_time' not in slv:
                    ws.append("-")
                else:
                    ws.append(f"{(slv['sol_time']/satddd['sol_time']):.1f}x")
            worst_instance_lines.append(" & ".join(ws) + " \\\\ ")

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


print("# total time")
print(alg_time)

print("# worst instances")
for l in worst_instance_lines:
    print(l)
