from collections import defaultdict
import sys

filename = sys.argv[1]
print("loading ", filename)

results = defaultdict(dict)

reading = False
with open(filename) as f:
    for line in f:
        if "origA1" in line:
            reading = True

        if not reading:
            continue

        x = line.strip().split()
        if len(x) == 0:
            continue

        assert len(x) == 5

        instance, obj_fun, solver, obj_val, time = x
        if not "solved" in results[solver]:
            results[solver]["solved"] = 0
        if not "time" in results[solver]:
            results[solver]["time"] = 0

        assert (obj_val=="9999") == (time == "9999")

        if time != "9999":
            results[solver]["solved"] += 1
            results[solver]["time"] += int(time)

for s,vals in results.items():
    print(s, vals["solved"], float(vals["time"])/float(vals["solved"]))
