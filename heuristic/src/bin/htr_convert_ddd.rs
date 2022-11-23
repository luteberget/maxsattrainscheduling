use ddd_problem::instances::txt_instances;

pub fn main() {
    txt_instances("../", |name, problem| {
        let _p = hprof::enter("convert");
        println!("Instance {}", name);
        // println!("{:?}", problem.problem);
        // println!("{:?}", problem.resource_names);

        // Convert ddd_problem to routing heuristic problem

        // assume that all resource ids >0 are exclusive, and that's all conflicts.
        let routing_problem = heuristic::problem::convert_ddd_problem(&problem);

        // println!("Routing problem:\n{:?}", routing_problem);
        // println!(" name {}", name);

        drop(_p);
        let _p =         hprof::enter("save");

        serde_json::to_writer(
            &std::fs::File::create(&format!("{}_rh.json", name)).unwrap(),
            &routing_problem,
        )
        .unwrap();
    });

    hprof::profiler().print_timing();
}
