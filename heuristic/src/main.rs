use ddd_problem::instances::txt_instances;

pub fn main() {
    txt_instances("../", |name, problem| {
        println!("Instance {}", name);
        println!("{:?}", problem.problem);
        println!("{:?}", problem.resource_names);
        panic!();
    })
}
