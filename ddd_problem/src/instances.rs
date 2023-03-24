use crate::{problem::{NamedProblem, self}, parser};


pub fn xml_instances(mut x: impl FnMut(String, NamedProblem)) {
    let a_instances = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10];

    // let a_instances = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10];
    let b_instances = [11, 12, 13, 14, 15, 16, 17, 18, 19, 20];
    #[allow(unused)]
    let c_instances = [21, 22, 23, 24];

    for instance_id in a_instances
        .into_iter()
        .chain(b_instances)
        .chain(c_instances)
    {
        let filename = format!("instances/Instance{}.xml", instance_id);
        println!("Reading {}", filename);
        #[allow(unused)]
        let problem = parser::read_xml_file(
            &filename,
            problem::DelayMeasurementType::FinalStationArrival,
        );
        x(format!("xml {}", instance_id), problem);
    }
}

pub fn txt_instances(path_prefix :&str, mut x: impl FnMut(String, NamedProblem)) {
    // let a_instances = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10];
    // let b_instances = [11, 12, 13, 14, 15, 16, 17, 18, 19, 20];
    // #[allow(unused)]
    // let c_instances = [21, 22, 23, 24];

    for (dir, shortname) in [
        // ("instances_original", "orig"),
        // ("instances_addtracktime", "track"),
        ("instances_addstationtime", "station"),
    ] {
        let instances = ["A" /* ,"B"*/]
            .iter()
            .flat_map(move |n| (12..=12).map(move |i| (n, i)));

        // let instances = instances.skip(16).take(1);

        for (infrastructure, number) in instances {
            let _p = hprof::enter("read");
            let filename = format!("{}{}/Instance{}{}.txt", path_prefix, dir, infrastructure, number);
            println!("Reading {}", filename);
            #[allow(unused)]
            let (problem, _) = parser::read_txt_file(
                &filename,
                problem::DelayMeasurementType::FinalStationArrival,
                false,
                None,
                |_| {},
            );
            drop(_p);
            x(
                format!("{}{}{}", shortname, infrastructure, number),
                problem,
            );
        }
    }
}

pub fn verify_instances(mut x: impl FnMut(String, NamedProblem, Vec<Vec<i32>>) -> Vec<Vec<i32>>) {
    let a_instances = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10];
    let b_instances = [11, 12, 13, 14, 15, 16, 17, 18, 19, 20];
    #[allow(unused)]
    let c_instances = [21, 22, 23, 24];
    let _instances = [20];

    for solvertype in ["BigMComplete", "BigMLazyCon"] {
        for instance_id in a_instances
            .iter()
            .chain(b_instances.iter())
            .chain(c_instances.iter())
        {
            let filename = format!("InstanceResults/{}Sol{}.txt", solvertype, instance_id);
            println!("Reading {}", filename);
            #[allow(unused)]
            let (problem, solution) = parser::read_txt_file(
                &filename,
                problem::DelayMeasurementType::FinalStationArrival,
                true,
                None,
                |_| {},
            );
            let _new_solution = x(
                format!("{} {}", solvertype, instance_id),
                problem,
                solution.unwrap(),
            );

            // let mut f = std::fs::File::create(&format!("{}.bl.txt", filename)).unwrap();
            // use std::io::Write;
            // parser::read_txt_file(
            //     &filename,
            //     problem::DelayMeasurementType::FinalStationArrival,
            //     true,
            //     Some(new_solution),
            //     |l| {
            //         writeln!(f, "{}", l).unwrap();
            //     },
            // );
        }
    }
}
