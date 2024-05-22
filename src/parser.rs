use crate::problem::{self, DelayMeasurementType, NamedProblem, Problem, Visit};
use log::debug;
use std::{collections::HashMap, mem::take};


pub fn read_txt_file(
    instance_fn: &str,
    measurement: DelayMeasurementType,
    with_solution: bool,
    output_solution: Option<Vec<Vec<i32>>>,
    mut write_solution: impl FnMut(&str),
) -> (NamedProblem, Option<Vec<Vec<i32>>>) {
    let instance_txt = std::fs::read_to_string(instance_fn).unwrap();
    let mut train_names = Vec::new();
    let mut resource_names = Vec::new();
    let mut resources = HashMap::new();
    let mut solution = Vec::new();

    let any_station_resource = 0;
    resource_names.push("Any station".to_string());

    let mut current_train: Option<Vec<Visit>> = None;
    let mut next_earliest: Option<i32> = None;
    let mut problem = Problem {
        name:instance_fn.to_string(),
        conflicts: Vec::new(),
        trains: Vec::new(),
    };

    let mut lines = instance_txt.lines().peekable();
    while let Some(line) = lines.next() {
        if !line.trim().is_empty() {
            let mut fields = line.split_ascii_whitespace();
            fn get_pair(fields: &mut std::str::SplitAsciiWhitespace) -> (String, i32) {
                // let name = fields.next().unwrap().to_string();
                // let value = fields.next().unwrap().parse::<i64>().unwrap();
                let pair = fields.next().unwrap();
                let mut split = pair.split('=');
                let name = split.next().unwrap().to_string();
                let value = split.next().unwrap().parse::<i32>().unwrap();
                (name, value)
            }

            match current_train.as_mut() {
                None => {
                    // Expect train header
                    write_solution(line);

                    let (train_id_field, train_id) = get_pair(&mut fields);
                    assert!(train_id_field == "TrainId");
                    let (delay_field, _delay) = get_pair(&mut fields);
                    assert!(delay_field == "Delay");
                    let (free_run_field, _free_run) = get_pair(&mut fields);
                    assert!(free_run_field == "FreeRun");

                    train_names.push(format!("Train{}", train_id));
                    if with_solution {
                        solution.push(Vec::new());
                    }
                    current_train = Some(Vec::new());
                    next_earliest = None;
                }
                Some(train) => {
                    let track_id = fields.next().unwrap();
                    let train_id = fields.next().unwrap();
                    assert!(train_names.last().map(String::as_str) == Some(train_id));

                    let (aimeddep_field, aimeddep) = get_pair(&mut fields);
                    assert!(aimeddep_field == "AimedDepartureTime");
                    let (waittime_field, wait_time) = get_pair(&mut fields);
                    assert!(waittime_field == "WaitTime");
                    let (basetime_field, base_time) = get_pair(&mut fields);
                    assert!(basetime_field == "BaseTime");
                    let (runtime_field, run_time) = get_pair(&mut fields);
                    assert!(runtime_field == "RunTime");

                    let sol_time = with_solution.then(|| {
                        let (soltime_field, sol_time) = get_pair(&mut fields);
                        assert!(soltime_field == "OptSolTime");
                        sol_time
                    });

                    if let Some(s) = output_solution.as_ref() {
                        let current_train_idx = problem.trains.len();
                        let current_visit_idx = train.len() + 1;
                        let t = s[current_train_idx][current_visit_idx];
                        write_solution(&format!("{} {} AimedDepartureTime={} WaitTime={} BaseTime={} RunTime={}\tOptSolTime={}", 
                            track_id, train_id, aimeddep, wait_time, base_time, run_time, t,
                        ));
                    }

                    let is_last_track = match lines.peek() {
                        None => true,
                        Some(l) => l.trim().is_empty(),
                    };

                    let resource_id = *resources.entry(track_id.to_string()).or_insert_with(|| {
                        let idx = resource_names.len();
                        resource_names.push(track_id.to_string());
                        idx
                    });

                    let (aimed_in, aimed_out) = match measurement {
                        DelayMeasurementType::AllStationArrivals => {
                            (Some(aimeddep - wait_time), None)
                        }
                        DelayMeasurementType::AllStationDepartures => (None, Some(aimeddep)),
                        DelayMeasurementType::FinalStationArrival => {
                            (None, is_last_track.then(|| aimeddep))
                        }
                        DelayMeasurementType::EverywhereEarliest => todo!(),
                    };

                    let earliest_in = next_earliest.unwrap_or(base_time as i32 - wait_time as i32);
                    let earliest_out = base_time as i32;
                    next_earliest = Some(earliest_out + run_time);

                    train.push(Visit {
                        earliest: earliest_in,
                        aimed: aimed_in,
                        resource_id: any_station_resource,
                        travel_time: wait_time as i32,
                    });

                    train.push(Visit {
                        earliest: earliest_out,
                        aimed: aimed_out,
                        resource_id,
                        travel_time: run_time as i32,
                    });

                    if let Some(sol_time) = sol_time {
                        if solution.last().unwrap().is_empty() {
                            solution.last_mut().unwrap().push(sol_time - wait_time);
                        }
                        solution.last_mut().unwrap().push(sol_time);
                        solution.last_mut().unwrap().push(sol_time + run_time);
                    }
                }
            }
        } else {
            write_solution(line);
        }

        if line.trim().is_empty() || lines.peek().is_none() {
            if let Some(mut visits) = current_train.take() {
                // add the final station visit
                let prev_visit = *visits.last().unwrap();
                visits.push(Visit {
                    earliest: prev_visit.earliest + prev_visit.travel_time,
                    aimed: None,
                    resource_id: any_station_resource,
                    travel_time: 0,
                });

                if with_solution {
                    let prev_time = *solution.last().unwrap().last().unwrap();
                    solution.last_mut().unwrap().push(prev_time);
                }

                problem.trains.push(crate::problem::Train { visits })
            }
        }
    }

    // All tracks are exclusive
    for (_, id) in resources.iter() {
        problem.conflicts.push((*id, *id));
    }

    (
        NamedProblem {
            problem,
            train_names,
            resource_names,
        },
        with_solution.then(|| solution),
    )
}

#[derive(Debug)]
pub struct TxtParseError;

#[derive(Debug)]
pub struct TxtTrain {
    pub name: String,
    pub delay: i32,
    pub final_delay: i32,
    pub visits: Vec<TxtVisit>,
}

#[derive(Debug)]
pub struct TxtVisit {
    // T24 TrainID=166 AimedDepartureTime=3118 WaitTime=0 Delay=0 RunTime=143 FinalTimeScheduled=3118
    pub track_name: String,
    pub aimed_departure_time: i32,
    pub wait_time: i32,
    pub delay: i32,
    pub run_time: i32,
    pub final_time_scheduled: i32,
}

pub fn parse_solution_txt(txt: &str) -> Result<Vec<TxtTrain>, TxtParseError> {
    let mut trains = Vec::new();

    let mut header: Option<(String, i32, i32)> = None;
    let mut visits = Vec::new();

    fn parse_kv(s: &str) -> Result<(&str, &str), TxtParseError> {
        let mut split = s.split('=');
        let key = split.next().ok_or(TxtParseError)?;
        let value = split.next().ok_or(TxtParseError)?;
        Ok((key, value))
    }

    let mut lines = txt.lines();
    let _headerline = lines.next().unwrap();
    let _empty = lines.next().unwrap();

    let mut lines = lines.peekable();
    while let Some(line) = lines.next() {
        let mut fields = line.split_ascii_whitespace();

        if header.is_none() {
            // Parse header
            // Train=166 Init Delay=0 FinalDelay=49

            let (train_key, train) = parse_kv(fields.next().ok_or(TxtParseError)?)?;
            assert!(train_key == "Train");

            fields.next().ok_or(TxtParseError)?;
            let (delay_key, delay) = parse_kv(fields.next().ok_or(TxtParseError)?)?;
            assert!(delay_key == "Delay");

            let (final_delay_key, final_delay) = parse_kv(fields.next().ok_or(TxtParseError)?)?;
            assert!(final_delay_key == "FinalDelay");

            header = Some((
                train.to_string(),
                delay.parse().map_err(|_| TxtParseError)?,
                final_delay.parse().map_err(|_| TxtParseError)?,
            ));
        } else {
            // Parse visit
            // T24 TrainID=166 AimedDepartureTime=3118 WaitTime=0 Delay=0 RunTime=143 FinalTimeScheduled=3118

            let track_name = fields.next().unwrap();

            let (train_id_key, train_id) = parse_kv(fields.next().ok_or(TxtParseError)?)?;
            assert!(train_id_key == "TrainID");
            assert!(train_id == header.as_ref().unwrap().0);

            let (aimed_dep_key, aimed_dep) = parse_kv(fields.next().ok_or(TxtParseError)?)?;
            assert!(aimed_dep_key == "AimedDepartureTime");

            let (wait_time_key, wait_time) = parse_kv(fields.next().ok_or(TxtParseError)?)?;
            assert!(wait_time_key == "WaitTime");

            let (delay_key, delay) = parse_kv(fields.next().ok_or(TxtParseError)?)?;
            assert!(delay_key == "Delay");

            assert!(delay == "0", "delay field is not in use?");

            let (run_time_key, run_time) = parse_kv(fields.next().ok_or(TxtParseError)?)?;
            assert!(run_time_key == "RunTime");

            let (final_time_key, final_time) = parse_kv(fields.next().ok_or(TxtParseError)?)?;
            assert!(final_time_key == "FinalTimeScheduled");

            visits.push(TxtVisit {
                track_name: track_name.to_string(),
                aimed_departure_time: aimed_dep.parse().map_err(|_| TxtParseError)?,
                wait_time: wait_time.parse().map_err(|_| TxtParseError)?,
                delay: delay.parse().map_err(|_| TxtParseError)?,
                run_time: run_time.parse().map_err(|_| TxtParseError)?,
                final_time_scheduled: final_time.parse().map_err(|_| TxtParseError)?,
            });
        }

        if lines.peek() == Some(&"") || lines.peek() == None {
            let (name, delay, final_delay) = header.take().unwrap();
            trains.push(TxtTrain {
                name,
                delay,
                final_delay,
                visits: take(&mut visits),
            });
            lines.next();
        }
    }

    Ok(trains)
}
