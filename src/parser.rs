use crate::problem::{DelayMeasurementType, NamedProblem, Problem, Visit};
use chrono::{Duration, NaiveDateTime};
use log::debug;
use std::{collections::HashMap, mem::take};

enum Pos<'a> {
    OnTrack(&'a str, NaiveDateTime, i32),
    InStation(&'a str, NaiveDateTime, i32),
    NotStarted(i32),
}

pub fn read_txt_file(instance_fn: &str, measurement: DelayMeasurementType) -> NamedProblem {
    let instance_txt = std::fs::read_to_string(instance_fn).unwrap();
    let mut train_names = Vec::new();
    let mut resource_names = Vec::new();
    let mut resources = HashMap::new();

    let any_station_resource = 0;
    resource_names.push("Any station".to_string());

    let mut current_train: Option<Vec<Visit>> = None;
    let mut problem = Problem {
        conflicts: Vec::new(),
        trains: Vec::new(),
    };

    let mut lines = instance_txt.lines().peekable();
    while let Some(line) = lines.next() {
        if line.trim().is_empty() {
            if let Some(visits) = current_train.take() {
                problem.trains.push(crate::problem::Train { visits })
            }
            continue;
        }

        let mut fields = line.split_ascii_whitespace();
        fn get_pair(fields: &mut std::str::SplitAsciiWhitespace) -> (String, i32) {
            // let name = fields.next().unwrap().to_string();
            // let value = fields.next().unwrap().parse::<i64>().unwrap();
            let pair = fields.next().unwrap();
            let mut split = pair.split("=");
            let name = split.next().unwrap().to_string();
            let value = split.next().unwrap().parse::<i32>().unwrap();
            (name, value)
        }

        match current_train.as_mut() {
            None => {
                // Expect train header

                let (train_id_field, train_id) = get_pair(&mut fields);
                assert!(train_id_field == "TrainId");
                let (delay_field, delay) = get_pair(&mut fields);
                assert!(delay_field == "Delay");
                let (free_run_field, free_run) = get_pair(&mut fields);
                assert!(free_run_field == "FreeRun");

                train_names.push(format!("Train{}", train_id));
                current_train = Some(Vec::new());
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
                    DelayMeasurementType::AllStationArrivals => (None, Some(aimeddep + run_time)),
                    DelayMeasurementType::AllStationDepartures => (Some(aimeddep), None),
                    DelayMeasurementType::FinalStationArrival => {
                        // (None, is_last_track.then(|| aimeddep + run_time))
                        (is_last_track.then(|| aimeddep), None)
                    }
                    DelayMeasurementType::EverywhereEarliest => todo!(),
                };

                train.push(Visit {
                    aimed: aimed_in,
                    earliest: base_time as i32,
                    resource_id,
                    travel_time: run_time as i32,
                });

                train.push(Visit {
                    aimed: aimed_out,
                    earliest: (base_time + run_time) as i32,
                    travel_time: wait_time,
                    resource_id: any_station_resource,
                });
            }
        }
    }

    // All tracks are exclusive
    for (_, id) in resources.iter() {
        problem.conflicts.push((*id, *id));
    }

    if let Some(visits) = current_train.take() {
        problem.trains.push(crate::problem::Train { visits })
    }

    NamedProblem {
        problem,
        train_names,
        resource_names,
    }
}

pub fn read_xml_file(instance_fn: &str, measurement: DelayMeasurementType) -> NamedProblem {
    let date_format = "%Y-%m-%dT%H:%M:%S";
    let parse_date = |d| chrono::NaiveDateTime::parse_from_str(d, date_format).unwrap();
    let instance_xml = std::fs::read_to_string(instance_fn).unwrap();
    let doc = roxmltree::Document::parse(&instance_xml).unwrap();
    let network_elem = doc
        .root_element()
        .children()
        .find(|n| n.tag_name().name() == "Network")
        .unwrap();
    let stations_elem = network_elem
        .children()
        .find(|n| n.tag_name().name() == "Stations")
        .unwrap();
    let tracks_elem = network_elem
        .children()
        .find(|n| n.tag_name().name() == "Tracks")
        .unwrap();

    let connection_ids = get_track_id_map(tracks_elem);

    #[derive(PartialEq, Eq, PartialOrd, Ord, Hash, Clone, Copy, Debug)]
    enum ResourceType {
        Station,
        Track,
    }

    // Create resource ids for stations in order
    let mut resource_order: Vec<(ResourceType, &str)> = Vec::new();
    for station in stations_elem.children().filter(|c| c.is_element()) {
        let id = station.attribute("StationId").unwrap();
        resource_order.push((ResourceType::Station, id));
    }

    // Insert tracks between stations
    for track in tracks_elem.children().filter(|c| c.is_element()) {
        let t = track.attribute("TrackId").unwrap();
        let sa = track.attribute("StationA").unwrap();
        let sb = track.attribute("StationB").unwrap();
        let p = (0..resource_order.len() - 1).find(|i| {
            (resource_order[*i] == (ResourceType::Station, sa)
                && resource_order[*i + 1] == (ResourceType::Station, sb))
                || (resource_order[*i] == (ResourceType::Station, sb)
                    && resource_order[*i + 1] == (ResourceType::Station, sa))
                || (*i < resource_order.len() - 1
                    && resource_order[*i] == (ResourceType::Station, sa)
                    && resource_order[*i + 2] == (ResourceType::Station, sb))
                || (*i < resource_order.len() - 1
                    && resource_order[*i] == (ResourceType::Station, sb)
                    && resource_order[*i + 2] == (ResourceType::Station, sa))
        });

        let idx = p.unwrap();
        resource_order.insert(idx + 1, (ResourceType::Track, t));
    }

    let minimum_running_times = get_runningtimes_map(&doc);
    let _objective_map = get_objective_map(&doc);
    let (time_now, train_positions) = get_train_pos(&doc, date_format);

    let timetable_node = doc
        .root_element()
        .children()
        .find(|n| n.tag_name().name() == "TimeTable")
        .unwrap();
    let train_schedules_noed = timetable_node
        .children()
        .find(|n| n.tag_name().name() == "TrainSchedules")
        .unwrap();

    let mut problem_trains = Vec::new();
    for train_schedule in train_schedules_noed.children().filter(|c| c.is_element()) {
        assert!(train_schedule.tag_name().name() == "TrainSchedule");
        let id = train_schedule.attribute("TrainId").unwrap();
        let speed_class = train_schedule.attribute("SpeedClass").unwrap();

        let _type_ = train_schedule.attribute("Type").unwrap();
        let _origin_id = train_schedule.attribute("OriginId").unwrap();
        let _destination_id = train_schedule.attribute("DestinationId").unwrap();
        let _length = train_schedule
            .attribute("Length")
            .unwrap()
            .parse::<u32>()
            .unwrap();

        let stop_nodes = train_schedule
            .children()
            .filter(|c| c.is_element())
            .collect::<Vec<_>>();
        if stop_nodes.len() <= 1 {
            debug!(
                "train {} is irrelevant as it visits less than two stations in the network",
                id
            );
            continue;
        }

        let last_station_name = stop_nodes.last().unwrap().attribute("StationId").unwrap();

        if let Some(pos) = train_positions.get(id) {
            let mut visits = Vec::new();

            // <TrainInfo TrainId="15" Position="InStation" StationId="XX.S7" TrackId="2" TimeIn="2020-12-17T12:20:24" DelayInSeconds="296" />

            let (first_stop_idx, mut earliest_time_cursor) = match pos {
                Pos::OnTrack(track, enter_time, _delay) => stop_nodes
                    .iter()
                    .zip(stop_nodes.iter().skip(1))
                    .enumerate()
                    .find_map(|(stop_idx, (n1, n2))| {
                        let prev_station = n1.attribute("StationId").unwrap();
                        let next_station = n2.attribute("StationId").unwrap();
                        (connection_ids[&(prev_station, next_station)] == *track).then(|| {
                            let already_traveled = time_now - *enter_time;
                            let minimum_running_time = Duration::seconds(
                                minimum_running_times[speed_class][*track] as i64,
                            );
                            let remaining_travel_time = minimum_running_time - already_traveled;
                            assert!(
                                time_now + remaining_travel_time
                                    == *enter_time + minimum_running_time
                            );
                            visits.push((
                                (ResourceType::Track, *track),
                                *enter_time,
                                minimum_running_time,
                                None,
                            ));
                            (stop_idx + 1, *enter_time + minimum_running_time)
                        })
                    })
                    .unwrap(),
                Pos::InStation(sx, enter_time, _delay) => stop_nodes
                    .iter()
                    .enumerate()
                    .find_map(|(stop_idx, n)| {
                        (*sx == n.attribute("StationId").unwrap()).then(|| (stop_idx, *enter_time))
                    })
                    .unwrap(),
                Pos::NotStarted(delay) => {
                    let aimed_arrival =
                        parse_date(stop_nodes[0].attribute("AimedArrivalTime").unwrap());
                    let delayed_arrival = aimed_arrival + chrono::Duration::seconds(*delay as i64);
                    (0, delayed_arrival)
                }
            };

            if id == "17" {
                println!("train 17 with {} nodes", stop_nodes[first_stop_idx..].len());
            }

            for (stop, next_stop) in stop_nodes[first_stop_idx..].iter().zip(
                stop_nodes[first_stop_idx..]
                    .iter()
                    .skip(1)
                    .map(Some)
                    .chain(std::iter::once(None)),
            ) {
                let station = stop.attribute("StationId").unwrap();
                let aimed_arrival = parse_date(stop.attribute("AimedArrivalTime").unwrap());
                let aimed_departure = parse_date(stop.attribute("AimedDepartureTime").unwrap());

                // println!("Train {} statino {}  aimed_arr {} aimed_dep {}", id, station, aimed_arrival, aimed_departure);

                // Now we enter the station at current_time.
                visits.push((
                    (ResourceType::Station, station),
                    earliest_time_cursor,
                    Duration::zero(),
                    Some(aimed_arrival),
                ));

                // Now the earliest time to exit the station, is the max of (the current time + 0) and
                // the aimed departure time.
                // earliest_time_cursor += Duration::zero();

                earliest_time_cursor = earliest_time_cursor.max(aimed_departure);

                // TODO check this with Anna Livia
                //earliest_time_cursor = earliest_time_cursor.max(time_now);

                if let Some(next_stop) = next_stop {
                    let next_station = next_stop.attribute("StationId").unwrap();
                    let track = *connection_ids.get(&(station, next_station)).unwrap();
                    let travel_time =
                        Duration::seconds(minimum_running_times[speed_class][track] as i64);
                    visits.push((
                        (ResourceType::Track, track),
                        earliest_time_cursor,
                        travel_time,
                        Some(aimed_departure),
                    ));
                    earliest_time_cursor += travel_time;
                }
            }

            // println!("Train {} visits {:?}", id, visits);
            problem_trains.push((id, visits, last_station_name));
        } else {
            // println!("Ignoring train {} has left the network.", id);
        }
    }

    let mut train_names = Vec::new();
    let resource_names = resource_order
        .iter()
        .map(|(_type, name)| name.to_string())
        .collect::<Vec<_>>();

    let resource_ids = resource_order
        .iter()
        .enumerate()
        .map(|(i, x)| (*x, i))
        .collect::<HashMap<_, _>>();

    let mut problem = crate::problem::Problem {
        trains: Vec::new(),
        conflicts: Vec::new(),
    };
    for ((res_type, _name), i) in resource_ids.iter() {
        if matches!(res_type, ResourceType::Track) {
            problem.conflicts.push((*i, *i));
        }
    }
    for (name, visits, last_station_name) in problem_trains.iter() {
        let mut t = Vec::new();
        for ((res_type, name), earliest, travel, aimed) in visits.iter() {
            // println!("Looking up resource {:?}", r);
            let resource_id = resource_ids[&(*res_type, *name)];
            //     let i = resource_idx;
            //     resource_names.push(match r {
            //         Ok(station) => format!("Station {}", station),
            //         Err(track) => format!("Track {}", track),
            //     });
            //     if r.is_err() {
            //         // Tracks are exclusive.
            //         problem.conflicts.push((i,i));
            //     }
            //     resource_idx += 1;
            //     i
            // });

            let aimed = match measurement {
                DelayMeasurementType::AllStationArrivals => {
                    matches!(res_type, ResourceType::Station).then(|| *aimed)
                }
                DelayMeasurementType::AllStationDepartures => {
                    matches!(res_type, ResourceType::Track).then(|| *aimed)
                }
                DelayMeasurementType::FinalStationArrival => {
                    (matches!(res_type, ResourceType::Station) && name == last_station_name)
                        .then(|| *aimed)
                }
                DelayMeasurementType::EverywhereEarliest => Some(Some(*earliest)),
            };

            let aimed = aimed.flatten().map(|a| (a - time_now).num_seconds() as i32);

            t.push(Visit {
                resource_id,
                earliest: (*earliest - time_now).num_seconds() as i32,
                travel_time: travel.num_seconds() as i32,
                aimed,
            });
        }

        problem.trains.push(crate::problem::Train { visits: t });
        train_names.push(name.to_string());
    }

    NamedProblem {
        problem,
        train_names,
        resource_names,
    }
}

#[allow(unused)]
fn station_info(stations: roxmltree::Node) {
    for station in stations.children().filter(|c| c.is_element()) {
        let id = station.attribute("StationId").unwrap();
        let maintrack = station.attribute("MainTrackId").unwrap();
        // println!("station {} {}", id, maintrack);

        let internal_tracks = station
            .children()
            .find(|n| n.tag_name().name() == "InternalTracks")
            .unwrap();
        for internal_track in internal_tracks.children().filter(|c| c.is_element()) {
            assert!(internal_track.tag_name().name() == "InternalTrack");
            let track_id = internal_track.attribute("TrackId").unwrap();
            let has_platform = internal_track
                .attribute("HasPlatform")
                .unwrap()
                .parse::<bool>()
                .unwrap();
            let length = internal_track
                .attribute("Length")
                .map(|l| l.parse::<u32>().unwrap());

            println!("  - track {} {} {:?}", track_id, has_platform, length);
        }
    }
}

fn get_train_pos<'a>(
    doc: &'a roxmltree::Document,
    date_format: &'_ str,
) -> (NaiveDateTime, HashMap<&'a str, Pos<'a>>) {
    let mut train_positions: HashMap<&str, Pos> = HashMap::new();
    let snapshots = doc
        .root_element()
        .children()
        .find(|n| n.tag_name().name() == "Snapshots")
        .unwrap();
    let snapshots = snapshots
        .children()
        .filter(|c| c.is_element())
        .collect::<Vec<_>>();
    assert!(snapshots.len() == 1);
    let snapshot = snapshots[0];
    assert!(snapshot.tag_name().name() == "Snapshot");
    let snapshot_index = snapshot.attribute("Index").unwrap();
    assert!(snapshot_index == "0");
    let time_now =
        chrono::NaiveDateTime::parse_from_str(snapshot.attribute("Now").unwrap(), date_format)
            .unwrap();
    let train_infos = snapshot
        .children()
        .filter(|c| c.is_element())
        .collect::<Vec<_>>();
    assert!(train_infos.len() == 1);
    let train_infos = train_infos[0];
    for train_info in train_infos.children().filter(|c| c.is_element()) {
        assert!(train_info.tag_name().name() == "TrainInfo");
        let train = train_info.attribute("TrainId").unwrap();
        let position = train_info.attribute("Position").unwrap();
        let delay = train_info
            .attribute("DelayInSeconds")
            .unwrap()
            .parse::<i32>()
            .unwrap();

        match position {
            "HasLeftTheNetwork" => {
                continue;
            } // uninteresting
            "OnConnection" => {
                let track = train_info.attribute("TrackId").unwrap();
                let time_in = chrono::NaiveDateTime::parse_from_str(
                    train_info.attribute("TimeIn").unwrap(),
                    date_format,
                )
                .unwrap();
                // println!(
                //     "  train{} {} delay{} track{} time{}",
                //     train, position, delay, track, time_in
                // );

                let inactive_time = (time_now - time_in).num_seconds();
                if inactive_time > 3600 {
                    println!(
                        "Warning: keeping inactive train {} (inactive on track {} for {} seconds)",
                        train, track, inactive_time
                    );
                }

                assert!(train_positions
                    .insert(train, Pos::OnTrack(track, time_in, delay))
                    .is_none());
            }
            "InStation" => {
                let station = train_info.attribute("StationId").unwrap();
                let time_in = chrono::NaiveDateTime::parse_from_str(
                    train_info.attribute("TimeIn").unwrap(),
                    date_format,
                )
                .unwrap();
                // println!(
                //     "  train{} {} delay{} station{} time{}",
                //     train, position, delay, station, time_in
                // );

                // If the train has been standing in the station for over one hour, we assume it is cancelled.
                let inactive_time = (time_now - time_in).num_seconds();
                if inactive_time > 3600 {
                    println!("Warning: removing inactive train {} (inactive n station {} for {} seconds)", train, station, inactive_time);
                    continue;
                }

                assert!(train_positions
                    .insert(train, Pos::InStation(station, time_in, delay))
                    .is_none());
            }
            "Offline" => {
                // println!("  train{} {} delay{}", train, position, delay);

                assert!(delay >= 0);
                assert!(train_positions
                    .insert(train, Pos::NotStarted(delay))
                    .is_none());
            }
            _ => panic!(),
        }
    }
    (time_now, train_positions)
}

#[allow(clippy::type_complexity)]
fn get_objective_map<'a>(
    doc: &'a roxmltree::Document<'a>,
) -> HashMap<(&'a str, &'a str), Vec<(i32, i32, f32, f32)>> {
    let mut objective_map: HashMap<(&str, &str), Vec<(i32, i32, f32, f32)>> = HashMap::new();
    let objective = doc
        .root_element()
        .children()
        .find(|n| n.tag_name().name() == "Objective")
        .unwrap();
    assert!(objective.attribute("Name") == Some("PiecewiseLinear"));
    for penalties in objective.children().filter(|c| c.is_element()) {
        assert!(penalties.tag_name().name() == "TrainDelayPenalties");
        let train = penalties.attribute("TrainId").unwrap();

        for penalty in penalties.children().filter(|c| c.is_element()) {
            assert!(penalty.tag_name().name() == "DelayPenalty");
            let station = penalty.attribute("StationId").unwrap();
            let from_seconds = penalty
                .attribute("FromSeconds")
                .unwrap()
                .parse::<i32>()
                .unwrap();
            let to_seconds = penalty
                .attribute("ToSeconds")
                .unwrap()
                .parse::<i32>()
                .unwrap();
            let from_value = penalty
                .attribute("FromValue")
                .unwrap()
                .parse::<f32>()
                .unwrap();
            let slope = penalty.attribute("Slope").unwrap().parse::<f32>().unwrap();

            objective_map.entry((train, station)).or_default().push((
                from_seconds,
                to_seconds,
                from_value,
                slope,
            ));

            // println!(
            //     "  obj t{} s{} {}-{}  {}-{}",
            //     train, station, from_seconds, to_seconds, from_value, slope
            // );
        }
    }
    objective_map
}

fn get_track_id_map<'a>(tracks: roxmltree::Node<'a, 'a>) -> HashMap<(&'a str, &'a str), &'a str> {
    let mut connection_ids: HashMap<(&str, &str), &str> = HashMap::new();
    for track in tracks.children().filter(|c| c.is_element()) {
        let id = track.attribute("TrackId").unwrap();
        let station_a = track.attribute("StationA").unwrap();
        let station_b = track.attribute("StationB").unwrap();
        // println!(" track {} {} {}", id, station_a, station_b);
        assert!(!track.children().any(|c| c.is_element()));

        assert!(connection_ids.insert((station_a, station_b), id).is_none());
        // assert!(connection_ids.insert((station_b, station_a), id).is_none());
    }
    for track in tracks.children().filter(|c| c.is_element()) {
        let id = track.attribute("TrackId").unwrap();
        let station_a = track.attribute("StationA").unwrap();
        let station_b = track.attribute("StationB").unwrap();

        connection_ids
            .entry((station_b, station_a))
            .or_insert_with(|| {
                // println!("{}-{} is not a double track", station_a, station_b);
                id
            });
        // assert!(connection_ids.insert((station_a, station_b), id).is_none());
    }

    connection_ids
}

fn get_runningtimes_map<'a>(
    doc: &'a roxmltree::Document<'a>,
) -> HashMap<&'a str, HashMap<&'a str, u32>> {
    let mut minimum_running_times: HashMap<&str, HashMap<&str, u32>> = HashMap::new();
    let running_times = doc
        .root_element()
        .children()
        .find(|n| n.tag_name().name() == "RunningTimes")
        .unwrap();
    for speed_class in running_times.children().filter(|c| c.is_element()) {
        let id = speed_class.attribute("Id").unwrap();
        let map = minimum_running_times.entry(id).or_default();
        let tracks = speed_class
            .children()
            .find(|c| c.tag_name().name() == "Tracks")
            .unwrap();
        for track_time in tracks.children().filter(|c| c.is_element()) {
            let track = track_time.attribute("TrackId").unwrap();
            let dt = track_time
                .attribute("MinimumRunningTimeInSeconds")
                .unwrap()
                .parse::<u32>()
                .unwrap();

            assert!(map.insert(track, dt).is_none());

            // println!(" sp{} tr{} dt{}", id, track, dt);
        }
    }
    minimum_running_times
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
    // println!("Parsing {}", headerline);
    let _empty = lines.next().unwrap();

    let mut lines = lines.peekable();
    while let Some(line) = lines.next() {
        let mut fields = line.split_ascii_whitespace();

        if header.is_none() {
            // println!(" parsing header {}", line);
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
            // println!("Header {:?}", header);
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
