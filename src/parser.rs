use crate::problem::Problem;
use chrono::{Duration, NaiveDateTime};
use std::{collections::HashMap, net::ToSocketAddrs};

enum Pos<'a> {
    OnTrack(&'a str, NaiveDateTime, i32),
    InStation(&'a str, NaiveDateTime, i32),
    NotStarted(i32),
}

pub fn read_file(instance_fn: &str) -> (Problem, Vec<String>, Vec<String>) {
    let date_format = "%Y-%m-%dT%H:%M:%S";
    let parse_date = |d| chrono::NaiveDateTime::parse_from_str(d, date_format).unwrap();
    let instance_xml = std::fs::read_to_string(instance_fn).unwrap();
    let doc = roxmltree::Document::parse(&instance_xml).unwrap();
    let network = doc
        .root_element()
        .children()
        .find(|n| n.tag_name().name() == "Network")
        .unwrap();
    let stations = network
        .children()
        .find(|n| n.tag_name().name() == "Stations")
        .unwrap();
    let tracks = network
        .children()
        .find(|n| n.tag_name().name() == "Tracks")
        .unwrap();

    let connection_ids = get_track_id_map(tracks);

    // Create resource ids for stations in order
    let mut resource_order = Vec::new();
    for station in stations.children().filter(|c| c.is_element()) {
        let id = station.attribute("StationId").unwrap();
        resource_order.push(Ok(id));
    }

    // Insert tracks between stations
    for track in tracks.children().filter(|c| c.is_element()) {
        let t = track.attribute("TrackId").unwrap();
        let sa = track.attribute("StationA").unwrap();
        let sb = track.attribute("StationB").unwrap();
        let p = (0..resource_order.len() - 1).find(|i| {
            (resource_order[*i] == Ok(sa) && resource_order[*i + 1] == Ok(sb))
            ||(resource_order[*i] == Ok(sb) && resource_order[*i + 1] == Ok(sa))
            || (*i < resource_order.len() - 1 && resource_order[*i] == Ok(sa)&& resource_order[*i + 2] == Ok(sb))
            || (*i < resource_order.len() - 1 && resource_order[*i] == Ok(sb)&& resource_order[*i + 2] == Ok(sa))
        });

        let idx = p.unwrap();
        resource_order.insert(idx + 1, Err(t));
    }

    let minimum_running_times = get_runningtimes_map(&doc);
    let _objective_map = get_objective_map(&doc);
    let (time_now, train_positions) = get_train_pos(&doc, date_format);

    let timetable = doc
        .root_element()
        .children()
        .find(|n| n.tag_name().name() == "TimeTable")
        .unwrap();
    let train_schedules = timetable
        .children()
        .find(|n| n.tag_name().name() == "TrainSchedules")
        .unwrap();

    let mut problem_trains = Vec::new();
    for train_schedule in train_schedules.children().filter(|c| c.is_element()) {
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
        if stop_nodes.len() < 2 {
            // println!("ignoring train {} too few stations", id);
            continue;
        }

        if let Some(pos) = train_positions.get(id) {
            let mut visits = Vec::new();

            let (first_stop_idx, mut current_time) = match pos {
                Pos::OnTrack(track, enter_time, _delay) => stop_nodes
                    .iter()
                    .zip(stop_nodes.iter().skip(1))
                    .enumerate()
                    .find_map(|(stop_idx, (n1, n2))| {
                        let prev_station = n1.attribute("StationId").unwrap();
                        let next_station = n2.attribute("StationId").unwrap();
                        (connection_ids[&(prev_station, next_station)] == *track).then(|| {
                            let travel_time = Duration::seconds(
                                minimum_running_times[speed_class][*track] as i64,
                            );
                            visits.push((Err(*track), *enter_time, travel_time));
                            (stop_idx + 1, *enter_time + travel_time)
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
                Pos::NotStarted(delay) => (
                    0,
                    parse_date(stop_nodes[0].attribute("AimedArrivalTime").unwrap())
                        + chrono::Duration::seconds(*delay as i64),
                ),
            };

            for (stop, next_stop) in stop_nodes[first_stop_idx..].iter().zip(
                stop_nodes[first_stop_idx..]
                    .iter()
                    .skip(1)
                    .map(Some)
                    .chain(std::iter::once(None)),
            ) {
                let station = stop.attribute("StationId").unwrap();
                let _aimed_arrival = parse_date(stop.attribute("AimedArrivalTime").unwrap());
                let aimed_departure = parse_date(stop.attribute("AimedDepartureTime").unwrap());

                // Now we enter the station at current_time.
                visits.push((Ok(station), current_time, Duration::zero()));

                // Now the earliest time to exit the station, is the max of (the current time + 0) and
                // the aimed departure time.
                current_time = (current_time + Duration::zero()).max(aimed_departure);

                if let Some(next_stop) = next_stop {
                    let next_station = next_stop.attribute("StationId").unwrap();
                    let track = *connection_ids.get(&(station, next_station)).unwrap();
                    let travel_time =
                        Duration::seconds(minimum_running_times[speed_class][track] as i64);
                    visits.push((Err(track), current_time, travel_time));
                    current_time += travel_time;
                }
            }

            problem_trains.push((id, visits));
        } else {
            // println!("Ignoring train {} has left the network.", id);
        }
    }

    let mut train_names = Vec::new();
    let resource_names = resource_order
        .iter()
        .map(|i| match i {
            Ok(x) => x.to_string(),
            Err(x) => x.to_string(),
        })
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
    for (n, i) in resource_ids.iter() {
        if n.is_err() {
            // track
            problem.conflicts.push((*i, *i));
        }
    }
    for (name, visits) in problem_trains.iter() {
        let mut t = Vec::new();
        for (r, earliest, travel) in visits.iter() {
            // println!("Looking up resource {:?}", r);
            let resource = resource_ids[r];
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

            t.push((
                resource,
                (*earliest - time_now).num_seconds() as i32,
                travel.num_seconds() as i32,
            ));
        }

        problem.trains.push(crate::problem::Train { visits: t });
        train_names.push(name.to_string());
    }

    (problem, train_names, resource_names)
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
                assert!(train_positions
                    .insert(train, Pos::InStation(station, time_in, delay))
                    .is_none());
            }
            "Offline" => {
                // println!("  train{} {} delay{}", train, position, delay);

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
