use crate::problem::Problem;
use chrono::NaiveDateTime;
use log::info;
use std::collections::{HashMap, HashSet};

struct RawProblem {
    trains: Vec<RawTrain>,
    conflicts: Vec<(usize, usize)>,
}

struct RawTrain {
    visit: Vec<RawVisit>,
}

struct RawVisit {
    resource_id: usize,
    earliest_in: i32,
    travel_time: i32,
}
enum Pos<'a> {
    OnTrack(&'a str, NaiveDateTime, i32),
    InStation(&'a str, NaiveDateTime, i32),
    NotStarted(i32),
}

pub fn read_file(instance_fn: &str) -> Problem {
    let date_format = "%Y-%m-%dT%H:%M:%S";
    let parse_date = |d| chrono::NaiveDateTime::parse_from_str(d, date_format).unwrap();
    let instance_xml = std::fs::read_to_string(instance_fn).unwrap();
    let doc = roxmltree::Document::parse(&instance_xml).unwrap();
    let network = doc
        .root_element()
        .children()
        .find(|n| n.tag_name().name() == "Network")
        .unwrap();
    let stations = network.children().find(|n| n.tag_name().name() == "Stations").unwrap();
    let tracks = network.children().find(|n| n.tag_name().name() == "Tracks").unwrap();

    let minimum_running_times = get_runningtimes_map(&doc);
    let connection_ids = get_track_id_map(tracks);
    let objective_map = get_objective_map(&doc);
    let train_positions = get_train_pos(&doc, date_format);

    // let mut ignore_trains = HashSet::new();

    // // for ch in doc.root_element().children().filter(|c| c.is_element()) {
    // //     println!("{:?}", ch.tag_name().name());
    // // }

    // for station in stations.children().filter(|c| c.is_element()) {
    //     let id = station.attribute("StationId").unwrap();
    //     let maintrack = station.attribute("MainTrackId").unwrap();
    //     println!("station {} {}", id, maintrack);

    //     let internal_tracks = station
    //         .children()
    //         .find(|n| n.tag_name().name() == "InternalTracks")
    //         .unwrap();
    //     for internal_track in internal_tracks.children().filter(|c| c.is_element()) {
    //         assert!(internal_track.tag_name().name() == "InternalTrack");
    //         let track_id = internal_track.attribute("TrackId").unwrap();
    //         let has_platform = internal_track
    //             .attribute("HasPlatform")
    //             .unwrap()
    //             .parse::<bool>()
    //             .unwrap();
    //         let length = internal_track.attribute("Length").map(|l| l.parse::<u32>().unwrap());

    //         println!("  - track {} {} {:?}", track_id, has_platform, length);
    //     }
    // }

    let timetable = doc
        .root_element()
        .children()
        .find(|n| n.tag_name().name() == "TimeTable")
        .unwrap();
    let train_schedules = timetable
        .children()
        .find(|n| n.tag_name().name() == "TrainSchedules")
        .unwrap();
    for train_schedule in train_schedules.children().filter(|c| c.is_element()) {
        assert!(train_schedule.tag_name().name() == "TrainSchedule");
        let id = train_schedule.attribute("TrainId").unwrap();
        let speed_class = train_schedule.attribute("SpeedClass").unwrap();
        let type_ = train_schedule.attribute("Type").unwrap();
        let origin_id = train_schedule.attribute("OriginId").unwrap();
        let destination_id = train_schedule.attribute("DestinationId").unwrap();
        let length = train_schedule.attribute("Length").unwrap().parse::<u32>().unwrap();

        let stop_nodes = train_schedule.children().filter(|c| c.is_element()).collect::<Vec<_>>();
        if stop_nodes.len() < 2 {
            println!("ignoring train {} too few stations", id);
            continue;
        }

        if let Some(pos) = train_positions.get(id) {
            let mut visits = Vec::new();

            let (stop_idx, current_time) = match pos {
                Pos::OnTrack(tx, enter_time, delay) => stop_nodes
                    .iter()
                    .zip(stop_nodes.iter().skip(1))
                    .find_map(|(n1, n2)| {
                        let prev_station = n1.attribute("StationId").unwrap();
                        let next_station = n2.attribute("StationId").unwrap();
                        if connection_ids[&(prev_station, next_station)] == *tx {
                            Some(todo!())
                        } else {
                            None
                        }
                    })
                    .unwrap(),
                Pos::InStation(sx, enter_time, delay) => stop_nodes
                    .iter()
                    .find_map(|n| {
                        if *sx == n.attribute("StationId").unwrap() {
                            Some(todo!())
                        } else {
                            None
                        }
                    })
                    .unwrap(),
                Pos::NotStarted(delay) => {
                    // start_time = true;
                    (
                        0,
                        parse_date(stop_nodes[0].attribute("AimedArrivalTime").unwrap())
                            + chrono::Duration::seconds(*delay as i64),
                    )
                }
            };

            for (stop, next_stop) in stop_nodes[stop_idx..].iter().zip(
                stop_nodes[stop_idx..]
                    .iter()
                    .skip(1)
                    .map(Some)
                    .chain(std::iter::once(None)),
            ) {
                let station = stop.attribute("StationId").unwrap();
                let aimed_arrival = parse_date(stop.attribute("AimedArrivalTime").unwrap());
                let aimed_departure = parse_date(stop.attribute("AimedDepartureTime").unwrap());

                // Now we enter the station at current_time.

                visits.push((Ok(station), current_time, 0));

                if let Some(next_stop) = next_stop {
                    let next_station = next_stop.attribute("StationId").unwrap();
                    let track = *connection_ids.get(&(station, next_station)).unwrap();

                    visits.push((Err(track), current_time, minimum_running_times[speed_class][track]));
                }
            }

            // if let Some(start_time) = start_time {

            //     if stop_idx > 0 {
            //         // add the track
            //         let prev_station = stop_nodes[stop_idx - 1].attribute("StationId").unwrap();
            //         let next_station = stop_nodes[stop_idx].attribute("StationId").unwrap();
            //         let track = connection_ids[&(prev_station, next_station)];

            //         let track_running_time = minimum_running_times[speed_class][track];

            //         visits.push((Err(track),0));

            //     }

            //     visits.push((Ok(station), 0));

            // }

            // stop_idx += 1;
            // if stop_idx >= stop_nodes.len() {
            //     break; // Done.
            // }

            // for stop in train_schedule.children().filter(|c| c.is_element()) {
            //     assert!(stop.tag_name().name() == "ScheduledStop");

            //     println!("  {} {} {}", station, aimed_arrival, aimed_departure);
            // }
        } else {
            println!("Ignoring train {} has left the network.", id);
        }
    }

    // println!("{}", &instance_xml[snapshots.range()]);

    todo!()
}

fn get_train_pos<'a>(doc: &'a roxmltree::Document, date_format: &'_ str) -> HashMap<&'a str, Pos<'a>> {
    let mut train_positions: HashMap<&str, Pos> = HashMap::new();
    let snapshots = doc
        .root_element()
        .children()
        .find(|n| n.tag_name().name() == "Snapshots")
        .unwrap();
    let snapshots = snapshots.children().filter(|c| c.is_element()).collect::<Vec<_>>();
    assert!(snapshots.len() == 1);
    let snapshot = snapshots[0];
    assert!(snapshot.tag_name().name() == "Snapshot");
    let snapshot_index = snapshot.attribute("Index").unwrap();
    assert!(snapshot_index == "0");
    let time_now = chrono::NaiveDateTime::parse_from_str(snapshot.attribute("Now").unwrap(), date_format).unwrap();
    let train_infos = snapshot.children().filter(|c| c.is_element()).collect::<Vec<_>>();
    assert!(train_infos.len() == 1);
    let train_infos = train_infos[0];
    for train_info in train_infos.children().filter(|c| c.is_element()) {
        assert!(train_info.tag_name().name() == "TrainInfo");
        let train = train_info.attribute("TrainId").unwrap();
        let position = train_info.attribute("Position").unwrap();
        let delay = train_info.attribute("DelayInSeconds").unwrap().parse::<i32>().unwrap();

        match position {
            "HasLeftTheNetwork" => {
                continue;
            } // uninteresting
            "OnConnection" => {
                let track = train_info.attribute("TrackId").unwrap();
                let time_in =
                    chrono::NaiveDateTime::parse_from_str(train_info.attribute("TimeIn").unwrap(), date_format)
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
                let time_in =
                    chrono::NaiveDateTime::parse_from_str(train_info.attribute("TimeIn").unwrap(), date_format)
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

                assert!(train_positions.insert(train, Pos::NotStarted(delay)).is_none());
            }
            _ => panic!(),
        }
    }
    train_positions
}

fn get_objective_map<'a>(doc: &'a roxmltree::Document<'a>) -> HashMap<(&'a str, &'a str), Vec<(i32, i32, f32, f32)>> {
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
            let from_seconds = penalty.attribute("FromSeconds").unwrap().parse::<i32>().unwrap();
            let to_seconds = penalty.attribute("ToSeconds").unwrap().parse::<i32>().unwrap();
            let from_value = penalty.attribute("FromValue").unwrap().parse::<f32>().unwrap();
            let slope = penalty.attribute("Slope").unwrap().parse::<f32>().unwrap();

            objective_map
                .entry((train, station))
                .or_default()
                .push((from_seconds, to_seconds, from_value, slope));

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
        println!(" track {} {} {}", id, station_a, station_b);
        assert!(!track.children().any(|c| c.is_element()));

        assert!(connection_ids.insert((station_a, station_b), id).is_none());
        assert!(connection_ids.insert((station_b, station_a), id).is_none());
    }
    connection_ids
}

fn get_runningtimes_map<'a>(doc: &'a roxmltree::Document<'a>) -> HashMap<&'a str, HashMap<&'a str, u32>> {
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
