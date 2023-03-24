use eframe::egui::Visuals;
use eframe::egui::{
    self,
    plot::{HLine, Line, PlotPoints, Polygon},
};
use eframe::epaint::Color32;
use heuristic::solvers::solver_brb::BnBConflictSolver;
use heuristic::solvers::solver_heurheur::HeurHeur;
use heuristic::solvers::solver_heurheur2::HeurHeur2;
use heuristic::solvers::solver_random::RandomHeuristic;
use heuristic::solvers::train_queue::QueueTrainSolver;
use heuristic::{problem, TrainSolver};
use heuristic::{problem::*, ConflictSolver};
use serde::Deserialize;
use std::collections::HashMap;

#[derive(Deserialize, Debug)]
pub struct Input {
    pub problem: Problem,
}

pub struct Model<Solver> {
    pub solver: Solver,
    pub selected_train: usize,
    pub current_cost: Option<i32>,
    pub locations: HashMap<String, i32>,
    pub ref_sol: Vec<Vec<i32>>,
    pub show_ref_sol: bool,
}

pub struct AutoColor {
    next_auto_color_idx: u32,
}

#[allow(clippy::new_without_default)]
impl AutoColor {
    pub fn new() -> Self {
        Self {
            next_auto_color_idx: 0,
        }
    }
    pub fn next_color(&mut self) -> Color32 {
        let i = self.next_auto_color_idx;
        self.next_auto_color_idx += 1;
        let golden_ratio = (5.0_f32.sqrt() - 1.0) / 2.0; // 0.61803398875
        let h = i as f32 * golden_ratio;
        eframe::epaint::color::Hsva::new(h, 0.85, 0.5, 1.0).into() // TODO(emilk): OkLab or some other perspective color space
    }
}

pub struct App<Solver> {
    pub model: Model<Solver>,
}

impl<Solver: ConflictSolver + StatusGui> App<Solver> {
    pub fn draw_gui(&mut self, ctx: &eframe::egui::Context) {
        egui::CentralPanel::default().show(ctx, |ui| {
            egui::SidePanel::left("left_panel")
                .width_range(300.0..=300.0)
                .show_inside(ui, |ui| {
                    egui::ScrollArea::vertical().show(ui, |ui| {
                        ui.label(&format!("Cost: {:?}", self.model.current_cost));
                        ui.checkbox(&mut self.model.show_ref_sol, "show ref sol");

                        if ui.button("Small step").clicked() {
                            if let Some((cost, _sol)) = self.model.solver.small_step() {
                                self.model.current_cost = Some(cost);
                            }
                        }
                        if ui.button("Big step").clicked() {
                            if let Some((cost, _sol)) = self.model.solver.big_step() {
                                self.model.current_cost = Some(cost);
                            }
                        }

                        self.model.solver.status_gui(ui);
                    })
                });
            egui::SidePanel::left("left_panel2")
                .width_range(300.0..=300.0)
                .show_inside(ui, |ui| {
                    egui::ScrollArea::vertical().show(ui, |ui| {
                        self.train_solver_ui(ui);
                    })
                });

            // ui.heading("infrastructure");
            // self.plot_infrastructure(ui);

            // ui.heading("train graph");
            self.plot_traingraph(ui);
        });
    }

    fn train_solver_ui(&mut self, ui: &mut egui::Ui) {
        ui.heading("Train solver");
        egui::ComboBox::from_label("Select one!")
            .selected_text(format!("Train {}", self.model.selected_train))
            .show_ui(ui, |ui| {
                for i in 0..self.model.solver.trainset().trains.len() {
                    ui.selectable_value(&mut self.model.selected_train, i, format!("Train {}", i));
                }
            });

        if let Some(train_solver) = self
            .model
            .solver
            .trainset()
            .trains
            .get(self.model.selected_train)
        {
            ui.label(&format!("Train index {:?}", self.model.selected_train));
            ui.label(&format!("Train status {:?}", train_solver.status()));
            ui.label(&format!("Occupations {:?}", train_solver.occupied));
            ui.label(&format!("Current node {:?}", train_solver.current_node));
            ui.label(&format!("Current time {}", train_solver.current_time()));
            ui.label(&format!(
                "Total number of nodes {}",
                train_solver.total_nodes
            ));
            ui.label(&format!("Nodes {:?}", train_solver.queued_nodes));
            ui.label(&format!("Solution {:?}", train_solver.solution));
        } else {
            ui.label(&format!("No train index {}", self.model.selected_train));
        }
    }

    fn plot_traingraph(&mut self, ui: &mut egui::Ui) {
        let mut autocolor = AutoColor::new();
        let plot = egui::plot::Plot::new("grph");
        plot.show(ui, |plot_ui| {
            for (_, y) in self.model.locations.iter() {
                for pt in [*y, *y + 1] {
                    plot_ui.hline(HLine::new(pt as f32).color(Color32::BLACK).width(1.0));
                }
            }

            for (train_idx, train_solver) in self.model.solver.trainset().trains.iter().enumerate()
            {
                let color = autocolor.next_color();
                let direction = guess_train_direction(train_solver, &self.model.locations);
                let mut curr = &train_solver.current_node;
                let mut prev_pos: Option<[f64; 2]> = None;
                while let Some(prev) = curr.parent.as_ref() {
                    let curr_time = if self.model.show_ref_sol && curr.block > 0 {
                        self.model.ref_sol[train_idx][curr.block as usize - 1]
                    } else {
                        curr.time
                    };
                    let prev_time = if self.model.show_ref_sol && prev.block > 0 {
                        self.model.ref_sol[train_idx][prev.block as usize - 1]
                    } else {
                        prev.time
                    };
                    for res in train_solver.train.blocks[prev.block as usize]
                        .resource_usage
                        .iter()
                    {
                        let trackname = res.track_name.as_ref();
                        if let Some(y) = trackname.and_then(|n| self.model.locations.get(n)) {
                            let ys = if direction > 0 {
                                (*y as f64, (*y + 1) as f64)
                            } else {
                                ((*y + 1) as f64, *y as f64)
                            };

                            if let Some(prev_pos) = prev_pos {
                                if prev_pos[1] == ys.1 {
                                    plot_ui.line(
                                        Line::new(PlotPoints::from_iter([
                                            prev_pos,
                                            [
                                                0.5 * (prev_pos[0] + curr_time as f64),
                                                0.5 * (prev_pos[1] + ys.1),
                                            ],
                                            [curr_time as f64, ys.1],
                                        ]))
                                        .color(color)
                                        .width(3.0)
                                        .name(&format!("Train {} Station", train_solver.id)),
                                    );
                                }
                            }

                            let weight = self
                                .model
                                .solver
                                .visit_weights()
                                .and_then(|w| {
                                    w.range(
                                        (train_idx as TrainRef, curr.block)
                                            ..(train_idx as TrainRef, u32::MAX),
                                    )
                                    .nth(0)
                                    .map(|(_, w)| *w)
                                })
                                .unwrap_or(0.);

                            plot_ui.line(
                                Line::new(PlotPoints::from_iter([
                                    [prev_time as f64, ys.0],
                                    [
                                        0.5 * ((prev_time as f64) + (curr_time as f64)),
                                        0.5 * (ys.0 + ys.1),
                                    ],
                                    [curr_time as f64, ys.1],
                                ]))
                                .color(color)
                                .width(2.0)
                                .name(&format!(
                                    "Train {} Track {:?} dir {} w={:.2}",
                                    train_solver.id, trackname, direction, weight
                                )),
                            );

                            prev_pos = Some([prev_time as f64, ys.0]);
                        } else if trackname.is_some() {
                            println!("Warning: no track name {:?}", trackname);
                        }
                    }

                    curr = prev;
                }
            }

            if let Some((resource_idx, (occ_a, occ_b))) = self
                .model
                .solver
                .conflicts()
                .conflicting_resource_set
                .iter()
                .map(|r| {
                    (
                        *r,
                        self.model.solver.conflicts().resources[*r as usize]
                            .get_conflict()
                            .unwrap(),
                    )
                })
                .min_by_key(|(_, c)| c.0.interval.time_start.min(c.1.interval.time_start))
            {
                for res in &self.model.solver.trainset().trains[occ_a.train as usize]
                    .train
                    .blocks[occ_a.block as usize]
                    .resource_usage
                {
                    let trackname = res.track_name.as_ref();
                    if let Some(y) = trackname.and_then(|n| self.model.locations.get(n)) {
                        for occ in [occ_a, occ_b] {
                            plot_ui.polygon(
                                Polygon::new(PlotPoints::from_iter([
                                    [occ.interval.time_start as f64, *y as f64],
                                    [occ.interval.time_end as f64, *y as f64],
                                    [occ.interval.time_end as f64, (*y + 1) as f64],
                                    [occ.interval.time_start as f64, (*y + 1) as f64],
                                ]))
                                .color(Color32::RED)
                                .fill_alpha(0.5)
                                .name(format!("res {} train {}", resource_idx, occ.train)),
                            );
                        }
                    }
                }
            }

            // for (resource_idx, occs) in self.model.solver.conflicts.resources.iter().enumerate() {
            //     for pt in [resource_idx, resource_idx + 1] {
            //         plot_ui.hline(HLine::new(pt as f32).color(Color32::BLACK).width(1.0));
            //     }

            //     let conflicts = occs.conflicting_resource_set_idx >= 0;
            //     for occ in occs.occupations.iter() {
            //         plot_ui.polygon(
            //             Polygon::new(PlotPoints::from_iter([
            //                 [occ.interval.time_start as f64, resource_idx as f64],
            //                 [occ.interval.time_end as f64, resource_idx as f64],
            //                 [occ.interval.time_end as f64, (resource_idx + 1) as f64],
            //                 [occ.interval.time_start as f64, (resource_idx + 1) as f64],
            //             ]))
            //             .color(if conflicts {
            //                 Color32::RED
            //             } else {
            //                 Color32::BLACK
            //             })
            //             .fill_alpha(0.5)
            //             .name(format!("res {} train {}", resource_idx, occ.train)),
            //         );
            //     }
            // }

            // for train_solver in
            //     self.model.solver.trains.iter()
            // {
            //     let mut curr = &train_solver.current_node;
            //     while let Some(prev) = curr.parent.as_ref() {
            //         if let Some(prev_draw) =
            //             (prev.track >= 0).then(|| &self.model.draw_tracks[prev.track as usize])
            //         {
            //             plot_ui.line(
            //                 Line::new(PlotPoints::from_iter([
            //                     [prev.time as f64, prev_draw.p_a[0] as f64],
            //                     [curr.time as f64, prev_draw.p_b[0] as f64],
            //                 ]))
            //                 .color(Color32::BLACK)
            //                 .width(2.0),
            //             );
            //         }

            //         curr = prev;
            //     }
            // }
        });
    }

    fn _plot_infrastructure(&mut self, _ui: &mut egui::Ui) {
        // let plot = egui::plot::Plot::new("infr").data_aspect(1.0).height(200.0);
        // plot.show(ui, |plot_ui| {
        //     for (_track, draw) in self
        //         .model
        //         .problem
        //         .tracks
        //         .iter()
        //         .zip(self.model.draw_tracks.iter())
        //     {
        //         plot_ui.line(
        //             Line::new(PlotPoints::from_iter([draw.p_a, draw.p_b]))
        //                 .color(Color32::BLACK)
        //                 .width(2.0),
        //         );

        //         plot_ui.text(Text::new(PlotPoint::from(draw.p_a), &draw.name));
        //     }
        // });
    }
}

fn guess_train_direction(
    train_solver: &heuristic::solvers::train_queue::QueueTrainSolver,
    locs: &HashMap<String, i32>,
) -> i32 {
    let mut dir = 1;
    let mut prev_y = None;
    let mut curr = &train_solver.current_node;
    'x: while let Some(prev) = curr.parent.as_ref() {
        for res in train_solver.train.blocks[prev.block as usize]
            .resource_usage
            .iter()
        {
            let trackname = res.track_name.as_ref();
            if let Some(&y) = trackname.and_then(|n| locs.get(n)) {
                if let Some(prev_y) = prev_y {
                    if prev_y < y {
                        // We are looking at the nodes backwards, so
                        // prev_y < y is a negative-direction train.
                        dir = -1;
                    } else {
                        dir = 1;
                    }
                    break 'x;
                } else {
                    prev_y = Some(y);
                }
            }
        }
        curr = prev;
    }
    dir
}

impl<Solver: ConflictSolver + StatusGui> eframe::App for App<Solver> {
    fn update(&mut self, ctx: &eframe::egui::Context, _frame: &mut eframe::Frame) {
        // self.get_messages();
        self.draw_gui(ctx);

        ctx.request_repaint_after(std::time::Duration::from_secs_f32(1.0));
    }
}

fn main() {
    pretty_env_logger::init();

    let input: problem::Problem =
        serde_json::from_str(&std::fs::read_to_string("origA11_rh.json").unwrap()).unwrap();

    let ref_sol: Vec<Vec<i32>> =
        serde_json::from_str(&std::fs::read_to_string("origA11_sol_3331.json").unwrap()).unwrap();
    // let ref_sol = vec![];

    // let input = examples::example_1();

    // let problem = Rc::new(input.problem);
    // let draw_tracks = Rc::new(input.draw_tracks);

    let mut locations = HashMap::new();
    for i in 1..=35 {
        locations.insert(format!("T{}", i), i);
        locations.insert(format!("T{}_S{}_to_S{}", i, i, i + 1), i);
    }

    locations.insert("T1_S1_to_S2".to_string(), 1);
    locations.insert("DT_T2_DT_S2_to_S3".to_string(), 2);
    locations.insert("DT_T4_DT_S2_to_S3".to_string(), 2);
    locations.insert("DT_T6_DT_S2_to_S3".to_string(), 2);
    locations.insert("DT_T6_DT_S4_to_S5X".to_string(), 4);
    locations.insert("DT_T4_DT_S3_to_S4X".to_string(), 3);
    locations.insert("DT_T2_DT_S2_to_S3X".to_string(), 2);
    locations.insert("DT_T4_DT_S3_to_S4".to_string(), 3);
    locations.insert("DT_T6_DT_S4_to_S5".to_string(), 4);
    locations.insert("T8_S5_to_S6".to_string(), 5);
    locations.insert("T9_S6_to_S7".to_string(), 6);
    locations.insert("T10_S7_to_S8".to_string(), 7);
    locations.insert("T11_S8_to_S9".to_string(), 8);
    locations.insert("T12_S9_to_S10".to_string(), 9);
    locations.insert("T13_S10_to_S11".to_string(), 10);
    locations.insert("T14_S11_to_S12".to_string(), 11);
    locations.insert("T15_S12_to_S13".to_string(), 12);
    locations.insert("T16_S13_to_S14".to_string(), 13);
    locations.insert("T17_S14_to_S15".to_string(), 14);
    locations.insert("T18_S15_to_S16".to_string(), 15);
    locations.insert("T19_S16_to_S17".to_string(), 16);
    locations.insert("T20_S17_to_S18".to_string(), 17);
    locations.insert("T21_S18_to_S19".to_string(), 18);
    locations.insert("T22_S19_to_S20".to_string(), 19);
    locations.insert("T23_S20_to_S21".to_string(), 20);
    locations.insert("T24_S21_to_S22".to_string(), 21);
    locations.insert("T25_S22_to_S23".to_string(), 22);
    locations.insert("T26_S23_to_S24".to_string(), 23);
    locations.insert("T27_S24_to_S25".to_string(), 24);
    locations.insert("T28_S25_to_S26".to_string(), 25);
    locations.insert("T29_S26_to_S27".to_string(), 26);
    locations.insert("T30_S27_to_S28".to_string(), 27);
    locations.insert("T31_S28_to_S29".to_string(), 28);
    locations.insert("T32_S29_to_S30".to_string(), 29);
    locations.insert("T33_S30_to_S31".to_string(), 30);

    let app = App {
        model: Model {
            solver: HeurHeur2::new(input),
            selected_train: 0,
            current_cost: None,
            locations,
            ref_sol,
            show_ref_sol: false,
        },
    };

    eframe::run_native(
        "heuristic train re-scheduling and re-routing",
        eframe::NativeOptions::default(),
        Box::new(|ctx| {
            ctx.egui_ctx.set_visuals(Visuals::light());
            Box::new(app)
        }),
    );
}

pub trait StatusGui {
    fn status_gui(&self, ui: &mut egui::Ui);
}

impl StatusGui for BnBConflictSolver<QueueTrainSolver> {
    fn status_gui(&self, ui: &mut egui::Ui) {
        ui.heading("Conflict solver");
        ui.label(&format!("Status: {:?}", self.status()));

        ui.label(&format!(
            "Total number of nodes {}",
            self.conflict_space.n_nodes_generated
        ));
        ui.label(&format!("Queued nodes: {}", self.queued_nodes.len()));
        ui.label(&format!(
            "Unsolved trains: {:?}",
            self.trainset.dirty_trains
        ));
        ui.label(&format!(
            "Total conflicts: {}",
            self.conflicts.conflicting_resource_set.len()
        ));

        ui.heading("Current resource occupations");
        for resource in self.conflicts.conflicting_resource_set.iter() {
            for occ in self.conflicts.resources[*resource as usize]
                .occupations
                .iter()
            {
                ui.label(&format!("res {} {:?}", resource, occ));
            }
        }
        ui.heading("Current node");
        ui.label(&format!("{:?}", self.conflict_space.current_node));
        ui.heading("Open nodes");
        for n in self.queued_nodes.iter() {
            ui.label(&format!("{:?}", n));
        }
    }
}

impl StatusGui for RandomHeuristic {
    fn status_gui(&self, _ui: &mut egui::Ui) {}
}

impl StatusGui for HeurHeur {
    fn status_gui(&self, _ui: &mut egui::Ui) {}
}

impl StatusGui for HeurHeur2 {
    fn status_gui(&self, ui: &mut egui::Ui) {
        ui.heading("Queue main");
        for x in self.queue_main.iter() {
            ui.label(&format!("{:?}", x));
        }
    }
}
