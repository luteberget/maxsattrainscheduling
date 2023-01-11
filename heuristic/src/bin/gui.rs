use eframe::egui::Visuals;
use eframe::egui::{
    self,
    plot::{HLine, Line, PlotPoint, PlotPoints, Polygon, Text},
};
use eframe::epaint::Color32;
use heuristic::{problem, TrainSolver};
use heuristic::problem::*;
use serde::Deserialize;
use std::collections::HashMap;

#[derive(Deserialize, Debug)]
pub struct Input {
    pub problem: Problem,
}

pub struct Model {
    pub solver: heuristic::solvers::bnb_solver::ConflictSolver<heuristic::solvers::queue_train::QueueTrainSolver>,
    pub selected_train: usize,
    pub current_cost: Option<i32>,
    pub locations: HashMap<String, i32>,
}

pub struct AutoColor {
    next_auto_color_idx: u32,
}

impl AutoColor {
    pub fn new() -> Self {
        Self {
            next_auto_color_idx: 0,
        }
    }
    pub fn next(&mut self) -> Color32 {
        let i = self.next_auto_color_idx;
        self.next_auto_color_idx += 1;
        let golden_ratio = (5.0_f32.sqrt() - 1.0) / 2.0; // 0.61803398875
        let h = i as f32 * golden_ratio;
        eframe::epaint::color::Hsva::new(h, 0.85, 0.5, 1.0).into() // TODO(emilk): OkLab or some other perspective color space
    }
}

pub struct App {
    pub model: Model,
}

impl App {
    pub fn draw_gui(&mut self, ctx: &eframe::egui::Context) {
        egui::CentralPanel::default().show(ctx, |ui| {
            egui::SidePanel::left("left_panel")
                .width_range(300.0..=300.0)
                .show_inside(ui, |ui| {
                    egui::ScrollArea::vertical().show(ui, |ui| {
                        self.conflict_solver_ui(ui);
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
                for i in 0..self.model.solver.trains.len() {
                    ui.selectable_value(&mut self.model.selected_train, i, format!("Train {}", i));
                }
            });

        if let Some(train_solver) = self.model.solver.trains.get(self.model.selected_train) {
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

    fn conflict_solver_ui(&mut self, ui: &mut egui::Ui) {
        ui.heading("Conflict solver");
        ui.label(&format!("Status: {:?}", self.model.solver.status()));
        ui.label(&format!("Cost: {:?}", self.model.current_cost));
        ui.label(&format!(
            "Total number of nodes {}",
            self.model.solver.total_nodes
        ));
        ui.label(&format!(
            "Queued nodes: {}",
            self.model.solver.queued_nodes.len()
        ));
        ui.label(&format!(
            "Unsolved trains: {:?}",
            self.model.solver.dirty_trains
        ));
        ui.label(&format!(
            "Total conflicts: {}",
            self.model.solver.conflicts.conflicting_resource_set.len()
        ));

        if ui.button("Step").clicked() {
            if let Some((cost, _sol)) = self.model.solver.solve_next() {
                self.model.current_cost = Some(cost);
            }
        }

        ui.heading("Current resource occupations");
        for resource in self.model.solver.conflicts.conflicting_resource_set.iter() {
            for occ in self.model.solver.conflicts.resources[*resource as usize]
                .occupations
                .iter()
            {
                ui.label(&format!("res {} {:?}", resource, occ));
            }
        }
        ui.heading("Current node");
        ui.label(&format!("{:?}", self.model.solver.current_node));
        ui.heading("Open nodes");
        for n in self.model.solver.queued_nodes.iter() {
            ui.label(&format!("{:?}", n));
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

            for train_solver in self.model.solver.trains.iter() {
                let color = autocolor.next();
                let direction = {
                    let mut dir = 1;
                    let mut prev_y = None;
                    let mut curr = &train_solver.current_node;
                    'x: while let Some(prev) = curr.parent.as_ref() {
                        for res in train_solver.train.blocks[prev.block as usize]
                            .resource_usage
                            .iter()
                        {
                            let trackname = res.track_name.as_ref();
                            if let Some(&y) = trackname.and_then(|n| self.model.locations.get(n)) {
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
                };

                let mut curr = &train_solver.current_node;
                let mut prev_pos: Option<[f64; 2]> = None;
                while let Some(prev) = curr.parent.as_ref() {
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
                                                0.5 * (prev_pos[0] + curr.time as f64),
                                                0.5 * (prev_pos[1] + ys.1),
                                            ],
                                            [curr.time as f64, ys.1],
                                        ]))
                                        .color(color)
                                        .width(3.0)
                                        .name(&format!("Train {} Station", train_solver.id)),
                                    );
                                }
                            }

                            plot_ui.line(
                                Line::new(PlotPoints::from_iter([
                                    [prev.time as f64, ys.0],
                                    [
                                        0.5 * ((prev.time as f64) + (curr.time as f64)),
                                        0.5 * (ys.0 + ys.1),
                                    ],
                                    [curr.time as f64, ys.1],
                                ]))
                                .color(color)
                                .width(2.0)
                                .name(&format!(
                                    "Train {} Track {:?} dir {}",
                                    train_solver.id, trackname, direction
                                )),
                            );

                            prev_pos = Some([prev.time as f64, ys.0]);
                        } else if trackname.is_some() {
                            println!("Warning: no track name {:?}", trackname);
                        }
                    }

                    curr = prev;
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

    fn plot_infrastructure(&mut self, ui: &mut egui::Ui) {
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

impl eframe::App for App {
    fn update(&mut self, ctx: &eframe::egui::Context, _frame: &mut eframe::Frame) {
        // self.get_messages();
        self.draw_gui(ctx);

        ctx.request_repaint_after(std::time::Duration::from_secs_f32(1.0));
    }
}

fn main() {
    pretty_env_logger::init();

    let input: problem::Problem =
        serde_json::from_str(&std::fs::read_to_string("trackB12_rh.json").unwrap())
            .unwrap();

    // let input = examples::example_1();

    // let problem = Rc::new(input.problem);
    // let draw_tracks = Rc::new(input.draw_tracks);

    let mut locations = HashMap::new();
    for i in 1..=25 {
        locations.insert(format!("T{}", i), i);
    }

    let app = App {
        model: Model {
            solver: heuristic::solvers::bnb_solver::ConflictSolver::new(input),
            selected_train: 0,
            current_cost: None,
            locations,
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
