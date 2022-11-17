use eframe::{
    egui::{
        self,
        plot::{HLine, Line, PlotPoint, PlotPoints, Polygon, Text},
    },
    epaint::Color32,
};

use crate::model::Model;

pub struct App {
    pub model: Model,
}

impl App {
    pub fn draw_gui(&mut self, ctx: &eframe::egui::Context) {
        egui::CentralPanel::default().show(ctx, |ui| {
            egui::SidePanel::left("left_panel")
                .width_range(400.0..=400.0)
                .show_inside(ui, |ui| {
                    self.conflict_solver_ui(ui);
                });
            egui::SidePanel::left("left_panel2")
                .width_range(400.0..=400.0)
                .show_inside(ui, |ui| {
                    self.train_solver_ui(ui);
                });

            ui.heading("infrastructure");
            self.plot_infrastructure(ui);

            ui.heading("train graph");
            self.plot_traingraph(ui);
        });
    }

    fn train_solver_ui(&mut self, ui: &mut egui::Ui) {
        ui.heading("Train solver");
        egui::ComboBox::from_label("Select one!")
            .selected_text(format!("Train {}", self.model.selected_train))
            .show_ui(ui, |ui| {
                for i in 0..self.model.problem.trains.len() {
                    ui.selectable_value(&mut self.model.selected_train, i, format!("Train {}", i));
                }
            });

        if let Some(train_solver) = self.model.solver.trains.get(self.model.selected_train) {
            ui.label(&format!("Train index {:?}", self.model.selected_train));
            ui.label(&format!("Train status {:?}", train_solver.status()));
            ui.label(&format!("Blocks {:?}", train_solver.blocks));
            ui.label(&format!("Current node {:?}", train_solver.current_node));
            ui.label(&format!("Current time {}", train_solver.current_time()));
            ui.label(&format!("Total number of nodes {}", train_solver.total_nodes));
            ui.label(&format!("Nodes {:?}", train_solver.queued_nodes));
            ui.label(&format!("Solution {:?}", train_solver.solution));
        } else {
            ui.label(&format!("No train index {}", self.model.selected_train));
        }
    }

    fn conflict_solver_ui(&mut self, ui: &mut egui::Ui) {
        ui.heading("Conflict solver");
        ui.label(&format!("Status: {:?}", self.model.solver.status()));
        ui.label(&format!("Total number of nodes {}", self.model.solver.total_nodes));
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
            self.model.solver.step();
        }
        ui.heading("Current resource occupations");
        for resource in self.model.solver.conflicts.conflicting_resource_set.iter() {
            for occ in self.model.solver.conflicts.resources[*resource as usize].occupations.iter() {
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
        let plot = egui::plot::Plot::new("grph");
        plot.show(ui, |plot_ui| {
            for ((_track, draw), occs) in self
                .model
                .problem
                .tracks
                .iter()
                .zip(self.model.draw_tracks.iter())
                .zip(self.model.solver.conflicts.resources.iter())
            {
                for pt in [draw.p_a[0], draw.p_b[0]] {
                    plot_ui.hline(HLine::new(pt).color(Color32::BLACK).width(1.0));
                }

                let conflicts = occs.conflicting_resource_set_idx >= 0;
                for occ in occs.occupations.iter() {
                    plot_ui.polygon(
                        Polygon::new(PlotPoints::from_iter([
                            [occ.interval.time_start as f64, draw.p_a[0] as f64],
                            [occ.interval.time_end as f64, draw.p_a[0] as f64],
                            [occ.interval.time_end as f64, draw.p_b[0] as f64],
                            [occ.interval.time_start as f64, draw.p_b[0] as f64],
                        ]))
                        .color(if conflicts {
                            Color32::RED
                        } else {
                            Color32::BLACK
                        })
                        .fill_alpha(0.5),
                    );
                }
            }

            for (_train, train_solver) in self
                .model
                .problem
                .trains
                .iter()
                .zip(self.model.solver.trains.iter())
            {
                let mut curr = &train_solver.current_node;
                while let Some(prev) = curr.parent.as_ref() {
                    if let Some(prev_draw) =
                        (prev.track >= 0).then(|| &self.model.draw_tracks[prev.track as usize])
                    {
                        plot_ui.line(
                            Line::new(PlotPoints::from_iter([
                                [prev.time as f64, prev_draw.p_a[0] as f64],
                                [curr.time as f64, prev_draw.p_b[0] as f64],
                            ]))
                            .color(Color32::BLACK)
                            .width(2.0),
                        );
                    }

                    curr = prev;
                }
            }
        });
    }

    fn plot_infrastructure(&mut self, ui: &mut egui::Ui) {
        let plot = egui::plot::Plot::new("infr").data_aspect(1.0).height(200.0);
        plot.show(ui, |plot_ui| {
            for (_track, draw) in self
                .model
                .problem
                .tracks
                .iter()
                .zip(self.model.draw_tracks.iter())
            {
                plot_ui.line(
                    Line::new(PlotPoints::from_iter([draw.p_a, draw.p_b]))
                        .color(Color32::BLACK)
                        .width(2.0),
                );

                plot_ui.text(Text::new(PlotPoint::from(draw.p_a), &draw.name));
            }
        });
    }
}

impl eframe::App for App {
    fn update(&mut self, ctx: &eframe::egui::Context, _frame: &mut eframe::Frame) {
        // self.get_messages();
        self.draw_gui(ctx);

        ctx.request_repaint_after(std::time::Duration::from_secs_f32(1.0));
    }
}
