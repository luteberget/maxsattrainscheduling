use eframe::{
    egui::{
        self,
        plot::{HLine, Line, PlotPoint, PlotPoints, Text},
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
                .default_width(500.0)
                .show_inside(ui, |ui| {
                    ui.heading("Conflict solver");
                    ui.label(&format!("Status: {:?}", self.model.solver.status()));
                    ui.label(&format!(
                        "Total conflicts: {}",
                        self.model
                            .solver
                            .conflicts
                            .iter()
                            .map(|c| c.conflicts.len())
                            .sum::<usize>()
                    ));
                });
            egui::SidePanel::left("left_panel2")
                .default_width(500.0)
                .show_inside(ui, |ui| {
                    ui.heading("Train solver");

                    egui::ComboBox::from_label("Select one!")
                        .selected_text(format!("Train {}", self.model.selected_train))
                        .show_ui(ui, |ui| {
                            for i in 0..self.model.problem.trains.len() {
                                ui.selectable_value(
                                    &mut self.model.selected_train,
                                    i,
                                    format!("Train {}", i),
                                );
                            }
                        });
                });

            ui.heading("infrastructure");
            self.plot_infrastructure(ui);

            ui.heading("train graph");
            self.plot_traingraph(ui);
        });
    }

    fn plot_traingraph(&mut self, ui: &mut egui::Ui) {
        let plot = egui::plot::Plot::new("grph");
        plot.show(ui, |plot_ui| {
            for (_track, draw) in self
                .model
                .problem
                .tracks
                .iter()
                .zip(self.model.draw_tracks.iter())
            {
                for pt in [draw.p_a[0], draw.p_b[0]] {
                    plot_ui.hline(HLine::new(pt).color(Color32::BLACK).width(1.0));
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
