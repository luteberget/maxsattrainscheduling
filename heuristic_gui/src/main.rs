use std::rc::Rc;

use eframe::egui::Visuals;

mod app;
mod model;

fn main() {

    pretty_env_logger::init();

    let input: model::Input =
        serde_json::from_str(&std::fs::read_to_string("example1.json").unwrap()).unwrap();

    if input.symmetric {
        todo!();
    }
    
    if input.problem.tracks.len() != input.draw_tracks.len() {
        panic!("draw tracks length");
    }

    let problem = Rc::new(input.problem);
    let draw_tracks = Rc::new(input.draw_tracks);

    let app = app::App {
        model: model::Model {
            solver: heuristic::solver::ConflictSolver::new(problem.clone()),
            problem,
            draw_tracks,
            selected_train: 0,
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
