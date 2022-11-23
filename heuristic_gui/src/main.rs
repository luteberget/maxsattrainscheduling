use eframe::egui::Visuals;
use heuristic::problem;

mod app;
pub mod examples;
mod model;

fn main() {
    pretty_env_logger::init();

    let input: problem::Problem =
        serde_json::from_str(&std::fs::read_to_string("origA1_rh.json").unwrap()).unwrap();

    // let input = examples::example_1();

    // let problem = Rc::new(input.problem);
    // let draw_tracks = Rc::new(input.draw_tracks);

    let app = app::App {
        model: model::Model {
            solver: heuristic::solver::ConflictSolver::new(input),
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
