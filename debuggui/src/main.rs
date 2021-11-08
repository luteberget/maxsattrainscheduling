use backend_glfw::imgui::{
    self, igBeginTooltip, igEndTooltip, igGetColorU32, ImDrawList_AddLine, ImDrawList_AddText,
    ImGuiCol__ImGuiCol_PlotHistogramHovered, ImGuiCol__ImGuiCol_PlotLines,
    ImGuiCol__ImGuiCol_PlotLinesHovered, ImGuiCol__ImGuiCol_TabActive, ImGuiCol__ImGuiCol_Text,
    ImVec2,
};
use ddd::{debug::*, problem};
use std::{collections::HashMap, sync::{mpsc, Arc}, thread};

mod widgets;

struct ProblemContext<'a> {
    problem: &'a ddd::problem::Problem,
    train_names: Vec<String>,
    res_names: Vec<String>,
}

fn main() {
    let mut frame_idx = -1isize;
    let mut frames = Vec::new();
    let (tx, rx) = mpsc::channel();
    let filename = "../instances/Instance1.xml";
    println!("Loading problem {}", filename);

    let (problem, train_names, res_names) = ddd::parser::read_file(filename);
    let problem = Arc::new(problem);

    {
        let problem = problem.clone();
        thread::spawn(move || {
            // let problem = problem::problem1();
            let result = ddd::solver::solve_debug(satcoder::solvers::minisat::Solver::new(), &problem, |d| tx.send(d).unwrap()).unwrap();

            println!("Finished. result={:?}", result);
        });
    }

    let problem_context = ProblemContext {
        problem: &problem,
        res_names,
        train_names,
    };

    backend_glfw::backend("ddd debug", None, 12.0, |action| {
        while let Ok(d) = rx.try_recv() {
            frames.push(d);
        }

        if let backend_glfw::SystemAction::Close = action {
            return false;
        }

        widgets::in_root_window(|| {
            if unsafe {
                imgui::igButton(
                    const_cstr::const_cstr!("<<").as_ptr(),
                    imgui::ImVec2::zero(),
                )
            } {
                frame_idx = 0;
            }

            unsafe { imgui::igSameLine(0., -1.) };
            if unsafe {
                imgui::igButton(const_cstr::const_cstr!("<").as_ptr(), imgui::ImVec2::zero())
            } {
                frame_idx -= 1;
            }
            unsafe { imgui::igSameLine(0., -1.) };

            if unsafe {
                imgui::igButton(const_cstr::const_cstr!(">").as_ptr(), imgui::ImVec2::zero())
            } {
                frame_idx += 1;
            }

            unsafe { imgui::igSameLine(0., -1.) };
            if unsafe {
                imgui::igButton(
                    const_cstr::const_cstr!(">>").as_ptr(),
                    imgui::ImVec2::zero(),
                )
            } {
                frame_idx = frames.len() as isize - 1;
            }
            unsafe { imgui::igSameLine(0., -1.) };

            if frame_idx < -1 {
                frame_idx = -1
            }
            if frame_idx >= frames.len() as isize {
                frame_idx = frames.len() as isize - 1;
            }

            widgets::show_text(&format!("frames={}/{}", frame_idx, frames.len()));
            unsafe { imgui::igSameLine(0., -1.) };

            if let Some(frame) = (frame_idx >= 0)
                .then(|| frames.get(frame_idx as usize))
                .flatten()
            {

                let mut count_action_types :HashMap<&str,usize>= HashMap::new();
                for action in frame.actions.iter() {
                    match action {
                        SolverAction::TravelTimeConflict(_) => {*count_action_types.entry("travel").or_default() += 1 },
                        SolverAction::ResourceConflict(_, _) => {*count_action_types.entry("resource").or_default() += 1 },
                        SolverAction::Core(_sz) => {*count_action_types.entry("core").or_default() += 1 },
                    }
                }
                widgets::show_text(&format!("{:?}", count_action_types));

                display_frame(&problem_context, frame);
            }
        });

        return true;
    })
    .unwrap();
}
use const_cstr::const_cstr;
use itertools::Itertools;
fn display_frame(problem_context: &ProblemContext, frame: &DebugInfo) {
    let size = unsafe { imgui::igGetContentRegionAvail_nonUDT2().into() };
    let draw = crate::widgets::canvas(size, 0, const_cstr!("draw").as_ptr());

    draw.begin_draw();

    let (min_x, max_x) = frame
        .solution
        .iter()
        .flatten()
        .minmax()
        .into_option()
        .map(|(x, y)| (*x, *y))
        .unwrap();
    let (min_y, max_y) = (0isize, problem_context.res_names.len() as isize);
    let xrel = |x| (x as f32 - min_x as f32) / (max_x as f32 - min_x as f32);
    let yrel = |y| (y as f32 - min_y as f32) / (max_y as f32 - min_y as f32);

    for y in min_y..max_y {
        let a = draw.relative_pt(xrel(min_x), yrel(y));
        let b = draw.relative_pt(xrel(max_x), yrel(y));

        let yp1 = draw.relative_pt(xrel(min_x), yrel(y + 1));

        // println!("Drawing {:?} {:?}", a,b );
        let col = unsafe { igGetColorU32(ImGuiCol__ImGuiCol_PlotLines as _, 0.75) };
        let hoveredcol = unsafe { igGetColorU32(ImGuiCol__ImGuiCol_PlotLinesHovered as _, 0.75) };
        let hovered = draw.pos.y + draw.mouse.y >= a.y && draw.pos.y + draw.mouse.y <= yp1.y;
        // println!("mouse {} min {} max {}", draw.pos.y + draw.mouse.y, a.y.min(b.y), a.y.max(b.y));
        let col = if hovered { hoveredcol } else { col };
        unsafe {
            // println!("draw {:?} {:?}, {:?}", a, b, col);
            ImDrawList_AddLine(draw.draw_list, a, b, col, 2.);
        }

        let text = problem_context.res_names[y as usize]
            .as_bytes()
            .as_ptr_range();
        unsafe {
            ImDrawList_AddText(
                draw.draw_list,
                a,
                col,
                text.start as *const _,
                text.end as *const _,
            );
        }
    }

    let mut closest: Option<(f32, (usize, usize))> = None;
    for (train_idx, train_times) in frame.solution.iter().enumerate() {
        let r1 = problem_context.problem.trains[train_idx].visits[0].0;
        let r2 = problem_context.problem.trains[train_idx].visits[1].0;
        let dir = if r1 < r2 { 1isize } else { -1 };

        for ((visit_idx, time1), (_, time2)) in train_times
            .iter()
            .enumerate()
            .zip(train_times.iter().enumerate().skip(1))
        {
            // println!("xmin {} xmax {} x {} xrel {}", min_x, max_x, *time1, xrel(*time1));

            let visit_resource = problem_context.problem.trains[train_idx].visits[visit_idx].0;
            let (resource1, resource2) = if r1 < r2 {
                (visit_resource as isize, visit_resource as isize + 1)
            } else {
                (visit_resource as isize + 1, visit_resource as isize)
            };

            let a = draw.relative_pt(xrel(*time1), yrel(resource1));
            let b = draw.relative_pt(xrel(*time2), yrel(resource2));

            let middle = ImVec2 {
                x: 0.5 * (a.x + b.x) - draw.pos.x,
                y: 0.5 * (a.y + b.y) - draw.pos.y,
            };
            let dx = middle.x - draw.mouse.x;
            let dy = middle.y - draw.mouse.y;
            let dist = dx * dx + dy * dy;
            if closest.is_none() || closest.as_ref().unwrap().0 > dist {
                closest = Some((dist, (train_idx, visit_idx)));
            }

            // println!("Drawing {:?} {:?}", a,b );
            unsafe {
                let col = igGetColorU32(ImGuiCol__ImGuiCol_PlotLinesHovered as _, 0.75);
                // println!("draw {:?} {:?}, {:?}", a, b, col);
                ImDrawList_AddLine(draw.draw_list, a, b, col, 2.);
            }
        }
    }

    if let Some((_, (train_idx, visit_idx))) = closest {
        let r1 = problem_context.problem.trains[train_idx].visits[0].0;
        let r2 = problem_context.problem.trains[train_idx].visits[1].0;
        let dir = if r1 < r2 { 1isize } else { -1 };
        let visit_resource = problem_context.problem.trains[train_idx].visits[visit_idx].0;

        let (resource1, resource2) = if r1 < r2 {
            (visit_resource as isize, visit_resource as isize + 1)
        } else {
            (visit_resource as isize + 1, visit_resource as isize)
        };

        let time1 = &frame.solution[train_idx][visit_idx];
        let time2 = &frame.solution[train_idx][visit_idx + 1];

        let a = draw.relative_pt(xrel(*time1), yrel(resource1));
        let b = draw.relative_pt(xrel(*time2), yrel(resource2));

        // println!("Drawing {:?} {:?}", a,b );
        unsafe {
            let col = igGetColorU32(ImGuiCol__ImGuiCol_TabActive as _, 0.75);
            // println!("draw {:?} {:?}, {:?}", a, b, col);
            ImDrawList_AddLine(draw.draw_list, a, b, col, 2.);
        }

        unsafe { igBeginTooltip() };

        widgets::show_text(&format!(
            "{} @ {} dir{}",
            problem_context.train_names[train_idx], problem_context.res_names[visit_resource], dir
        ));

        unsafe { igEndTooltip() };
    }

    draw.end_draw();
}
