use backend_glfw::imgui::*;
use const_cstr::const_cstr;

pub fn show_text(s: &str) {
    unsafe {
        igTextSlice(s.as_ptr() as _, s.as_ptr().offset(s.len() as _) as _);
    }
}

pub fn in_root_window(f: impl FnOnce()) {
    unsafe {
        let zero = ImVec2 { x: 0.0, y: 0.0 };
        let io = igGetIO();
        igSetNextWindowPos(zero, ImGuiCond__ImGuiCond_Always as _, zero);
        igSetNextWindowSize((*io).DisplaySize, ImGuiCond__ImGuiCond_Always as _);
        igPushStyleVarFloat(ImGuiStyleVar__ImGuiStyleVar_WindowRounding as _, 0.0);
        let win_flags = ImGuiWindowFlags__ImGuiWindowFlags_NoTitleBar
            | ImGuiWindowFlags__ImGuiWindowFlags_NoCollapse
            | ImGuiWindowFlags__ImGuiWindowFlags_NoResize
            | ImGuiWindowFlags__ImGuiWindowFlags_NoMove
            | ImGuiWindowFlags__ImGuiWindowFlags_NoBringToFrontOnFocus
            | ImGuiWindowFlags__ImGuiWindowFlags_NoNavFocus
            | ImGuiWindowFlags__ImGuiWindowFlags_MenuBar;
        igBegin(
            const_cstr!("root").as_ptr(),
            std::ptr::null_mut(),
            win_flags as _,
        );
        f();
        igEnd();
        igPopStyleVar(1);
    }
}

pub struct Draw {
    pub draw_list: *mut ImDrawList,
    pub pos: ImVec2,
    pub size: ImVec2,
    pub mouse: ImVec2,
}

impl Draw {
    pub fn begin_draw(&self) {
        unsafe {
            ImDrawList_PushClipRect(self.draw_list, self.pos, self.pos + self.size, true);
        }
    }

    pub fn end_draw(&self) {
        unsafe {
            ImDrawList_PopClipRect(self.draw_list);
        }
    }

    pub fn relative_pt(&self, x: f32, y: f32) -> ImVec2 {
        ImVec2 {
            x: self.pos.x + (self.size.x) * x,
            y: self.pos.y + (self.size.y) * y,
        }
    }
}

pub fn canvas(mut size: ImVec2, color: u32, name: *const i8) -> Draw {
    unsafe {
        let pos: ImVec2 = igGetCursorScreenPos_nonUDT2().into();
        let draw_list = igGetWindowDrawList();
        ImDrawList_AddRectFilled(draw_list, pos, pos + size, color, 0.0, 0);
        size.y -= igGetFrameHeightWithSpacing() - igGetFrameHeight();
        let _clicked = igInvisibleButton(name, size);
        igSetItemAllowOverlap();
        let mouse = (*igGetIO()).MousePos - pos;
        Draw {
            pos,
            size,
            draw_list,
            mouse,
        }
    }
}
