#![cfg_attr(not(debug_assertions), windows_subsystem = "windows")] // hide console window on Windows in release

use eframe::egui::{
    self, widgets::text_edit::TextEdit, Align, FontId, Layout, Ui, Vec2,
};

fn main() {
    let options = eframe::NativeOptions::default();
    eframe::run_native(
        "fidget",
        options,
        Box::new(|_cc| Box::new(MyApp::default())),
    );
}

struct MyApp {
    script: String,
}

impl Default for MyApp {
    fn default() -> Self {
        Self {
            script: "// hello, world".to_owned(),
        }
    }
}

impl eframe::App for MyApp {
    fn update(&mut self, ctx: &egui::Context, _frame: &mut eframe::Frame) {
        egui::SidePanel::left("root").show(ctx, |ui| {
            ui.add_sized(
                ui.available_size(),
                TextEdit::multiline(&mut self.script)
                    .font(FontId::monospace(16.0)),
            );
        });
    }
}
