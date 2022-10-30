// Based on the egui demo library:
// https://github.com/emilk/egui/blob/master/crates/egui_demo_lib/src/syntax_highlighting.rs
//
// The original source is released under MIT OR Apache-2.0,
// and copyright (c) 2018-2021 Emil Ernerfeldt.
use eframe::egui::{self, text::LayoutJob, Color32, FontId, Response, Vec2};

const THEME_NAME: &str = "Solarized (dark)";

/// View some code with syntax highlighting and selection.
#[allow(clippy::ptr_arg)]
pub fn code_view_ui(
    ui: &mut egui::Ui,
    code: &mut String,
    size: Vec2,
) -> Response {
    let mut layouter = |ui: &egui::Ui, string: &str, _wrap_width: f32| {
        let layout_job = highlight(ui.ctx(), string);
        ui.fonts().layout_job(layout_job)
    };

    ui.add_sized(
        size,
        egui::TextEdit::multiline(code)
            .layouter(&mut layouter)
            .lock_focus(true),
    )
}

/// Memoized Code highlighting
pub fn highlight(ctx: &egui::Context, code: &str) -> LayoutJob {
    type HighlightCache = egui::util::cache::FrameCache<LayoutJob, Highlighter>;

    let mut memory = ctx.memory();
    let highlight_cache = memory.caches.cache::<HighlightCache>();
    highlight_cache.get(code)
}

pub fn get_theme() -> syntect::highlighting::Theme {
    let h = Highlighter::default();
    h.ts.themes[THEME_NAME].clone()
}

struct Highlighter {
    ps: syntect::parsing::SyntaxSet,
    ts: syntect::highlighting::ThemeSet,
}

impl egui::util::cache::ComputerMut<&str, LayoutJob> for Highlighter {
    fn compute(&mut self, code: &str) -> LayoutJob {
        self.highlight(code)
    }
}

impl Default for Highlighter {
    fn default() -> Self {
        use syntect::parsing::{SyntaxDefinition, SyntaxSetBuilder};
        let mut builder = SyntaxSetBuilder::new();
        builder.add(
            SyntaxDefinition::load_from_str(
                include_str!("../syntax/rhai.sublime-syntax"),
                true,
                None,
            )
            .unwrap(),
        );

        Self {
            ps: builder.build(),
            ts: syntect::highlighting::ThemeSet::load_defaults(),
        }
    }
}

impl Highlighter {
    fn highlight(&self, code: &str) -> LayoutJob {
        self.highlight_impl(code).unwrap_or_else(|| {
            // Fallback:
            LayoutJob::simple(
                code.into(),
                FontId::monospace(16.0),
                Color32::DARK_GRAY,
                f32::INFINITY,
            )
        })
    }

    fn highlight_impl(&self, text: &str) -> Option<LayoutJob> {
        use syntect::easy::HighlightLines;
        use syntect::highlighting::FontStyle;
        use syntect::util::LinesWithEndings;

        let syntax = self.ps.find_syntax_by_name("Rhai").unwrap();

        let mut h = HighlightLines::new(syntax, &self.ts.themes[THEME_NAME]);

        use egui::text::{LayoutSection, TextFormat};

        let mut job = LayoutJob {
            text: text.into(),
            ..Default::default()
        };

        for line in LinesWithEndings::from(text) {
            for (style, range) in h.highlight_line(line, &self.ps).ok()? {
                let fg = style.foreground;
                let text_color = egui::Color32::from_rgb(fg.r, fg.g, fg.b);
                let italics = style.font_style.contains(FontStyle::ITALIC);
                let underline = style.font_style.contains(FontStyle::ITALIC);
                let underline = if underline {
                    egui::Stroke::new(1.0, text_color)
                } else {
                    egui::Stroke::none()
                };
                job.sections.push(LayoutSection {
                    leading_space: 0.0,
                    byte_range: as_byte_range(text, range),
                    format: TextFormat {
                        font_id: egui::FontId::monospace(16.0),
                        color: text_color,
                        italics,
                        underline,
                        ..Default::default()
                    },
                });
            }
        }

        Some(job)
    }
}

fn as_byte_range(whole: &str, range: &str) -> std::ops::Range<usize> {
    let whole_start = whole.as_ptr() as usize;
    let range_start = range.as_ptr() as usize;
    assert!(whole_start <= range_start);
    assert!(range_start + range.len() <= whole_start + whole.len());
    let offset = range_start - whole_start;
    offset..(offset + range.len())
}
