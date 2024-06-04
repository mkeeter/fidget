use anyhow::Result;
use clap::Parser;
use eframe::egui;
use egui::{emath, pos2, Pos2, Rect, Sense, Shape, Vec2};
use env_logger::Env;
use fidget::eval::MathFunction;
use log::{info, trace};
use std::collections::HashMap;

/// Minimal demo of constraint solving
#[derive(Parser, Debug)]
#[clap(author, version, about, long_about = None)]
struct Args;

/// Container holding an abstract variable and its current value
#[derive(Copy, Clone, Debug)]
struct Value {
    var: fidget::var::Var,
    cur: f32,
}

impl Value {
    fn new(cur: f32) -> Self {
        let var = fidget::var::Var::new();
        Value { var, cur }
    }
}

/// Container holding two values
#[derive(Copy, Clone, Debug)]
struct Point {
    x: Value,
    y: Value,
}

impl Point {
    fn new(x: f32, y: f32) -> Self {
        Point {
            x: Value::new(x),
            y: Value::new(y),
        }
    }
}

#[derive(Default)]
struct ConstraintsApp {
    /// Handles to variables within our equations
    points: Vec<Point>,

    /// Built-in constraints
    constraints: Vec<fidget::vm::VmFunction>,
}

impl ConstraintsApp {
    fn new() -> Self {
        let points = vec![
            Point::new(50.0, 50.0),
            Point::new(100.0, 50.0),
            Point::new(200.0, 200.0),
            Point::new(250.0, 50.0),
        ];

        let mut ctx = fidget::Context::new();
        let mut constraints = vec![];

        // add a constraint matching the Y position of two values
        let w = ctx.sub(points[0].y.var, points[1].y.var).unwrap();
        let f = fidget::vm::VmFunction::new(&ctx, w).unwrap();
        constraints.push(f);

        // Add a constraint that point 2 must have the same X and Y values
        let w = ctx.sub(points[2].x.var, points[2].y.var).unwrap();
        let f = fidget::vm::VmFunction::new(&ctx, w).unwrap();
        constraints.push(f);

        Self {
            points,
            constraints,
        }
    }
}

impl eframe::App for ConstraintsApp {
    fn update(&mut self, ctx: &egui::Context, _frame: &mut eframe::Frame) {
        egui::CentralPanel::default().show(ctx, |ui| {
            // The central panel the region left after adding TopPanel's and SidePanel's
            ui.heading("eframe template");

            ui.horizontal(|ui| {
                ui.label("Write something: ");
            });

            let (response, painter) = ui.allocate_painter(
                Vec2::new(ui.available_width(), 300.0),
                Sense::hover(),
            );

            let to_screen = emath::RectTransform::from_to(
                Rect::from_min_size(Pos2::ZERO, response.rect.size()),
                response.rect,
            );

            let control_point_radius = 8.0;

            // Do interaction, storing target positions
            let mut changed = false;
            let r = self
                .points
                .iter_mut()
                .enumerate()
                .map(|(i, point)| {
                    let size = Vec2::splat(2.0 * control_point_radius);

                    let point_pos = pos2(point.x.cur, point.y.cur);
                    let point_in_screen = to_screen.transform_pos(point_pos);
                    let point_rect =
                        Rect::from_center_size(point_in_screen, size);
                    let point_id = response.id.with(i);
                    let point_response =
                        ui.interact(point_rect, point_id, Sense::drag());
                    let stroke = ui.style().interact(&point_response).fg_stroke;

                    let dragged = point_response.dragged();
                    changed |= dragged;

                    let new_pos = to_screen
                        .from()
                        .clamp(point_pos + point_response.drag_delta());
                    (stroke, new_pos, dragged)
                })
                .collect::<Vec<_>>();

            // Constraint solving!
            if changed {
                trace!("points have changed, running constraint solver");
                // Initial values for parameters
                let mut parameters = HashMap::new();

                // Do an initial solve that forces the point to the cursor
                for (point, (_, pos, dragged)) in self.points.iter().zip(&r) {
                    for (var, p) in [(point.x.var, pos.x), (point.y.var, pos.y)]
                    {
                        parameters.insert(
                            var,
                            if *dragged {
                                fidget::solver::Parameter::Fixed(p)
                            } else {
                                fidget::solver::Parameter::Free(p)
                            },
                        );
                    }
                }
                let sol = fidget::solver::solve(&self.constraints, &parameters)
                    .unwrap();
                // Update positions, either to the drag pos or the solver pos
                for (point, (_, pos, dragged)) in self.points.iter_mut().zip(&r)
                {
                    if *dragged {
                        point.x.cur = pos.x;
                        point.y.cur = pos.y;
                    } else {
                        for (var, p) in [
                            (point.x.var, &mut point.x.cur),
                            (point.y.var, &mut point.y.cur),
                        ] {
                            if let Some(v) = sol.get(&var) {
                                *p = *v;
                            }
                        }
                    }
                }

                // A second solve which frees the dragged variable to fully
                // minimize the constraints
                let mut parameters = HashMap::new();
                for point in self.points.iter() {
                    for v in [point.x, point.y] {
                        parameters.insert(
                            v.var,
                            fidget::solver::Parameter::Free(v.cur),
                        );
                    }
                }
                let sol = fidget::solver::solve(&self.constraints, &parameters)
                    .unwrap();
                for point in self.points.iter_mut() {
                    for (var, p) in [
                        (point.x.var, &mut point.x.cur),
                        (point.y.var, &mut point.y.cur),
                    ] {
                        if let Some(v) = sol.get(&var) {
                            *p = *v;
                        }
                    }
                }
            }

            // Draw the points
            painter.extend(self.points.iter().zip(&r).map(
                |(point, (stroke, _, _))| {
                    let point_in_screen =
                        to_screen.transform_pos(pos2(point.x.cur, point.y.cur));

                    Shape::circle_stroke(
                        point_in_screen,
                        control_point_radius,
                        *stroke,
                    )
                },
            ));

            response
        });
    }
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    env_logger::Builder::from_env(Env::default().default_filter_or("info"))
        .init();
    let _args = Args::parse();

    let options = eframe::NativeOptions::default();
    info!("starting...");
    eframe::run_native(
        "Fidget",
        options,
        Box::new(move |_cc| Box::new(ConstraintsApp::new())),
    )?;

    Ok(())
}
