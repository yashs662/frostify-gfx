//! Scalar animation primitives.
//!
//! Stage-1 scope: tween `Signal<f32>` values over a timeline. Curves
//! cover the four shapes a real UI app needs — linear, ease-in-out,
//! cubic-bezier (css-style timing), and a 1-DOF damped spring. Vector
//! quantities (colors, positions) are driven by derivation: the app
//! reads the scalar inside a refresh closure and pushes the derived
//! value through the tracked setter.
//!
//! The timeline does not touch winit or the tree. It exposes
//! `tick(now)` which returns a `TickResult` containing `updated` (any
//! signal value actually changed) and `next_deadline` (the recommended
//! `ControlFlow::WaitUntil` target). Caller wiring:
//!
//! ```ignore
//! let res = timeline.tick(Instant::now());
//! if res.updated {
//!     refresh_derived_values();
//!     flush_tree();
//!     request_redraw();
//! }
//! match res.next_deadline {
//!     Some(deadline) => event_loop.set_control_flow(WaitUntil(deadline)),
//!     None => event_loop.set_control_flow(Wait),
//! }
//! ```

use std::time::{Duration, Instant};

use crate::signal::Signal;

/// Easing curves. All stored by value — `Copy`.
///
/// `Spring` ignores the tween duration except as a hard safety cap;
/// the motion is driven by elapsed seconds through a closed-form
/// damped harmonic oscillator with unit mass. Use stiffness ~150 and
/// damping ~18 for a snappy UI spring, or ~80/12 for a softer one.
#[derive(Copy, Clone, Debug)]
pub enum Curve {
    Linear,
    EaseInOut,
    /// CSS-style cubic bezier timing: `[x1, y1, x2, y2]`. Control
    /// points (0,0) and (1,1) are implicit.
    CubicBezier([f32; 4]),
    Spring {
        stiffness: f32,
        damping: f32,
    },
}

impl Curve {
    /// Evaluate at normalized `t ∈ [0,1]`. Panics for `Spring`, which
    /// is evaluated by elapsed seconds inside `Tween::sample` instead.
    pub fn eval(&self, t: f32) -> f32 {
        match *self {
            Curve::Linear => t.clamp(0.0, 1.0),
            Curve::EaseInOut => {
                let t = t.clamp(0.0, 1.0);
                t * t * (3.0 - 2.0 * t)
            }
            Curve::CubicBezier([x1, y1, x2, y2]) => {
                let t = t.clamp(0.0, 1.0);
                cubic_bezier_y_at_x(x1, y1, x2, y2, t)
            }
            Curve::Spring { .. } => unreachable!("spring sampled via elapsed seconds"),
        }
    }
}

/// Solve `Bx(t) = x` for `t` via Newton's method, then evaluate `By(t)`.
///
/// Control points are `(0,0)`, `(x1,y1)`, `(x2,y2)`, `(1,1)` — i.e.
/// standard CSS timing function form. 8 iterations is plenty: the
/// function is smooth and almost always converges in 3–4.
fn cubic_bezier_y_at_x(x1: f32, y1: f32, x2: f32, y2: f32, x: f32) -> f32 {
    let bx = |t: f32| {
        let it = 1.0 - t;
        3.0 * it * it * t * x1 + 3.0 * it * t * t * x2 + t * t * t
    };
    let by = |t: f32| {
        let it = 1.0 - t;
        3.0 * it * it * t * y1 + 3.0 * it * t * t * y2 + t * t * t
    };
    let dbx = |t: f32| {
        let it = 1.0 - t;
        3.0 * it * it * x1 + 6.0 * it * t * (x2 - x1) + 3.0 * t * t * (1.0 - x2)
    };
    let mut t = x;
    for _ in 0..8 {
        let err = bx(t) - x;
        if err.abs() < 1e-4 {
            break;
        }
        let slope = dbx(t);
        if slope.abs() < 1e-6 {
            break;
        }
        t = (t - err / slope).clamp(0.0, 1.0);
    }
    by(t)
}

/// Closed-form unit-step response of a 1-DOF mass-spring-damper with
/// unit mass, initial position 0, initial velocity 0, target 1.
/// Returns `(position, velocity)` at elapsed time `t`.
fn spring_eval(k: f32, c: f32, t: f32) -> (f32, f32) {
    let w0 = k.max(1e-6).sqrt();
    let zeta = c / (2.0 * w0);
    if (zeta - 1.0).abs() < 1e-4 {
        // Critically damped.
        let e = (-w0 * t).exp();
        let x = 1.0 - e * (1.0 + w0 * t);
        let v = e * (w0 * w0 * t);
        (x, v)
    } else if zeta < 1.0 {
        // Underdamped — the UI default.
        let wd = w0 * (1.0 - zeta * zeta).sqrt();
        let a = zeta * w0;
        let e = (-a * t).exp();
        let cos = (wd * t).cos();
        let sin = (wd * t).sin();
        let x = 1.0 - e * (cos + (a / wd) * sin);
        let v = (w0 * w0 / wd) * e * sin;
        (x, v)
    } else {
        // Overdamped.
        let disc = (zeta * zeta - 1.0).sqrt();
        let r1 = -w0 * (zeta - disc);
        let r2 = -w0 * (zeta + disc);
        let a1 = -r2 / (r1 - r2);
        let a2 = r1 / (r1 - r2);
        let x = 1.0 - a1 * (r1 * t).exp() - a2 * (r2 * t).exp();
        let v = -a1 * r1 * (r1 * t).exp() - a2 * r2 * (r2 * t).exp();
        (x, v)
    }
}

/// One in-flight tween. Owned by a `Timeline`.
#[derive(Clone, Debug)]
pub struct Tween {
    /// User-facing tag. `Timeline::start` replaces any existing tween
    /// with the same key, which is how hover-on/hover-off interrupt
    /// each other smoothly.
    pub key: u32,
    pub signal: Signal<f32>,
    pub from: f32,
    pub to: f32,
    pub curve: Curve,
    pub start: Instant,
    pub duration: Duration,
}

impl Tween {
    /// Returns `(value, done)` at `now`. The caller snaps to `to` when
    /// `done` is true.
    fn sample(&self, now: Instant) -> (f32, bool) {
        let elapsed = now.saturating_duration_since(self.start);
        match self.curve {
            Curve::Spring { stiffness, damping } => {
                let t = elapsed.as_secs_f32();
                let (x, vel) = spring_eval(stiffness, damping, t);
                let val = self.from + (self.to - self.from) * x;
                let settled = (x - 1.0).abs() < 1e-3 && vel.abs() < 1e-3;
                (val, settled || elapsed >= self.duration)
            }
            _ => {
                let total = self.duration.as_secs_f32().max(1e-6);
                let t_norm = elapsed.as_secs_f32() / total;
                let done = t_norm >= 1.0;
                let eased = self.curve.eval(t_norm.clamp(0.0, 1.0));
                let val = self.from + (self.to - self.from) * eased;
                (val, done)
            }
        }
    }
}

/// Result of one `Timeline::tick`.
#[derive(Copy, Clone, Debug, Default)]
pub struct TickResult {
    /// True if any tween's signal actually changed value.
    pub updated: bool,
    /// Recommended wake-up time. `None` when the timeline is idle —
    /// the caller should fall back to `ControlFlow::Wait`.
    pub next_deadline: Option<Instant>,
}

/// Collection of running tweens. The tick cadence is fixed at ~16 ms
/// (≈60 Hz) in stage 1. Once GPU timestamps + adaptive vsync land in
/// M7/M9 this can become dynamic.
pub struct Timeline {
    tweens: Vec<Tween>,
    tick_interval: Duration,
}

impl Timeline {
    pub fn new() -> Self {
        Self {
            tweens: Vec::new(),
            tick_interval: Duration::from_millis(16),
        }
    }

    pub fn active(&self) -> bool {
        !self.tweens.is_empty()
    }

    pub fn len(&self) -> usize {
        self.tweens.len()
    }

    pub fn is_empty(&self) -> bool {
        self.tweens.is_empty()
    }

    pub fn tick_interval(&self) -> Duration {
        self.tick_interval
    }

    pub fn set_tick_interval(&mut self, dt: Duration) {
        self.tick_interval = dt;
    }

    /// Start (or restart) a tween for `key`. `from` is read from the
    /// signal's current value so mid-flight interrupts land smoothly.
    /// No-op if `from == to`.
    pub fn start(
        &mut self,
        key: u32,
        signal: Signal<f32>,
        to: f32,
        curve: Curve,
        duration: Duration,
        now: Instant,
    ) {
        self.tweens.retain(|t| t.key != key);
        let from = signal.get();
        if from == to {
            return;
        }
        self.tweens.push(Tween {
            key,
            signal,
            from,
            to,
            curve,
            start: now,
            duration,
        });
    }

    pub fn stop(&mut self, key: u32) {
        self.tweens.retain(|t| t.key != key);
    }

    pub fn clear(&mut self) {
        self.tweens.clear();
    }

    /// Advance every tween to `now`. Writes to each tween's signal
    /// via `Signal::set` (which deduplicates no-op writes). Completed
    /// tweens snap to `to` and are removed.
    pub fn tick(&mut self, now: Instant) -> TickResult {
        let mut updated = false;
        self.tweens.retain_mut(|tw| {
            let (val, done) = tw.sample(now);
            let final_val = if done { tw.to } else { val };
            if tw.signal.set(final_val) {
                updated = true;
            }
            !done
        });
        let next_deadline = if self.tweens.is_empty() {
            None
        } else {
            Some(now + self.tick_interval)
        };
        TickResult {
            updated,
            next_deadline,
        }
    }
}

impl Default for Timeline {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn linear_end_to_end() {
        let s = Signal::new(0.0f32);
        let mut tl = Timeline::new();
        let now = Instant::now();
        tl.start(
            0,
            s.clone(),
            1.0,
            Curve::Linear,
            Duration::from_millis(100),
            now,
        );
        assert!(tl.active());
        let mid = tl.tick(now + Duration::from_millis(50));
        assert!(mid.updated);
        assert!((s.get() - 0.5).abs() < 1e-3);
        let end = tl.tick(now + Duration::from_millis(120));
        assert!(end.updated);
        assert!(!tl.active());
        assert_eq!(s.get(), 1.0);
    }

    #[test]
    fn ease_in_out_monotonic() {
        let c = Curve::EaseInOut;
        let mut prev = -1.0;
        for i in 0..=10 {
            let y = c.eval(i as f32 / 10.0);
            assert!(y >= prev);
            prev = y;
        }
        assert!((c.eval(0.0)).abs() < 1e-4);
        assert!((c.eval(1.0) - 1.0).abs() < 1e-4);
    }

    #[test]
    fn cubic_bezier_endpoints() {
        let c = Curve::CubicBezier([0.42, 0.0, 0.58, 1.0]);
        assert!(c.eval(0.0).abs() < 1e-3);
        assert!((c.eval(1.0) - 1.0).abs() < 1e-3);
    }

    #[test]
    fn spring_settles_near_target() {
        // Underdamped spring should hit ~1.0 within a few seconds.
        let (x, _v) = spring_eval(180.0, 20.0, 2.0);
        assert!((x - 1.0).abs() < 1e-2);
    }

    #[test]
    fn restart_picks_up_current_value() {
        let s = Signal::new(0.0f32);
        let mut tl = Timeline::new();
        let now = Instant::now();
        tl.start(
            0,
            s.clone(),
            1.0,
            Curve::Linear,
            Duration::from_millis(100),
            now,
        );
        tl.tick(now + Duration::from_millis(40));
        let mid_val = s.get();
        // Reverse mid-flight.
        tl.start(
            0,
            s.clone(),
            0.0,
            Curve::Linear,
            Duration::from_millis(100),
            now + Duration::from_millis(40),
        );
        assert_eq!(tl.len(), 1);
        // New tween starts from `mid_val`, not 1.0.
        let tw = &tl.tweens[0];
        assert_eq!(tw.from, mid_val);
        assert_eq!(tw.to, 0.0);
    }
}
