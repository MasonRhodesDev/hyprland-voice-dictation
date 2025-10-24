use keyframe::{ease, functions::*};
use std::time::Instant;

const HEIGHT_TRANSITION_DURATION: f32 = 0.2;
const FADE_DURATION: f32 = 0.3;
const COLLAPSE_DURATION: f32 = 0.5;

#[derive(Clone, Copy)]
pub struct HeightAnimation {
    start_height: f32,
    target_height: f32,
    start_time: Instant,
}

impl HeightAnimation {
    pub fn new(start: f32, target: f32) -> Self {
        Self { start_height: start, target_height: target, start_time: Instant::now() }
    }

    pub fn current_value(&self) -> f32 {
        let elapsed = self.start_time.elapsed().as_secs_f32();
        let progress = (elapsed / HEIGHT_TRANSITION_DURATION).min(1.0);

        let eased_progress = ease(EaseOutCubic, 0.0, 1.0, progress);

        self.start_height + (self.target_height - self.start_height) * eased_progress
    }

    pub fn is_complete(&self) -> bool {
        self.start_time.elapsed().as_secs_f32() >= HEIGHT_TRANSITION_DURATION
    }
}

#[derive(Clone, Copy)]
pub struct FadeAnimation {
    start_time: Instant,
    fade_in: bool,
}

impl FadeAnimation {
    pub fn fade_in() -> Self {
        Self { start_time: Instant::now(), fade_in: true }
    }

    pub fn fade_out() -> Self {
        Self { start_time: Instant::now(), fade_in: false }
    }

    pub fn current_alpha(&self) -> f32 {
        let elapsed = self.start_time.elapsed().as_secs_f32();
        let progress = (elapsed / FADE_DURATION).min(1.0);

        let eased_progress = ease(EaseInOutQuad, 0.0, 1.0, progress);

        if self.fade_in {
            eased_progress
        } else {
            1.0 - eased_progress
        }
    }

    pub fn is_complete(&self) -> bool {
        self.start_time.elapsed().as_secs_f32() >= FADE_DURATION
    }
}

#[derive(Clone, Copy)]
pub struct CollapseAnimation {
    start_time: Instant,
}

impl CollapseAnimation {
    pub fn new() -> Self {
        Self { start_time: Instant::now() }
    }

    pub fn scale_factor(&self) -> f32 {
        let elapsed = self.start_time.elapsed().as_secs_f32();
        let progress = (elapsed / COLLAPSE_DURATION).min(1.0);

        // Ease in cubic for smooth collapse
        let eased_progress = ease(EaseInCubic, 0.0, 1.0, progress);

        1.0 - eased_progress
    }

    pub fn is_complete(&self) -> bool {
        self.start_time.elapsed().as_secs_f32() >= COLLAPSE_DURATION
    }
}

pub fn ease_spinner_rotation(time: f32) -> f32 {
    // Smooth continuous rotation
    time * 2.0
}
