pub mod correction;
pub mod engine_runner;
pub mod recording;

use anyhow::{Context, Result};
use std::path::PathBuf;
use std::sync::{Arc, Mutex};

slint::include_modules!();

pub fn run(recording_path: Option<&str>) -> Result<()> {
    let recordings = recording::list_recordings().unwrap_or_default();

    let initial_index: usize = if let Some(path) = recording_path {
        let wav = PathBuf::from(path);
        recordings
            .iter()
            .position(|r| r.wav_path == wav)
            .unwrap_or(if recordings.is_empty() { 0 } else { recordings.len() - 1 })
    } else {
        if recordings.is_empty() {
            0
        } else {
            recordings.len() - 1
        }
    };

    let recordings = Arc::new(Mutex::new(recordings));
    let current_index = Arc::new(Mutex::new(initial_index));
    let loaded_recording: Arc<Mutex<Option<recording::Recording>>> = Arc::new(Mutex::new(None));

    let ui = TestLoop::new().context("creating Slint window")?;
    let ui_weak = ui.as_weak();

    // Populate picker with initial recording info
    {
        let recs = recordings.lock().unwrap();
        let count = recs.len() as i32;
        ui.set_recording_count(count);
        if count > 0 {
            let idx = *current_index.lock().unwrap();
            update_picker(&ui, &recs, idx);
        }
    }

    // Navigate callback
    {
        let recordings = Arc::clone(&recordings);
        let current_index = Arc::clone(&current_index);
        let ui_weak = ui_weak.clone();
        ui.on_navigate(move |delta| {
            let recs = recordings.lock().unwrap();
            if recs.is_empty() {
                return;
            }
            let mut idx = current_index.lock().unwrap();
            let count = recs.len();
            if delta < 0 {
                *idx = if *idx == 0 { count - 1 } else { *idx - 1 };
            } else {
                *idx = (*idx + 1) % count;
            }
            let i = *idx;
            drop(idx);
            if let Some(ui) = ui_weak.upgrade() {
                update_picker(&ui, &recs, i);
            }
        });
    }

    // Load recording callback
    {
        let recordings = Arc::clone(&recordings);
        let current_index = Arc::clone(&current_index);
        let loaded_recording = Arc::clone(&loaded_recording);
        let ui_weak = ui_weak.clone();
        ui.on_load_recording(move |_idx| {
            let recs = recordings.lock().unwrap();
            let idx = *current_index.lock().unwrap();
            if let Some(rec) = recs.get(idx) {
                let rec = rec.clone();
                drop(recs);
                *loaded_recording.lock().unwrap() = Some(rec.clone());
                if let Some(ui) = ui_weak.upgrade() {
                    populate_review(&ui, &rec);
                }
            }
        });
    }

    // Select word callback
    {
        let ui_weak = ui_weak.clone();
        let loaded_recording = Arc::clone(&loaded_recording);
        ui.on_select_word(move |word_idx| {
            let rec_guard = loaded_recording.lock().unwrap();
            if let Some(rec) = rec_guard.as_ref() {
                let words: Vec<&str> = rec.metadata.final_text.split_whitespace().collect();
                let raw_word = words
                    .get(word_idx as usize)
                    .copied()
                    .unwrap_or("")
                    .to_string();
                // Strip leading/trailing punctuation so hotwords and substitutions are clean
                let clean = raw_word.trim_matches(|c: char| !c.is_alphanumeric()).to_string();
                if let Some(ui) = ui_weak.upgrade() {
                    ui.set_selected_word_index(word_idx);
                    ui.set_correction_text(clean.clone().into());
                    ui.set_spoken_form_text(clean.into());
                    ui.set_hotword_added(false);
                    ui.set_action_status("".into());
                }
            }
        });
    }

    // Re-run engine callback
    {
        let loaded_recording = Arc::clone(&loaded_recording);
        let ui_weak = ui_weak.clone();
        ui.on_rerun_engine(move || {
            let wav_path = {
                let guard = loaded_recording.lock().unwrap();
                guard.as_ref().map(|r| r.wav_path.clone())
            };
            if let Some(path) = wav_path {
                let ui_weak = ui_weak.clone();
                std::thread::spawn(move || {
                    let result = engine_runner::rerun_on_wav(&path)
                        .unwrap_or_else(|e| format!("[error] {}", e));
                    let _ = slint::invoke_from_event_loop(move || {
                        if let Some(ui) = ui_weak.upgrade() {
                            ui.set_rerun_result(result.into());
                            ui.set_has_rerun(true);
                        }
                    });
                });
            }
        });
    }

    // Add substitution callback
    {
        let ui_weak = ui_weak.clone();
        ui.on_add_substitution(move |spoken, replacement| {
            match correction::append_substitution(spoken.as_str(), replacement.as_str()) {
                Ok(_) => {
                    if let Some(ui) = ui_weak.upgrade() {
                        ui.set_action_status(
                            format!("✓ Added: \"{}\" → \"{}\"", spoken, replacement).into(),
                        );
                    }
                }
                Err(e) => {
                    if let Some(ui) = ui_weak.upgrade() {
                        ui.set_action_status(format!("Error: {}", e).into());
                    }
                }
            }
        });
    }

    // Add hotword callback
    {
        let ui_weak = ui_weak.clone();
        ui.on_add_hotword(move |word, score| {
            match correction::append_hotword(word.as_str(), score) {
                Ok(_) => {
                    if let Some(ui) = ui_weak.upgrade() {
                        ui.set_hotword_added(true);
                        ui.set_action_status(format!("✓ Hotword added: \"{}\"", word).into());
                    }
                }
                Err(e) => {
                    if let Some(ui) = ui_weak.upgrade() {
                        ui.set_action_status(format!("Error: {}", e).into());
                    }
                }
            }
        });
    }

    ui.run().context("running Slint event loop")?;
    Ok(())
}

fn update_picker(ui: &TestLoop, recs: &[recording::Recording], idx: usize) {
    if let Some(rec) = recs.get(idx) {
        let ts = rec
            .metadata
            .timestamp
            .format("%Y-%m-%d %H:%M:%S")
            .to_string();
        let dur = format!("{:.1}s", rec.metadata.duration_ms as f64 / 1000.0);
        let preview = if rec.metadata.final_text.len() > 60 {
            format!("{}...", &rec.metadata.final_text[..57])
        } else {
            rec.metadata.final_text.clone()
        };
        ui.set_recording_timestamp(ts.into());
        ui.set_recording_duration(dur.into());
        ui.set_recording_preview(preview.into());
        ui.set_recording_index(idx as i32);
        ui.set_recording_count(recs.len() as i32);
    }
}

fn populate_review(ui: &TestLoop, rec: &recording::Recording) {
    let ts = rec
        .metadata
        .timestamp
        .format("%Y-%m-%d %H:%M:%S")
        .to_string();
    let dur = format!("{:.1}s", rec.metadata.duration_ms as f64 / 1000.0);
    let engine_info = format!(
        "engine: {} | accurate: {}",
        rec.metadata.preview_engine, rec.metadata.accurate_engine
    );

    ui.set_recording_timestamp(ts.into());
    ui.set_recording_duration(dur.into());
    ui.set_engine_info(engine_info.into());
    ui.set_final_text(rec.metadata.final_text.clone().into());
    ui.set_recording_loaded(true);
    ui.set_selected_word_index(-1);
    ui.set_has_rerun(false);
    ui.set_rerun_result("".into());

    let word_tiles: Vec<WordTile> = rec
        .metadata
        .final_text
        .split_whitespace()
        .enumerate()
        .map(|(i, w)| WordTile {
            word: w.to_string().into(),
            index: i as i32,
        })
        .collect();
    ui.set_words(word_tiles.as_slice().into());
}
