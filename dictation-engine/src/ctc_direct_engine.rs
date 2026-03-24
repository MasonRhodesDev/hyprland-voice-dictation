//! CTC Direct transcription engine with hotword-boosted beam search.
//!
//! Bypasses `parakeet-rs` entirely: loads the ONNX model directly via `ort`,
//! extracts mel features via `ctc_features`, and runs CTC prefix beam search
//! with hotword trie boosting to produce the final transcript.
//!
//! Model path: `~/.config/voice-dictation/models/parakeet-ctc/model.onnx`
//! Tokenizer:  `~/.config/voice-dictation/models/parakeet-ctc/tokenizer.json`
//! Hotwords:   `~/.local/share/voice-dictation/hotwords.txt`

use anyhow::Result;
use ndarray::Array2;
use ort::session::Session;
use std::path::PathBuf;
use std::sync::{Arc, Mutex};
use tokenizers::Tokenizer;
use tracing::{debug, info};

use crate::chunking::{transcribe_chunked_with_timestamps, ChunkConfig, TimestampedChunkResult};
use crate::ctc_features;
use crate::engine::TranscriptionEngine;
use crate::hotword_trie::{self, HotwordTrie};

// ---------------------------------------------------------------------------
// Constants
// ---------------------------------------------------------------------------

const BLANK_TOKEN_ID: u32 = 1024;
const DEFAULT_BEAM_WIDTH: usize = 10;

/// Minimum number of samples required before attempting transcription (~0.15 s)
const MIN_AUDIO_SAMPLES: usize = 2400;
/// Number of new samples required before re-transcribing for preview (~0.3 s)
const RETRANSCRIBE_THRESHOLD: usize = 4800;

// ---------------------------------------------------------------------------
// Engine struct
// ---------------------------------------------------------------------------

/// CTC Direct engine: ONNX model + HF tokenizer + CTC beam search + hotword boosting.
pub struct CtcDirectEngine {
    /// ONNX inference session
    session: Arc<Mutex<Session>>,
    /// HuggingFace BPE tokenizer for decoding token IDs → text
    tokenizer: Arc<Tokenizer>,
    /// Hotword prefix trie for beam search score boosting
    hotword_trie: Arc<HotwordTrie>,
    /// Beam width for CTC prefix beam search
    beam_width: usize,
    /// Buffered audio samples (i16, mono, 16 kHz)
    audio_buffer: Arc<Mutex<Vec<i16>>>,
    /// Configured sample rate (must be 16000)
    sample_rate: u32,
    /// Cached preview text
    current_text: Arc<Mutex<String>>,
    /// Length of audio buffer at last transcription
    last_transcribed_len: Arc<Mutex<usize>>,
    /// Chunking configuration (30s chunks, 2s overlap)
    chunk_config: ChunkConfig,
}

impl CtcDirectEngine {
    /// Create a new `CtcDirectEngine`.
    ///
    /// # Arguments
    /// * `model_dir` – Directory containing `model.onnx` and `tokenizer.json`
    /// * `sample_rate` – Must be 16000
    pub fn new(model_dir: PathBuf, sample_rate: u32) -> Result<Self> {
        if sample_rate != 16000 {
            anyhow::bail!(
                "CtcDirectEngine requires 16 kHz audio, got {} Hz",
                sample_rate
            );
        }

        info!("Loading CtcDirectEngine from {:?}", model_dir);

        // Load ONNX model (pattern copied from parakeet-rs model.rs)
        let model_path = model_dir.join("model.onnx");
        if !model_path.exists() {
            anyhow::bail!("ONNX model not found: {:?}", model_path);
        }
        let session = Session::builder()?.commit_from_file(&model_path)?;
        info!("ONNX session loaded from {:?}", model_path);

        // Load HuggingFace tokenizer
        let tokenizer_path = model_dir.join("tokenizer.json");
        if !tokenizer_path.exists() {
            anyhow::bail!("Tokenizer not found: {:?}", tokenizer_path);
        }
        let tokenizer = Tokenizer::from_file(&tokenizer_path)
            .map_err(|e| anyhow::anyhow!("Failed to load tokenizer: {}", e))?;
        info!("Tokenizer loaded from {:?}", tokenizer_path);

        // Load hotword trie
        let hotwords_path = hotword_trie::default_hotwords_path();
        let hotword_trie = load_hotword_trie(Some(hotwords_path), &tokenizer_path)?;
        info!("Hotword trie loaded ({} entries)", hotword_trie.len());

        let chunk_config = ChunkConfig::new(30, 2, sample_rate);

        Ok(Self {
            session: Arc::new(Mutex::new(session)),
            tokenizer: Arc::new(tokenizer),
            hotword_trie: Arc::new(hotword_trie),
            beam_width: DEFAULT_BEAM_WIDTH,
            audio_buffer: Arc::new(Mutex::new(Vec::with_capacity(480_000))),
            sample_rate,
            current_text: Arc::new(Mutex::new(String::new())),
            last_transcribed_len: Arc::new(Mutex::new(0)),
            chunk_config,
        })
    }

    // -----------------------------------------------------------------------
    // Private helpers
    // -----------------------------------------------------------------------

    /// Transcribe a single chunk of audio.
    fn transcribe_chunk(&self, samples: &[i16]) -> Result<String> {
        if samples.is_empty() {
            return Ok(String::new());
        }

        // Step 1: Mel spectrogram features [T, 80]
        let features = ctc_features::extract_features(samples, self.sample_rate)?;
        let num_frames = features.shape()[0];
        if num_frames == 0 {
            return Ok(String::new());
        }

        debug!(
            "CtcDirect: {} samples → {} frames",
            samples.len(),
            num_frames
        );

        // Step 2: Run ONNX model
        // Input shape: input_features [1, T, 80], attention_mask [1, T] (all ones i64)
        let input = features
            .to_shape((1, num_frames, 80))
            .map_err(|e| anyhow::anyhow!("Failed to reshape features: {}", e))?
            .to_owned();

        let attention_mask = ndarray::Array2::<i64>::ones((1, num_frames));

        let input_value = ort::value::Value::from_array(input)?;
        let mask_value = ort::value::Value::from_array(attention_mask)?;

        // Step 3: Run model and extract logits [1, T', 1025] → [T', 1025]
        let logits_2d = {
            let mut session = self
                .session
                .lock()
                .map_err(|e| anyhow::anyhow!("Session lock poisoned: {}", e))?;
            let outputs = session.run(ort::inputs!(
                "input_features" => input_value,
                "attention_mask" => mask_value
            ))?;

            let (shape, data) = outputs["logits"].try_extract_tensor::<f32>()
                .map_err(|e| anyhow::anyhow!("Failed to extract logits: {}", e))?;
            // Shape derefs to &[i64]
            if shape.len() != 3 {
                anyhow::bail!("Expected 3D logits, got {} dims", shape.len());
            }
            let time_out = shape[1] as usize;
            let vocab = shape[2] as usize;
            Array2::from_shape_vec((time_out, vocab), data.to_vec())
                .map_err(|e| anyhow::anyhow!("Failed to create logits array: {}", e))?
        };

        // Step 4: log_softmax along vocab axis (done inside beam search)
        // Step 5: CTC prefix beam search with hotword boosting
        let token_ids = ctc_beam_search_decode(&logits_2d, &self.hotword_trie, self.beam_width)?;

        if token_ids.is_empty() {
            return Ok(String::new());
        }

        // Step 6: Decode token IDs → text
        let text = self
            .tokenizer
            .decode(&token_ids, true)
            .map_err(|e| anyhow::anyhow!("Tokenizer decode failed: {}", e))?;

        Ok(text.trim().to_string())
    }

    /// Transcribe a single chunk and return with (empty) word timestamps.
    /// Used by the chunked-transcription path.
    fn transcribe_chunk_with_timestamps(&self, samples: &[i16]) -> Result<TimestampedChunkResult> {
        let text = self.transcribe_chunk(samples)?;
        Ok(TimestampedChunkResult {
            text,
            words: Vec::new(),
        })
    }

    /// Transcribe the full audio buffer, chunking if necessary.
    fn transcribe_buffer(&self, samples: &[i16]) -> Result<String> {
        if samples.is_empty() {
            debug!("CtcDirect transcribe_buffer: empty");
            return Ok(String::new());
        }

        let duration_secs = samples.len() as f32 / self.sample_rate as f32;
        debug!(
            "CtcDirect transcribe_buffer: {} samples ({:.2}s)",
            samples.len(),
            duration_secs
        );

        if self.chunk_config.needs_chunking(samples) {
            return transcribe_chunked_with_timestamps(samples, &self.chunk_config, |chunk| {
                self.transcribe_chunk_with_timestamps(chunk)
            });
        }

        self.transcribe_chunk(samples)
    }
}

// ---------------------------------------------------------------------------
// TranscriptionEngine impl — mirrors parakeet_engine.rs / ctc_engine.rs
// ---------------------------------------------------------------------------

impl TranscriptionEngine for CtcDirectEngine {
    fn process_audio(&self, samples: &[i16]) -> Result<()> {
        let mut buffer = self
            .audio_buffer
            .lock()
            .map_err(|e| anyhow::anyhow!("Audio buffer lock poisoned: {}", e))?;
        buffer.extend_from_slice(samples);
        Ok(())
    }

    fn get_current_text(&self) -> Result<String> {
        let buffer = self
            .audio_buffer
            .lock()
            .map_err(|e| anyhow::anyhow!("Audio buffer lock poisoned: {}", e))?;

        if buffer.is_empty() || buffer.len() < MIN_AUDIO_SAMPLES {
            return Ok(String::new());
        }

        let current_len = buffer.len();
        let last_len_val = {
            let last_len = self
                .last_transcribed_len
                .lock()
                .map_err(|e| anyhow::anyhow!("Lock poisoned: {}", e))?;
            *last_len
        };

        if current_len <= last_len_val + RETRANSCRIBE_THRESHOLD {
            let cached = self
                .current_text
                .lock()
                .map_err(|e| anyhow::anyhow!("Lock poisoned: {}", e))?;
            return Ok(cached.clone());
        }

        let full_audio = buffer.clone();
        drop(buffer);

        debug!(
            "CtcDirect preview: {} samples ({:.2}s)",
            full_audio.len(),
            full_audio.len() as f32 / 16000.0
        );

        let full_text = self.transcribe_buffer(&full_audio)?;

        {
            let mut cached = self
                .current_text
                .lock()
                .map_err(|e| anyhow::anyhow!("Lock poisoned: {}", e))?;
            *cached = full_text.clone();
        }
        {
            let mut last_len = self
                .last_transcribed_len
                .lock()
                .map_err(|e| anyhow::anyhow!("Lock poisoned: {}", e))?;
            *last_len = current_len;
        }

        Ok(full_text)
    }

    fn get_final_result(&self) -> Result<String> {
        let buffer = self
            .audio_buffer
            .lock()
            .map_err(|e| anyhow::anyhow!("Audio buffer lock poisoned: {}", e))?;
        let samples = buffer.clone();
        drop(buffer);
        self.transcribe_buffer(&samples)
    }

    fn get_cached_text(&self) -> String {
        self.current_text
            .lock()
            .map(|g| g.clone())
            .unwrap_or_default()
    }

    fn get_audio_buffer(&self) -> Vec<i16> {
        self.audio_buffer
            .lock()
            .map(|g| g.clone())
            .unwrap_or_default()
    }

    fn reset(&self) {
        if let Ok(mut buffer) = self.audio_buffer.lock() {
            buffer.clear();
        }
        if let Ok(mut text) = self.current_text.lock() {
            text.clear();
        }
        if let Ok(mut last_len) = self.last_transcribed_len.lock() {
            *last_len = 0;
        }
    }
}

// ---------------------------------------------------------------------------
// CTC prefix beam search (copied & adapted from ctc_engine.rs)
//
// Adaptation: accepts raw logits [T, 1025] and returns Vec<u32> of token IDs
// so the caller can decode with a HuggingFace tokenizer instead of the
// ParakeetDecoder (which requires rebuilding fake logits).
// ---------------------------------------------------------------------------

/// A hypothesis in the CTC beam search.
#[derive(Debug, Clone)]
struct BeamHypothesis {
    tokens: Vec<u32>,
    blank_score: f32,
    non_blank_score: f32,
    hotword_context: Vec<u32>,
}

impl BeamHypothesis {
    fn total_score(&self) -> f32 {
        log_add(self.blank_score, self.non_blank_score)
    }
}

/// CTC prefix beam search decoder with hotword trie boosting.
///
/// # Arguments
/// * `logits` – Shape `[T, vocab_size]` raw logits from the CTC encoder
/// * `hotword_trie` – Trie of hotword token sequences for boosting
/// * `beam_width` – Number of hypotheses to maintain
///
/// # Returns
/// Token ID sequence (CTC collapsed, blanks removed) for the best hypothesis.
fn ctc_beam_search_decode(
    logits: &Array2<f32>,
    hotword_trie: &HotwordTrie,
    beam_width: usize,
) -> Result<Vec<u32>> {
    let time_steps = logits.shape()[0];
    let blank_id = BLANK_TOKEN_ID;

    // log-softmax along vocab axis
    let log_probs = log_softmax(logits);

    let initial = BeamHypothesis {
        tokens: Vec::new(),
        blank_score: 0.0,
        non_blank_score: f32::NEG_INFINITY,
        hotword_context: Vec::new(),
    };

    let mut beam = vec![initial];

    for t in 0..time_steps {
        let frame_log_probs = log_probs.row(t);

        let mut next_beam: std::collections::HashMap<Vec<u32>, BeamHypothesis> =
            std::collections::HashMap::new();

        for hyp in &beam {
            // --- Blank extension ---
            {
                let blank_log_prob = frame_log_probs[blank_id as usize];
                let new_blank_score =
                    log_add(hyp.blank_score, hyp.non_blank_score) + blank_log_prob;

                let entry = next_beam
                    .entry(hyp.tokens.clone())
                    .or_insert_with(|| BeamHypothesis {
                        tokens: hyp.tokens.clone(),
                        blank_score: f32::NEG_INFINITY,
                        non_blank_score: f32::NEG_INFINITY,
                        hotword_context: hyp.hotword_context.clone(),
                    });
                entry.blank_score = log_add(entry.blank_score, new_blank_score);
            }

            // --- Non-blank extensions ---
            let top_tokens = top_k_tokens(
                frame_log_probs.as_slice().unwrap(),
                beam_width * 3,
                blank_id,
            );

            for &(token_id, token_log_prob) in &top_tokens {
                let last_token = hyp.tokens.last().copied();

                if Some(token_id) == last_token {
                    // Same token: only extend via blank path (CTC rule)
                    let new_non_blank_score = hyp.blank_score + token_log_prob;

                    let boost = if !hotword_trie.is_empty() {
                        hotword_trie.boost_for_token(&hyp.hotword_context, token_id)
                    } else {
                        0.0
                    };

                    let entry = next_beam
                        .entry(hyp.tokens.clone())
                        .or_insert_with(|| BeamHypothesis {
                            tokens: hyp.tokens.clone(),
                            blank_score: f32::NEG_INFINITY,
                            non_blank_score: f32::NEG_INFINITY,
                            hotword_context: hyp.hotword_context.clone(),
                        });
                    entry.non_blank_score =
                        log_add(entry.non_blank_score, new_non_blank_score + boost);
                } else {
                    // New token: extend hypothesis
                    let new_non_blank_score =
                        log_add(hyp.blank_score, hyp.non_blank_score) + token_log_prob;

                    let mut new_tokens = hyp.tokens.clone();
                    new_tokens.push(token_id);

                    // Update hotword context
                    let mut new_context = hyp.hotword_context.clone();
                    new_context.push(token_id);

                    let boost = if !hotword_trie.is_empty() {
                        hotword_trie.boost_for_token(&hyp.hotword_context, token_id)
                    } else {
                        0.0
                    };

                    // Check if context still matches any hotword prefix
                    let trie_match = hotword_trie.query(&new_context);
                    if !trie_match.is_prefix && !trie_match.is_complete {
                        // Try starting a new hotword from just this token
                        let single = hotword_trie.query(&[token_id]);
                        new_context = if single.is_prefix || single.is_complete {
                            vec![token_id]
                        } else {
                            Vec::new()
                        };
                    }

                    let entry = next_beam
                        .entry(new_tokens.clone())
                        .or_insert_with(|| BeamHypothesis {
                            tokens: new_tokens,
                            blank_score: f32::NEG_INFINITY,
                            non_blank_score: f32::NEG_INFINITY,
                            hotword_context: new_context.clone(),
                        });
                    entry.non_blank_score =
                        log_add(entry.non_blank_score, new_non_blank_score + boost);
                }
            }
        }

        // Prune beam to top-K
        beam = next_beam.into_values().collect();
        beam.sort_by(|a, b| {
            b.total_score()
                .partial_cmp(&a.total_score())
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        beam.truncate(beam_width);
    }

    // Best hypothesis
    let best = beam
        .into_iter()
        .max_by(|a, b| {
            a.total_score()
                .partial_cmp(&b.total_score())
                .unwrap_or(std::cmp::Ordering::Equal)
        })
        .unwrap_or(BeamHypothesis {
            tokens: Vec::new(),
            blank_score: f32::NEG_INFINITY,
            non_blank_score: f32::NEG_INFINITY,
            hotword_context: Vec::new(),
        });

    Ok(best.tokens)
}

// ---------------------------------------------------------------------------
// Beam search utilities (copied from ctc_engine.rs)
// ---------------------------------------------------------------------------

fn log_softmax(logits: &Array2<f32>) -> Array2<f32> {
    let time_steps = logits.shape()[0];
    let vocab_size = logits.shape()[1];
    let mut result = Array2::<f32>::zeros((time_steps, vocab_size));

    for t in 0..time_steps {
        let row = logits.row(t);
        let max_val = row.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
        let log_sum_exp: f32 =
            row.iter().map(|&x| (x - max_val).exp()).sum::<f32>().ln() + max_val;
        for v in 0..vocab_size {
            result[[t, v]] = logits[[t, v]] - log_sum_exp;
        }
    }

    result
}

fn log_add(a: f32, b: f32) -> f32 {
    if a == f32::NEG_INFINITY {
        return b;
    }
    if b == f32::NEG_INFINITY {
        return a;
    }
    let max = a.max(b);
    max + ((a - max).exp() + (b - max).exp()).ln()
}

fn top_k_tokens(log_probs: &[f32], k: usize, blank_id: u32) -> Vec<(u32, f32)> {
    let mut indexed: Vec<(u32, f32)> = log_probs
        .iter()
        .enumerate()
        .filter(|(i, _)| *i != blank_id as usize)
        .map(|(i, &p)| (i as u32, p))
        .collect();
    indexed.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
    indexed.truncate(k);
    indexed
}

// ---------------------------------------------------------------------------
// Hotword trie loader
// ---------------------------------------------------------------------------

fn load_hotword_trie(
    hotwords_path: Option<PathBuf>,
    tokenizer_path: &std::path::Path,
) -> Result<HotwordTrie> {
    let path = hotwords_path.unwrap_or_else(hotword_trie::default_hotwords_path);

    if !path.exists() {
        info!(
            "No hotwords file at {}, hotword boosting disabled",
            path.display()
        );
        return Ok(HotwordTrie::empty());
    }

    let entries = hotword_trie::parse_hotwords_file(&path)?;
    if entries.is_empty() {
        return Ok(HotwordTrie::empty());
    }

    let tokenizer = Tokenizer::from_file(tokenizer_path)
        .map_err(|e| anyhow::anyhow!("Failed to load tokenizer for hotwords: {}", e))?;

    let mut token_sequences: Vec<(hotword_trie::HotwordEntry, Vec<u32>)> = Vec::new();
    for entry in entries {
        match tokenizer.encode(entry.text.as_str(), false) {
            Ok(encoding) => {
                let ids: Vec<u32> = encoding.get_ids().to_vec();
                debug!(
                    "Hotword '{}' -> {} tokens: {:?}",
                    entry.text,
                    ids.len(),
                    ids
                );
                token_sequences.push((entry, ids));
            }
            Err(e) => {
                tracing::warn!("Failed to tokenize hotword '{}': {}", entry.text, e);
            }
        }
    }

    Ok(HotwordTrie::from_token_sequences(&token_sequences))
}

// ---------------------------------------------------------------------------
// Unit tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    /// Minimal smoke test: beam search on synthetic logits (no real model needed).
    ///
    /// Logits are constructed so that greedy decoding would yield tokens [0, 2]
    /// with blank (id=4) frames between them.
    #[test]
    fn test_beam_search_synthetic() {
        let vocab_size = 5; // tokens 0-3, blank=4
        let blank_id = 4u32;
        // 5 frames: tok0, blank, tok2, blank, blank
        let mut logits = Array2::<f32>::from_elem((5, vocab_size), -10.0);
        logits[[0, 0]] = 10.0;
        logits[[1, blank_id as usize]] = 10.0;
        logits[[2, 2]] = 10.0;
        logits[[3, blank_id as usize]] = 10.0;
        logits[[4, blank_id as usize]] = 10.0;

        let trie = HotwordTrie::empty();
        // Override BLANK_TOKEN_ID — test uses a small vocab so we patch logits only
        // The fn uses BLANK_TOKEN_ID = 1024, but our logits only have 5 cols.
        // Use a tiny wrapper that exercises the algorithm with correct blank_id.
        let result = beam_search_with_blank(logits, &trie, 5, blank_id);
        assert!(result.is_ok());
        let tokens = result.unwrap();
        // Should produce [0, 2]
        assert_eq!(tokens, vec![0u32, 2u32]);
    }

    /// Version of beam search callable with a custom blank_id for unit testing.
    fn beam_search_with_blank(
        logits: Array2<f32>,
        hotword_trie: &HotwordTrie,
        beam_width: usize,
        blank_id: u32,
    ) -> Result<Vec<u32>> {
        let time_steps = logits.shape()[0];
        let log_probs = log_softmax(&logits);

        let initial = BeamHypothesis {
            tokens: Vec::new(),
            blank_score: 0.0,
            non_blank_score: f32::NEG_INFINITY,
            hotword_context: Vec::new(),
        };
        let mut beam = vec![initial];

        for t in 0..time_steps {
            let frame_log_probs = log_probs.row(t);
            let mut next_beam: std::collections::HashMap<Vec<u32>, BeamHypothesis> =
                std::collections::HashMap::new();

            for hyp in &beam {
                // Blank extension
                {
                    let blank_log_prob = frame_log_probs[blank_id as usize];
                    let new_blank_score =
                        log_add(hyp.blank_score, hyp.non_blank_score) + blank_log_prob;
                    let entry =
                        next_beam
                            .entry(hyp.tokens.clone())
                            .or_insert_with(|| BeamHypothesis {
                                tokens: hyp.tokens.clone(),
                                blank_score: f32::NEG_INFINITY,
                                non_blank_score: f32::NEG_INFINITY,
                                hotword_context: hyp.hotword_context.clone(),
                            });
                    entry.blank_score = log_add(entry.blank_score, new_blank_score);
                }

                // Non-blank extensions
                let mut indexed: Vec<(u32, f32)> = frame_log_probs
                    .iter()
                    .enumerate()
                    .filter(|(i, _)| *i != blank_id as usize)
                    .map(|(i, &p)| (i as u32, p))
                    .collect();
                indexed
                    .sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
                indexed.truncate(beam_width * 3);

                for &(token_id, token_log_prob) in &indexed {
                    let last_token = hyp.tokens.last().copied();
                    if Some(token_id) == last_token {
                        let new_non_blank_score = hyp.blank_score + token_log_prob;
                        let boost = if !hotword_trie.is_empty() {
                            hotword_trie.boost_for_token(&hyp.hotword_context, token_id)
                        } else {
                            0.0
                        };
                        let entry = next_beam
                            .entry(hyp.tokens.clone())
                            .or_insert_with(|| BeamHypothesis {
                                tokens: hyp.tokens.clone(),
                                blank_score: f32::NEG_INFINITY,
                                non_blank_score: f32::NEG_INFINITY,
                                hotword_context: hyp.hotword_context.clone(),
                            });
                        entry.non_blank_score =
                            log_add(entry.non_blank_score, new_non_blank_score + boost);
                    } else {
                        let new_non_blank_score =
                            log_add(hyp.blank_score, hyp.non_blank_score) + token_log_prob;
                        let mut new_tokens = hyp.tokens.clone();
                        new_tokens.push(token_id);
                        let boost = if !hotword_trie.is_empty() {
                            hotword_trie.boost_for_token(&hyp.hotword_context, token_id)
                        } else {
                            0.0
                        };
                        let mut new_context = hyp.hotword_context.clone();
                        new_context.push(token_id);
                        let trie_match = hotword_trie.query(&new_context);
                        if !trie_match.is_prefix && !trie_match.is_complete {
                            let single = hotword_trie.query(&[token_id]);
                            new_context = if single.is_prefix || single.is_complete {
                                vec![token_id]
                            } else {
                                Vec::new()
                            };
                        }
                        let entry = next_beam
                            .entry(new_tokens.clone())
                            .or_insert_with(|| BeamHypothesis {
                                tokens: new_tokens,
                                blank_score: f32::NEG_INFINITY,
                                non_blank_score: f32::NEG_INFINITY,
                                hotword_context: new_context.clone(),
                            });
                        entry.non_blank_score =
                            log_add(entry.non_blank_score, new_non_blank_score + boost);
                    }
                }
            }

            beam = next_beam.into_values().collect();
            beam.sort_by(|a, b| {
                b.total_score()
                    .partial_cmp(&a.total_score())
                    .unwrap_or(std::cmp::Ordering::Equal)
            });
            beam.truncate(beam_width);
        }

        let best = beam
            .into_iter()
            .max_by(|a, b| {
                a.total_score()
                    .partial_cmp(&b.total_score())
                    .unwrap_or(std::cmp::Ordering::Equal)
            })
            .unwrap_or(BeamHypothesis {
                tokens: Vec::new(),
                blank_score: f32::NEG_INFINITY,
                non_blank_score: f32::NEG_INFINITY,
                hotword_context: Vec::new(),
            });

        Ok(best.tokens)
    }

    #[test]
    fn test_log_add_identity() {
        assert_eq!(log_add(f32::NEG_INFINITY, 1.0), 1.0);
        assert_eq!(log_add(1.0, f32::NEG_INFINITY), 1.0);
        let result = log_add(0.0, 0.0);
        assert!((result - 2.0_f32.ln()).abs() < 1e-5);
    }

    #[test]
    fn test_log_softmax_sums_to_one() {
        let logits = Array2::from_shape_vec((1, 4), vec![1.0f32, 2.0, 3.0, 4.0]).unwrap();
        let lsm = log_softmax(&logits);
        let sum: f32 = (0..4).map(|i| lsm[[0, i]].exp()).sum();
        assert!((sum - 1.0).abs() < 1e-5);
    }

    #[test]
    fn test_top_k_tokens_excludes_blank() {
        let log_probs = vec![-1.0, 0.0, -2.0, -3.0]; // token 1 highest
        let top = top_k_tokens(&log_probs, 3, 1); // blank_id=1
        assert!(top.iter().all(|(id, _)| *id != 1));
        assert_eq!(top[0].0, 0);
    }
}
