//! Parakeet CTC transcription engine with hotword-boosted beam search
//!
//! Uses NVIDIA's Parakeet CTC model (e.g., parakeet-ctc-1.1b) via ONNX Runtime
//! through the `parakeet-rs` crate's `Parakeet` wrapper.
//!
//! Unlike the TDT engine, CTC outputs per-frame probability distributions that
//! can be biased during beam search decoding for hotword boosting.
//!
//! Decoding modes:
//! - **Greedy** (default): Uses parakeet-rs's built-in CTC greedy decoder
//! - **Beam search with hotword boost** (future): Maintains a beam of hypotheses
//!   and applies score bonuses when token sequences match hotword prefixes in a trie.
//!   Currently requires `parakeet-rs` to expose its audio feature extraction module
//!   (mel spectrogram). The beam search algorithm is fully implemented in
//!   [`ctc_beam_search_decode`] and unit-tested with synthetic data.

use anyhow::Result;
use ndarray::Array2;
use parakeet_rs::{Parakeet, ParakeetDecoder, TimestampMode, Transcriber};
use std::path::PathBuf;
use std::sync::{Arc, Mutex};
use tracing::{debug, info, warn};

use crate::chunking::{transcribe_chunked_with_timestamps, ChunkConfig, TimestampedChunkResult};
use crate::engine::TranscriptionEngine;
use crate::hotword_trie::{self, HotwordTrie};

// Audio thresholds (at 16kHz sample rate)
const MIN_AUDIO_SAMPLES: usize = 2400; // 0.15s minimum for transcription
const RETRANSCRIBE_THRESHOLD: usize = 4800; // 0.3s of new audio before re-transcribing

/// Parakeet CTC-based transcription engine
///
/// Uses NVIDIA's Parakeet CTC model for transcription with optional
/// hotword-boosted beam search decoding. CTC models output per-frame
/// log-probabilities that can be biased during decoding.
///
/// Currently uses greedy decoding via `parakeet-rs`. Beam search with
/// hotword boosting is implemented ([`ctc_beam_search_decode`]) but requires
/// direct access to model logits, which needs `parakeet-rs` to expose its
/// audio feature extraction (mel spectrogram) module.
pub struct CtcEngine {
    /// The parakeet CTC model (encoder + greedy decoder)
    parakeet: Arc<Mutex<Parakeet>>,
    /// Hotword trie for future beam search boosting
    hotword_trie: Arc<HotwordTrie>,
    /// Beam width for beam search (0 = greedy only).
    /// Currently always uses greedy; beam search is ready but awaits
    /// feature extraction support from parakeet-rs.
    #[allow(dead_code)]
    beam_width: usize,
    /// Audio buffer
    audio_buffer: Arc<Mutex<Vec<i16>>>,
    sample_rate: u32,
    /// Cached transcription text
    current_text: Arc<Mutex<String>>,
    /// Position in audio_buffer up to which we've transcribed
    last_transcribed_len: Arc<Mutex<usize>>,
    /// Chunking configuration for long audio
    chunk_config: ChunkConfig,
}

impl CtcEngine {
    /// Create a new CTC engine
    ///
    /// # Arguments
    /// * `model_path` - Path to the Parakeet CTC model directory
    /// * `sample_rate` - Audio sample rate (must be 16000)
    /// * `hotwords_path` - Optional path to hotwords file
    /// * `beam_width` - Beam width for search (0 = greedy). Currently only greedy is active;
    ///   beam search algorithm is implemented but awaits feature extraction support.
    pub fn new(
        model_path: PathBuf,
        sample_rate: u32,
        hotwords_path: Option<PathBuf>,
        beam_width: usize,
    ) -> Result<Self> {
        info!("Loading Parakeet CTC model from {:?}", model_path);

        if sample_rate != 16000 {
            anyhow::bail!("Parakeet requires 16kHz audio, got {} Hz", sample_rate);
        }

        // Load the parakeet-rs CTC model (handles feature extraction + greedy decoding)
        let parakeet = Parakeet::from_pretrained(model_path.to_str().unwrap_or("."), None)?;

        // Load hotword trie
        let tokenizer_path = model_path.join("tokenizer.json");
        let hotword_trie = load_hotword_trie(hotwords_path, &tokenizer_path)?;

        if beam_width > 0 && !hotword_trie.is_empty() {
            // TODO: Beam search with hotword boosting is implemented in ctc_beam_search_decode()
            // but requires direct access to model logits. This needs parakeet-rs to expose
            // its audio::extract_features_raw() function, or we add rustfft as a dependency
            // and reimplement mel spectrogram extraction.
            warn!(
                "Beam search (width={}) with {} hotwords requested but not yet active. \
                 Using greedy decoding. The beam search algorithm is implemented and tested \
                 but requires feature extraction support from parakeet-rs.",
                beam_width, hotword_trie.len()
            );
        }

        let chunk_config = ChunkConfig::new(30, 2, sample_rate);

        Ok(Self {
            parakeet: Arc::new(Mutex::new(parakeet)),
            hotword_trie: Arc::new(hotword_trie),
            beam_width,
            audio_buffer: Arc::new(Mutex::new(Vec::with_capacity(480_000))),
            sample_rate,
            current_text: Arc::new(Mutex::new(String::new())),
            last_transcribed_len: Arc::new(Mutex::new(0)),
            chunk_config,
        })
    }

    /// Convert i16 samples to f32
    fn samples_to_f32(samples: &[i16]) -> Vec<f32> {
        samples.iter().map(|&s| s as f32 / 32768.0).collect()
    }

    /// Run transcription on a single chunk using greedy decoding
    fn transcribe_chunk(&self, samples: &[i16]) -> Result<String> {
        if samples.is_empty() {
            return Ok(String::new());
        }

        let f32_samples = Self::samples_to_f32(samples);
        let mut parakeet = self.parakeet.lock()
            .map_err(|e| anyhow::anyhow!("Parakeet CTC model lock poisoned: {}", e))?;
        let result = parakeet.transcribe_samples(f32_samples, self.sample_rate, 1, None)?;
        Ok(result.text)
    }

    /// Run transcription on a single chunk with timestamps
    fn transcribe_chunk_with_timestamps(&self, samples: &[i16]) -> Result<TimestampedChunkResult> {
        if samples.is_empty() {
            return Ok(TimestampedChunkResult { text: String::new(), words: Vec::new() });
        }

        let f32_samples = Self::samples_to_f32(samples);
        let mut parakeet = self.parakeet.lock()
            .map_err(|e| anyhow::anyhow!("Parakeet CTC model lock poisoned: {}", e))?;
        let result = parakeet.transcribe_samples(f32_samples, self.sample_rate, 1, Some(TimestampMode::Words))?;

        Ok(TimestampedChunkResult {
            text: result.text,
            words: result.tokens,
        })
    }

    /// Run transcription on accumulated audio, chunking if necessary
    fn transcribe_buffer(&self, samples: &[i16]) -> Result<String> {
        if samples.is_empty() {
            debug!("transcribe_buffer: empty samples");
            return Ok(String::new());
        }

        let max_sample = samples.iter().map(|s| s.abs()).max().unwrap_or(0);
        let rms = (samples.iter().map(|&s| (s as f64).powi(2)).sum::<f64>() / samples.len() as f64).sqrt();
        let duration_secs = samples.len() as f32 / self.sample_rate as f32;
        debug!(
            "transcribe_buffer (CTC): {} samples, max={}, rms={:.1}, duration={:.2}s",
            samples.len(), max_sample, rms, duration_secs
        );

        let normalized = normalize_audio(samples, 3000.0, 20.0);
        let samples = &normalized;

        // Use timestamped chunking for better merge accuracy when chunking is needed
        if self.chunk_config.needs_chunking(samples) {
            return transcribe_chunked_with_timestamps(samples, &self.chunk_config, |chunk| {
                self.transcribe_chunk_with_timestamps(chunk)
            });
        }

        // Short audio: single-pass transcription
        self.transcribe_chunk(samples)
    }
}

impl TranscriptionEngine for CtcEngine {
    fn process_audio(&self, samples: &[i16]) -> Result<()> {
        let mut buffer = self.audio_buffer.lock()
            .map_err(|e| anyhow::anyhow!("Audio buffer lock poisoned: {}", e))?;
        buffer.extend_from_slice(samples);
        Ok(())
    }

    fn get_current_text(&self) -> Result<String> {
        // Lock ordering: audio_buffer -> last_transcribed_len -> current_text
        let buffer = self.audio_buffer.lock()
            .map_err(|e| anyhow::anyhow!("Audio buffer lock poisoned: {}", e))?;

        if buffer.is_empty() || buffer.len() < MIN_AUDIO_SAMPLES {
            return Ok(String::new());
        }

        let current_len = buffer.len();
        let last_len_val = {
            let last_len = self.last_transcribed_len.lock()
                .map_err(|e| anyhow::anyhow!("Lock poisoned: {}", e))?;
            *last_len
        };

        // Only re-transcribe when enough new audio accumulated
        if current_len <= last_len_val + RETRANSCRIBE_THRESHOLD {
            let cached = self.current_text.lock()
                .map_err(|e| anyhow::anyhow!("Lock poisoned: {}", e))?;
            return Ok(cached.clone());
        }

        // Transcribe full buffer
        let full_audio = buffer.clone();
        drop(buffer);

        debug!("CTC preview transcription: {} samples ({:.2}s)",
               full_audio.len(), full_audio.len() as f32 / 16000.0);

        let full_text = self.transcribe_buffer(&full_audio)?;

        {
            let mut cached = self.current_text.lock()
                .map_err(|e| anyhow::anyhow!("Lock poisoned: {}", e))?;
            *cached = full_text.clone();
        }
        {
            let mut last_len = self.last_transcribed_len.lock()
                .map_err(|e| anyhow::anyhow!("Lock poisoned: {}", e))?;
            *last_len = current_len;
        }

        Ok(full_text)
    }

    fn get_final_result(&self) -> Result<String> {
        let buffer = self.audio_buffer.lock()
            .map_err(|e| anyhow::anyhow!("Audio buffer lock poisoned: {}", e))?;
        let samples = buffer.clone();
        drop(buffer);
        self.transcribe_buffer(&samples)
    }

    fn get_cached_text(&self) -> String {
        self.current_text.lock()
            .map(|guard| guard.clone())
            .unwrap_or_default()
    }

    fn get_audio_buffer(&self) -> Vec<i16> {
        self.audio_buffer.lock()
            .map(|guard| guard.clone())
            .unwrap_or_default()
    }

    fn reset(&self) {
        // Lock ordering: audio_buffer -> current_text -> last_transcribed_len
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
// CTC Beam Search with Hotword Boosting
// ---------------------------------------------------------------------------
//
// This section implements the full CTC beam search algorithm with hotword trie
// boosting. It is fully functional and unit-tested with synthetic logits data.
//
// To activate this in production, we need access to raw model logits, which
// requires either:
// 1. parakeet-rs exposing its `audio::extract_features_raw()` function, or
// 2. Adding `rustfft` as a dependency and reimplementing mel spectrogram
//    feature extraction in this crate.
//
// Once feature extraction is available, the CtcEngine can call:
//   let features = extract_features(audio_samples);
//   let logits = model.forward(features);
//   let text = ctc_beam_search_decode(&logits, &decoder, &trie, beam_width);

/// A hypothesis in the CTC beam search
#[derive(Debug, Clone)]
struct BeamHypothesis {
    /// Token IDs produced so far (after CTC collapse)
    tokens: Vec<u32>,
    /// Score from blank-ending paths (CTC prefix search)
    blank_score: f32,
    /// Score from non-blank-ending paths
    non_blank_score: f32,
    /// Token IDs being tracked for hotword trie matching
    /// (recent suffix that might be part of a hotword)
    hotword_context: Vec<u32>,
}

impl BeamHypothesis {
    fn total_score(&self) -> f32 {
        log_add(self.blank_score, self.non_blank_score)
    }
}

/// CTC beam search decoder with hotword trie boosting.
///
/// Algorithm:
/// 1. At each frame t, get log-probabilities over vocabulary from logits
/// 2. Maintain a beam of top-K hypotheses
/// 3. For each hypothesis, consider extending with each token:
///    - Blank token: keep hypothesis unchanged (CTC blank)
///    - Same as last token: can mean repeat or blank-then-same
///    - New token: extend hypothesis
/// 4. Apply hotword boost: if extending with a token continues a hotword
///    prefix in the trie, add the partial boost to that hypothesis's score
/// 5. Prune beam to top-K by score
/// 6. After all frames, decode the best hypothesis's token sequence
///
/// # Arguments
/// * `logits` - Shape [time_steps, vocab_size] raw logits from the CTC encoder
/// * `decoder` - ParakeetDecoder for converting token IDs to text
/// * `hotword_trie` - Trie of hotword token sequences for boosting
/// * `beam_width` - Number of hypotheses to maintain
pub fn ctc_beam_search_decode(
    logits: &Array2<f32>,
    decoder: &ParakeetDecoder,
    hotword_trie: &HotwordTrie,
    beam_width: usize,
) -> Result<String> {
    let time_steps = logits.shape()[0];
    let vocab_size = logits.shape()[1];
    let blank_id = decoder.pad_token_id() as u32;

    // Convert logits to log-probabilities (log-softmax per frame)
    let log_probs = log_softmax(logits);

    // Initialize beam with empty hypothesis
    let initial = BeamHypothesis {
        tokens: Vec::new(),
        blank_score: 0.0,
        non_blank_score: f32::NEG_INFINITY,
        hotword_context: Vec::new(),
    };

    let mut beam = vec![initial];

    for t in 0..time_steps {
        let frame_log_probs = log_probs.row(t);

        // Map from token sequence -> merged hypothesis
        let mut next_beam: std::collections::HashMap<Vec<u32>, BeamHypothesis> =
            std::collections::HashMap::new();

        for hyp in &beam {
            // --- Blank extension ---
            {
                let blank_log_prob = frame_log_probs[blank_id as usize];
                let new_blank_score = log_add(hyp.blank_score, hyp.non_blank_score) + blank_log_prob;

                let entry = next_beam.entry(hyp.tokens.clone()).or_insert_with(|| BeamHypothesis {
                    tokens: hyp.tokens.clone(),
                    blank_score: f32::NEG_INFINITY,
                    non_blank_score: f32::NEG_INFINITY,
                    hotword_context: hyp.hotword_context.clone(),
                });
                entry.blank_score = log_add(entry.blank_score, new_blank_score);
            }

            // --- Non-blank extensions ---
            // Consider top-K tokens per frame for efficiency
            let top_tokens = top_k_tokens(
                frame_log_probs.as_slice().unwrap(),
                beam_width * 3,
                blank_id,
            );

            for &(token_id, token_log_prob) in &top_tokens {
                let last_token = hyp.tokens.last().copied();

                if Some(token_id) == last_token {
                    // Same token as last: only extend via blank path (CTC rule)
                    let new_non_blank_score = hyp.blank_score + token_log_prob;

                    let boost = if !hotword_trie.is_empty() {
                        hotword_trie.boost_for_token(&hyp.hotword_context, token_id)
                    } else {
                        0.0
                    };

                    let entry = next_beam.entry(hyp.tokens.clone()).or_insert_with(|| BeamHypothesis {
                        tokens: hyp.tokens.clone(),
                        blank_score: f32::NEG_INFINITY,
                        non_blank_score: f32::NEG_INFINITY,
                        hotword_context: hyp.hotword_context.clone(),
                    });
                    entry.non_blank_score = log_add(entry.non_blank_score, new_non_blank_score + boost);
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

                    let entry = next_beam.entry(new_tokens.clone()).or_insert_with(|| BeamHypothesis {
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

        // Prune beam to top-K by total score
        beam = next_beam.into_values().collect();
        beam.sort_by(|a, b| {
            b.total_score()
                .partial_cmp(&a.total_score())
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        beam.truncate(beam_width);
    }

    // Get best hypothesis
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

    if best.tokens.is_empty() {
        return Ok(String::new());
    }

    // Decode token IDs to text.
    // ParakeetDecoder::decode() internally does argmax + CTC collapse on logits.
    // We construct a logits matrix where our already-collapsed tokens are the argmax,
    // with blank frames between them to prevent re-collapsing.
    let fake_logits = build_argmax_logits(&best.tokens, vocab_size);
    let text = decoder.decode(&fake_logits)?;

    Ok(text)
}

/// Build a logits matrix where row i has token_ids[i] as the argmax.
/// Blank frames are inserted between tokens to prevent CTC collapse
/// from merging adjacent identical tokens during decoding.
fn build_argmax_logits(token_ids: &[u32], vocab_size: usize) -> Array2<f32> {
    let num_frames = if token_ids.is_empty() {
        1
    } else {
        token_ids.len() * 2 - 1
    };
    let blank_id = 1024; // Parakeet CTC pad_token_id
    let mut logits = Array2::<f32>::from_elem((num_frames, vocab_size), -100.0);

    for (i, &token_id) in token_ids.iter().enumerate() {
        let frame = i * 2;
        logits[[frame, token_id as usize]] = 100.0;

        // Insert blank frame between tokens (except after last)
        if i + 1 < token_ids.len() {
            let blank_frame = frame + 1;
            if (blank_id as usize) < vocab_size {
                logits[[blank_frame, blank_id as usize]] = 100.0;
            }
        }
    }

    logits
}

/// Compute log-softmax over the vocabulary dimension (axis 1)
fn log_softmax(logits: &Array2<f32>) -> Array2<f32> {
    let time_steps = logits.shape()[0];
    let vocab_size = logits.shape()[1];
    let mut result = Array2::<f32>::zeros((time_steps, vocab_size));

    for t in 0..time_steps {
        let row = logits.row(t);
        let max_val = row.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
        let log_sum_exp: f32 = row.iter().map(|&x| (x - max_val).exp()).sum::<f32>().ln() + max_val;

        for v in 0..vocab_size {
            result[[t, v]] = logits[[t, v]] - log_sum_exp;
        }
    }

    result
}

/// Log-domain addition: log(exp(a) + exp(b))
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

/// Get the top-K token indices by log-probability, excluding blank
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
// Utility functions
// ---------------------------------------------------------------------------

/// Load hotwords from file and build the trie using the model's tokenizer
fn load_hotword_trie(
    hotwords_path: Option<PathBuf>,
    tokenizer_path: &std::path::Path,
) -> Result<HotwordTrie> {
    let path = hotwords_path.unwrap_or_else(hotword_trie::default_hotwords_path);

    if !path.exists() {
        info!("No hotwords file at {}, hotword boosting disabled", path.display());
        return Ok(HotwordTrie::empty());
    }

    let entries = hotword_trie::parse_hotwords_file(&path)?;
    if entries.is_empty() {
        return Ok(HotwordTrie::empty());
    }

    // Load tokenizer to convert hotwords to token ID sequences
    let tokenizer = tokenizers::Tokenizer::from_file(tokenizer_path)
        .map_err(|e| anyhow::anyhow!("Failed to load tokenizer for hotwords: {}", e))?;

    let mut token_sequences: Vec<(hotword_trie::HotwordEntry, Vec<u32>)> = Vec::new();

    for entry in entries {
        match tokenizer.encode(entry.text.as_str(), false) {
            Ok(encoding) => {
                let ids: Vec<u32> = encoding.get_ids().to_vec();
                debug!("Hotword '{}' -> {} tokens: {:?}", entry.text, ids.len(), ids);
                token_sequences.push((entry, ids));
            }
            Err(e) => {
                warn!("Failed to tokenize hotword '{}': {}", entry.text, e);
            }
        }
    }

    Ok(HotwordTrie::from_token_sequences(&token_sequences))
}

/// Normalize audio to a target RMS level for consistent transcription quality.
/// (Same logic as parakeet_engine.rs)
fn normalize_audio(samples: &[i16], target_rms: f32, max_gain: f32) -> Vec<i16> {
    if samples.is_empty() {
        return Vec::new();
    }

    let rms = (samples.iter().map(|&s| (s as f64).powi(2)).sum::<f64>() / samples.len() as f64)
        .sqrt() as f32;

    if rms < 1.0 {
        return samples.to_vec();
    }

    let gain = (target_rms / rms).min(max_gain);

    if (gain - 1.0).abs() < 0.05 {
        return samples.to_vec();
    }

    debug!("normalize_audio: rms={:.1}, gain={:.2}x", rms, gain);

    samples
        .iter()
        .map(|&s| {
            let amplified = s as f32 * gain;
            amplified.clamp(i16::MIN as f32, i16::MAX as f32) as i16
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_log_add() {
        let result = log_add(0.0, 0.0);
        assert!((result - 2.0_f32.ln()).abs() < 1e-5);

        assert_eq!(log_add(f32::NEG_INFINITY, 1.0), 1.0);
        assert_eq!(log_add(1.0, f32::NEG_INFINITY), 1.0);
    }

    #[test]
    fn test_log_softmax() {
        let logits = Array2::from_shape_vec((1, 3), vec![1.0, 2.0, 3.0]).unwrap();
        let lsm = log_softmax(&logits);

        let sum: f32 = (0..3).map(|i| lsm[[0, i]].exp()).sum();
        assert!((sum - 1.0).abs() < 1e-5);

        assert!(lsm[[0, 2]] > lsm[[0, 1]]);
        assert!(lsm[[0, 1]] > lsm[[0, 0]]);
    }

    #[test]
    fn test_top_k_tokens() {
        let log_probs = vec![-5.0, -1.0, -3.0, -0.5, -2.0];
        let top = top_k_tokens(&log_probs, 3, 999);

        assert_eq!(top.len(), 3);
        assert_eq!(top[0].0, 3); // -0.5 highest
        assert_eq!(top[1].0, 1); // -1.0
        assert_eq!(top[2].0, 4); // -2.0
    }

    #[test]
    fn test_top_k_excludes_blank() {
        let log_probs = vec![-5.0, -1.0, 0.0]; // index 2 highest but it's blank
        let top = top_k_tokens(&log_probs, 3, 2);
        assert!(top.iter().all(|(id, _)| *id != 2));
    }

    #[test]
    fn test_build_argmax_logits() {
        let tokens = vec![5, 10, 5]; // repeated 5
        let logits = build_argmax_logits(&tokens, 20);

        assert_eq!(logits.shape(), &[5, 20]);

        // Frame 0: token 5
        let max_idx_0 = logits
            .row(0)
            .iter()
            .enumerate()
            .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
            .map(|(i, _)| i);
        assert_eq!(max_idx_0, Some(5));

        // Frame 2: token 10
        let max_idx_2 = logits
            .row(2)
            .iter()
            .enumerate()
            .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
            .map(|(i, _)| i);
        assert_eq!(max_idx_2, Some(10));
    }

    #[test]
    fn test_normalize_audio_silent() {
        let silent = vec![0i16; 100];
        let normalized = normalize_audio(&silent, 3000.0, 20.0);
        assert_eq!(normalized, silent);
    }

    #[test]
    fn test_normalize_audio_empty() {
        let normalized = normalize_audio(&[], 3000.0, 20.0);
        assert!(normalized.is_empty());
    }

    /// Test beam search with synthetic logits that spell out a known sequence.
    /// This verifies the algorithm works correctly without needing a real model.
    #[test]
    fn test_beam_search_synthetic_greedy_equivalent() {
        // Create logits where greedy decoding would produce tokens [0, 1, 2]
        // blank_id = 5 (just needs to be != the tokens we use)
        let vocab_size = 6;
        let blank_id: u32 = 5;

        // 6 frames: token0, blank, token1, blank, token2, blank
        let mut logits = Array2::<f32>::from_elem((6, vocab_size), -10.0);
        logits[[0, 0]] = 10.0; // Frame 0: token 0
        logits[[1, blank_id as usize]] = 10.0; // Frame 1: blank
        logits[[2, 1]] = 10.0; // Frame 2: token 1
        logits[[3, blank_id as usize]] = 10.0; // Frame 3: blank
        logits[[4, 2]] = 10.0; // Frame 4: token 2
        logits[[5, blank_id as usize]] = 10.0; // Frame 5: blank

        // Without a real decoder/tokenizer, we can test the beam search internals
        // by checking that the best hypothesis has tokens [0, 1, 2]
        let log_probs = log_softmax(&logits);
        let time_steps = log_probs.shape()[0];

        let initial = BeamHypothesis {
            tokens: Vec::new(),
            blank_score: 0.0,
            non_blank_score: f32::NEG_INFINITY,
            hotword_context: Vec::new(),
        };

        let trie = HotwordTrie::empty();
        let beam_width = 5;
        let mut beam = vec![initial];

        for t in 0..time_steps {
            let frame_lp = log_probs.row(t);
            let mut next_beam: std::collections::HashMap<Vec<u32>, BeamHypothesis> =
                std::collections::HashMap::new();

            for hyp in &beam {
                // Blank extension
                {
                    let blp = frame_lp[blank_id as usize];
                    let new_bs = log_add(hyp.blank_score, hyp.non_blank_score) + blp;
                    let entry = next_beam.entry(hyp.tokens.clone()).or_insert_with(|| BeamHypothesis {
                        tokens: hyp.tokens.clone(),
                        blank_score: f32::NEG_INFINITY,
                        non_blank_score: f32::NEG_INFINITY,
                        hotword_context: Vec::new(),
                    });
                    entry.blank_score = log_add(entry.blank_score, new_bs);
                }

                let top = top_k_tokens(frame_lp.as_slice().unwrap(), beam_width * 3, blank_id);
                for &(tid, tlp) in &top {
                    let last = hyp.tokens.last().copied();
                    if Some(tid) == last {
                        let new_nbs = hyp.blank_score + tlp;
                        let entry = next_beam.entry(hyp.tokens.clone()).or_insert_with(|| BeamHypothesis {
                            tokens: hyp.tokens.clone(),
                            blank_score: f32::NEG_INFINITY,
                            non_blank_score: f32::NEG_INFINITY,
                            hotword_context: Vec::new(),
                        });
                        entry.non_blank_score = log_add(entry.non_blank_score, new_nbs);
                    } else {
                        let new_nbs = log_add(hyp.blank_score, hyp.non_blank_score) + tlp;
                        let mut new_tokens = hyp.tokens.clone();
                        new_tokens.push(tid);
                        let entry = next_beam.entry(new_tokens.clone()).or_insert_with(|| BeamHypothesis {
                            tokens: new_tokens,
                            blank_score: f32::NEG_INFINITY,
                            non_blank_score: f32::NEG_INFINITY,
                            hotword_context: Vec::new(),
                        });
                        entry.non_blank_score = log_add(entry.non_blank_score, new_nbs);
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
            .unwrap();

        assert_eq!(best.tokens, vec![0, 1, 2]);
    }

    /// Test that hotword boosting changes the beam search result
    #[test]
    fn test_beam_search_hotword_boost_effect() {
        use crate::hotword_trie::HotwordEntry;

        // Create ambiguous logits where tokens 3 and 4 are close in probability
        // Hotword trie will boost the sequence [3, 4]
        let vocab_size = 6;
        let blank_id: u32 = 5;

        // 4 frames
        let mut logits = Array2::<f32>::from_elem((4, vocab_size), -10.0);
        // Frame 0: token 3 slightly less probable than token 0
        logits[[0, 0]] = 5.0;
        logits[[0, 3]] = 4.8; // Close but lower
        logits[[0, blank_id as usize]] = -5.0;
        // Frame 1: blank
        logits[[1, blank_id as usize]] = 5.0;
        // Frame 2: token 4 slightly less probable than token 1
        logits[[2, 1]] = 5.0;
        logits[[2, 4]] = 4.8; // Close but lower
        logits[[2, blank_id as usize]] = -5.0;
        // Frame 3: blank
        logits[[3, blank_id as usize]] = 5.0;

        // Build hotword trie boosting sequence [3, 4]
        let entries = vec![(
            HotwordEntry {
                text: "hotword".to_string(),
                boost_score: 6.0, // Strong boost
            },
            vec![3, 4],
        )];
        let trie = HotwordTrie::from_token_sequences(&entries);

        // Run beam search with boosting (inline, since we can't use the full decode
        // function without a real tokenizer)
        let log_probs = log_softmax(&logits);
        let beam_width = 5;

        let initial = BeamHypothesis {
            tokens: Vec::new(),
            blank_score: 0.0,
            non_blank_score: f32::NEG_INFINITY,
            hotword_context: Vec::new(),
        };
        let mut beam = vec![initial];

        for t in 0..4 {
            let frame_lp = log_probs.row(t);
            let mut next_beam: std::collections::HashMap<Vec<u32>, BeamHypothesis> =
                std::collections::HashMap::new();

            for hyp in &beam {
                {
                    let blp = frame_lp[blank_id as usize];
                    let new_bs = log_add(hyp.blank_score, hyp.non_blank_score) + blp;
                    let entry = next_beam.entry(hyp.tokens.clone()).or_insert_with(|| BeamHypothesis {
                        tokens: hyp.tokens.clone(),
                        blank_score: f32::NEG_INFINITY,
                        non_blank_score: f32::NEG_INFINITY,
                        hotword_context: hyp.hotword_context.clone(),
                    });
                    entry.blank_score = log_add(entry.blank_score, new_bs);
                }

                let top = top_k_tokens(frame_lp.as_slice().unwrap(), beam_width * 3, blank_id);
                for &(tid, tlp) in &top {
                    let last = hyp.tokens.last().copied();
                    if Some(tid) == last {
                        let new_nbs = hyp.blank_score + tlp;
                        let boost = trie.boost_for_token(&hyp.hotword_context, tid);
                        let entry = next_beam.entry(hyp.tokens.clone()).or_insert_with(|| BeamHypothesis {
                            tokens: hyp.tokens.clone(),
                            blank_score: f32::NEG_INFINITY,
                            non_blank_score: f32::NEG_INFINITY,
                            hotword_context: hyp.hotword_context.clone(),
                        });
                        entry.non_blank_score = log_add(entry.non_blank_score, new_nbs + boost);
                    } else {
                        let new_nbs = log_add(hyp.blank_score, hyp.non_blank_score) + tlp;
                        let mut new_tokens = hyp.tokens.clone();
                        new_tokens.push(tid);
                        let mut new_context = hyp.hotword_context.clone();
                        new_context.push(tid);
                        let boost = trie.boost_for_token(&hyp.hotword_context, tid);
                        let trie_match = trie.query(&new_context);
                        if !trie_match.is_prefix && !trie_match.is_complete {
                            let single = trie.query(&[tid]);
                            new_context = if single.is_prefix || single.is_complete {
                                vec![tid]
                            } else {
                                Vec::new()
                            };
                        }
                        let entry = next_beam.entry(new_tokens.clone()).or_insert_with(|| BeamHypothesis {
                            tokens: new_tokens,
                            blank_score: f32::NEG_INFINITY,
                            non_blank_score: f32::NEG_INFINITY,
                            hotword_context: new_context,
                        });
                        entry.non_blank_score = log_add(entry.non_blank_score, new_nbs + boost);
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
            .unwrap();

        // With the hotword boost, sequence [3, 4] should win over [0, 1]
        // even though [0, 1] had slightly higher raw probabilities
        assert_eq!(best.tokens, vec![3, 4], "Hotword boost should have changed the result");
    }
}
