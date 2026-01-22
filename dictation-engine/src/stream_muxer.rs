use anyhow::Result;
use chrono::Utc;
use crossbeam_channel::Sender;
use hound::{SampleFormat, WavSpec, WavWriter};
use std::collections::{HashMap, VecDeque};
use std::fs::{self, File};
use std::io::BufWriter;
use std::path::PathBuf;
use std::time::{Duration, Instant};
use tracing::{debug, info, warn};

/// Unique identifier for an audio stream (typically device name).
pub type StreamId = String;

/// Configuration for the stream muxer.
#[derive(Clone)]
pub struct MuxerConfig {
    /// How long to stay on a stream before considering switching (ms).
    pub sticky_duration_ms: u64,
    /// Cooldown after switching before considering another switch (ms).
    pub cooldown_ms: u64,
    /// New stream must be this fraction better to trigger switch.
    pub switch_threshold: f32,
    /// Window size for quality scoring (ms).
    pub scoring_window_ms: u64,
    /// Sample rate for audio.
    pub sample_rate: u32,
    /// Enable debug WAV recording.
    pub debug_audio: bool,
}

impl Default for MuxerConfig {
    fn default() -> Self {
        Self {
            sticky_duration_ms: 500,
            cooldown_ms: 200,
            switch_threshold: 0.15,
            scoring_window_ms: 100,
            sample_rate: 16000,
            debug_audio: false,
        }
    }
}

/// Per-stream circular buffer using VecDeque for efficient operations.
struct PerStreamBuffer {
    samples: VecDeque<i16>,
    max_samples: usize,
    /// Samples received since last scoring (for throttling)
    samples_since_score: usize,
}

impl PerStreamBuffer {
    fn new(max_samples: usize) -> Self {
        Self {
            samples: VecDeque::with_capacity(max_samples),
            max_samples,
            samples_since_score: 0,
        }
    }

    fn extend(&mut self, new_samples: &[i16]) {
        self.samples_since_score += new_samples.len();

        for &sample in new_samples {
            if self.samples.len() >= self.max_samples {
                self.samples.pop_front();
            }
            self.samples.push_back(sample);
        }
    }

    fn len(&self) -> usize {
        self.samples.len()
    }

    /// Get samples as contiguous slice for scoring.
    /// Uses make_contiguous() to avoid allocation - rearranges internal
    /// VecDeque storage and returns a slice reference.
    fn as_contiguous_slice(&mut self) -> &[i16] {
        self.samples.make_contiguous()
    }

    /// Reset the samples-since-score counter.
    fn reset_score_counter(&mut self) {
        self.samples_since_score = 0;
    }

    /// Get samples received since last scoring.
    fn samples_since_score(&self) -> usize {
        self.samples_since_score
    }
}

/// Scores audio quality using RMS energy and envelope variance.
///
/// Speech has high envelope variance (amplitude changes over time).
/// Noise has low envelope variance (flat signal).
pub struct QualityScorer {
    window_samples: usize,
    chunk_samples: usize, // 10ms chunks for envelope
}

impl QualityScorer {
    pub fn new(sample_rate: u32, window_ms: u64) -> Self {
        let window_samples = (sample_rate as u64 * window_ms / 1000) as usize;
        let chunk_samples = (sample_rate as usize) / 100; // 10ms chunks
        Self {
            window_samples,
            chunk_samples,
        }
    }

    pub fn window_samples(&self) -> usize {
        self.window_samples
    }

    /// Calculate quality score from audio samples.
    ///
    /// Returns combined score of RMS energy (30%) and coefficient of variation (70%).
    pub fn score(&self, samples: &[i16]) -> f32 {
        if samples.is_empty() {
            return 0.0;
        }

        let rms = self.calculate_rms(samples);
        let cv = self.calculate_coefficient_of_variation(samples);

        // Normalize RMS to 0-1 range (assuming i16 audio)
        let normalized_rms = (rms / 32768.0).min(1.0);

        // CV is already normalized (std_dev / mean), typically 0-2 for speech
        // Clamp to 0-1 range
        let normalized_cv = (cv / 2.0).min(1.0);

        // Combined score: energy + speech-likeness
        // CV is weighted more heavily as it better distinguishes speech from noise
        normalized_rms * 0.3 + normalized_cv * 0.7
    }

    fn calculate_rms(&self, samples: &[i16]) -> f32 {
        if samples.is_empty() {
            return 0.0;
        }
        let sum_squares: f64 = samples.iter().map(|&s| (s as f64).powi(2)).sum();
        (sum_squares / samples.len() as f64).sqrt() as f32
    }

    /// Calculate coefficient of variation of the envelope.
    /// CV = std_dev / mean, which is scale-independent.
    fn calculate_coefficient_of_variation(&self, samples: &[i16]) -> f32 {
        if samples.len() < self.chunk_samples * 2 {
            return 0.0;
        }

        // Calculate RMS of each chunk (envelope)
        let envelope: Vec<f32> = samples
            .chunks(self.chunk_samples)
            .filter(|chunk| chunk.len() >= self.chunk_samples / 2)
            .map(|chunk| self.calculate_rms(chunk))
            .collect();

        if envelope.len() < 2 {
            return 0.0;
        }

        // Calculate mean and standard deviation
        let mean: f32 = envelope.iter().sum::<f32>() / envelope.len() as f32;

        // Avoid division by zero
        if mean < 1.0 {
            return 0.0;
        }

        let variance: f32 = envelope.iter().map(|&e| (e - mean).powi(2)).sum::<f32>()
            / envelope.len() as f32;
        let std_dev = variance.sqrt();

        // Coefficient of variation: std_dev / mean
        std_dev / mean
    }
}

/// Stream selector with hysteresis to prevent rapid switching.
pub struct StreamSelector {
    current_stream: Option<StreamId>,
    last_switch: Instant,
    sticky_duration: Duration,
    cooldown: Duration,
    switch_threshold: f32,
}

impl StreamSelector {
    pub fn new(sticky_duration_ms: u64, cooldown_ms: u64, switch_threshold: f32) -> Self {
        Self {
            current_stream: None,
            last_switch: Instant::now(),
            sticky_duration: Duration::from_millis(sticky_duration_ms),
            cooldown: Duration::from_millis(cooldown_ms),
            switch_threshold,
        }
    }

    /// Select the best stream based on quality scores.
    ///
    /// Uses hysteresis with two time constraints:
    /// - sticky_duration: minimum time to stay on a stream before ANY switch
    /// - cooldown: minimum time after a switch before considering another
    /// - switch_threshold: new stream must be this fraction better
    pub fn select(&mut self, scores: &HashMap<StreamId, f32>) -> Option<StreamId> {
        if scores.is_empty() {
            return self.current_stream.clone();
        }

        // Find best stream, handling NaN safely
        let best = scores
            .iter()
            .filter(|(_, &score)| score.is_finite())
            .max_by(|a, b| a.1.partial_cmp(b.1).unwrap_or(std::cmp::Ordering::Equal));

        let (best_id, best_score) = match best {
            Some((id, score)) => (id, *score),
            None => return self.current_stream.clone(),
        };

        match &self.current_stream {
            Some(current) => {
                let current_score = *scores.get(current).unwrap_or(&0.0);
                let time_since_switch = self.last_switch.elapsed();

                // Check sticky duration: don't switch at all during sticky period
                if time_since_switch < self.sticky_duration {
                    return self.current_stream.clone();
                }

                // Check cooldown: need to wait after sticky period too
                let past_cooldown = time_since_switch > self.sticky_duration + self.cooldown;

                // Hysteresis: only switch if significantly better AND past cooldown
                let is_significantly_better = best_id != current
                    && best_score > current_score * (1.0 + self.switch_threshold);

                // Also switch if current stream has no score (disconnected?)
                let current_missing = !scores.contains_key(current);

                if (is_significantly_better && past_cooldown) || current_missing {
                    debug!(
                        "Switching stream: {} ({:.4}) -> {} ({:.4})",
                        current, current_score, best_id, best_score
                    );
                    self.current_stream = Some(best_id.clone());
                    self.last_switch = Instant::now();
                }
            }
            None => {
                debug!("Initial stream selection: {} ({:.4})", best_id, best_score);
                self.current_stream = Some(best_id.clone());
                self.last_switch = Instant::now();
            }
        }

        self.current_stream.clone()
    }

    /// Get the currently selected stream.
    #[allow(dead_code)] // Public API for debugging
    pub fn current(&self) -> Option<&StreamId> {
        self.current_stream.as_ref()
    }
}

/// Records individual streams to WAV files for debugging.
pub struct DebugRecorder {
    output_dir: PathBuf,
    writers: HashMap<StreamId, WavWriter<BufWriter<File>>>,
    sample_rate: u32,
}

impl DebugRecorder {
    pub fn new(sample_rate: u32) -> Result<Self> {
        let session_id = Utc::now().format("%Y%m%d_%H%M%S").to_string();
        let output_dir = dirs::data_dir()
            .ok_or_else(|| anyhow::anyhow!("Could not determine data directory"))?
            .join("voice-dictation")
            .join("debug")
            .join(&session_id);

        fs::create_dir_all(&output_dir)?;
        info!("Debug audio recording to: {}", output_dir.display());

        Ok(Self {
            output_dir,
            writers: HashMap::new(),
            sample_rate,
        })
    }

    /// Get or create WAV writer for a stream.
    fn get_writer(&mut self, stream_id: &StreamId) -> Result<&mut WavWriter<BufWriter<File>>> {
        if !self.writers.contains_key(stream_id) {
            // Sanitize stream ID for filename
            let safe_name: String = stream_id
                .chars()
                .map(|c| if c.is_alphanumeric() || c == '-' || c == '_' { c } else { '_' })
                .collect();
            let path = self.output_dir.join(format!("{}.wav", safe_name));

            let spec = WavSpec {
                channels: 1,
                sample_rate: self.sample_rate,
                bits_per_sample: 16,
                sample_format: SampleFormat::Int,
            };

            let writer = WavWriter::create(&path, spec)?;
            info!("Created debug WAV: {}", path.display());
            self.writers.insert(stream_id.clone(), writer);
        }

        self.writers.get_mut(stream_id)
            .ok_or_else(|| anyhow::anyhow!("Failed to get WAV writer for stream '{}'", stream_id))
    }

    /// Record samples from a stream.
    pub fn record(&mut self, stream_id: &StreamId, samples: &[i16]) -> Result<()> {
        let writer = self.get_writer(stream_id)?;
        for &sample in samples {
            writer.write_sample(sample)?;
        }
        Ok(())
    }

    /// Finalize all recordings (flush and close files).
    #[allow(dead_code)] // Called when debug recording is enabled
    pub fn finalize(self) -> Result<()> {
        for (stream_id, writer) in self.writers {
            if let Err(e) = writer.finalize() {
                warn!("Failed to finalize WAV for {}: {}", stream_id, e);
            }
        }
        info!("Debug recording finalized");
        Ok(())
    }
}

/// Orchestrates multi-stream audio selection.
///
/// Routes audio from multiple input streams through quality scoring,
/// selects the best stream, and forwards only that stream's audio
/// to the output channel.
pub struct StreamMuxer {
    streams: HashMap<StreamId, PerStreamBuffer>,
    selector: StreamSelector,
    scorer: QualityScorer,
    debug_recorder: Option<DebugRecorder>,
    output_tx: Sender<Vec<i16>>,
    config: MuxerConfig,
    /// Pre-allocated scores map to avoid allocation per push
    scores_cache: HashMap<StreamId, f32>,
    /// Minimum samples between scoring operations (throttle)
    score_interval_samples: usize,
}

impl StreamMuxer {
    pub fn new(output_tx: Sender<Vec<i16>>, config: MuxerConfig) -> Result<Self> {
        let scorer = QualityScorer::new(config.sample_rate, config.scoring_window_ms);
        let selector = StreamSelector::new(
            config.sticky_duration_ms,
            config.cooldown_ms,
            config.switch_threshold,
        );

        let debug_recorder = if config.debug_audio {
            Some(DebugRecorder::new(config.sample_rate)?)
        } else {
            None
        };

        // Score every ~50ms worth of samples (reduces overhead significantly)
        let score_interval_samples = (config.sample_rate as usize) / 20;

        Ok(Self {
            streams: HashMap::new(),
            selector,
            scorer,
            debug_recorder,
            output_tx,
            config,
            scores_cache: HashMap::with_capacity(8),
            score_interval_samples,
        })
    }

    /// Register a new audio stream.
    pub fn add_stream(&mut self, id: StreamId) {
        let buffer_samples =
            (self.config.sample_rate as u64 * self.config.scoring_window_ms * 2 / 1000) as usize;
        self.streams.insert(id.clone(), PerStreamBuffer::new(buffer_samples));
        info!("StreamMuxer: added stream '{}'", id);
    }

    /// Remove an audio stream.
    #[allow(dead_code)] // Public API for hot-plug support
    pub fn remove_stream(&mut self, id: &StreamId) {
        self.streams.remove(id);
        self.scores_cache.remove(id);
        info!("StreamMuxer: removed stream '{}'", id);
    }

    /// Process incoming audio samples from a stream.
    ///
    /// 1. Stores samples in per-stream buffer
    /// 2. Records to debug file if enabled
    /// 3. Periodically scores streams (throttled)
    /// 4. Selects best stream with hysteresis
    /// 5. Forwards samples if this is the selected stream
    pub fn push_samples(&mut self, stream_id: &StreamId, samples: &[i16]) {
        // 1. Store in per-stream buffer (auto-register if needed)
        if !self.streams.contains_key(stream_id) {
            self.add_stream(stream_id.clone());
        }

        if let Some(buffer) = self.streams.get_mut(stream_id) {
            buffer.extend(samples);
        }

        // 2. Record to debug file if enabled
        if let Some(ref mut recorder) = self.debug_recorder {
            if let Err(e) = recorder.record(stream_id, samples) {
                warn!("Debug recording error for {}: {}", stream_id, e);
            }
        }

        // 3. Check if we should score (throttled to reduce CPU overhead)
        let should_score = self.streams.get(stream_id)
            .map(|b| b.samples_since_score() >= self.score_interval_samples)
            .unwrap_or(false);

        if should_score {
            // Score all streams that have enough data
            let window_samples = self.scorer.window_samples();
            self.scores_cache.clear();

            for (id, buffer) in &mut self.streams {
                if buffer.len() >= window_samples {
                    // Use make_contiguous() to avoid allocation - returns &[i16]
                    let samples_slice = buffer.as_contiguous_slice();
                    let score = self.scorer.score(samples_slice);
                    self.scores_cache.insert(id.clone(), score);
                    buffer.reset_score_counter();
                }
            }

            // 4. Select best stream (only when we have scores)
            if !self.scores_cache.is_empty() {
                self.selector.select(&self.scores_cache);
            }
        }

        // 5. Forward samples if this is the selected stream
        if let Some(selected) = self.selector.current() {
            if selected == stream_id {
                let _ = self.output_tx.try_send(samples.to_vec());
            }
        }
    }

    /// Get the currently selected stream ID.
    #[allow(dead_code)] // Public API for debugging
    pub fn current_stream(&self) -> Option<&StreamId> {
        self.selector.current()
    }

    /// Flush all buffered samples from all streams.
    ///
    /// Called during stop to ensure no audio is lost.
    /// Only flushes non-selected streams since the selected stream already forwarded its samples.
    pub fn flush(&mut self) {
        let current_stream = self.selector.current();
        debug!("StreamMuxer: flushing non-selected stream buffers (current: {:?})", current_stream);

        let mut flushed_count = 0;
        for (stream_id, buffer) in &mut self.streams {
            // Skip the currently selected stream - it already forwarded its samples
            if current_stream.map_or(false, |s| s == stream_id) {
                debug!("StreamMuxer: skipping flush of selected stream '{}'", stream_id);
                continue;
            }

            if buffer.len() > 0 {
                let samples = buffer.as_contiguous_slice().to_vec();
                debug!("StreamMuxer: flushing {} samples from stream '{}'", samples.len(), stream_id);
                // Use blocking send to ensure delivery
                if let Err(e) = self.output_tx.send(samples) {
                    warn!("Failed to flush stream '{}': {}", stream_id, e);
                } else {
                    flushed_count += 1;
                }
            }
        }

        self.streams.clear();
        debug!("StreamMuxer: flush complete ({} non-selected streams flushed)", flushed_count);
    }

    /// Finalize debug recording (call at end of session).
    #[allow(dead_code)] // Called when debug recording is enabled
    pub fn finalize(self) -> Result<()> {
        if let Some(recorder) = self.debug_recorder {
            recorder.finalize()?;
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_quality_scorer_empty() {
        let scorer = QualityScorer::new(16000, 100);
        assert_eq!(scorer.score(&[]), 0.0);
    }

    #[test]
    fn test_quality_scorer_silence() {
        let scorer = QualityScorer::new(16000, 100);
        let silence = vec![0i16; 1600]; // 100ms of silence
        let score = scorer.score(&silence);
        assert!(score < 0.01, "Silence should have very low score: {}", score);
    }

    #[test]
    fn test_quality_scorer_loud_signal() {
        let scorer = QualityScorer::new(16000, 100);
        // Alternating loud signal (high variance)
        let signal: Vec<i16> = (0..1600)
            .map(|i| if i % 160 < 80 { 10000 } else { -10000 })
            .collect();
        let score = scorer.score(&signal);
        // Score should be significantly higher than silence (which is ~0)
        assert!(score > 0.05, "Loud varying signal should have high score: {}", score);
    }

    #[test]
    fn test_quality_scorer_speech_like() {
        let scorer = QualityScorer::new(16000, 100);
        // Simulate speech-like signal with varying amplitude
        let signal: Vec<i16> = (0..1600)
            .map(|i| {
                let envelope = ((i as f32 / 160.0).sin() * 0.5 + 0.5) * 10000.0;
                (envelope * (i as f32 * 0.1).sin()) as i16
            })
            .collect();
        let score = scorer.score(&signal);
        assert!(score > 0.01, "Speech-like signal should have positive score: {}", score);
    }

    #[test]
    fn test_stream_selector_initial() {
        let mut selector = StreamSelector::new(500, 200, 0.15);
        let mut scores = HashMap::new();
        scores.insert("stream1".to_string(), 0.5);
        scores.insert("stream2".to_string(), 0.3);

        let selected = selector.select(&scores);
        assert_eq!(selected, Some("stream1".to_string()));
    }

    #[test]
    fn test_stream_selector_sticky_duration() {
        let mut selector = StreamSelector::new(100, 50, 0.15); // 100ms sticky, 50ms cooldown

        // Initial selection
        let mut scores = HashMap::new();
        scores.insert("stream1".to_string(), 0.5);
        scores.insert("stream2".to_string(), 0.4);
        selector.select(&scores);
        assert_eq!(selector.current(), Some(&"stream1".to_string()));

        // Even much better stream2 shouldn't cause switch during sticky period
        scores.insert("stream2".to_string(), 0.9);
        let selected = selector.select(&scores);
        assert_eq!(selected, Some("stream1".to_string()), "Should not switch during sticky period");
    }

    #[test]
    fn test_stream_selector_hysteresis() {
        let mut selector = StreamSelector::new(0, 0, 0.15); // No sticky/cooldown for this test

        // Initial selection
        let mut scores = HashMap::new();
        scores.insert("stream1".to_string(), 0.5);
        scores.insert("stream2".to_string(), 0.4);
        selector.select(&scores);

        // Slightly better stream2 shouldn't cause switch (within threshold)
        scores.insert("stream2".to_string(), 0.55);
        let selected = selector.select(&scores);
        assert_eq!(selected, Some("stream1".to_string()));

        // Much better stream2 should cause switch (> 15% better)
        scores.insert("stream2".to_string(), 0.7);
        let selected = selector.select(&scores);
        assert_eq!(selected, Some("stream2".to_string()));
    }

    #[test]
    fn test_stream_selector_nan_handling() {
        let mut selector = StreamSelector::new(0, 0, 0.15);
        let mut scores = HashMap::new();
        scores.insert("stream1".to_string(), f32::NAN);
        scores.insert("stream2".to_string(), 0.5);

        let selected = selector.select(&scores);
        assert_eq!(selected, Some("stream2".to_string()), "Should ignore NaN scores");
    }

    #[test]
    fn test_per_stream_buffer() {
        let mut buffer = PerStreamBuffer::new(100);
        buffer.extend(&[1, 2, 3]);
        assert_eq!(buffer.len(), 3);
        assert_eq!(buffer.as_contiguous_slice(), &[1, 2, 3]);

        // Test overflow - VecDeque should handle this efficiently
        buffer.extend(&(0..150).map(|i| i as i16).collect::<Vec<_>>());
        assert_eq!(buffer.len(), 100);

        // Should have the last 100 values (50-149)
        let recent = buffer.as_contiguous_slice();
        assert_eq!(recent[0], 50);
        assert_eq!(recent[99], 149);
    }

    #[test]
    fn test_per_stream_buffer_score_counter() {
        let mut buffer = PerStreamBuffer::new(100);
        assert_eq!(buffer.samples_since_score(), 0);

        buffer.extend(&[1, 2, 3]);
        assert_eq!(buffer.samples_since_score(), 3);

        buffer.extend(&[4, 5]);
        assert_eq!(buffer.samples_since_score(), 5);

        buffer.reset_score_counter();
        assert_eq!(buffer.samples_since_score(), 0);
    }
}
