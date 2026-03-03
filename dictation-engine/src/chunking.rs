//! Audio chunking utilities for long transcriptions
//!
//! Provides reusable chunking and merging logic for transcription engines
//! that have context length limits.

use parakeet_rs::TimedToken;
use tracing::debug;

/// Configuration for audio chunking
#[derive(Debug, Clone)]
pub struct ChunkConfig {
    /// Maximum chunk duration in seconds
    pub max_chunk_seconds: u32,
    /// Overlap between chunks in seconds (to avoid cutting words)
    pub overlap_seconds: u32,
    /// Sample rate in Hz
    pub sample_rate: u32,
}

impl Default for ChunkConfig {
    fn default() -> Self {
        Self {
            max_chunk_seconds: 30,
            overlap_seconds: 2,
            sample_rate: 16000,
        }
    }
}

impl ChunkConfig {
    /// Create a new chunk config
    pub fn new(max_chunk_seconds: u32, overlap_seconds: u32, sample_rate: u32) -> Self {
        Self {
            max_chunk_seconds,
            overlap_seconds,
            sample_rate,
        }
    }

    /// Maximum samples per chunk
    pub fn max_chunk_samples(&self) -> usize {
        (self.max_chunk_seconds * self.sample_rate) as usize
    }

    /// Overlap samples between chunks
    pub fn overlap_samples(&self) -> usize {
        (self.overlap_seconds * self.sample_rate) as usize
    }

    /// Check if audio needs chunking
    pub fn needs_chunking(&self, samples: &[i16]) -> bool {
        samples.len() > self.max_chunk_samples()
    }
}

/// Iterator over audio chunks with overlap
#[allow(dead_code)]
pub struct AudioChunks<'a> {
    samples: &'a [i16],
    config: ChunkConfig,
    offset: usize,
    chunk_num: usize,
}

impl<'a> AudioChunks<'a> {
    /// Create a new chunk iterator
    #[allow(dead_code)]
    pub fn new(samples: &'a [i16], config: ChunkConfig) -> Self {
        Self {
            samples,
            config,
            offset: 0,
            chunk_num: 0,
        }
    }
}

impl<'a> Iterator for AudioChunks<'a> {
    type Item = (usize, &'a [i16]); // (chunk_number, samples)

    fn next(&mut self) -> Option<Self::Item> {
        if self.offset >= self.samples.len() {
            return None;
        }

        let max_samples = self.config.max_chunk_samples();
        let overlap = self.config.overlap_samples();

        let chunk_end = (self.offset + max_samples).min(self.samples.len());
        let chunk = &self.samples[self.offset..chunk_end];
        let chunk_num = self.chunk_num;

        // Advance for next iteration
        self.offset += max_samples - overlap;
        self.chunk_num += 1;

        Some((chunk_num, chunk))
    }
}

/// Result of transcribing a chunk with word-level timestamps
#[derive(Debug, Clone)]
pub struct TimestampedChunkResult {
    pub text: String,
    pub words: Vec<TimedToken>,
}

/// Find the quietest frame near `target_pos` within a search window.
///
/// Scans for the lowest-energy frame to avoid splitting audio mid-word.
/// Returns the best split position (start of the quietest frame).
fn find_silence_boundary(samples: &[i16], target_pos: usize, search_window: usize, frame_size: usize) -> usize {
    let search_start = target_pos.saturating_sub(search_window / 2);
    let search_end = (target_pos + search_window / 2).min(samples.len());

    if search_start >= search_end || search_end - search_start < frame_size {
        return target_pos.min(samples.len());
    }

    let mut best_pos = target_pos;
    let mut best_energy = f64::MAX;

    let mut pos = search_start;
    while pos + frame_size <= search_end {
        let frame = &samples[pos..pos + frame_size];
        let energy: f64 = frame.iter().map(|&s| (s as f64).powi(2)).sum::<f64>() / frame_size as f64;

        if energy < best_energy {
            best_energy = energy;
            best_pos = pos;
        }
        pos += frame_size;
    }

    debug!("find_silence_boundary: target={}, best={}, energy={:.1}", target_pos, best_pos, best_energy);
    best_pos
}

/// Generate VAD-aware chunk boundaries that prefer silence points over fixed splits
fn chunk_boundaries_vad(samples: &[i16], config: &ChunkConfig) -> Vec<(usize, usize)> {
    let max_samples = config.max_chunk_samples();
    let overlap = config.overlap_samples();
    // 1 second search window, 25ms frames at configured sample rate
    let search_window = config.sample_rate as usize;
    let frame_size = config.sample_rate as usize / 40; // 25ms

    let mut boundaries = Vec::new();
    let mut offset = 0;

    while offset < samples.len() {
        let ideal_end = (offset + max_samples).min(samples.len());

        // For the last chunk or if very close to the end, just take it all
        let chunk_end = if ideal_end >= samples.len() || samples.len() - ideal_end < config.sample_rate as usize {
            samples.len()
        } else {
            find_silence_boundary(samples, ideal_end, search_window, frame_size)
        };

        boundaries.push((offset, chunk_end));

        if chunk_end >= samples.len() {
            break;
        }

        // Next chunk starts overlap before the boundary
        offset = chunk_end.saturating_sub(overlap);
    }

    boundaries
}

/// Merge timestamped chunk results using word time positions instead of text matching.
///
/// For overlapping regions, keeps chunk N's words (more left context) and appends
/// chunk N+1's words that start after the overlap zone.
fn merge_chunks_timestamped(chunks: &[TimestampedChunkResult], overlap_seconds: f32) -> String {
    if chunks.is_empty() {
        return String::new();
    }
    if chunks.len() == 1 {
        return chunks[0].text.clone();
    }

    let mut all_words: Vec<&TimedToken> = Vec::new();

    for (i, chunk) in chunks.iter().enumerate() {
        if i == 0 {
            // First chunk: keep all words
            all_words.extend(chunk.words.iter());
            continue;
        }

        if chunk.words.is_empty() {
            continue;
        }

        // Find where the previous chunk's time coverage ends
        let prev_end_time = all_words.last()
            .map(|w| w.end)
            .unwrap_or(0.0);

        // Skip words from this chunk that fall within the overlap zone
        // (they were already covered by the previous chunk with better context)
        let overlap_cutoff = prev_end_time - overlap_seconds;

        for word in &chunk.words {
            // Keep words that start after the overlap zone
            if word.start >= overlap_cutoff {
                // Avoid duplicating the last word if timestamps are very close
                if let Some(last) = all_words.last() {
                    if last.text.eq_ignore_ascii_case(&word.text)
                        && (word.start - last.start).abs() < overlap_seconds
                    {
                        continue;
                    }
                }
                all_words.push(word);
            }
        }
    }

    // Build text from collected words
    let mut result = String::new();
    for (i, word) in all_words.iter().enumerate() {
        let is_standalone_punct = word.text.len() == 1
            && word.text.chars().all(|c| matches!(c, '.' | ',' | '!' | '?' | ';' | ':'));
        if i > 0 && !is_standalone_punct {
            result.push(' ');
        }
        result.push_str(&word.text);
    }

    result
}

/// Process long audio in chunks using timestamped transcription for accurate merging.
///
/// Uses VAD-aware boundaries and word timestamps to merge chunks without
/// the fragility of text-based overlap matching.
pub fn transcribe_chunked_with_timestamps<F>(
    samples: &[i16],
    config: &ChunkConfig,
    transcribe_fn: F,
) -> anyhow::Result<String>
where
    F: Fn(&[i16]) -> anyhow::Result<TimestampedChunkResult>,
{
    let duration_secs = samples.len() as f32 / config.sample_rate as f32;
    tracing::info!(
        "transcribe_chunked_with_timestamps: chunking {:.1}s audio into ~{}s segments",
        duration_secs,
        config.max_chunk_seconds
    );

    let boundaries = chunk_boundaries_vad(samples, config);
    let mut results: Vec<TimestampedChunkResult> = Vec::new();

    for (chunk_num, &(start, end)) in boundaries.iter().enumerate() {
        let chunk = &samples[start..end];
        let chunk_start_sec = start as f32 / config.sample_rate as f32;
        let chunk_end_sec = end as f32 / config.sample_rate as f32;

        debug!(
            "transcribe_chunked_ts: chunk {} ({:.1}s - {:.1}s, {} samples)",
            chunk_num, chunk_start_sec, chunk_end_sec, chunk.len()
        );

        match transcribe_fn(chunk) {
            Ok(mut result) => {
                // Offset timestamps to absolute positions
                for word in &mut result.words {
                    word.start += chunk_start_sec;
                    word.end += chunk_start_sec;
                }
                if !result.text.is_empty() {
                    debug!("transcribe_chunked_ts: chunk {} -> '{}' ({} words)",
                           chunk_num, result.text, result.words.len());
                    results.push(result);
                }
            }
            Err(e) => {
                debug!("transcribe_chunked_ts: chunk {} error: {}", chunk_num, e);
            }
        }
    }

    let overlap_secs = config.overlap_seconds as f32;
    let merged = merge_chunks_timestamped(&results, overlap_secs);
    tracing::info!(
        "transcribe_chunked_with_timestamps: merged {} chunks into {} chars",
        results.len(),
        merged.len()
    );

    Ok(merged)
}

/// Merge transcription chunks, removing duplicate words at overlap boundaries
#[allow(dead_code)]
pub fn merge_chunks(chunks: &[String]) -> String {
    if chunks.is_empty() {
        return String::new();
    }
    if chunks.len() == 1 {
        return chunks[0].clone();
    }

    let mut result = chunks[0].clone();

    for (i, chunk) in chunks[1..].iter().enumerate() {
        let merged = merge_two_chunks(&result, chunk);
        debug!(
            "merge_chunks: merged chunk {} -> {} chars",
            i + 1,
            merged.len()
        );
        result = merged;
    }

    result
}

/// Merge two adjacent transcription chunks
#[allow(dead_code)]
fn merge_two_chunks(first: &str, second: &str) -> String {
    let first_words: Vec<&str> = first.split_whitespace().collect();
    let second_words: Vec<&str> = second.split_whitespace().collect();

    if first_words.is_empty() {
        return second.to_string();
    }
    if second_words.is_empty() {
        return first.to_string();
    }

    // Look for overlap: find where second starts that matches end of first
    // Check last N words of first against first N words of second
    let max_overlap_words = 10.min(first_words.len()).min(second_words.len());
    let mut best_overlap = 0;

    for overlap_len in 1..=max_overlap_words {
        let first_end = &first_words[first_words.len() - overlap_len..];
        let second_start = &second_words[..overlap_len];

        // Case-insensitive comparison for better matching
        let matches = first_end
            .iter()
            .zip(second_start.iter())
            .all(|(a, b)| a.eq_ignore_ascii_case(b));

        if matches {
            best_overlap = overlap_len;
        }
    }

    // Build result
    let mut result = first.to_string();

    if best_overlap > 0 {
        debug!("merge_two_chunks: found {} word overlap", best_overlap);
        let new_words = &second_words[best_overlap..];
        if !new_words.is_empty() {
            result.push(' ');
            result.push_str(&new_words.join(" "));
        }
    } else {
        // No overlap found, just append with space
        result.push(' ');
        result.push_str(second);
    }

    result
}

/// Process long audio in chunks using a provided transcription function
///
/// # Arguments
/// * `samples` - Audio samples to transcribe
/// * `config` - Chunking configuration
/// * `transcribe_fn` - Function that transcribes a single chunk
///
/// # Returns
/// Merged transcription result
#[allow(dead_code)]
pub fn transcribe_chunked<F>(
    samples: &[i16],
    config: &ChunkConfig,
    transcribe_fn: F,
) -> anyhow::Result<String>
where
    F: Fn(&[i16]) -> anyhow::Result<String>,
{
    // Short audio: transcribe directly
    if !config.needs_chunking(samples) {
        debug!("transcribe_chunked: short audio, single pass");
        return transcribe_fn(samples);
    }

    let duration_secs = samples.len() as f32 / config.sample_rate as f32;
    tracing::info!(
        "transcribe_chunked: chunking {:.1}s audio into ~{}s segments",
        duration_secs,
        config.max_chunk_seconds
    );

    let mut results: Vec<String> = Vec::new();

    for (chunk_num, chunk) in AudioChunks::new(samples, config.clone()) {
        let chunk_start = chunk_num as f32 * (config.max_chunk_seconds - config.overlap_seconds) as f32;
        let chunk_end = chunk_start + chunk.len() as f32 / config.sample_rate as f32;

        debug!(
            "transcribe_chunked: chunk {} ({:.1}s - {:.1}s, {} samples)",
            chunk_num,
            chunk_start,
            chunk_end,
            chunk.len()
        );

        match transcribe_fn(chunk) {
            Ok(text) => {
                if !text.is_empty() {
                    debug!("transcribe_chunked: chunk {} -> '{}'", chunk_num, text);
                    results.push(text);
                }
            }
            Err(e) => {
                debug!("transcribe_chunked: chunk {} error: {}", chunk_num, e);
            }
        }
    }

    let merged = merge_chunks(&results);
    tracing::info!(
        "transcribe_chunked: merged {} chunks into {} chars",
        results.len(),
        merged.len()
    );

    Ok(merged)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_merge_no_overlap() {
        let chunks = vec!["hello world".to_string(), "foo bar".to_string()];
        let merged = merge_chunks(&chunks);
        assert_eq!(merged, "hello world foo bar");
    }

    #[test]
    fn test_merge_with_overlap() {
        let chunks = vec![
            "hello world foo".to_string(),
            "foo bar baz".to_string(),
        ];
        let merged = merge_chunks(&chunks);
        assert_eq!(merged, "hello world foo bar baz");
    }

    #[test]
    fn test_merge_multi_word_overlap() {
        let chunks = vec![
            "one two three four".to_string(),
            "three four five six".to_string(),
        ];
        let merged = merge_chunks(&chunks);
        assert_eq!(merged, "one two three four five six");
    }

    #[test]
    fn test_chunk_iterator() {
        let samples = vec![0i16; 80000]; // 5 seconds at 16kHz
        let config = ChunkConfig::new(2, 0, 16000); // 2 second chunks, no overlap

        let chunks: Vec<_> = AudioChunks::new(&samples, config).collect();
        assert_eq!(chunks.len(), 3); // ceil(5/2) = 3 chunks
    }

    #[test]
    fn test_needs_chunking() {
        let config = ChunkConfig::new(30, 2, 16000);

        // 20 seconds - no chunking needed
        let short = vec![0i16; 320000];
        assert!(!config.needs_chunking(&short));

        // 60 seconds - needs chunking
        let long = vec![0i16; 960000];
        assert!(config.needs_chunking(&long));
    }

    // === EDGE CASE TESTS ===

    #[test]
    fn test_empty_samples() {
        let config = ChunkConfig::default();
        assert!(!config.needs_chunking(&[]));

        let chunks: Vec<_> = AudioChunks::new(&[], config).collect();
        assert_eq!(chunks.len(), 0);
    }

    #[test]
    fn test_very_short_samples_below_threshold() {
        // Test samples shorter than minimum threshold (2400 samples = 0.15s)
        let config = ChunkConfig::new(30, 2, 16000);

        // 1000 samples = 0.0625s - too short
        let tiny = vec![0i16; 1000];
        assert!(!config.needs_chunking(&tiny));

        // 2400 samples = 0.15s - exactly at threshold
        let threshold = vec![0i16; 2400];
        assert!(!config.needs_chunking(&threshold));
    }

    #[test]
    fn test_long_sample_chunking_count() {
        let config = ChunkConfig::new(30, 2, 16000);

        // 60 seconds = 960000 samples
        let samples = vec![0i16; 960000];
        assert!(config.needs_chunking(&samples));

        // With 30s chunks and 2s overlap, effective chunk step = 28s
        // 60s / 28s = ~2.14, so 3 chunks
        let chunks: Vec<_> = AudioChunks::new(&samples, config).collect();
        assert_eq!(chunks.len(), 3);
    }

    #[test]
    fn test_chunk_overlap_boundaries() {
        // 3 seconds of audio with 1s chunks and 0.5s overlap
        let config = ChunkConfig::new(1, 0, 16000); // 1s chunks, no overlap for simplicity
        let samples = vec![0i16; 48000]; // 3 seconds

        let chunks: Vec<_> = AudioChunks::new(&samples, config).collect();
        assert_eq!(chunks.len(), 3);

        // Each chunk should be exactly 16000 samples
        for (_, chunk) in &chunks {
            assert_eq!(chunk.len(), 16000);
        }
    }

    #[test]
    fn test_silent_audio_detection() {
        // Helper to check if audio is silent (all zeros or very low amplitude)
        fn is_silent(samples: &[i16]) -> bool {
            if samples.is_empty() {
                return true;
            }
            let max_sample = samples.iter().map(|s| s.abs()).max().unwrap_or(0);
            let rms = (samples.iter().map(|&s| (s as f64).powi(2)).sum::<f64>()
                / samples.len() as f64)
                .sqrt();
            max_sample < 100 && rms < 50.0
        }

        // Silent audio
        let silent = vec![0i16; 16000];
        assert!(is_silent(&silent));

        // Audio with noise
        let noisy: Vec<i16> = (0..16000).map(|i| (i % 100) as i16 * 100).collect();
        assert!(!is_silent(&noisy));

        // Very low amplitude (background noise level)
        let low_noise: Vec<i16> = (0..16000).map(|i| (i % 50) as i16).collect();
        assert!(is_silent(&low_noise));
    }

    #[test]
    fn test_transcribe_chunked_empty() {
        let config = ChunkConfig::default();
        let result = transcribe_chunked(&[], &config, |_| Ok("test".to_string()));
        assert!(result.is_ok());
        assert_eq!(result.unwrap(), "test"); // Empty passes to transcribe_fn
    }

    #[test]
    fn test_transcribe_chunked_short_passthrough() {
        let config = ChunkConfig::new(30, 2, 16000);
        let samples = vec![0i16; 16000]; // 1 second

        let result = transcribe_chunked(&samples, &config, |chunk| {
            assert_eq!(chunk.len(), 16000);
            Ok("short audio".to_string())
        });

        assert!(result.is_ok());
        assert_eq!(result.unwrap(), "short audio");
    }

    #[test]
    fn test_transcribe_chunked_long_merges() {
        use std::sync::atomic::{AtomicUsize, Ordering};

        let config = ChunkConfig::new(1, 0, 16000); // 1s chunks, no overlap
        let samples = vec![0i16; 48000]; // 3 seconds

        let chunk_count = AtomicUsize::new(0);
        let result = transcribe_chunked(&samples, &config, |_| {
            let count = chunk_count.fetch_add(1, Ordering::SeqCst) + 1;
            Ok(format!("chunk{}", count))
        });

        assert_eq!(chunk_count.load(Ordering::SeqCst), 3);
        assert!(result.is_ok());
        // Chunks should be merged with spaces
        assert!(result.unwrap().contains("chunk1"));
    }

    #[test]
    fn test_merge_empty_chunks() {
        // Empty vec
        assert_eq!(merge_chunks(&[]), "");

        // Single empty string
        assert_eq!(merge_chunks(&["".to_string()]), "");

        // Mix of empty and non-empty
        let chunks = vec!["hello".to_string(), "".to_string(), "world".to_string()];
        let merged = merge_chunks(&chunks);
        assert!(merged.contains("hello"));
        assert!(merged.contains("world"));
    }

    #[test]
    fn test_merge_case_insensitive_overlap() {
        // Overlap detection should be case-insensitive
        let chunks = vec![
            "Hello World".to_string(),
            "world foo".to_string(),
        ];
        let merged = merge_chunks(&chunks);
        // Should detect "World" and "world" as overlap
        assert_eq!(merged, "Hello World foo");
    }

    // === TIMESTAMP MERGING TESTS ===

    #[test]
    fn test_merge_timestamped_single_chunk() {
        let chunks = vec![TimestampedChunkResult {
            text: "Hello world.".to_string(),
            words: vec![
                TimedToken { text: "Hello".to_string(), start: 0.0, end: 0.5 },
                TimedToken { text: "world".to_string(), start: 0.5, end: 1.0 },
                TimedToken { text: ".".to_string(), start: 1.0, end: 1.1 },
            ],
        }];
        let merged = merge_chunks_timestamped(&chunks, 2.0);
        assert_eq!(merged, "Hello world.");
    }

    #[test]
    fn test_merge_timestamped_overlap_dedup() {
        let chunks = vec![
            TimestampedChunkResult {
                text: "Hello world foo".to_string(),
                words: vec![
                    TimedToken { text: "Hello".to_string(), start: 0.0, end: 0.5 },
                    TimedToken { text: "world".to_string(), start: 0.5, end: 1.0 },
                    TimedToken { text: "foo".to_string(), start: 1.0, end: 1.5 },
                ],
            },
            TimestampedChunkResult {
                // Overlap region re-transcribed "foo" then continues
                text: "foo bar baz".to_string(),
                words: vec![
                    TimedToken { text: "foo".to_string(), start: 1.0, end: 1.5 },
                    TimedToken { text: "bar".to_string(), start: 1.5, end: 2.0 },
                    TimedToken { text: "baz".to_string(), start: 2.0, end: 2.5 },
                ],
            },
        ];
        let merged = merge_chunks_timestamped(&chunks, 2.0);
        assert_eq!(merged, "Hello world foo bar baz");
    }

    #[test]
    fn test_merge_timestamped_no_overlap() {
        let chunks = vec![
            TimestampedChunkResult {
                text: "Hello".to_string(),
                words: vec![
                    TimedToken { text: "Hello".to_string(), start: 0.0, end: 0.5 },
                ],
            },
            TimestampedChunkResult {
                text: "world".to_string(),
                words: vec![
                    TimedToken { text: "world".to_string(), start: 2.0, end: 2.5 },
                ],
            },
        ];
        let merged = merge_chunks_timestamped(&chunks, 0.0);
        assert_eq!(merged, "Hello world");
    }

    // === SILENCE BOUNDARY TESTS ===

    #[test]
    fn test_find_silence_boundary_prefers_quiet() {
        // Create audio with a loud section and a quiet gap
        let mut samples = vec![10000i16; 16000]; // 1s of loud audio
        // Insert 25ms of silence at position 8000
        for s in &mut samples[8000..8400] {
            *s = 0;
        }

        let boundary = find_silence_boundary(&samples, 8000, 1600, 400);
        // Should find the quiet region near 8000
        assert!((boundary as i64 - 8000).abs() < 800);
    }

    #[test]
    fn test_find_silence_boundary_edge_cases() {
        let samples = vec![0i16; 100];
        // Target beyond array
        let boundary = find_silence_boundary(&samples, 200, 100, 10);
        assert_eq!(boundary, 100); // Clamped to len

        // Empty
        let boundary = find_silence_boundary(&[], 0, 100, 10);
        assert_eq!(boundary, 0);
    }

    #[test]
    fn test_chunk_boundaries_vad_short_audio() {
        let config = ChunkConfig::new(30, 2, 16000);
        let samples = vec![0i16; 16000]; // 1 second, no chunking needed
        let boundaries = chunk_boundaries_vad(&samples, &config);
        assert_eq!(boundaries.len(), 1);
        assert_eq!(boundaries[0], (0, 16000));
    }

    #[test]
    fn test_chunk_boundaries_vad_long_audio() {
        let config = ChunkConfig::new(1, 0, 16000); // 1s chunks
        let samples = vec![0i16; 48000]; // 3 seconds
        let boundaries = chunk_boundaries_vad(&samples, &config);
        assert!(boundaries.len() >= 2);
        // Last boundary should reach end
        assert_eq!(boundaries.last().unwrap().1, 48000);
    }

    #[test]
    fn test_audio_statistics_helper() {
        // Test helper for calculating audio statistics
        fn audio_stats(samples: &[i16]) -> (i16, f64, f32) {
            if samples.is_empty() {
                return (0, 0.0, 0.0);
            }
            let max = samples.iter().map(|s| s.abs()).max().unwrap_or(0);
            let rms = (samples.iter().map(|&s| (s as f64).powi(2)).sum::<f64>()
                / samples.len() as f64)
                .sqrt();
            let duration = samples.len() as f32 / 16000.0;
            (max, rms, duration)
        }

        // Silent
        let silent = vec![0i16; 16000];
        let (max, rms, dur) = audio_stats(&silent);
        assert_eq!(max, 0);
        assert_eq!(rms, 0.0);
        assert!((dur - 1.0).abs() < 0.01);

        // Full amplitude
        let loud: Vec<i16> = vec![i16::MAX; 16000];
        let (max, rms, _) = audio_stats(&loud);
        assert_eq!(max, i16::MAX);
        assert!(rms > 30000.0);
    }
}
