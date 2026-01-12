//! Audio chunking utilities for long transcriptions
//!
//! Provides reusable chunking and merging logic for transcription engines
//! that have context length limits.

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
pub struct AudioChunks<'a> {
    samples: &'a [i16],
    config: ChunkConfig,
    offset: usize,
    chunk_num: usize,
}

impl<'a> AudioChunks<'a> {
    /// Create a new chunk iterator
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

/// Merge transcription chunks, removing duplicate words at overlap boundaries
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
