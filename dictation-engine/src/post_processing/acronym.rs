use super::TextProcessor;
use anyhow::Result;
use std::collections::HashSet;

/// Acronym detection and conversion processor.
///
/// Converts letter-by-letter patterns to acronyms:
/// - "a p i" → "API"
/// - "h t t p" → "HTTP"
/// - "u r l" → "URL"
///
/// Uses a curated dictionary of common acronyms plus pattern matching
/// for generic 2-5 letter sequences.
pub struct AcronymProcessor {
    known_acronyms: HashSet<String>,
}

impl AcronymProcessor {
    /// Create a new acronym processor with curated dictionary.
    pub fn new() -> Self {
        let mut known_acronyms = HashSet::new();

        // Programming & Web
        known_acronyms.insert("API".to_string());
        known_acronyms.insert("HTTP".to_string());
        known_acronyms.insert("HTTPS".to_string());
        known_acronyms.insert("URL".to_string());
        known_acronyms.insert("URI".to_string());
        known_acronyms.insert("JSON".to_string());
        known_acronyms.insert("XML".to_string());
        known_acronyms.insert("HTML".to_string());
        known_acronyms.insert("CSS".to_string());
        known_acronyms.insert("SQL".to_string());
        known_acronyms.insert("REST".to_string());
        known_acronyms.insert("CRUD".to_string());
        known_acronyms.insert("CLI".to_string());
        known_acronyms.insert("GUI".to_string());
        known_acronyms.insert("SDK".to_string());
        known_acronyms.insert("IDE".to_string());

        // File formats & protocols
        known_acronyms.insert("PDF".to_string());
        known_acronyms.insert("CSV".to_string());
        known_acronyms.insert("SVG".to_string());
        known_acronyms.insert("PNG".to_string());
        known_acronyms.insert("JPG".to_string());
        known_acronyms.insert("JPEG".to_string());
        known_acronyms.insert("GIF".to_string());
        known_acronyms.insert("SSH".to_string());
        known_acronyms.insert("FTP".to_string());
        known_acronyms.insert("SMTP".to_string());
        known_acronyms.insert("TCP".to_string());
        known_acronyms.insert("UDP".to_string());
        known_acronyms.insert("IP".to_string());
        known_acronyms.insert("DNS".to_string());

        // Development tools & concepts
        known_acronyms.insert("GIT".to_string());
        known_acronyms.insert("NPM".to_string());
        known_acronyms.insert("CI".to_string());
        known_acronyms.insert("CD".to_string());
        known_acronyms.insert("AWS".to_string());
        known_acronyms.insert("VPN".to_string());
        known_acronyms.insert("RAM".to_string());
        known_acronyms.insert("CPU".to_string());
        known_acronyms.insert("GPU".to_string());
        known_acronyms.insert("SSD".to_string());
        known_acronyms.insert("HDD".to_string());
        known_acronyms.insert("USB".to_string());

        // Common tech acronyms
        known_acronyms.insert("AI".to_string());
        known_acronyms.insert("ML".to_string());
        known_acronyms.insert("NLP".to_string());
        known_acronyms.insert("UI".to_string());
        known_acronyms.insert("UX".to_string());
        known_acronyms.insert("QA".to_string());
        known_acronyms.insert("DB".to_string());
        known_acronyms.insert("OS".to_string());
        known_acronyms.insert("VM".to_string());

        Self { known_acronyms }
    }
}

impl TextProcessor for AcronymProcessor {
    fn process(&self, text: &str) -> Result<String> {
        if text.is_empty() {
            return Ok(String::new());
        }

        let words: Vec<&str> = text.split_whitespace().collect();
        let mut result = Vec::new();
        let mut i = 0;

        while i < words.len() {
            // Try to match an acronym pattern starting at position i
            if let Some((acronym, consumed)) = self.try_match_acronym(&words[i..]) {
                result.push(acronym);
                i += consumed;
            } else {
                result.push(words[i].to_string());
                i += 1;
            }
        }

        Ok(result.join(" "))
    }
}

impl AcronymProcessor {
    /// Try to match an acronym pattern starting from the beginning of the slice.
    ///
    /// Returns (acronym_string, number_of_words_consumed) if successful.
    fn try_match_acronym(&self, words: &[&str]) -> Option<(String, usize)> {
        if words.is_empty() {
            return None;
        }

        // Try to match 2-5 letter sequences
        for length in (2..=5.min(words.len())).rev() {
            let candidate_words = &words[..length];

            // Check if all words are single characters (letters)
            if !candidate_words
                .iter()
                .all(|w| w.len() == 1 && w.chars().next().unwrap().is_alphabetic())
            {
                continue;
            }

            // Build the acronym
            let acronym: String = candidate_words
                .iter()
                .map(|w| w.to_uppercase())
                .collect::<Vec<_>>()
                .join("");

            // Check if it's in our known acronyms dictionary
            if self.known_acronyms.contains(&acronym) {
                return Some((acronym, length));
            }
        }

        None
    }
}

impl Default for AcronymProcessor {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_empty_string() {
        let processor = AcronymProcessor::new();
        let result = processor.process("").unwrap();
        assert_eq!(result, "");
    }

    #[test]
    fn test_api_pattern() {
        let processor = AcronymProcessor::new();
        let result = processor.process("testing a p i integration").unwrap();
        assert_eq!(result, "testing API integration");
    }

    #[test]
    fn test_http_pattern() {
        let processor = AcronymProcessor::new();
        let result = processor.process("h t t p request").unwrap();
        assert_eq!(result, "HTTP request");
    }

    #[test]
    fn test_url_pattern() {
        let processor = AcronymProcessor::new();
        let result = processor.process("the u r l is valid").unwrap();
        assert_eq!(result, "the URL is valid");
    }

    #[test]
    fn test_multiple_acronyms() {
        let processor = AcronymProcessor::new();
        let result = processor.process("a p i uses h t t p").unwrap();
        assert_eq!(result, "API uses HTTP");
    }

    #[test]
    fn test_no_false_positives() {
        let processor = AcronymProcessor::new();
        let result = processor.process("i want a p e n").unwrap();
        // "a p e n" is not a known acronym, so should stay as-is
        assert_eq!(result, "i want a p e n");
    }

    #[test]
    fn test_mixed_content() {
        let processor = AcronymProcessor::new();
        let result = processor.process("the a p i needs better error handling").unwrap();
        assert_eq!(result, "the API needs better error handling");
    }

    #[test]
    fn test_already_capitalized() {
        let processor = AcronymProcessor::new();
        let result = processor.process("API is working").unwrap();
        assert_eq!(result, "API is working");
    }

    #[test]
    fn test_json_xml() {
        let processor = AcronymProcessor::new();
        let result = processor.process("j s o n and x m l formats").unwrap();
        assert_eq!(result, "JSON and XML formats");
    }

    #[test]
    fn test_two_letter_acronym() {
        let processor = AcronymProcessor::new();
        let result = processor.process("a i model").unwrap();
        assert_eq!(result, "AI model");
    }

    #[test]
    fn test_preserve_non_acronyms() {
        let processor = AcronymProcessor::new();
        let result = processor.process("hello world testing").unwrap();
        assert_eq!(result, "hello world testing");
    }
}
