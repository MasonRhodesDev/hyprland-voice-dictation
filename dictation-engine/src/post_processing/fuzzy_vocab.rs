use super::TextProcessor;
use crate::user_dictionary::UserDictionary;
use anyhow::Result;
use std::collections::HashMap;
use std::sync::Arc;

/// Fuzzy vocabulary post-processor.
///
/// Snaps transcribed tokens onto the user's glossary (the user dictionary) so
/// vendor/product names and acronyms come out right regardless of the engine.
/// Unlike [`WordSubstitutionProcessor`](super::WordSubstitutionProcessor),
/// which needs an exact spoken→written rule, this corrects *unseen* near-misses.
///
/// Two matching strategies, tried per position, longest run first:
///
/// 1. **Exact (normalized) n-gram** — join up to [`MAX_JOIN`] spoken tokens,
///    strip case/punctuation/hyphens, and look the result up in the glossary.
///    This deterministically fixes splits (`node red` → `node-red`), multi-part
///    names (`aws agent tools` → `aws-agent-tools`), and normalizes casing to
///    the dictionary form (`LifeMD` → `lifemd`). Zero false-positive risk.
///
/// Canonical output uses the dictionary's spelling, which is always lowercase
/// (the dictionary lowercases on load), so this corrects recognition — not
/// mixed-case styling.
/// 2. **Guarded single-token fuzzy** — for longer tokens only, accept a
///    glossary term when the Jaro-Winkler similarity is high, or when the
///    phonetic keys match and similarity is at least moderate. This catches
///    spelling/ASR variants (`hyperland` → `hyprland`).
///
/// The glossary is read fresh on every call, so `dict add` takes effect on the
/// next utterance without a restart.
pub struct FuzzyVocabularyProcessor {
    dictionary: Arc<UserDictionary>,
}

/// Minimum normalized length for a term to be eligible for *fuzzy* (non-exact)
/// matching. Short tokens/acronyms match only exactly, so common words like
/// "add" are never rewritten to a short glossary entry like "ad".
const MIN_FUZZY_LEN: usize = 5;
/// Jaro-Winkler score above which a single-token fuzzy match is accepted on
/// spelling similarity alone.
const HIGH_SIMILARITY: f64 = 0.92;
/// Lower Jaro-Winkler bound required when the phonetic keys also match.
const PHONETIC_SIMILARITY: f64 = 0.82;
/// Longest run of spoken tokens joined when matching a multi-part glossary
/// term (e.g. "aws agent tools" -> "aws-agent-tools").
const MAX_JOIN: usize = 3;

struct Glossary {
    /// normalized form -> canonical spelling (for exact n-gram lookup).
    exact: HashMap<String, String>,
    /// Entries eligible for single-token fuzzy matching (normalized len >= MIN).
    fuzzy: Vec<FuzzyEntry>,
}

struct FuzzyEntry {
    canonical: String,
    normalized: String,
    phonetic: String,
}

impl FuzzyVocabularyProcessor {
    pub fn new(dictionary: Arc<UserDictionary>) -> Self {
        Self { dictionary }
    }

    fn build_glossary(&self) -> Glossary {
        let mut exact = HashMap::new();
        let mut fuzzy = Vec::new();
        for word in self.dictionary.app_words() {
            let normalized = normalize(&word);
            if normalized.is_empty() {
                continue;
            }
            // First writer wins if two canonical forms normalize identically.
            exact.entry(normalized.clone()).or_insert_with(|| word.clone());
            if normalized.chars().count() >= MIN_FUZZY_LEN {
                fuzzy.push(FuzzyEntry {
                    canonical: word.clone(),
                    phonetic: phonetic_key(&normalized),
                    normalized,
                });
            }
        }
        Glossary { exact, fuzzy }
    }
}

impl TextProcessor for FuzzyVocabularyProcessor {
    fn process(&self, text: &str) -> Result<String> {
        if text.is_empty() {
            return Ok(String::new());
        }

        let glossary = self.build_glossary();
        if glossary.exact.is_empty() {
            return Ok(text.to_string());
        }

        let tokens: Vec<Token> = text.split_whitespace().map(Token::split).collect();
        let mut out: Vec<String> = Vec::with_capacity(tokens.len());
        let mut i = 0;

        while i < tokens.len() {
            // 1. Exact normalized n-gram, longest run first.
            let mut matched = false;
            let max_n = MAX_JOIN.min(tokens.len() - i);
            for n in (1..=max_n).rev() {
                if n > 1 && !joinable(&tokens[i..i + n]) {
                    continue;
                }
                let joined: String =
                    tokens[i..i + n].iter().map(|t| normalize(&t.core)).collect();
                if joined.is_empty() {
                    continue;
                }
                if let Some(canonical) = glossary.exact.get(&joined) {
                    out.push(format!(
                        "{}{}{}",
                        tokens[i].prefix,
                        canonical,
                        tokens[i + n - 1].suffix
                    ));
                    i += n;
                    matched = true;
                    break;
                }
            }
            if matched {
                continue;
            }

            // 2. Guarded single-token fuzzy match.
            let token = &tokens[i];
            let norm = normalize(&token.core);
            if let Some(canonical) = best_fuzzy_match(&norm, &glossary.fuzzy) {
                out.push(format!("{}{}{}", token.prefix, canonical, token.suffix));
            } else {
                out.push(token.original.clone());
            }
            i += 1;
        }

        Ok(out.join(" "))
    }
}

/// A whitespace-delimited token split into leading punctuation, an alphanumeric
/// "core", and trailing punctuation, so replacements preserve surrounding marks.
struct Token {
    original: String,
    prefix: String,
    core: String,
    suffix: String,
}

impl Token {
    fn split(raw: &str) -> Self {
        let start = raw.find(|c: char| c.is_alphanumeric());
        let (prefix, rest) = match start {
            Some(s) => raw.split_at(s),
            None => (raw, ""), // no alphanumerics at all
        };
        let end = rest.rfind(|c: char| c.is_alphanumeric());
        let (core, suffix) = match end {
            Some(e) => rest.split_at(e + rest[e..].chars().next().map(char::len_utf8).unwrap_or(1)),
            None => ("", rest),
        };
        Token {
            original: raw.to_string(),
            prefix: prefix.to_string(),
            core: core.to_string(),
            suffix: suffix.to_string(),
        }
    }
}

/// A run of tokens can be joined only when no interior boundary carries
/// punctuation (the first token may have a prefix, the last a suffix).
fn joinable(run: &[Token]) -> bool {
    let n = run.len();
    // No trailing punctuation on any token but the last.
    if run[..n - 1].iter().any(|t| !t.suffix.is_empty()) {
        return false;
    }
    // No leading punctuation on any token but the first.
    if run[1..].iter().any(|t| !t.prefix.is_empty()) {
        return false;
    }
    // Every core must be non-empty.
    run.iter().all(|t| !t.core.is_empty())
}

/// Lowercase and drop everything but alphanumerics — the comparison key that
/// makes casing, hyphens, and split words all compare equal.
fn normalize(s: &str) -> String {
    s.chars().filter(|c| c.is_alphanumeric()).flat_map(|c| c.to_lowercase()).collect()
}

/// A Soundex-style phonetic signature: first letter, then the sequence of
/// consonant-group codes with adjacent duplicates collapsed and vowels dropped.
/// Kept full-length (not truncated to 4) to reduce collisions.
fn phonetic_key(normalized: &str) -> String {
    let mut out = String::new();
    let mut last_code: Option<char> = None;
    for (idx, ch) in normalized.chars().enumerate() {
        let code = soundex_code(ch);
        if idx == 0 {
            out.push(ch);
            last_code = code;
            continue;
        }
        match code {
            Some(c) => {
                if Some(c) != last_code {
                    out.push(c);
                }
                last_code = Some(c);
            }
            None => last_code = None,
        }
    }
    out
}

fn soundex_code(c: char) -> Option<char> {
    match c.to_ascii_lowercase() {
        'b' | 'f' | 'p' | 'v' => Some('1'),
        'c' | 'g' | 'j' | 'k' | 'q' | 's' | 'x' | 'z' => Some('2'),
        'd' | 't' => Some('3'),
        'l' => Some('4'),
        'm' | 'n' => Some('5'),
        'r' => Some('6'),
        _ => None,
    }
}

/// Best glossary canonical for a normalized token, or `None` if nothing clears
/// the confidence bar. Exact matches are handled earlier, so a token equal to a
/// glossary term returns `None` here (no change needed).
fn best_fuzzy_match(norm: &str, entries: &[FuzzyEntry]) -> Option<String> {
    if norm.chars().count() < MIN_FUZZY_LEN {
        return None;
    }
    let token_phon = phonetic_key(norm);
    let mut best: Option<(f64, &str)> = None;
    for entry in entries {
        if entry.normalized == norm {
            return None; // exact — leave it (already correct)
        }
        let sim = strsim::jaro_winkler(norm, &entry.normalized);
        let accept = sim >= HIGH_SIMILARITY
            || (token_phon == entry.phonetic && sim >= PHONETIC_SIMILARITY);
        if accept && best.map(|(b, _)| sim > b).unwrap_or(true) {
            best = Some((sim, &entry.canonical));
        }
    }
    best.map(|(_, canonical)| canonical.to_string())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::user_dictionary::UserDictionary;
    use std::io::Write;
    use tempfile::NamedTempFile;

    fn processor_with(words: &[&str]) -> FuzzyVocabularyProcessor {
        let mut file = NamedTempFile::new().unwrap();
        for w in words {
            writeln!(file, "{w}").unwrap();
        }
        file.flush().unwrap();
        let dict = UserDictionary::with_paths(file.path().to_path_buf(), None).unwrap();
        // Words are loaded eagerly at construction, so the temp file can go now.
        drop(file);
        FuzzyVocabularyProcessor::new(Arc::new(dict))
    }

    #[test]
    fn empty_input() {
        let p = processor_with(&["lifemd"]);
        assert_eq!(p.process("").unwrap(), "");
    }

    #[test]
    fn passthrough_without_glossary() {
        let p = processor_with(&[]);
        assert_eq!(p.process("hello world").unwrap(), "hello world");
    }

    #[test]
    fn normalizes_to_dictionary_form() {
        // The dictionary stores lowercase, so canonical output is lowercase.
        let p = processor_with(&["lifemd"]);
        assert_eq!(p.process("i work at LifeMD today").unwrap(), "i work at lifemd today");
    }

    #[test]
    fn split_word_join() {
        let p = processor_with(&["lifemd"]);
        assert_eq!(p.process("open life md now").unwrap(), "open lifemd now");
    }

    #[test]
    fn hyphenated_multipart_join() {
        let p = processor_with(&["aws-agent-tools"]);
        assert_eq!(p.process("run aws agent tools please").unwrap(), "run aws-agent-tools please");
    }

    #[test]
    fn fuzzy_spelling_variant() {
        let p = processor_with(&["hyprland"]);
        assert_eq!(p.process("using hyperland here").unwrap(), "using hyprland here");
    }

    #[test]
    fn short_words_not_fuzzy_matched() {
        // "add" must not be rewritten to the short acronym "ad".
        let p = processor_with(&["ad"]);
        assert_eq!(p.process("please add this").unwrap(), "please add this");
    }

    #[test]
    fn preserves_trailing_punctuation() {
        let p = processor_with(&["chezmoi"]);
        assert_eq!(p.process("edit chezmoy, now").unwrap(), "edit chezmoi, now");
    }

    #[test]
    fn leaves_unrelated_words_alone() {
        let p = processor_with(&["kubernetes"]);
        assert_eq!(p.process("the weather is nice").unwrap(), "the weather is nice");
    }
}
