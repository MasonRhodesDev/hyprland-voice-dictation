mod acronym;
mod fuzzy_vocab;
mod grammar;
mod punctuation;
mod sanitize;
mod word_substitution;

use crate::user_dictionary::UserDictionary;
use anyhow::Result;
use std::sync::Arc;

pub use acronym::AcronymProcessor;
pub use fuzzy_vocab::FuzzyVocabularyProcessor;
pub use grammar::GrammarProcessor;
pub use punctuation::PunctuationProcessor;
pub use sanitize::SanitizationProcessor;
pub use sanitize::SanitizationRules;
pub use word_substitution::WordSubstitutionProcessor;

/// Trait for text post-processors.
///
/// Processors transform transcribed text by applying corrections,
/// punctuation, capitalization, or other transformations.
pub trait TextProcessor: Send + Sync {
    /// Process the input text and return the transformed result.
    fn process(&self, text: &str) -> Result<String>;
}

/// Pipeline that orchestrates multiple text processors.
///
/// Processors are applied in sequence, with each processor
/// receiving the output of the previous one.
pub struct Pipeline {
    processors: Vec<Box<dyn TextProcessor>>,
}

impl Pipeline {
    /// Create an empty pipeline.
    pub fn new() -> Self {
        Self { processors: Vec::new() }
    }

    /// Add a processor to the pipeline.
    pub fn add_processor(&mut self, processor: Box<dyn TextProcessor>) {
        self.processors.push(processor);
    }

    /// Create a pipeline from configuration.
    ///
    /// Enables processors based on configuration flags.
    /// Processors are applied in order: acronyms → punctuation → word substitution → grammar.
    pub fn from_config(
        enable_acronyms: bool,
        enable_punctuation: bool,
        enable_grammar: bool,
    ) -> Self {
        Self::from_config_with_dict(
            enable_acronyms,
            enable_punctuation,
            enable_grammar,
            None,
            false,
            None,
            false,
        )
    }

    /// Create a pipeline from configuration with optional user dictionary and word substitution.
    ///
    /// Enables processors based on configuration flags. Processors are applied
    /// in order: acronyms → punctuation → word substitution → fuzzy vocabulary
    /// → grammar. Fuzzy vocabulary runs before grammar so corrected proper
    /// nouns are already in the user dictionary Harper trusts.
    #[allow(clippy::too_many_arguments)]
    pub fn from_config_with_dict(
        enable_acronyms: bool,
        enable_punctuation: bool,
        enable_grammar: bool,
        user_dict: Option<Arc<UserDictionary>>,
        enable_word_substitution: bool,
        word_sub: Option<WordSubstitutionProcessor>,
        enable_fuzzy_vocab: bool,
    ) -> Self {
        let mut pipeline = Self::new();

        // Apply acronym detection first (a p i → API)
        if enable_acronyms {
            pipeline.add_processor(Box::new(AcronymProcessor::new()));
        }

        // Then apply punctuation (capitalization)
        if enable_punctuation {
            pipeline.add_processor(Box::new(PunctuationProcessor::new()));
        }

        // Apply exact word substitutions (shay moy → chezmoi)
        if enable_word_substitution {
            if let Some(ws) = word_sub {
                pipeline.add_processor(Box::new(ws));
            }
        }

        // Snap remaining near-misses onto the user's glossary (life md → lifemd,
        // hyperland → hyprland). Needs the dictionary as its glossary source.
        if enable_fuzzy_vocab {
            if let Some(ref dict) = user_dict {
                pipeline
                    .add_processor(Box::new(FuzzyVocabularyProcessor::new(Arc::clone(dict))));
            }
        }

        // Finally apply grammar checking
        if enable_grammar {
            if let Some(dict) = user_dict {
                pipeline.add_processor(Box::new(GrammarProcessor::new_with_user_dictionary(dict)));
            } else {
                pipeline.add_processor(Box::new(GrammarProcessor::new()));
            }
        }

        pipeline
    }

    /// Process text through all processors in the pipeline.
    ///
    /// Returns the final processed result, or the original text
    /// if no processors are enabled.
    pub fn process(&self, text: &str) -> Result<String> {
        let mut result = text.to_string();

        for processor in &self.processors {
            result = processor.process(&result)?;
        }

        Ok(result)
    }

    /// Check if the pipeline has any processors.
    pub fn is_empty(&self) -> bool {
        self.processors.is_empty()
    }
}

impl Default for Pipeline {
    fn default() -> Self {
        Self::new()
    }
}
