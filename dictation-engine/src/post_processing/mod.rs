mod acronym;
mod grammar;
mod punctuation;

use crate::user_dictionary::UserDictionary;
use anyhow::Result;
use std::sync::Arc;

pub use acronym::AcronymProcessor;
pub use grammar::GrammarProcessor;
pub use punctuation::PunctuationProcessor;

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
        Self {
            processors: Vec::new(),
        }
    }

    /// Add a processor to the pipeline.
    pub fn add_processor(&mut self, processor: Box<dyn TextProcessor>) {
        self.processors.push(processor);
    }

    /// Create a pipeline from configuration.
    ///
    /// Enables processors based on configuration flags.
    /// Processors are applied in order: acronyms → punctuation → grammar.
    pub fn from_config(
        enable_acronyms: bool,
        enable_punctuation: bool,
        enable_grammar: bool,
    ) -> Self {
        Self::from_config_with_dict(enable_acronyms, enable_punctuation, enable_grammar, None)
    }

    /// Create a pipeline from configuration with optional user dictionary.
    ///
    /// Enables processors based on configuration flags.
    /// Processors are applied in order: acronyms → punctuation → grammar.
    pub fn from_config_with_dict(
        enable_acronyms: bool,
        enable_punctuation: bool,
        enable_grammar: bool,
        user_dict: Option<Arc<UserDictionary>>,
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

        // Finally apply grammar checking
        if enable_grammar {
            if let Some(dict) = user_dict {
                pipeline.add_processor(Box::new(GrammarProcessor::new_with_user_dictionary(
                    dict,
                )));
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
