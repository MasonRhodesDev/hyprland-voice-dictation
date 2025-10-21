// Voice Activity Detection

pub struct VadDetector {
    energy_threshold_db: f32,
    speech_trigger_frames: usize,
    silence_trigger_frames: usize,
    
    speech_frames: usize,
    silence_frames: usize,
    is_speaking: bool,
}

impl VadDetector {
    pub fn new(energy_threshold_db: f32) -> Self {
        Self {
            energy_threshold_db,
            speech_trigger_frames: 3,
            silence_trigger_frames: 24,
            speech_frames: 0,
            silence_frames: 0,
            is_speaking: false,
        }
    }
    
    pub fn process_frame(&mut self, samples: &[f32]) -> VadEvent {
        let is_speech = self.detect_speech(samples);
        
        if is_speech {
            self.speech_frames += 1;
            self.silence_frames = 0;
            
            if !self.is_speaking && self.speech_frames >= self.speech_trigger_frames {
                self.is_speaking = true;
                return VadEvent::SpeechStart;
            }
        } else {
            self.silence_frames += 1;
            self.speech_frames = 0;
            
            if self.is_speaking && self.silence_frames >= self.silence_trigger_frames {
                self.is_speaking = false;
                return VadEvent::SpeechEnd;
            }
        }
        
        VadEvent::None
    }
    
    fn detect_speech(&self, samples: &[f32]) -> bool {
        let rms = calculate_rms(samples);
        let db = 20.0 * rms.log10();
        db > self.energy_threshold_db
    }
}

fn calculate_rms(samples: &[f32]) -> f32 {
    let sum: f32 = samples.iter().map(|s| s * s).sum();
    (sum / samples.len() as f32).sqrt()
}

#[derive(Debug, PartialEq)]
pub enum VadEvent {
    None,
    SpeechStart,
    SpeechEnd,
}
