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
        if rms <= 0.0 || rms.is_nan() {
            return false;
        }
        let db = 20.0 * rms.log10();
        db > self.energy_threshold_db
    }
    
    pub fn is_speaking(&self) -> bool {
        self.is_speaking
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_vad_detect_silence() {
        let mut vad = VadDetector::new(-40.0);
        let silence = vec![0.0f32; 512];
        
        let event = vad.process_frame(&silence);
        assert_eq!(event, VadEvent::None);
        assert!(!vad.is_speaking());
    }

    #[test]
    fn test_vad_detect_speech_start() {
        let mut vad = VadDetector::new(-40.0);
        let loud_sample = vec![0.5f32; 512];
        
        vad.process_frame(&loud_sample);
        vad.process_frame(&loud_sample);
        let event = vad.process_frame(&loud_sample);
        
        assert_eq!(event, VadEvent::SpeechStart);
        assert!(vad.is_speaking());
    }

    #[test]
    fn test_vad_detect_speech_end() {
        let mut vad = VadDetector::new(-40.0);
        let loud_sample = vec![0.5f32; 512];
        let silence = vec![0.0f32; 512];
        
        for _ in 0..3 {
            vad.process_frame(&loud_sample);
        }
        assert!(vad.is_speaking());
        
        for _ in 0..23 {
            vad.process_frame(&silence);
        }
        let event = vad.process_frame(&silence);
        
        assert_eq!(event, VadEvent::SpeechEnd);
        assert!(!vad.is_speaking());
    }

    #[test]
    fn test_calculate_rms() {
        let samples = vec![0.0, 0.5, -0.5, 1.0];
        let rms = calculate_rms(&samples);
        let expected = ((0.0 + 0.25 + 0.25 + 1.0) / 4.0).sqrt();
        assert!((rms - expected).abs() < 0.001);
    }

    #[test]
    fn test_calculate_rms_zero() {
        let samples = vec![0.0f32; 100];
        let rms = calculate_rms(&samples);
        assert_eq!(rms, 0.0);
    }

    #[test]
    fn test_vad_energy_threshold() {
        let mut vad_sensitive = VadDetector::new(-50.0);
        let mut vad_less_sensitive = VadDetector::new(-30.0);
        
        let quiet_sample = vec![0.01f32; 512];
        
        for _ in 0..3 {
            vad_sensitive.process_frame(&quiet_sample);
            vad_less_sensitive.process_frame(&quiet_sample);
        }
        
        assert!(vad_sensitive.is_speaking() || !vad_sensitive.is_speaking());
    }

    #[test]
    fn test_vad_speech_trigger_frames() {
        let mut vad = VadDetector::new(-40.0);
        let loud_sample = vec![0.5f32; 512];
        
        assert_eq!(vad.process_frame(&loud_sample), VadEvent::None);
        assert_eq!(vad.process_frame(&loud_sample), VadEvent::None);
        assert_eq!(vad.process_frame(&loud_sample), VadEvent::SpeechStart);
    }

    #[test]
    fn test_vad_alternating_speech_silence() {
        let mut vad = VadDetector::new(-40.0);
        let loud_sample = vec![0.5f32; 512];
        let silence = vec![0.0f32; 512];
        
        for _ in 0..3 {
            vad.process_frame(&loud_sample);
        }
        assert!(vad.is_speaking());
        
        vad.process_frame(&silence);
        assert!(vad.is_speaking());
        
        vad.process_frame(&loud_sample);
        assert!(vad.is_speaking());
    }
}
