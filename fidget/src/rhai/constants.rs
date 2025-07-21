//! Mathematical constants for Rhai scripts
use std::f64::consts;

/// Get a mathematical constant by name
pub fn get_constant(name: &str) -> Option<f64> {
    match name {
        // Basic mathematical constants
        "PI" => Some(consts::PI),
        "E" => Some(consts::E),
        "TAU" => Some(consts::TAU),
        
        // Square roots
        "SQRT_2" => Some(consts::SQRT_2),
        
        // Logarithms
        "LN_2" => Some(consts::LN_2),
        "LN_10" => Some(consts::LN_10),
        "LOG2_E" => Some(consts::LOG2_E),
        "LOG10_E" => Some(consts::LOG10_E),
        
        // Fractions of PI
        "FRAC_PI_2" => Some(consts::FRAC_PI_2),
        "FRAC_PI_3" => Some(consts::FRAC_PI_3),
        "FRAC_PI_4" => Some(consts::FRAC_PI_4),
        "FRAC_PI_6" => Some(consts::FRAC_PI_6),
        "FRAC_PI_8" => Some(consts::FRAC_PI_8),
        
        // Reciprocals of PI
        "FRAC_1_PI" => Some(consts::FRAC_1_PI),
        "FRAC_2_PI" => Some(consts::FRAC_2_PI),
        "FRAC_2_SQRT_PI" => Some(consts::FRAC_2_SQRT_PI),
        
        // Golden ratio and related constants
        "PHI" | "GOLDEN_RATIO" => Some(1.618033988749895_f64), // Golden ratio (1 + sqrt(5)) / 2
        
        // Common fractions
        "FRAC_1_SQRT_2" => Some(consts::FRAC_1_SQRT_2),
        
        _ => None,
    }
}

#[cfg(test)]
mod test {
    use super::*;
    
    #[test]
    fn test_get_constant() {
        // Test basic constants
        assert!((get_constant("PI").unwrap() - std::f64::consts::PI).abs() < f64::EPSILON);
        assert!((get_constant("E").unwrap() - std::f64::consts::E).abs() < f64::EPSILON);
        assert!((get_constant("TAU").unwrap() - std::f64::consts::TAU).abs() < f64::EPSILON);
        
        // Test golden ratio
        assert!((get_constant("PHI").unwrap() - 1.618033988749895_f64).abs() < f64::EPSILON);
        assert!((get_constant("GOLDEN_RATIO").unwrap() - 1.618033988749895_f64).abs() < f64::EPSILON);
        
        // Test fractions of PI
        assert!((get_constant("FRAC_PI_2").unwrap() - std::f64::consts::FRAC_PI_2).abs() < f64::EPSILON);
        
        // Test unknown constant
        assert!(get_constant("UNKNOWN").is_none());
    }
}
