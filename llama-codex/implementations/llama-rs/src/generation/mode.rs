use std::fmt;
use std::str::FromStr;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum GenerationMode {
    #[default]
    Generate,
    Chat,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ParseGenerationModeError {
    value: String,
}

impl GenerationMode {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Generate => "generate",
            Self::Chat => "chat",
        }
    }
}

impl FromStr for GenerationMode {
    type Err = ParseGenerationModeError;

    fn from_str(value: &str) -> Result<Self, Self::Err> {
        match value {
            "generate" | "gen" => Ok(Self::Generate),
            "chat" => Ok(Self::Chat),
            _ => Err(ParseGenerationModeError {
                value: value.to_string(),
            }),
        }
    }
}

impl fmt::Display for GenerationMode {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.as_str())
    }
}

impl fmt::Display for ParseGenerationModeError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "invalid generation mode `{}`; expected `generate` or `chat`",
            self.value
        )
    }
}

impl std::error::Error for ParseGenerationModeError {}

#[cfg(test)]
mod tests {
    use super::GenerationMode;
    use std::str::FromStr;

    #[test]
    fn defaults_to_generate() {
        assert_eq!(GenerationMode::default(), GenerationMode::Generate);
    }

    #[test]
    fn parses_supported_modes() {
        assert_eq!(
            GenerationMode::from_str("generate").expect("mode should parse"),
            GenerationMode::Generate
        );
        assert_eq!(
            GenerationMode::from_str("chat").expect("mode should parse"),
            GenerationMode::Chat
        );
    }

    #[test]
    fn rejects_unknown_modes() {
        assert!(GenerationMode::from_str("other").is_err());
    }
}
