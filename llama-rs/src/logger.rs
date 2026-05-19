// @trace-pilot 5f50038dae75a7ab6c556f586a9adb5d86c3b026
use std::fmt;
use std::fs::{File, OpenOptions};
use std::io::Write;
use std::sync::OnceLock;
use std::time::{SystemTime, UNIX_EPOCH};

static LOG_LEVEL: OnceLock<LogLevel> = OnceLock::new();
static LOG_FILE: OnceLock<File> = OnceLock::new();

#[derive(Clone, Copy, Debug, Eq, Ord, PartialEq, PartialOrd)]
pub enum LogLevel {
    Error,
    Warn,
    Info,
    Debug,
    Trace,
}

impl LogLevel {
    fn as_str(self) -> &'static str {
        match self {
            Self::Error => "ERROR",
            Self::Warn => "WARN",
            Self::Info => "INFO",
            Self::Debug => "DEBUG",
            Self::Trace => "TRACE",
        }
    }

    fn from_str(level: &str) -> Option<Self> {
        match level.to_ascii_lowercase().as_str() {
            "error" => Some(Self::Error),
            "warn" | "warning" => Some(Self::Warn),
            "info" => Some(Self::Info),
            "debug" => Some(Self::Debug),
            "trace" => Some(Self::Trace),
            _ => None,
        }
    }
}

pub fn init() {
    let level = std::env::var("LLAMA_RS_LOG")
        .ok()
        .as_deref()
        .and_then(LogLevel::from_str)
        .unwrap_or(LogLevel::Info);
    let _ = LOG_LEVEL.set(level);
    let _ = LOG_FILE.set(open_log_file());
}

pub fn init_with_level(level: LogLevel) {
    let _ = LOG_LEVEL.set(level);
    let _ = LOG_FILE.set(open_log_file());
}

pub fn enabled(level: LogLevel) -> bool {
    level <= current_level()
}

pub fn log(level: LogLevel, module_path: &str, file: &str, line: u32, args: fmt::Arguments<'_>) {
    if !enabled(level) {
        return;
    }

    let timestamp = timestamp_millis();
    let mut log_file = LOG_FILE.get_or_init(open_log_file);
    let _ = writeln!(
        &mut log_file,
        "[{}] {:>5} {} {}:{} {}",
        timestamp,
        level.as_str(),
        module_path,
        file,
        line,
        args
    );
}

fn current_level() -> LogLevel {
    *LOG_LEVEL.get_or_init(|| LogLevel::Info)
}

fn open_log_file() -> File {
    OpenOptions::new()
        .create(true)
        .append(true)
        .open("log.txt")
        .expect("failed to open log.txt")
}

fn timestamp_millis() -> u128 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|duration| duration.as_millis())
        .unwrap_or(0)
}

#[macro_export]
macro_rules! log_error {
    ($($arg:tt)*) => {
        $crate::logger::log(
            $crate::logger::LogLevel::Error,
            module_path!(),
            file!(),
            line!(),
            format_args!($($arg)*),
        )
    };
}

#[macro_export]
macro_rules! log_warn {
    ($($arg:tt)*) => {
        $crate::logger::log(
            $crate::logger::LogLevel::Warn,
            module_path!(),
            file!(),
            line!(),
            format_args!($($arg)*),
        )
    };
}

#[macro_export]
macro_rules! log_info {
    ($($arg:tt)*) => {
        $crate::logger::log(
            $crate::logger::LogLevel::Info,
            module_path!(),
            file!(),
            line!(),
            format_args!($($arg)*),
        )
    };
}

#[macro_export]
macro_rules! log_debug {
    ($($arg:tt)*) => {
        $crate::logger::log(
            $crate::logger::LogLevel::Debug,
            module_path!(),
            file!(),
            line!(),
            format_args!($($arg)*),
        )
    };
}

#[macro_export]
macro_rules! log_trace {
    ($($arg:tt)*) => {
        $crate::logger::log(
            $crate::logger::LogLevel::Trace,
            module_path!(),
            file!(),
            line!(),
            format_args!($($arg)*),
        )
    };
}
