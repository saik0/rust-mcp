use serde::{Deserialize, Serialize};
use std::str::FromStr;
use std::{
    collections::{BTreeMap, HashMap},
    env,
    path::{Path, PathBuf},
    sync::{Arc, Mutex, OnceLock},
    time::Duration,
};
use tokio::sync::{Mutex as AsyncMutex, OwnedMutexGuard};

pub const DEFAULT_TARGET_DIR: &str = "target/mcp-inspections";
const DEFAULT_TIMEOUT_SECONDS: u64 = 60;
const DEFAULT_MAX_OUTPUT_BYTES: usize = 2 * 1024 * 1024;
const DEFAULT_MAX_OUTPUT_LINES: usize = 20_000;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum GatingMode {
    Strict,
    Lenient,
}

impl Default for GatingMode {
    fn default() -> Self {
        GatingMode::Strict
    }
}

impl FromStr for GatingMode {
    type Err = ();

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s.to_ascii_lowercase().as_str() {
            "lenient" => Ok(GatingMode::Lenient),
            "strict" => Ok(GatingMode::Strict),
            _ => Err(()),
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum ToolchainChannel {
    Stable,
    Nightly,
    Dev,
}

impl ToolchainChannel {
    pub fn is_nightly_like(self) -> bool {
        matches!(self, ToolchainChannel::Nightly | ToolchainChannel::Dev)
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct InspectionLimits {
    pub timeout_seconds: u64,
    pub max_output_bytes: usize,
    pub max_output_lines: usize,
}

impl Default for InspectionLimits {
    fn default() -> Self {
        Self {
            timeout_seconds: DEFAULT_TIMEOUT_SECONDS,
            max_output_bytes: DEFAULT_MAX_OUTPUT_BYTES,
            max_output_lines: DEFAULT_MAX_OUTPUT_LINES,
        }
    }
}

impl InspectionLimits {
    pub fn timeout(&self) -> Duration {
        Duration::from_secs(self.timeout_seconds)
    }
}

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct TruncationSummary {
    pub original_bytes: usize,
    pub original_lines: usize,
    pub kept_bytes: usize,
    pub kept_lines: usize,
    pub max_bytes: usize,
    pub max_lines: usize,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct InspectionProvenance {
    pub workspace_root: PathBuf,
    pub target_dir: PathBuf,
    pub env: BTreeMap<String, String>,
    pub gating_mode: GatingMode,
    pub toolchain_channel: ToolchainChannel,
    pub workspace_locked: bool,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub rustc_verbose_version: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub rust_analyzer_version: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub command: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub truncation: Option<TruncationSummary>,
}

impl InspectionProvenance {
    pub fn with_command(mut self, command: impl Into<String>) -> Self {
        self.command = Some(command.into());
        self
    }

    pub fn with_truncation(mut self, truncation: Option<TruncationSummary>) -> Self {
        self.truncation = truncation;
        self
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct InspectionResult {
    pub view: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub symbol: Option<String>,
    pub text: String,
    pub truncated: bool,
    #[serde(default)]
    pub diagnostics: Vec<String>,
    pub provenance: InspectionProvenance,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct InspectionCapabilities {
    pub toolchain_channel: ToolchainChannel,
    pub gating_mode: GatingMode,
    pub views: Vec<String>,
    pub limits: InspectionLimits,
    #[serde(default)]
    pub diagnostics: Vec<String>,
    pub provenance: InspectionProvenance,
}

#[derive(Debug, Clone)]
pub struct InspectionView {
    pub name: &'static str,
    pub description: &'static str,
    pub requires_nightly: bool,
    pub emit: Option<&'static str>,
    pub unpretty: Option<&'static str>,
}

impl InspectionView {
    pub fn curated() -> Vec<Self> {
        vec![
            InspectionView {
                name: "def",
                description: "Definition location and symbol identity",
                requires_nightly: false,
                emit: None,
                unpretty: None,
            },
            InspectionView {
                name: "types",
                description: "Type hierarchy for the symbol",
                requires_nightly: false,
                emit: None,
                unpretty: None,
            },
            InspectionView {
                name: "llvm-ir",
                description: "Lowered LLVM IR for a symbol",
                requires_nightly: false,
                emit: Some("llvm-ir"),
                unpretty: None,
            },
            InspectionView {
                name: "asm",
                description: "Assembly for a symbol",
                requires_nightly: false,
                emit: Some("asm"),
                unpretty: None,
            },
            InspectionView {
                name: "mir",
                description: "MIR for a symbol",
                requires_nightly: true,
                emit: None,
                unpretty: Some("mir"),
            },
        ]
    }

    pub fn find(name: &str) -> Option<Self> {
        Self::curated()
            .into_iter()
            .find(|view| view.name.eq_ignore_ascii_case(name))
    }
}

pub fn is_view_advertised(
    view: &InspectionView,
    channel: ToolchainChannel,
    gating_mode: GatingMode,
) -> bool {
    if view.requires_nightly
        && !channel.is_nightly_like()
        && matches!(gating_mode, GatingMode::Strict)
    {
        return false;
    }
    true
}

pub fn is_view_runnable(view: &InspectionView, channel: ToolchainChannel) -> bool {
    !(view.requires_nightly && !channel.is_nightly_like())
}

#[derive(Clone)]
pub struct InspectionContext {
    limits: InspectionLimits,
    gating_mode: GatingMode,
    toolchain_channel: ToolchainChannel,
    workspace_root: PathBuf,
    rustc_verbose_version: Option<String>,
    rust_analyzer_version: Option<String>,
    env: BTreeMap<String, String>,
    workspace_lock: Arc<AsyncMutex<()>>,
}

impl InspectionContext {
    pub fn new(workspace_root: impl Into<PathBuf>) -> Self {
        let root = workspace_root.into();
        let mut env = BTreeMap::new();
        env.insert(
            "CARGO_TARGET_DIR".to_string(),
            DEFAULT_TARGET_DIR.to_string(),
        );

        let toolchain = detect_toolchain_details();

        Self {
            limits: InspectionLimits::default(),
            gating_mode: default_gating_mode_from_env(),
            toolchain_channel: toolchain.channel,
            workspace_root: root.clone(),
            rustc_verbose_version: toolchain.rustc_verbose_version,
            rust_analyzer_version: detect_rust_analyzer_version(),
            env,
            workspace_lock: workspace_lock_for(&root),
        }
    }

    pub fn with_gating_mode(mut self, gating_mode: GatingMode) -> Self {
        self.gating_mode = gating_mode;
        self
    }

    pub fn limits(&self) -> &InspectionLimits {
        &self.limits
    }

    pub fn gating_mode(&self) -> GatingMode {
        self.gating_mode
    }

    pub fn toolchain_channel(&self) -> ToolchainChannel {
        self.toolchain_channel
    }

    pub fn env(&self) -> &BTreeMap<String, String> {
        &self.env
    }

    pub fn target_dir(&self) -> PathBuf {
        self.env
            .get("CARGO_TARGET_DIR")
            .map(|value| PathBuf::from(value))
            .unwrap_or_else(|| PathBuf::from(DEFAULT_TARGET_DIR))
    }

    pub fn workspace_root(&self) -> &Path {
        &self.workspace_root
    }

    pub async fn lock_workspace(&self) -> WorkspaceLockGuard {
        WorkspaceLockGuard {
            guard: Some(self.workspace_lock.clone().lock_owned().await),
        }
    }

    pub fn provenance(&self) -> InspectionProvenance {
        InspectionProvenance {
            workspace_root: self.workspace_root.clone(),
            target_dir: self.target_dir(),
            env: self.env.clone(),
            gating_mode: self.gating_mode,
            toolchain_channel: self.toolchain_channel,
            workspace_locked: false,
            rustc_verbose_version: self.rustc_verbose_version.clone(),
            rust_analyzer_version: self.rust_analyzer_version.clone(),
            command: None,
            truncation: None,
        }
    }
}

pub struct WorkspaceLockGuard {
    guard: Option<OwnedMutexGuard<()>>,
}

impl Drop for WorkspaceLockGuard {
    fn drop(&mut self) {
        self.guard.take();
    }
}

fn workspace_lock_for(workspace_root: &Path) -> Arc<AsyncMutex<()>> {
    static LOCKS: OnceLock<Mutex<HashMap<PathBuf, Arc<AsyncMutex<()>>>>> = OnceLock::new();
    let locks = LOCKS.get_or_init(|| Mutex::new(HashMap::new()));
    let mut guard = locks.lock().expect("workspace locks poisoned");
    guard
        .entry(workspace_root.to_path_buf())
        .or_insert_with(|| Arc::new(AsyncMutex::new(())))
        .clone()
}

fn default_gating_mode_from_env() -> GatingMode {
    match env::var("MCP_GATING_MODE") {
        Ok(value) => GatingMode::from_str(&value).unwrap_or_default(),
        Err(_) => GatingMode::Strict,
    }
}

#[derive(Clone, Debug)]
struct ToolchainDetails {
    channel: ToolchainChannel,
    rustc_verbose_version: Option<String>,
}

pub fn detect_toolchain_channel() -> ToolchainChannel {
    detect_toolchain_details().channel
}

fn detect_toolchain_details() -> ToolchainDetails {
    static DETAILS: OnceLock<ToolchainDetails> = OnceLock::new();

    DETAILS
        .get_or_init(|| {
            let output = std::process::Command::new("rustc").arg("-Vv").output();

            let stdout = match output {
                Ok(output) => String::from_utf8_lossy(&output.stdout).to_string(),
                Err(_) => {
                    return ToolchainDetails {
                        channel: ToolchainChannel::Stable,
                        rustc_verbose_version: None,
                    };
                }
            };

            let mut channel = ToolchainChannel::Stable;
            for line in stdout.lines() {
                if let Some(release) = line.strip_prefix("release:") {
                    if release.contains("nightly") {
                        channel = ToolchainChannel::Nightly;
                        break;
                    }

                    if release.contains("dev") {
                        channel = ToolchainChannel::Dev;
                        break;
                    }
                }
            }

            ToolchainDetails {
                channel,
                rustc_verbose_version: Some(stdout),
            }
        })
        .clone()
}

fn detect_rust_analyzer_version() -> Option<String> {
    static VERSION: OnceLock<Option<String>> = OnceLock::new();

    VERSION
        .get_or_init(|| {
            let ra_path = rust_analyzer_path();
            let output = std::process::Command::new(&ra_path)
                .arg("--version")
                .output()
                .ok()?;

            let text = String::from_utf8_lossy(&output.stdout).trim().to_string();
            if text.is_empty() { None } else { Some(text) }
        })
        .clone()
}

fn rust_analyzer_path() -> String {
    env::var("RUST_ANALYZER_PATH").unwrap_or_else(|_| {
        let home = env::var("HOME").unwrap_or_else(|_| ".".to_string());
        format!("{home}/.cargo/bin/rust-analyzer")
    })
}

pub fn truncate_with_limits(
    text: &str,
    limits: &InspectionLimits,
) -> (String, bool, Option<TruncationSummary>) {
    let original_bytes = text.as_bytes().len();
    let original_lines = text.lines().count();

    if original_bytes <= limits.max_output_bytes && original_lines <= limits.max_output_lines {
        return (text.to_string(), false, None);
    }

    let mut kept_bytes = 0usize;
    let mut kept_lines = 0usize;
    let mut truncated_output = String::new();

    for line in text.lines() {
        let line_with_newline = format!("{line}\n");
        let next_bytes = kept_bytes + line_with_newline.as_bytes().len();
        let next_lines = kept_lines + 1;

        if next_bytes > limits.max_output_bytes || next_lines > limits.max_output_lines {
            break;
        }

        truncated_output.push_str(&line_with_newline);
        kept_bytes = next_bytes;
        kept_lines = next_lines;
    }

    let marker = format!(
        "\n[truncated after {} lines/{} bytes; original {} lines/{} bytes; limits {} lines/{} bytes]",
        kept_lines,
        kept_bytes,
        original_lines,
        original_bytes,
        limits.max_output_lines,
        limits.max_output_bytes
    );
    truncated_output.push_str(&marker);

    let summary = TruncationSummary {
        original_bytes,
        original_lines,
        kept_bytes,
        kept_lines,
        max_bytes: limits.max_output_bytes,
        max_lines: limits.max_output_lines,
    };

    (truncated_output, true, Some(summary))
}
