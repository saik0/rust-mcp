use crate::inspection::InspectionLimits;
use anyhow::{Context, Result};
use std::{
    collections::{BTreeMap, HashSet},
    fmt,
    path::{Path, PathBuf},
    process::Stdio,
    time::Duration,
};
use tokio::{fs, io::AsyncReadExt, process::Command, time::timeout};

/// Runs `cargo rustc` with an inspection-friendly configuration.
///
/// The runner keeps builds isolated in `target/mcp-inspections` (or a custom target
/// directory) and exposes a narrowly scoped API so callers can explicitly opt in to
/// compilation work. No background daemons are spawned – the command executes once
/// per request.
#[derive(Debug, Clone)]
pub struct CompilerRunner {
    target_dir: PathBuf,
}

#[derive(Debug)]
pub enum RunnerError {
    Timeout(Duration),
}

impl fmt::Display for RunnerError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            RunnerError::Timeout(duration) => {
                write!(
                    f,
                    "compiler run exceeded the {}s timeout",
                    duration.as_secs()
                )
            }
        }
    }
}

impl std::error::Error for RunnerError {}

impl Default for CompilerRunner {
    fn default() -> Self {
        Self::new()
    }
}

impl CompilerRunner {
    /// Creates a runner that writes artifacts to `target/mcp-inspections`.
    pub fn new() -> Self {
        Self {
            target_dir: PathBuf::from("target/mcp-inspections"),
        }
    }

    /// Creates a runner with a custom target directory.
    pub fn with_target_dir<T: Into<PathBuf>>(target_dir: T) -> Self {
        Self {
            target_dir: target_dir.into(),
        }
    }

    /// Execute `cargo rustc` and capture compiler output alongside any new artifacts
    /// produced in the configured target directory.
    ///
    /// - Runs in read-only mode with respect to the workspace by using an isolated
    ///   target directory (no background services).
    /// - Allows callers to select the target triple and optimization level.
    /// - Supports forwarding `--emit` and `-Zunpretty` flags to rustc.
    pub async fn run(&self, request: RunRequest, limits: &InspectionLimits) -> Result<RunResult> {
        fs::create_dir_all(&self.target_dir)
            .await
            .with_context(|| format!("creating target dir {}", self.target_dir.display()))?;

        let before = collect_files(&self.target_dir).await.unwrap_or_default();

        let mut command_line = vec!["cargo".to_string(), "rustc".to_string()];
        let mut command = Command::new("cargo");
        command.arg("rustc");
        command.env("CARGO_TARGET_DIR", &self.target_dir);
        command.arg("--offline");
        command_line.push("--offline".to_string());
        command.stdout(Stdio::piped());
        command.stderr(Stdio::piped());

        if let Some(manifest_path) = request.manifest_path {
            command.arg("--manifest-path");
            command.arg(&manifest_path);
            command_line.push("--manifest-path".to_string());
            command_line.push(manifest_path.display().to_string());
        }

        if let Some(package) = request.package {
            command.arg("--package");
            command.arg(&package);
            command_line.push("--package".to_string());
            command_line.push(package);
        }

        if let Some(target_triple) = request.target_triple {
            command.arg("--target");
            command.arg(&target_triple);
            command_line.push("--target".to_string());
            command_line.push(target_triple);
        }

        if let Some(opt_level) = request.opt_level {
            command.arg("--");
            command.arg(format!("-Copt-level={opt_level}"));
            command_line.push("--".to_string());
            command_line.push(format!("-Copt-level={opt_level}"));
        } else {
            command.arg("--");
            command_line.push("--".to_string());
        }

        if let Some(emit) = request.emit {
            command.arg(format!("--emit={emit}"));
            command_line.push(format!("--emit={emit}"));
        }

        if let Some(unpretty) = request.unpretty {
            command.arg(format!("-Zunpretty={unpretty}"));
            command_line.push(format!("-Zunpretty={unpretty}"));
        }

        for arg in request.additional_rustc_args.iter() {
            command.arg(arg);
            command_line.push(arg.clone());
        }

        for (key, value) in request.env {
            command.env(key, value);
        }
        command.env("CARGO_TARGET_DIR", &self.target_dir);

        let mut child = command
            .spawn()
            .context("running cargo rustc with inspection settings")?;

        let mut stdout = child
            .stdout
            .take()
            .ok_or_else(|| anyhow::anyhow!("failed to capture compiler stdout"))?;
        let mut stderr = child
            .stderr
            .take()
            .ok_or_else(|| anyhow::anyhow!("failed to capture compiler stderr"))?;

        let stdout_task = tokio::spawn(async move {
            let mut buf = Vec::new();
            stdout.read_to_end(&mut buf).await?;
            Ok::<_, anyhow::Error>(buf)
        });
        let stderr_task = tokio::spawn(async move {
            let mut buf = Vec::new();
            stderr.read_to_end(&mut buf).await?;
            Ok::<_, anyhow::Error>(buf)
        });

        let status = match timeout(limits.timeout(), child.wait()).await {
            Ok(result) => result.context("running cargo rustc with inspection settings")?,
            Err(_) => {
                let _ = child.kill().await;
                let _ = child.wait().await;
                return Err(RunnerError::Timeout(limits.timeout()).into());
            }
        };

        let stdout = stdout_task
            .await
            .context("joining compiler stdout task")?
            .context("reading compiler stdout")?;
        let stderr = stderr_task
            .await
            .context("joining compiler stderr task")?
            .context("reading compiler stderr")?;

        let after = collect_files(&self.target_dir).await.unwrap_or_default();
        let artifacts = diff_paths(before, after, &self.target_dir);

        Ok(RunResult {
            status,
            stdout: String::from_utf8_lossy(&stdout).into_owned(),
            stderr: String::from_utf8_lossy(&stderr).into_owned(),
            artifacts,
            command: command_line,
        })
    }
}

/// Parameters for a compiler run.
#[derive(Debug, Clone, Default)]
pub struct RunRequest {
    pub manifest_path: Option<PathBuf>,
    pub package: Option<String>,
    pub target_triple: Option<String>,
    pub opt_level: Option<String>,
    pub emit: Option<String>,
    pub unpretty: Option<String>,
    pub additional_rustc_args: Vec<String>,
    pub env: BTreeMap<String, String>,
}

/// Result of invoking `cargo rustc`.
#[derive(Debug, Clone)]
pub struct RunResult {
    pub status: std::process::ExitStatus,
    pub stdout: String,
    pub stderr: String,
    pub artifacts: Vec<PathBuf>,
    pub command: Vec<String>,
}

async fn collect_files(root: &Path) -> Result<HashSet<PathBuf>> {
    let mut stack = vec![root.to_path_buf()];
    let mut files = HashSet::new();

    while let Some(path) = stack.pop() {
        let mut entries = match fs::read_dir(&path).await {
            Ok(entries) => entries,
            Err(_) => continue,
        };

        while let Some(entry) = entries.next_entry().await? {
            let entry_path = entry.path();
            if entry.file_type().await?.is_dir() {
                stack.push(entry_path);
            } else if let Ok(relative) = entry_path.strip_prefix(root) {
                files.insert(relative.to_path_buf());
            }
        }
    }

    Ok(files)
}

fn diff_paths(before: HashSet<PathBuf>, after: HashSet<PathBuf>, root: &Path) -> Vec<PathBuf> {
    after
        .difference(&before)
        .map(|rel| root.join(rel))
        .collect()
}
