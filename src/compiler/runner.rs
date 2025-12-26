use anyhow::{Context, Result};
use std::{
    collections::HashSet,
    path::{Path, PathBuf},
};
use tokio::{fs, process::Command};

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
    pub async fn run(&self, request: RunRequest) -> Result<RunResult> {
        fs::create_dir_all(&self.target_dir)
            .await
            .with_context(|| format!("creating target dir {}", self.target_dir.display()))?;

        let before = collect_files(&self.target_dir).await.unwrap_or_default();

        let mut command = Command::new("cargo");
        command.arg("rustc");
        command.env("CARGO_TARGET_DIR", &self.target_dir);
        command.arg("--offline");

        if let Some(manifest_path) = request.manifest_path {
            command.arg("--manifest-path");
            command.arg(manifest_path);
        }

        if let Some(package) = request.package {
            command.arg("--package");
            command.arg(package);
        }

        if let Some(target_triple) = request.target_triple {
            command.arg("--target");
            command.arg(target_triple);
        }

        if let Some(opt_level) = request.opt_level {
            command.arg("--");
            command.arg(format!("-Copt-level={opt_level}"));
        } else {
            command.arg("--");
        }

        if let Some(emit) = request.emit {
            command.arg(format!("--emit={emit}"));
        }

        if let Some(unpretty) = request.unpretty {
            command.arg(format!("-Zunpretty={unpretty}"));
        }

        for arg in request.additional_rustc_args {
            command.arg(arg);
        }

        let output = command
            .output()
            .await
            .context("running cargo rustc with inspection settings")?;

        let after = collect_files(&self.target_dir).await.unwrap_or_default();
        let artifacts = diff_paths(before, after, &self.target_dir);

        Ok(RunResult {
            status: output.status,
            stdout: String::from_utf8_lossy(&output.stdout).into_owned(),
            stderr: String::from_utf8_lossy(&output.stderr).into_owned(),
            artifacts,
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
}

/// Result of invoking `cargo rustc`.
#[derive(Debug, Clone)]
pub struct RunResult {
    pub status: std::process::ExitStatus,
    pub stdout: String,
    pub stderr: String,
    pub artifacts: Vec<PathBuf>,
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
