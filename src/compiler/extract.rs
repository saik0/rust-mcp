use crate::analyzer::symbol::SymbolIdentity;
use anyhow::{Result, bail};

/// Normalized representation of a symbol used when matching compiler artifacts.
///
/// The `def_name` follows Rust path semantics (e.g. `crate::module::item`).
/// `mangled` can be provided when the fully qualified mangled name is known,
/// otherwise the extractor will fall back to a best-effort prefix derived from
/// the path segments.
#[derive(Debug, Clone)]
pub struct NormalizedSymbol {
    pub def_name: String,
    pub item_name: String,
    pub mangled: Option<String>,
    pub target: Option<String>,
    mangled_prefix: String,
}

impl NormalizedSymbol {
    /// Build a normalized symbol from an existing [`SymbolIdentity`].
    ///
    /// The def-name is assembled as `crate::module::item`, and a mangling
    /// prefix is derived from the same segments.
    pub fn from_identity(identity: &SymbolIdentity) -> Self {
        let mut segments = vec![identity.crate_name.clone()];
        segments.extend(identity.module_path.clone());
        segments.push(identity.item_name.clone());

        let def_name = segments.join("::");
        let mangled_prefix = encode_rust_mangled_prefix(&segments);

        Self {
            def_name,
            item_name: identity.item_name.clone(),
            mangled: None,
            target: None,
            mangled_prefix,
        }
    }

    /// Attach a known mangled symbol name.
    pub fn with_mangled(mut self, mangled: impl Into<String>) -> Self {
        self.mangled = Some(mangled.into());
        self
    }

    /// Attach an explicit target triple hint.
    pub fn with_target(mut self, target: impl Into<String>) -> Self {
        self.target = Some(target.into());
        self
    }

    fn mangled_prefix(&self) -> &str {
        &self.mangled_prefix
    }
}

/// Extract MIR for a symbol using the def-name if available, otherwise falling
/// back to the item name.
pub fn extract_mir(mir_outputs: &[String], symbol: &NormalizedSymbol) -> Result<String> {
    let mut def_matches = Vec::new();
    let mut name_matches = Vec::new();

    for output in mir_outputs {
        for block in split_mir_blocks(output) {
            let header = block_header(&block);
            if block.contains(&symbol.def_name) {
                def_matches.push(Candidate {
                    header,
                    content: block.clone(),
                });
            } else if block.contains(&symbol.item_name) {
                name_matches.push(Candidate {
                    header,
                    content: block,
                });
            }
        }
    }

    let matches = if !def_matches.is_empty() {
        def_matches
    } else {
        name_matches
    };

    select_unique_match(matches, "MIR", symbol)
}

/// Extract LLVM IR for a symbol, preferring an exact mangled match and falling
/// back to mangled prefix or def-name in comments.
pub fn extract_llvm_ir(llvm_outputs: &[String], symbol: &NormalizedSymbol) -> Result<String> {
    let mut exact_matches = Vec::new();
    let mut prefix_matches = Vec::new();
    let mut def_name_matches = Vec::new();

    for output in llvm_outputs {
        for (name, block) in split_llvm_blocks(output) {
            let header = name.clone();
            if let Some(mangled) = &symbol.mangled {
                if name.contains(mangled) {
                    exact_matches.push(Candidate {
                        header,
                        content: block.clone(),
                    });
                    continue;
                }
            }

            if !symbol.mangled_prefix().is_empty() && name.contains(symbol.mangled_prefix()) {
                prefix_matches.push(Candidate {
                    header,
                    content: block.clone(),
                });
                continue;
            }

            if block.contains(&symbol.def_name) {
                def_name_matches.push(Candidate {
                    header,
                    content: block,
                });
            }
        }
    }

    if !exact_matches.is_empty() {
        return select_unique_match(exact_matches, "LLVM IR", symbol);
    }

    if !prefix_matches.is_empty() {
        return select_unique_match(prefix_matches, "LLVM IR (prefix)", symbol);
    }

    select_unique_match(def_name_matches, "LLVM IR", symbol)
}

/// Extract assembly for a symbol within the given target triple. Uses mangled
/// name, then prefix, then def-name matches.
pub fn extract_asm(
    assemblies: &[TargetedAssembly],
    symbol: &NormalizedSymbol,
    target_triple: &str,
) -> Result<String> {
    let mut exact_matches = Vec::new();
    let mut prefix_matches = Vec::new();
    let mut name_matches = Vec::new();

    let mut found_target = false;

    for asm in assemblies.iter().filter(|asm| asm.target == target_triple) {
        found_target = true;
        for (label, block) in split_asm_blocks(&asm.content) {
            let header = label.clone();
            if let Some(mangled) = &symbol.mangled {
                if label.contains(mangled) {
                    exact_matches.push(Candidate {
                        header,
                        content: block.clone(),
                    });
                    continue;
                }
            }

            if !symbol.mangled_prefix().is_empty() && label.contains(symbol.mangled_prefix()) {
                prefix_matches.push(Candidate {
                    header,
                    content: block.clone(),
                });
                continue;
            }

            if block.contains(&symbol.def_name) || block.contains(&symbol.item_name) {
                name_matches.push(Candidate {
                    header,
                    content: block,
                });
            }
        }
    }

    if !found_target {
        bail!(
            "No assembly artifacts available for target `{}` while searching for `{}`",
            target_triple,
            symbol.def_name
        );
    }

    if !exact_matches.is_empty() {
        return select_unique_match(exact_matches, "assembly", symbol);
    }

    if !prefix_matches.is_empty() {
        return select_unique_match(prefix_matches, "assembly (prefix)", symbol);
    }

    select_unique_match(name_matches, "assembly", symbol)
}

/// Assembly output tagged by target triple.
#[derive(Debug, Clone)]
pub struct TargetedAssembly {
    pub target: String,
    pub content: String,
}

#[derive(Debug)]
struct Candidate {
    header: String,
    content: String,
}

fn split_mir_blocks(output: &str) -> Vec<String> {
    let mut blocks = Vec::new();
    let mut current = String::new();
    let mut capturing = false;

    for line in output.lines() {
        if is_mir_header(line) {
            if capturing && !current.is_empty() {
                blocks.push(current.trim().to_string());
                current.clear();
            }
            capturing = true;
        }

        if capturing {
            if !current.is_empty() {
                current.push('\n');
            }
            current.push_str(line);
        }
    }

    if capturing && !current.is_empty() {
        blocks.push(current.trim().to_string());
    }

    blocks
}

fn split_llvm_blocks(output: &str) -> Vec<(String, String)> {
    let mut blocks = Vec::new();
    let mut current_name: Option<String> = None;
    let mut current_lines: Vec<&str> = Vec::new();

    for line in output.lines() {
        if line.trim_start().starts_with("define") {
            if let Some(name) = current_name.take() {
                blocks.push((name, current_lines.join("\n")));
                current_lines.clear();
            }
            current_name = extract_llvm_symbol_name(line);
        }

        if current_name.is_some() {
            current_lines.push(line);
        }
    }

    if let Some(name) = current_name {
        blocks.push((name, current_lines.join("\n")));
    }

    blocks
        .into_iter()
        .map(|(name, block)| (name, block))
        .collect()
}

fn split_asm_blocks(output: &str) -> Vec<(String, String)> {
    let mut blocks = Vec::new();
    let mut current_label: Option<String> = None;
    let mut current_lines: Vec<&str> = Vec::new();

    for line in output.lines() {
        let trimmed = line.trim();
        if trimmed.ends_with(':') && !trimmed.starts_with('#') {
            if let Some(label) = current_label.take() {
                blocks.push((label, current_lines.join("\n")));
                current_lines.clear();
            }
            current_label = Some(trimmed.trim_end_matches(':').trim_matches('"').to_string());
        }

        if current_label.is_some() {
            current_lines.push(line);
        }
    }

    if let Some(label) = current_label {
        blocks.push((label, current_lines.join("\n")));
    }

    blocks
        .into_iter()
        .map(|(label, block)| (label, block))
        .collect()
}

fn extract_llvm_symbol_name(line: &str) -> Option<String> {
    let after_at = line.split('@').nth(1)?;
    let name_part = after_at.split('(').next()?;
    Some(name_part.trim_matches('"').to_string())
}

fn is_mir_header(line: &str) -> bool {
    let trimmed = line.trim_start();
    trimmed.starts_with("fn ")
        || trimmed.starts_with("const ")
        || trimmed.starts_with("static ")
        || trimmed.starts_with("promoted[")
}

fn block_header(block: &str) -> String {
    block.lines().next().unwrap_or_default().trim().to_string()
}

fn select_unique_match(
    matches: Vec<Candidate>,
    what: &str,
    symbol: &NormalizedSymbol,
) -> Result<String> {
    if matches.is_empty() {
        let mut looked_for = vec![
            format!("def-name `{}`", symbol.def_name),
            format!("item name `{}`", symbol.item_name),
        ];

        if let Some(mangled) = &symbol.mangled {
            looked_for.push(format!("mangled `{mangled}`"));
        } else if !symbol.mangled_prefix().is_empty() {
            looked_for.push(format!("mangled prefix `{}`", symbol.mangled_prefix()));
        }

        bail!(
            "No {} match found for `{}` (looked for {})",
            what,
            symbol.def_name,
            looked_for.join(", ")
        );
    }

    if matches.len() > 1 {
        let headers: Vec<String> = matches.iter().map(|m| m.header.clone()).collect();
        bail!(
            "Multiple {} candidates matched `{}`: {}",
            what,
            symbol.def_name,
            headers.join(", ")
        );
    }

    Ok(matches[0].content.clone())
}

fn encode_rust_mangled_prefix(segments: &[String]) -> String {
    let mut encoded = String::from("_ZN");
    for segment in segments {
        encoded.push_str(&format!("{}{}", segment.len(), segment));
    }
    encoded
}

#[cfg(test)]
mod tests {
    use super::{NormalizedSymbol, TargetedAssembly, extract_asm, extract_llvm_ir, extract_mir};
    use crate::analyzer::symbol::{SymbolIdentity, SymbolKind};

    fn demo_symbol() -> NormalizedSymbol {
        let identity = SymbolIdentity {
            crate_name: "demo".to_string(),
            module_path: vec!["utils".to_string()],
            item_name: "do_thing".to_string(),
            kind: SymbolKind::FreeFunction,
        };

        NormalizedSymbol::from_identity(&identity)
    }

    #[test]
    fn builds_mangled_prefix_from_segments() {
        let identity = demo_symbol();
        assert_eq!(identity.mangled_prefix(), "_ZN4demo5utils8do_thing");
    }

    #[test]
    fn extracts_mir_by_def_name() {
        let mir = r#"
fn demo::utils::do_thing(_1: i32) -> i32 {
    bb0: {
        _0 = _1;
        return;
    }
}

fn demo::utils::other(_1: i32) -> i32 {
    bb0: { return; }
}
        "#;

        let symbol = demo_symbol();
        let extracted = extract_mir(&[mir.to_string()], &symbol).expect("mir extracted");
        assert!(extracted.contains("do_thing"));
        assert!(!extracted.contains("other(_1"));
    }

    #[test]
    fn extracts_llvm_ir_with_mangled_prefix() {
        let llvm = r#"
; ModuleID = 'demo'
source_filename = "demo"

define dso_local void @_ZN4demo5utils8do_thing17h1234abcdE() #0 {
entry-block:
  ret void
}

define dso_local void @_ZN4demo5utils9do_other17h99999999E() #0 {
entry-block:
  ret void
}
        "#;

        let symbol = demo_symbol();
        let extracted = extract_llvm_ir(&[llvm.to_string()], &symbol).expect("llvm extracted");
        assert!(extracted.contains("_ZN4demo5utils8do_thing"));
        assert!(!extracted.contains("do_other17h"));
    }

    #[test]
    fn extracts_assembly_for_target() {
        let asm = TargetedAssembly {
            target: "x86_64-unknown-linux-gnu".to_string(),
            content: r#"
    .section    .text
    .globl  _ZN4demo5utils8do_thing17h1234abcdE
_ZN4demo5utils8do_thing17h1234abcdE:
    retq

_ZN4demo5utils9do_other17h99999999E:
    retq
            "#
            .to_string(),
        };

        let symbol = demo_symbol().with_mangled("_ZN4demo5utils8do_thing17h1234abcdE");
        let extracted =
            extract_asm(&[asm], &symbol, "x86_64-unknown-linux-gnu").expect("asm extracted");
        assert!(extracted.contains("_ZN4demo5utils8do_thing17h1234abcdE:"));
        assert!(!extracted.contains("do_other17h"));
    }

    #[test]
    fn errors_when_target_missing() {
        let asm = TargetedAssembly {
            target: "aarch64-unknown-linux-gnu".to_string(),
            content: "_ZN4demo5utils8do_thing17h1234abcdE:\nret".to_string(),
        };

        let symbol = demo_symbol().with_mangled("_ZN4demo5utils8do_thing17h1234abcdE");
        let err = extract_asm(&[asm], &symbol, "x86_64-unknown-linux-gnu").unwrap_err();
        assert!(err.to_string().contains("No assembly artifacts"));
    }
}
