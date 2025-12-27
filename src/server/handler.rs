use anyhow::Result;
use rmcp::{
    ServerHandler,
    handler::server::{router::tool::ToolRouter, tool::Parameters},
    model::{ErrorData as McpError, *},
    tool, tool_handler, tool_router,
};
use serde::Serialize;
use serde_json::{Value, json};
use std::{
    path::{Path, PathBuf},
    str::FromStr,
    sync::Arc,
};
use tokio::{fs, sync::Mutex};

use crate::analyzer::{
    RustAnalyzerClient,
    symbol::{SymbolIdentity, SymbolKind, identity_from_definition},
};
use crate::compiler::{
    CompilerRunner, RunRequest, RunResult, RunnerError,
    extract::{NormalizedSymbol, TargetedAssembly, extract_asm, extract_llvm_ir, extract_mir},
};
use crate::inspection::{
    GatingMode, InspectionCapabilities, InspectionContext, InspectionLimits, InspectionResult,
    InspectionView, TruncationSummary, is_view_advertised, is_view_runnable, truncate_with_limits,
};
use crate::server::parameters::*;
use crate::tools::{execute_tool, get_tools};

struct ResolvedDefinition {
    symbol: Option<SymbolIdentity>,
    text: String,
}

#[derive(Clone)]
pub struct RustMcpServer {
    analyzer: Arc<Mutex<RustAnalyzerClient>>,
    tool_router: ToolRouter<RustMcpServer>,
    inspection: InspectionContext,
}

impl Default for RustMcpServer {
    fn default() -> Self {
        Self::new()
    }
}

#[tool_router]
impl RustMcpServer {
    pub fn new() -> Self {
        let workspace_root = std::env::current_dir().unwrap_or_else(|_| PathBuf::from("."));
        Self {
            analyzer: Arc::new(Mutex::new(RustAnalyzerClient::new())),
            tool_router: Self::tool_router(),
            inspection: InspectionContext::new(workspace_root),
        }
    }

    pub async fn start(&mut self) -> Result<()> {
        let mut analyzer = self.analyzer.lock().await;
        analyzer.start().await
    }

    pub fn list_tools(&self) -> Vec<crate::tools::ToolDefinition> {
        get_tools()
    }

    pub async fn call_tool(&mut self, name: &str, args: Value) -> Result<crate::tools::ToolResult> {
        let mut analyzer = self.analyzer.lock().await;
        execute_tool(name, args, &mut analyzer).await
    }

    #[tool(description = "Discover supported inspection presets and limits")]
    async fn capabilities(
        &self,
        Parameters(CapabilitiesParams { gating_mode }): Parameters<CapabilitiesParams>,
    ) -> Result<CallToolResult, McpError> {
        let context = self.inspection_context(gating_mode.as_deref());

        let views = InspectionView::curated()
            .into_iter()
            .filter(|view| {
                is_view_advertised(view, context.toolchain_channel(), context.gating_mode())
            })
            .map(|view| view.name.to_string())
            .collect::<Vec<_>>();

        let mut diagnostics = Vec::new();
        if matches!(context.gating_mode(), GatingMode::Lenient)
            && !context.toolchain_channel().is_nightly_like()
        {
            diagnostics.extend(
                InspectionView::curated()
                    .into_iter()
                    .filter(|view| view.requires_nightly)
                    .map(|view| format!("View '{}' requires nightly", view.name)),
            );
        }

        let capabilities = InspectionCapabilities {
            toolchain_channel: context.toolchain_channel(),
            gating_mode: context.gating_mode(),
            views,
            limits: context.limits().clone(),
            diagnostics,
            provenance: context.provenance(),
        };

        Ok(CallToolResult::success(vec![json_content(capabilities)?]))
    }

    #[tool(description = "Inspect compiler artifacts using curated presets")]
    async fn inspect(
        &self,
        Parameters(InspectParams {
            view,
            file_path,
            line,
            character,
            symbol_name,
            opt_level,
            target,
            gating_mode,
        }): Parameters<InspectParams>,
    ) -> Result<CallToolResult, McpError> {
        let context = self.inspection_context(gating_mode.as_deref());
        let result = self
            .perform_inspection(
                &context,
                &view,
                &file_path,
                Some(line),
                Some(character),
                symbol_name,
                opt_level,
                target,
            )
            .await?;

        Ok(CallToolResult::success(vec![json_content(result)?]))
    }

    fn inspection_context(&self, gating_override: Option<&str>) -> InspectionContext {
        let mut context = self.inspection.clone();
        if let Some(mode) = gating_override.and_then(|value| GatingMode::from_str(value).ok()) {
            context = context.with_gating_mode(mode);
        }
        context
    }

    #[tool(description = "Find the definition of a symbol at a given position")]
    async fn find_definition(
        &self,
        Parameters(FindDefinitionParams {
            file_path,
            line,
            character,
        }): Parameters<FindDefinitionParams>,
    ) -> Result<CallToolResult, McpError> {
        let args = serde_json::json!({
            "file_path": file_path,
            "line": line,
            "character": character
        });

        let mut analyzer = self.analyzer.lock().await;
        match execute_tool("find_definition", args, &mut analyzer).await {
            Ok(result) => {
                if let Some(content) = result.content.first() {
                    if let Some(text) = content.get("text") {
                        return Ok(CallToolResult::success(vec![Content::text(
                            text.as_str().unwrap_or("No result"),
                        )]));
                    }
                }
                Ok(CallToolResult::success(vec![Content::text(
                    "No definition found",
                )]))
            }
            Err(e) => Ok(CallToolResult::success(vec![Content::text(format!(
                "Error: {e}"
            ))])),
        }
    }

    #[tool(description = "Find all references to a symbol at a given position")]
    async fn find_references(
        &self,
        Parameters(FindReferencesParams {
            file_path,
            line,
            character,
        }): Parameters<FindReferencesParams>,
    ) -> Result<CallToolResult, McpError> {
        let args = serde_json::json!({
            "file_path": file_path,
            "line": line,
            "character": character
        });

        let mut analyzer = self.analyzer.lock().await;
        match execute_tool("find_references", args, &mut analyzer).await {
            Ok(result) => {
                if let Some(content) = result.content.first() {
                    if let Some(text) = content.get("text") {
                        return Ok(CallToolResult::success(vec![Content::text(
                            text.as_str().unwrap_or("No result"),
                        )]));
                    }
                }
                Ok(CallToolResult::success(vec![Content::text(
                    "No references found",
                )]))
            }
            Err(e) => Ok(CallToolResult::success(vec![Content::text(format!(
                "Error: {e}"
            ))])),
        }
    }

    #[tool(description = "Get compiler diagnostics for a file")]
    async fn get_diagnostics(
        &self,
        Parameters(GetDiagnosticsParams { file_path }): Parameters<GetDiagnosticsParams>,
    ) -> Result<CallToolResult, McpError> {
        let args = serde_json::json!({
            "file_path": file_path
        });

        let mut analyzer = self.analyzer.lock().await;
        match execute_tool("get_diagnostics", args, &mut analyzer).await {
            Ok(result) => {
                if let Some(content) = result.content.first() {
                    if let Some(text) = content.get("text") {
                        return Ok(CallToolResult::success(vec![Content::text(
                            text.as_str().unwrap_or("No result"),
                        )]));
                    }
                }
                Ok(CallToolResult::success(vec![Content::text(
                    "No diagnostics found",
                )]))
            }
            Err(e) => Ok(CallToolResult::success(vec![Content::text(format!(
                "Error: {e}"
            ))])),
        }
    }

    #[tool(description = "Search for symbols in the workspace")]
    async fn workspace_symbols(
        &self,
        Parameters(WorkspaceSymbolsParams { query }): Parameters<WorkspaceSymbolsParams>,
    ) -> Result<CallToolResult, McpError> {
        let args = serde_json::json!({
            "query": query
        });

        let mut analyzer = self.analyzer.lock().await;
        match execute_tool("workspace_symbols", args, &mut analyzer).await {
            Ok(result) => {
                if let Some(content) = result.content.first() {
                    if let Some(text) = content.get("text") {
                        return Ok(CallToolResult::success(vec![Content::text(
                            text.as_str().unwrap_or("No result"),
                        )]));
                    }
                }
                Ok(CallToolResult::success(vec![Content::text(
                    "No symbols found",
                )]))
            }
            Err(e) => Ok(CallToolResult::success(vec![Content::text(format!(
                "Error: {e}"
            ))])),
        }
    }

    #[tool(description = "Rename a symbol with scope awareness")]
    async fn rename_symbol(
        &self,
        Parameters(RenameSymbolParams {
            file_path,
            line,
            character,
            new_name,
        }): Parameters<RenameSymbolParams>,
    ) -> Result<CallToolResult, McpError> {
        let args = serde_json::json!({
            "file_path": file_path,
            "line": line,
            "character": character,
            "new_name": new_name
        });

        let mut analyzer = self.analyzer.lock().await;
        match execute_tool("rename_symbol", args, &mut analyzer).await {
            Ok(result) => {
                if let Some(content) = result.content.first() {
                    if let Some(text) = content.get("text") {
                        return Ok(CallToolResult::success(vec![Content::text(
                            text.as_str().unwrap_or("No result"),
                        )]));
                    }
                }
                Ok(CallToolResult::success(vec![Content::text(
                    "Rename operation completed",
                )]))
            }
            Err(e) => Ok(CallToolResult::success(vec![Content::text(format!(
                "Error: {e}"
            ))])),
        }
    }

    #[tool(description = "Apply rustfmt formatting to a file")]
    async fn format_code(
        &self,
        Parameters(FormatCodeParams { file_path }): Parameters<FormatCodeParams>,
    ) -> Result<CallToolResult, McpError> {
        let args = serde_json::json!({
            "file_path": file_path
        });

        let mut analyzer = self.analyzer.lock().await;
        match execute_tool("format_code", args, &mut analyzer).await {
            Ok(result) => {
                if let Some(content) = result.content.first() {
                    if let Some(text) = content.get("text") {
                        return Ok(CallToolResult::success(vec![Content::text(
                            text.as_str().unwrap_or("No result"),
                        )]));
                    }
                }
                Ok(CallToolResult::success(vec![Content::text(
                    "Format operation completed",
                )]))
            }
            Err(e) => Ok(CallToolResult::success(vec![Content::text(format!(
                "Error: {e}"
            ))])),
        }
    }

    #[tool(description = "Parse and analyze Cargo.toml file")]
    async fn analyze_manifest(
        &self,
        Parameters(AnalyzeManifestParams { manifest_path }): Parameters<AnalyzeManifestParams>,
    ) -> Result<CallToolResult, McpError> {
        let args = serde_json::json!({
            "manifest_path": manifest_path
        });

        let mut analyzer = self.analyzer.lock().await;
        match execute_tool("analyze_manifest", args, &mut analyzer).await {
            Ok(result) => {
                if let Some(content) = result.content.first() {
                    if let Some(text) = content.get("text") {
                        return Ok(CallToolResult::success(vec![Content::text(
                            text.as_str().unwrap_or("No result"),
                        )]));
                    }
                }
                Ok(CallToolResult::success(vec![Content::text(
                    "Analysis completed",
                )]))
            }
            Err(e) => Ok(CallToolResult::success(vec![Content::text(format!(
                "Error: {e}"
            ))])),
        }
    }

    #[tool(description = "Execute cargo check and parse errors")]
    async fn run_cargo_check(
        &self,
        Parameters(RunCargoCheckParams { workspace_path }): Parameters<RunCargoCheckParams>,
    ) -> Result<CallToolResult, McpError> {
        let args = serde_json::json!({
            "workspace_path": workspace_path
        });

        let mut analyzer = self.analyzer.lock().await;
        match execute_tool("run_cargo_check", args, &mut analyzer).await {
            Ok(result) => {
                if let Some(content) = result.content.first() {
                    if let Some(text) = content.get("text") {
                        return Ok(CallToolResult::success(vec![Content::text(
                            text.as_str().unwrap_or("No result"),
                        )]));
                    }
                }
                Ok(CallToolResult::success(vec![Content::text(
                    "Cargo check completed",
                )]))
            }
            Err(e) => Ok(CallToolResult::success(vec![Content::text(format!(
                "Error: {e}"
            ))])),
        }
    }

    #[tool(description = "Extract selected code into a new function")]
    async fn extract_function(
        &self,
        Parameters(ExtractFunctionParams {
            file_path,
            start_line,
            start_character,
            end_line,
            end_character,
            function_name,
        }): Parameters<ExtractFunctionParams>,
    ) -> Result<CallToolResult, McpError> {
        let args = serde_json::json!({
            "file_path": file_path,
            "start_line": start_line,
            "start_character": start_character,
            "end_line": end_line,
            "end_character": end_character,
            "function_name": function_name
        });

        let mut analyzer = self.analyzer.lock().await;
        match execute_tool("extract_function", args, &mut analyzer).await {
            Ok(result) => {
                if let Some(content) = result.content.first() {
                    if let Some(text) = content.get("text") {
                        return Ok(CallToolResult::success(vec![Content::text(
                            text.as_str().unwrap_or("No result"),
                        )]));
                    }
                }
                Ok(CallToolResult::success(vec![Content::text(
                    "Function extracted successfully",
                )]))
            }
            Err(e) => Ok(CallToolResult::success(vec![Content::text(format!(
                "Error: {e}"
            ))])),
        }
    }

    #[tool(description = "Generate a struct with specified fields and derives")]
    async fn generate_struct(
        &self,
        Parameters(GenerateStructParams {
            struct_name,
            fields,
            derives,
            file_path,
        }): Parameters<GenerateStructParams>,
    ) -> Result<CallToolResult, McpError> {
        let args = serde_json::json!({
            "struct_name": struct_name,
            "fields": fields,
            "derives": derives,
            "file_path": file_path
        });

        let mut analyzer = self.analyzer.lock().await;
        match execute_tool("generate_struct", args, &mut analyzer).await {
            Ok(result) => {
                if let Some(content) = result.content.first() {
                    if let Some(text) = content.get("text") {
                        return Ok(CallToolResult::success(vec![Content::text(
                            text.as_str().unwrap_or("No result"),
                        )]));
                    }
                }
                Ok(CallToolResult::success(vec![Content::text(
                    "Struct generated successfully",
                )]))
            }
            Err(e) => Ok(CallToolResult::success(vec![Content::text(format!(
                "Error: {e}"
            ))])),
        }
    }

    #[tool(description = "Generate an enum with specified variants and derives")]
    async fn generate_enum(
        &self,
        Parameters(GenerateEnumParams {
            enum_name,
            variants,
            derives,
            file_path,
        }): Parameters<GenerateEnumParams>,
    ) -> Result<CallToolResult, McpError> {
        let args = serde_json::json!({
            "enum_name": enum_name,
            "variants": variants,
            "derives": derives,
            "file_path": file_path
        });

        let mut analyzer = self.analyzer.lock().await;
        match execute_tool("generate_enum", args, &mut analyzer).await {
            Ok(result) => {
                if let Some(content) = result.content.first() {
                    if let Some(text) = content.get("text") {
                        return Ok(CallToolResult::success(vec![Content::text(
                            text.as_str().unwrap_or("No result"),
                        )]));
                    }
                }
                Ok(CallToolResult::success(vec![Content::text(
                    "Enum generated successfully",
                )]))
            }
            Err(e) => Ok(CallToolResult::success(vec![Content::text(format!(
                "Error: {e}"
            ))])),
        }
    }

    #[tool(description = "Generate a trait implementation for a struct")]
    async fn generate_trait_impl(
        &self,
        Parameters(GenerateTraitImplParams {
            trait_name,
            struct_name,
            file_path,
        }): Parameters<GenerateTraitImplParams>,
    ) -> Result<CallToolResult, McpError> {
        let args = serde_json::json!({
            "trait_name": trait_name,
            "struct_name": struct_name,
            "file_path": file_path
        });

        let mut analyzer = self.analyzer.lock().await;
        match execute_tool("generate_trait_impl", args, &mut analyzer).await {
            Ok(result) => {
                if let Some(content) = result.content.first() {
                    if let Some(text) = content.get("text") {
                        return Ok(CallToolResult::success(vec![Content::text(
                            text.as_str().unwrap_or("No result"),
                        )]));
                    }
                }
                Ok(CallToolResult::success(vec![Content::text(
                    "Trait implementation generated successfully",
                )]))
            }
            Err(e) => Ok(CallToolResult::success(vec![Content::text(format!(
                "Error: {e}"
            ))])),
        }
    }

    #[tool(description = "Generate unit tests for a function")]
    async fn generate_tests(
        &self,
        Parameters(GenerateTestsParams {
            target_function,
            file_path,
            test_cases,
        }): Parameters<GenerateTestsParams>,
    ) -> Result<CallToolResult, McpError> {
        let args = serde_json::json!({
            "target_function": target_function,
            "file_path": file_path,
            "test_cases": test_cases
        });

        let mut analyzer = self.analyzer.lock().await;
        match execute_tool("generate_tests", args, &mut analyzer).await {
            Ok(result) => {
                if let Some(content) = result.content.first() {
                    if let Some(text) = content.get("text") {
                        return Ok(CallToolResult::success(vec![Content::text(
                            text.as_str().unwrap_or("No result"),
                        )]));
                    }
                }
                Ok(CallToolResult::success(vec![Content::text(
                    "Tests generated successfully",
                )]))
            }
            Err(e) => Ok(CallToolResult::success(vec![Content::text(format!(
                "Error: {e}"
            ))])),
        }
    }

    #[tool(description = "Inline a function call at specified position")]
    async fn inline_function(
        &self,
        Parameters(InlineFunctionParams {
            file_path,
            line,
            character,
        }): Parameters<InlineFunctionParams>,
    ) -> Result<CallToolResult, McpError> {
        let args = serde_json::json!({
            "file_path": file_path,
            "line": line,
            "character": character
        });

        let mut analyzer = self.analyzer.lock().await;
        match execute_tool("inline_function", args, &mut analyzer).await {
            Ok(result) => {
                if let Some(content) = result.content.first() {
                    if let Some(text) = content.get("text") {
                        return Ok(CallToolResult::success(vec![Content::text(
                            text.as_str().unwrap_or("No result"),
                        )]));
                    }
                }
                Ok(CallToolResult::success(vec![Content::text(
                    "Function inlined successfully",
                )]))
            }
            Err(e) => Ok(CallToolResult::success(vec![Content::text(format!(
                "Error: {e}"
            ))])),
        }
    }

    #[tool(description = "Change the signature of a function")]
    async fn change_signature(
        &self,
        Parameters(ChangeSignatureParams {
            file_path,
            line,
            character,
            new_signature,
        }): Parameters<ChangeSignatureParams>,
    ) -> Result<CallToolResult, McpError> {
        let args = serde_json::json!({
            "file_path": file_path,
            "line": line,
            "character": character,
            "new_signature": new_signature
        });

        let mut analyzer = self.analyzer.lock().await;
        match execute_tool("change_signature", args, &mut analyzer).await {
            Ok(result) => {
                if let Some(content) = result.content.first() {
                    if let Some(text) = content.get("text") {
                        return Ok(CallToolResult::success(vec![Content::text(
                            text.as_str().unwrap_or("No result"),
                        )]));
                    }
                }
                Ok(CallToolResult::success(vec![Content::text(
                    "Signature changed successfully",
                )]))
            }
            Err(e) => Ok(CallToolResult::success(vec![Content::text(format!(
                "Error: {e}"
            ))])),
        }
    }

    #[tool(description = "Organize and sort import statements in a file")]
    async fn organize_imports(
        &self,
        Parameters(OrganizeImportsParams { file_path }): Parameters<OrganizeImportsParams>,
    ) -> Result<CallToolResult, McpError> {
        let args = serde_json::json!({
            "file_path": file_path
        });

        let mut analyzer = self.analyzer.lock().await;
        match execute_tool("organize_imports", args, &mut analyzer).await {
            Ok(result) => {
                if let Some(content) = result.content.first() {
                    if let Some(text) = content.get("text") {
                        return Ok(CallToolResult::success(vec![Content::text(
                            text.as_str().unwrap_or("No result"),
                        )]));
                    }
                }
                Ok(CallToolResult::success(vec![Content::text(
                    "Imports organized successfully",
                )]))
            }
            Err(e) => Ok(CallToolResult::success(vec![Content::text(format!(
                "Error: {e}"
            ))])),
        }
    }

    #[tool(description = "Apply clippy lint suggestions to improve code quality")]
    async fn apply_clippy_suggestions(
        &self,
        Parameters(ApplyClippySuggestionsParams { file_path }): Parameters<
            ApplyClippySuggestionsParams,
        >,
    ) -> Result<CallToolResult, McpError> {
        let args = serde_json::json!({
            "file_path": file_path
        });

        let mut analyzer = self.analyzer.lock().await;
        match execute_tool("apply_clippy_suggestions", args, &mut analyzer).await {
            Ok(result) => {
                if let Some(content) = result.content.first() {
                    if let Some(text) = content.get("text") {
                        return Ok(CallToolResult::success(vec![Content::text(
                            text.as_str().unwrap_or("No result"),
                        )]));
                    }
                }
                Ok(CallToolResult::success(vec![Content::text(
                    "Clippy suggestions applied successfully",
                )]))
            }
            Err(e) => Ok(CallToolResult::success(vec![Content::text(format!(
                "Error: {e}"
            ))])),
        }
    }

    #[tool(description = "Validate and suggest lifetime annotations")]
    async fn validate_lifetimes(
        &self,
        Parameters(ValidateLifetimesParams { file_path }): Parameters<ValidateLifetimesParams>,
    ) -> Result<CallToolResult, McpError> {
        let args = serde_json::json!({
            "file_path": file_path
        });

        let mut analyzer = self.analyzer.lock().await;
        match execute_tool("validate_lifetimes", args, &mut analyzer).await {
            Ok(result) => {
                if let Some(content) = result.content.first() {
                    if let Some(text) = content.get("text") {
                        return Ok(CallToolResult::success(vec![Content::text(
                            text.as_str().unwrap_or("No result"),
                        )]));
                    }
                }
                Ok(CallToolResult::success(vec![Content::text(
                    "Lifetimes validated successfully",
                )]))
            }
            Err(e) => Ok(CallToolResult::success(vec![Content::text(format!(
                "Error: {e}"
            ))])),
        }
    }

    #[tool(description = "Get type hierarchy for a symbol at specified position")]
    async fn get_type_hierarchy(
        &self,
        Parameters(GetTypeHierarchyParams {
            file_path,
            line,
            character,
        }): Parameters<GetTypeHierarchyParams>,
    ) -> Result<CallToolResult, McpError> {
        let args = serde_json::json!({
            "file_path": file_path,
            "line": line,
            "character": character
        });

        let mut analyzer = self.analyzer.lock().await;
        match execute_tool("get_type_hierarchy", args, &mut analyzer).await {
            Ok(result) => {
                if let Some(content) = result.content.first() {
                    if let Some(text) = content.get("text") {
                        return Ok(CallToolResult::success(vec![Content::text(
                            text.as_str().unwrap_or("No result"),
                        )]));
                    }
                }
                Ok(CallToolResult::success(vec![Content::text(
                    "Type hierarchy retrieved successfully",
                )]))
            }
            Err(e) => Ok(CallToolResult::success(vec![Content::text(format!(
                "Error: {e}"
            ))])),
        }
    }

    #[tool(description = "Suggest crate dependencies based on code patterns")]
    async fn suggest_dependencies(
        &self,
        Parameters(SuggestDependenciesParams {
            query,
            workspace_path,
        }): Parameters<SuggestDependenciesParams>,
    ) -> Result<CallToolResult, McpError> {
        let args = serde_json::json!({
            "query": query,
            "workspace_path": workspace_path
        });

        let mut analyzer = self.analyzer.lock().await;
        match execute_tool("suggest_dependencies", args, &mut analyzer).await {
            Ok(result) => {
                if let Some(content) = result.content.first() {
                    if let Some(text) = content.get("text") {
                        return Ok(CallToolResult::success(vec![Content::text(
                            text.as_str().unwrap_or("No result"),
                        )]));
                    }
                }
                Ok(CallToolResult::success(vec![Content::text(
                    "Dependencies suggested successfully",
                )]))
            }
            Err(e) => Ok(CallToolResult::success(vec![Content::text(format!(
                "Error: {e}"
            ))])),
        }
    }

    #[tool(description = "Create a new Rust module with optional visibility")]
    async fn create_module(
        &self,
        Parameters(CreateModuleParams {
            module_name,
            module_path,
            is_public,
        }): Parameters<CreateModuleParams>,
    ) -> Result<CallToolResult, McpError> {
        let args = serde_json::json!({
            "module_name": module_name,
            "module_path": module_path,
            "is_public": is_public
        });

        let mut analyzer = self.analyzer.lock().await;
        match execute_tool("create_module", args, &mut analyzer).await {
            Ok(result) => {
                if let Some(content) = result.content.first() {
                    if let Some(text) = content.get("text") {
                        return Ok(CallToolResult::success(vec![Content::text(
                            text.as_str().unwrap_or("No result"),
                        )]));
                    }
                }
                Ok(CallToolResult::success(vec![Content::text(
                    "Module created successfully",
                )]))
            }
            Err(e) => Ok(CallToolResult::success(vec![Content::text(format!(
                "Error: {e}"
            ))])),
        }
    }

    #[tool(description = "Move code items from one file to another")]
    async fn move_items(
        &self,
        Parameters(MoveItemsParams {
            source_file,
            target_file,
            item_names,
        }): Parameters<MoveItemsParams>,
    ) -> Result<CallToolResult, McpError> {
        let args = serde_json::json!({
            "source_file": source_file,
            "target_file": target_file,
            "item_names": item_names
        });

        let mut analyzer = self.analyzer.lock().await;
        match execute_tool("move_items", args, &mut analyzer).await {
            Ok(result) => {
                if let Some(content) = result.content.first() {
                    if let Some(text) = content.get("text") {
                        return Ok(CallToolResult::success(vec![Content::text(
                            text.as_str().unwrap_or("No result"),
                        )]));
                    }
                }
                Ok(CallToolResult::success(vec![Content::text(
                    "Items moved successfully",
                )]))
            }
            Err(e) => Ok(CallToolResult::success(vec![Content::text(format!(
                "Error: {e}"
            ))])),
        }
    }

    #[tool(description = "Inspect MIR for a symbol or position")]
    async fn inspect_mir(
        &self,
        Parameters(InspectMirParams {
            file_path,
            line,
            character,
            symbol_name,
            opt_level,
            target,
        }): Parameters<InspectMirParams>,
    ) -> Result<CallToolResult, McpError> {
        let context = self.inspection_context(None);
        let result = self
            .perform_inspection(
                &context,
                "mir",
                &file_path,
                line,
                character,
                symbol_name,
                opt_level,
                target,
            )
            .await?;

        Ok(CallToolResult::success(vec![json_content(result)?]))
    }

    #[tool(description = "Inspect LLVM IR for a symbol or position")]
    async fn inspect_llvm_ir(
        &self,
        Parameters(InspectLlvmIrParams {
            file_path,
            line,
            character,
            symbol_name,
            opt_level,
            target,
        }): Parameters<InspectLlvmIrParams>,
    ) -> Result<CallToolResult, McpError> {
        let context = self.inspection_context(None);
        let result = self
            .perform_inspection(
                &context,
                "llvm-ir",
                &file_path,
                line,
                character,
                symbol_name,
                opt_level,
                target,
            )
            .await?;

        Ok(CallToolResult::success(vec![json_content(result)?]))
    }

    #[tool(description = "Inspect assembly for a symbol or position")]
    async fn inspect_asm(
        &self,
        Parameters(InspectAsmParams {
            file_path,
            line,
            character,
            symbol_name,
            opt_level,
            target,
        }): Parameters<InspectAsmParams>,
    ) -> Result<CallToolResult, McpError> {
        let context = self.inspection_context(None);
        let result = self
            .perform_inspection(
                &context,
                "asm",
                &file_path,
                line,
                character,
                symbol_name,
                opt_level,
                target,
            )
            .await?;

        Ok(CallToolResult::success(vec![json_content(result)?]))
    }

    async fn perform_inspection(
        &self,
        context: &InspectionContext,
        view_name: &str,
        file_path: &str,
        line: Option<u32>,
        character: Option<u32>,
        symbol_name: Option<String>,
        opt_level: Option<String>,
        target: Option<String>,
    ) -> Result<InspectionResult, McpError> {
        let Some(view) = InspectionView::find(view_name) else {
            return Err(mcp_error(
                ErrorCode::INVALID_PARAMS,
                format!("Unknown inspection view `{view_name}`"),
                None,
            ));
        };

        if !is_view_advertised(&view, context.toolchain_channel(), context.gating_mode()) {
            return Err(mcp_error(
                ErrorCode::INVALID_PARAMS,
                format!(
                    "View `{}` is not available under {:?} gating for {:?}",
                    view.name,
                    context.gating_mode(),
                    context.toolchain_channel()
                ),
                None,
            ));
        }

        let mut provenance = context.provenance();

        if !is_view_runnable(&view, context.toolchain_channel()) {
            return Ok(InspectionResult {
                view: view.name.to_string(),
                symbol: None,
                text: String::new(),
                truncated: false,
                diagnostics: vec![format!(
                    "View `{}` requires a nightly toolchain (detected {:?})",
                    view.name,
                    context.toolchain_channel()
                )],
                provenance,
            });
        }

        let workspace_guard = context.lock_workspace().await;
        provenance.workspace_locked = true;

        let mut diagnostics = Vec::new();
        let (output_text, symbol_name_out) = match view.name {
            "def" => {
                let resolved = self
                    .resolve_definition(file_path, line, character, symbol_name)
                    .await?;

                (
                    resolved.text,
                    resolved.symbol.map(|sym| sym.item_name.clone()),
                )
            }
            "types" => {
                let resolved = self
                    .resolve_types(file_path, line, character, symbol_name)
                    .await?;

                (
                    resolved.text,
                    resolved.symbol.map(|sym| sym.item_name.clone()),
                )
            }
            _ => {
                let mut symbol = self
                    .resolve_normalized_symbol(
                        file_path,
                        line,
                        character,
                        symbol_name,
                        target.clone(),
                    )
                    .await?;

                let run_result = self
                    .run_compiler(context, opt_level, target.clone(), view.emit, view.unpretty)
                    .await?;
                provenance = provenance.with_command(run_result.command.join(" "));

                if !run_result.stderr.trim().is_empty() {
                    let (stderr, truncated_stderr, _) =
                        truncate_with_limits(&run_result.stderr, context.limits());
                    let prefix = if truncated_stderr {
                        "Compiler stderr (truncated):\n"
                    } else {
                        "Compiler stderr:\n"
                    };
                    diagnostics.push(format!("{prefix}{stderr}"));
                }

                let output = match view.name {
                    "mir" => {
                        let mir_outputs = vec![run_result.stdout.clone()];
                        extract_mir(&mir_outputs, &symbol).map_err(|e| {
                            mcp_error(
                                ErrorCode::RESOURCE_NOT_FOUND,
                                format!("Unable to locate MIR for symbol: {e}"),
                                None,
                            )
                        })?
                    }
                    "llvm-ir" => {
                        let llvm_outputs =
                            read_artifacts(&run_result.artifacts, &["ll"], context.limits())
                                .await?;
                        if llvm_outputs.is_empty() {
                            return Err(mcp_error(
                                ErrorCode::INTERNAL_ERROR,
                                "No LLVM IR artifacts were produced by the compiler",
                                None,
                            ));
                        }

                        extract_llvm_ir(&llvm_outputs, &symbol).map_err(|e| {
                            mcp_error(
                                ErrorCode::RESOURCE_NOT_FOUND,
                                format!("Unable to locate LLVM IR for symbol: {e}"),
                                None,
                            )
                        })?
                    }
                    "asm" => {
                        let assemblies = load_assembly_artifacts(
                            &run_result.artifacts,
                            target.as_ref(),
                            context.limits(),
                        )
                        .await?;
                        if assemblies.is_empty() {
                            return Err(mcp_error(
                                ErrorCode::INTERNAL_ERROR,
                                "No assembly artifacts were produced by the compiler",
                                None,
                            ));
                        }

                        let target_triple = target
                            .clone()
                            .or_else(|| assemblies.first().map(|asm| asm.target.clone()))
                            .unwrap_or_else(|| "host".to_string());
                        symbol = symbol.with_target(target_triple.clone());

                        extract_asm(&assemblies, &symbol, &target_triple).map_err(|e| {
                            mcp_error(
                                ErrorCode::RESOURCE_NOT_FOUND,
                                format!("Unable to locate assembly for symbol: {e}"),
                                None,
                            )
                        })?
                    }
                    _ => {
                        return Err(mcp_error(
                            ErrorCode::INVALID_PARAMS,
                            format!("Unsupported inspection view `{}`", view.name),
                            None,
                        ));
                    }
                };

                (output, Some(symbol.item_name.clone()))
            }
        };

        drop(workspace_guard);

        let (text, truncated, truncation) = truncate_with_limits(&output_text, context.limits());
        if let Some(summary) = &truncation {
            diagnostics.push(truncation_note(summary));
        }

        Ok(InspectionResult {
            view: view.name.to_string(),
            symbol: symbol_name_out,
            text,
            truncated,
            diagnostics,
            provenance: provenance.with_truncation(truncation),
        })
    }

    async fn resolve_definition(
        &self,
        file_path: &str,
        line: Option<u32>,
        character: Option<u32>,
        symbol_name: Option<String>,
    ) -> Result<ResolvedDefinition, McpError> {
        let (line, character) = match (line, character) {
            (Some(line), Some(character)) => (line, character),
            _ => {
                return Err(mcp_error(
                    ErrorCode::INVALID_PARAMS,
                    "Both line and character are required to resolve a symbol",
                    None,
                ));
            }
        };

        let mut analyzer = self.analyzer.lock().await;
        let details = analyzer
            .definition_details(file_path, line, character)
            .await
            .map_err(|e| {
                mcp_error(
                    ErrorCode::INTERNAL_ERROR,
                    format!("Failed to resolve symbol: {e}"),
                    None,
                )
            })?
            .ok_or_else(|| symbol_not_found_error(file_path, line, character))?;

        let mut identity = identity_from_definition(&details.location.uri, &details.symbol_path)
            .ok_or_else(|| symbol_not_found_error(file_path, line, character))?;

        if let Some(explicit) = symbol_name {
            identity.item_name = explicit;
        }

        let symbol_path = details
            .symbol_path
            .iter()
            .map(|segment| segment.name.clone())
            .collect::<Vec<_>>()
            .join("::");

        let text = format!(
            "Definition: {}:{}:{} ({})",
            details.location.uri,
            details.location.range.start.line + 1,
            details.location.range.start.character + 1,
            symbol_path
        );

        Ok(ResolvedDefinition {
            symbol: Some(identity),
            text,
        })
    }

    async fn resolve_types(
        &self,
        file_path: &str,
        line: Option<u32>,
        character: Option<u32>,
        symbol_name: Option<String>,
    ) -> Result<ResolvedDefinition, McpError> {
        let (line, character) = match (line, character) {
            (Some(line), Some(character)) => (line, character),
            _ => {
                return Err(mcp_error(
                    ErrorCode::INVALID_PARAMS,
                    "Both line and character are required to resolve a symbol",
                    None,
                ));
            }
        };

        let mut analyzer = self.analyzer.lock().await;
        let details = analyzer
            .definition_details(file_path, line, character)
            .await
            .map_err(|e| {
                mcp_error(
                    ErrorCode::INTERNAL_ERROR,
                    format!("Failed to resolve symbol: {e}"),
                    None,
                )
            })?
            .ok_or_else(|| symbol_not_found_error(file_path, line, character))?;

        let mut identity = identity_from_definition(&details.location.uri, &details.symbol_path)
            .ok_or_else(|| symbol_not_found_error(file_path, line, character))?;

        if let Some(explicit) = symbol_name {
            identity.item_name = explicit;
        }

        let symbol_path = details
            .symbol_path
            .iter()
            .map(|segment| segment.name.clone())
            .collect::<Vec<_>>()
            .join("::");

        let type_info = analyzer
            .get_type_hierarchy(file_path, line, character)
            .await
            .map_err(|e| {
                mcp_error(
                    ErrorCode::INTERNAL_ERROR,
                    format!("Failed to fetch type hierarchy: {e}"),
                    None,
                )
            })?;

        let text = format!(
            "Types: {}:{}:{} ({})\n{type_info}",
            details.location.uri,
            details.location.range.start.line + 1,
            details.location.range.start.character + 1,
            symbol_path
        );

        Ok(ResolvedDefinition {
            symbol: Some(identity),
            text,
        })
    }

    async fn resolve_normalized_symbol(
        &self,
        file_path: &str,
        line: Option<u32>,
        character: Option<u32>,
        symbol_name: Option<String>,
        target: Option<String>,
    ) -> Result<NormalizedSymbol, McpError> {
        let (line, character) = match (line, character) {
            (Some(line), Some(character)) => (line, character),
            _ => {
                return Err(mcp_error(
                    ErrorCode::INVALID_PARAMS,
                    "Both line and character are required to resolve a symbol",
                    None,
                ));
            }
        };

        let identity = {
            let mut analyzer = self.analyzer.lock().await;
            let details = analyzer
                .definition_details(file_path, line, character)
                .await
                .map_err(|e| {
                    mcp_error(
                        ErrorCode::INTERNAL_ERROR,
                        format!("Failed to resolve symbol: {e}"),
                        None,
                    )
                })?
                .ok_or_else(|| symbol_not_found_error(file_path, line, character))?;

            identity_from_definition(&details.location.uri, &details.symbol_path)
                .ok_or_else(|| symbol_not_found_error(file_path, line, character))?
        };

        if !matches!(identity.kind, SymbolKind::FreeFunction | SymbolKind::Method) {
            return Err(non_function_error(&identity));
        }

        let mut identity = identity;
        if let Some(name) = symbol_name {
            identity.item_name = name;
        }

        let mut normalized = NormalizedSymbol::from_identity(&identity);
        if let Some(target) = target {
            normalized = normalized.with_target(target);
        }

        Ok(normalized)
    }

    async fn run_compiler(
        &self,
        context: &InspectionContext,
        opt_level: Option<String>,
        target: Option<String>,
        emit: Option<&str>,
        unpretty: Option<&str>,
    ) -> Result<RunResult, McpError> {
        let runner = CompilerRunner::with_target_dir(context.target_dir());
        let request = RunRequest {
            manifest_path: None,
            package: None,
            target_triple: target,
            opt_level,
            emit: emit.map(|emit| emit.to_string()),
            unpretty: unpretty.map(|unpretty| unpretty.to_string()),
            additional_rustc_args: Vec::new(),
            env: context.env().clone(),
        };

        let result = runner.run(request, context.limits()).await.map_err(|e| {
            if let Some(runner_error) = e.downcast_ref::<RunnerError>() {
                match runner_error {
                    RunnerError::Timeout(duration) => mcp_error(
                        ErrorCode::INTERNAL_ERROR,
                        format!(
                            "Compiler run timed out after {} seconds. Try narrowing the request or limiting emitted artifacts.",
                            duration.as_secs()
                        ),
                        Some(json!({
                            "timeout_seconds": duration.as_secs()
                        })),
                    ),
                }
            } else {
                mcp_error(
                    ErrorCode::INTERNAL_ERROR,
                    format!("{e:#}"),
                    None,
                )
            }
        })?;

        if !result.status.success() {
            return Err(compiler_failure_error(&result));
        }

        Ok(result)
    }
}

fn truncation_note(summary: &TruncationSummary) -> String {
    format!(
        "Output truncated to {} lines/{} bytes from {} lines/{} bytes",
        summary.kept_lines, summary.kept_bytes, summary.original_lines, summary.original_bytes
    )
}

fn json_content<T: Serialize>(value: T) -> Result<Content, McpError> {
    Content::json(value).map_err(|e| {
        mcp_error(
            ErrorCode::INTERNAL_ERROR,
            format!("Failed to serialize response: {e}"),
            None,
        )
    })
}

fn mcp_error(code: ErrorCode, message: impl Into<String>, data: Option<Value>) -> McpError {
    McpError::new(code, message.into(), data)
}

fn enforce_artifact_limit(
    path: &Path,
    size: usize,
    limits: &InspectionLimits,
) -> Result<(), McpError> {
    if size > limits.max_output_bytes {
        return Err(mcp_error(
            ErrorCode::INTERNAL_ERROR,
            format!(
                "Artifact {} exceeded the size limit ({} bytes > {} bytes). Request a smaller output (e.g., a single symbol or target).",
                path.display(),
                size,
                limits.max_output_bytes
            ),
            Some(json!({
                "artifact": path,
                "limit_bytes": limits.max_output_bytes,
                "observed_bytes": size
            })),
        ));
    }

    Ok(())
}

fn symbol_not_found_error(file_path: &str, line: u32, character: u32) -> McpError {
    mcp_error(
        ErrorCode::RESOURCE_NOT_FOUND,
        format!("No symbol found at {}:{}:{}", file_path, line, character),
        Some(json!({
            "file_path": file_path,
            "line": line,
            "character": character
        })),
    )
}

fn non_function_error(identity: &SymbolIdentity) -> McpError {
    mcp_error(
        ErrorCode::INVALID_PARAMS,
        format!(
            "Item at position is not a function (found {:?})",
            identity.kind
        ),
        Some(json!({
            "kind": format!("{:?}", identity.kind)
        })),
    )
}

fn compiler_failure_error(result: &RunResult) -> McpError {
    mcp_error(
        ErrorCode::INTERNAL_ERROR,
        "Compiler run failed",
        Some(json!({
            "status": result.status.code(),
            "stdout": result.stdout,
            "stderr": result.stderr,
            "command": result.command
        })),
    )
}

async fn read_artifacts(
    paths: &[PathBuf],
    extensions: &[&str],
    limits: &InspectionLimits,
) -> Result<Vec<String>, McpError> {
    let mut outputs = Vec::new();

    for path in paths {
        let Some(ext) = path.extension().and_then(|ext| ext.to_str()) else {
            continue;
        };

        if extensions
            .iter()
            .any(|wanted| ext.eq_ignore_ascii_case(wanted))
        {
            let content = fs::read_to_string(path).await.map_err(|e| {
                mcp_error(
                    ErrorCode::INTERNAL_ERROR,
                    format!("Failed to read artifact {}: {e}", path.display()),
                    Some(json!({
                        "artifact": path
                    })),
                )
            })?;

            enforce_artifact_limit(path, content.len(), limits)?;

            outputs.push(content);
        }
    }

    Ok(outputs)
}

async fn load_assembly_artifacts(
    paths: &[PathBuf],
    target_hint: Option<&String>,
    limits: &InspectionLimits,
) -> Result<Vec<TargetedAssembly>, McpError> {
    let mut assemblies = Vec::new();

    for path in paths {
        let Some(ext) = path.extension().and_then(|ext| ext.to_str()) else {
            continue;
        };

        if !(ext.eq_ignore_ascii_case("s") || ext.eq_ignore_ascii_case("asm")) {
            continue;
        }

        let content = fs::read_to_string(path).await.map_err(|e| {
            mcp_error(
                ErrorCode::INTERNAL_ERROR,
                format!("Failed to read assembly artifact {}: {e}", path.display()),
                Some(json!({
                    "artifact": path
                })),
            )
        })?;

        enforce_artifact_limit(path, content.len(), limits)?;

        let target = infer_target_from_path(path)
            .or_else(|| target_hint.cloned())
            .unwrap_or_else(|| "unknown".to_string());

        assemblies.push(TargetedAssembly { target, content });
    }

    Ok(assemblies)
}

fn infer_target_from_path(path: &Path) -> Option<String> {
    let mut components = path.components().peekable();
    while let Some(component) = components.next() {
        if component.as_os_str() == "mcp-inspections" {
            if let Some(next) = components.next() {
                let comp = next.as_os_str().to_string_lossy().into_owned();
                if comp == "debug" || comp == "release" {
                    return None;
                }
                return Some(comp);
            }
        }
    }
    None
}

#[tool_handler]
impl ServerHandler for RustMcpServer {
    fn get_info(&self) -> ServerInfo {
        ServerInfo {
            protocol_version: ProtocolVersion::V_2024_11_05,
            capabilities: ServerCapabilities::builder()
                .enable_tools()
                .build(),
            server_info: Implementation::from_build_env(),
            instructions: Some("Rust MCP Server providing rust-analyzer integration for idiomatic Rust development tools. Provides code analysis, refactoring, and project management capabilities.".to_string()),
        }
    }
}
