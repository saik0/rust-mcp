use serde::{Deserialize, Serialize};
use serde_json::{Value, json};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Position {
    pub line: u32,
    pub character: u32,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Range {
    pub start: Position,
    pub end: Position,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Location {
    pub uri: String,
    pub range: Range,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LocationLink {
    #[serde(
        rename = "originSelectionRange",
        skip_serializing_if = "Option::is_none"
    )]
    pub origin_selection_range: Option<Range>,
    #[serde(rename = "targetUri")]
    pub target_uri: String,
    #[serde(rename = "targetRange")]
    pub target_range: Range,
    #[serde(rename = "targetSelectionRange")]
    pub target_selection_range: Range,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TextDocumentIdentifier {
    pub uri: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TextDocumentPositionParams {
    #[serde(rename = "textDocument")]
    pub text_document: TextDocumentIdentifier,
    pub position: Position,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(untagged)]
pub enum DefinitionResponse {
    SingleLocation(Location),
    LocationArray(Vec<Location>),
    LocationLinks(Vec<LocationLink>),
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DocumentSymbolParams {
    #[serde(rename = "textDocument")]
    pub text_document: TextDocumentIdentifier,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DocumentSymbol {
    pub name: String,
    pub detail: Option<String>,
    pub kind: u32,
    pub range: Range,
    #[serde(rename = "selectionRange")]
    pub selection_range: Range,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub children: Option<Vec<DocumentSymbol>>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SymbolInformation {
    pub name: String,
    pub kind: u32,
    pub location: Location,
    #[serde(rename = "containerName")]
    pub container_name: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(untagged)]
pub enum DocumentSymbolResponse {
    DocumentSymbols(Vec<DocumentSymbol>),
    SymbolInformation(Vec<SymbolInformation>),
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SymbolPathSegment {
    pub name: String,
    pub kind: u32,
}

pub type SymbolPath = Vec<SymbolPathSegment>;

pub fn create_text_document_position_params(file_path: &str, line: u32, character: u32) -> Value {
    json!({
        "textDocument": {
            "uri": format!("file://{}", file_path)
        },
        "position": {
            "line": line,
            "character": character
        }
    })
}

pub fn create_references_params(file_path: &str, line: u32, character: u32) -> Value {
    json!({
        "textDocument": {
            "uri": format!("file://{}", file_path)
        },
        "position": {
            "line": line,
            "character": character
        },
        "context": {
            "includeDeclaration": true
        }
    })
}

pub fn create_workspace_symbol_params(query: &str) -> Value {
    json!({
        "query": query
    })
}

pub fn create_rename_params(file_path: &str, line: u32, character: u32, new_name: &str) -> Value {
    json!({
        "textDocument": {
            "uri": format!("file://{}", file_path)
        },
        "position": {
            "line": line,
            "character": character
        },
        "newName": new_name
    })
}

pub fn create_formatting_params(file_path: &str) -> Value {
    json!({
        "textDocument": {
            "uri": format!("file://{}", file_path)
        },
        "options": {
            "tabSize": 4,
            "insertSpaces": true
        }
    })
}
