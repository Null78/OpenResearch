//! Alibaba Cloud (token plan) harness.
//!
//! Chat: **no CLI child at all**. The plan's OpenAI-compatible endpoint
//! (`https://token-plan.<region>.maas.aliyuncs.com/compatible-mode/v1`, overridable
//! with `ALIBABA_TOKEN_PLAN_BASE_URL`) is called directly with the `reqwest`
//! client the process already owns, so one model step is one streamed HTTP
//! request and its SSE frames are folded into wire parts.
//!
//! Three things the process-spawning adapters get for free have to be built
//! here, and each has one deliberate design decision:
//!
//! * **Context** — there is no native session to resume, so a turn replays the
//!   session's own transcript (the store is the system of record for every
//!   harness anyway) plus the current turn's prepared input. Tool activity is
//!   replayed as compact `<tool-activity>` blocks, since the wire transcript
//!   keeps a tool part's input/output but no API-shaped `tool_call_id` to
//!   rebuild a real tool-call exchange from a previous turn.
//! * **Tools** — the agent loop (bash / read_file / write_file / edit_file)
//!   runs in-process: one plan of tool calls per model step, executed in the
//!   session worktree, results fed back as `role: "tool"` messages.
//! * **Approvals** — the shared bridge is reused verbatim
//!   ([`ChatHost::mint_gate_token`] + [`ChatHost::request_permission`]), so an
//!   approval under **Ask** surfaces the same card the CLI harnesses use and
//!   blocks the tool call until the user answers
//!   ([`Harness::resume_from_prompt`] settles it).
//!
//! Detection is env-only: the harness is "installed" always (there is nothing to
//! install) and "ready" when the plan key resolves. The model catalog is the
//! plan's own `GET /models`, filtered to models that can hold a chat turn —
//! image/TTS/realtime entries are served by the same endpoint but cannot.

use std::collections::HashSet;
use std::path::{Component, Path, PathBuf};
use std::sync::Mutex;
use std::time::{Duration, Instant};

use async_trait::async_trait;
use futures::StreamExt;
use serde_json::{json, Value};
use tokio::process::Command;

use super::detect::{api_key, HarnessAuthState, HarnessInfo, ModelInfo};
use super::options::{HarnessOptions, OptionChoice, PermissionMode};
use super::{Harness, OneShot, ResumeAction, TurnFailure, TurnOutcome, TurnResult};
use crate::error::{anyhow, Result};
use crate::local::chat::{
    self, ContextUsage, DeliveryState, PermissionDecision, PromptAnswer, ResumeCtx, TurnCtx,
    WirePart, WirePrompt,
};
use crate::local::opencode::ensure_playbook;

/// Canonical harness id, on the wire and in the store.
pub const HARNESS_ID: &str = "alibaba-token-plan";
const HARNESS_NAME: &str = "Alibaba Cloud";

/// The plan's OpenAI-compatible root. The region is part of the host, so a user
/// on another plan region overrides it with `ALIBABA_TOKEN_PLAN_BASE_URL`.
const DEFAULT_BASE_URL: &str =
    "https://token-plan.ap-southeast-1.maas.aliyuncs.com/compatible-mode/v1";

/// Credential variables, in priority order. `ALIBABA_TOKEN_PLAN_API_KEY` is the
/// one the plan's own dashboard hands out; `DASHSCOPE_API_KEY` is the standard
/// Alibaba Cloud/DashScope name, so a machine already set up for DashScope works
/// without a second secret.
const KEY_VARS: [&str; 2] = ["ALIBABA_TOKEN_PLAN_API_KEY", "DASHSCOPE_API_KEY"];
const BASE_URL_VAR: &str = "ALIBABA_TOKEN_PLAN_BASE_URL";

/// The model a turn runs when the session has picked none. The plan serves a
/// router under this id, which is exactly the "let the plan decide" contract.
const DEFAULT_MODEL: &str = "auto";

/// Worktree-relative dir the session's `SKILL.md` set is written to. No CLI
/// discovers this dir — the turn inlines the skills into the system prompt — but
/// writing them keeps one source of truth (the trait) and puts the full bodies
/// on disk for an agent that wants to re-read one.
pub const SESSION_SKILLS_DIR: &str = ".openresearch/agent/skills";

/// Budgets. The API itself has no turn length; these bound the damage one wedged
/// turn can do.
const CATALOG_TIMEOUT: Duration = Duration::from_secs(8);
const CHAT_TIMEOUT: Duration = Duration::from_secs(30 * 60);
const STALL_TIMEOUT: Duration = Duration::from_secs(5 * 60);
const COMMAND_TIMEOUT: Duration = Duration::from_secs(10 * 60);
const MAX_STEPS: usize = 60;
const MAX_TOOL_OUTPUT: usize = 16_000;
const MAX_FILE_LINES: usize = 2_000;
const MAX_READ_BYTES: usize = 256 * 1024;
const MAX_WRITE_BYTES: usize = 2 * 1024 * 1024;
const MAX_IMAGE_BYTES: u64 = 5 * 1024 * 1024;
const SKILL_INLINE_CAP: usize = 8_000;
const SKILL_TOTAL_CAP: usize = 60_000;
const CATALOG_TTL: Duration = Duration::from_secs(300);

/// Models the endpoint serves that cannot hold a chat turn. Matched as
/// substrings: the plan names them `wan2.7-image`, `qwen-audio-3.0-tts-plus`,
/// `qwen-audio-3.0-realtime-plus`.
const NON_CHAT_MARKERS: [&str; 5] = ["image", "tts", "audio", "realtime", "video"];

/// The plan's chat models, as measured on this box. Used only when there is no
/// key to ask `GET /models` with (a picker still has to show something
/// selectable) and as the catalog's floor if that call fails.
const KNOWN_MODELS: [&str; 19] = [
    "auto",
    "qwen3.8-max",
    "qwen3.8-flash",
    "qwen3.7-max",
    "qwen3.7-plus",
    "qwen3.6-plus",
    "qwen3.6-flash",
    "deepseek-v4-pro",
    "deepseek-v4.1-flash",
    "deepseek-v4-flash",
    "deepseek-v4-flash-0731",
    "deepseek-v3.2",
    "kimi-k2.7-code",
    "kimi-k2.6",
    "kimi-k2.5",
    "glm-5.3",
    "glm-5.2",
    "glm-5.1",
    "glm-5",
];

pub struct AlibabaTokenPlan;

// --- settings ------------------------------------------------------------------

/// Where the plan lives and what it is authenticated with.
#[derive(Debug, Clone, PartialEq, Eq)]
struct Settings {
    base_url: String,
    key: String,
}

/// Resolve the plan endpoint + key from the environment (and the synced env file
/// `orx` maintains, which `detect::api_key` also reads).
fn settings() -> Option<Settings> {
    let key = KEY_VARS.iter().find_map(|var| api_key(var))?;
    Some(Settings {
        base_url: base_url()?,
        key,
    })
}

/// The endpoint, refusing anything that would send the plan key in the clear.
/// An override is honored (another plan region, or a gateway), but plain http is
/// only allowed back to loopback — the same rule `local_models` applies to a
/// local server address.
fn base_url() -> Option<String> {
    base_url_from(api_key(BASE_URL_VAR))
}

/// Pure half of [`base_url`], so the scheme rule is testable without touching
/// the process environment.
fn base_url_from(override_url: Option<String>) -> Option<String> {
    let raw = override_url.unwrap_or_else(|| DEFAULT_BASE_URL.to_string());
    let trimmed = raw.trim().trim_end_matches('/').to_string();
    let parsed = reqwest::Url::parse(&trimmed).ok()?;
    let ok = match parsed.scheme() {
        "https" => true,
        "http" => crate::local::local_models::is_loopback_url(&trimmed),
        _ => false,
    };
    ok.then_some(trimmed)
}

/// Why the key is missing, in the words the composer should show.
fn missing_key_note() -> String {
    format!(
        "Set {} (or DASHSCOPE_API_KEY) to the API key from your Alibaba Cloud token plan, \
         then reload `orx up`.",
        KEY_VARS[0]
    )
}

/// A model id that can hold a chat turn on this plan.
fn is_chat_model(id: &str) -> bool {
    let lowered = id.to_ascii_lowercase();
    !NON_CHAT_MARKERS
        .iter()
        .any(|marker| lowered.contains(marker))
}

fn known_models() -> Vec<ModelInfo> {
    KNOWN_MODELS.iter().map(|id| ModelInfo::new(*id)).collect()
}

// --- catalog --------------------------------------------------------------------

/// In-process cache of one `GET /models` answer: the endpoint it came from, when
/// it was fetched, and the ids. The picker asks for the catalog on every
/// `/api/harnesses` (and its snapshot/full passes), and the endpoint is a network
/// round-trip away — a cold, unreachable plan must not make the dashboard slow
/// to paint.
type CatalogCache = Mutex<Option<(String, Instant, Vec<String>)>>;

fn catalog_cache() -> &'static CatalogCache {
    static CACHE: std::sync::OnceLock<CatalogCache> = std::sync::OnceLock::new();
    CACHE.get_or_init(|| Mutex::new(None))
}

fn cached_catalog(base_url: &str) -> Option<Vec<String>> {
    let cache = catalog_cache().lock().unwrap();
    let (url, at, ids) = cache.as_ref()?;
    (url == base_url && at.elapsed() < CATALOG_TTL).then(|| ids.clone())
}

fn store_catalog(base_url: &str, ids: Vec<String>) {
    *catalog_cache().lock().unwrap() = Some((base_url.to_string(), Instant::now(), ids));
}

/// What one catalog probe decided: the ids, and — when the plan refused the key
/// — that the credential is the problem rather than the network.
struct Catalog {
    ids: Vec<String>,
    rejected_key: bool,
    error: Option<String>,
}

/// `GET /models`, filtered to chat-capable ids. Never fatal: a failure returns
/// the reason so detection can say what happened instead of silently dropping
/// the catalog.
async fn fetch_catalog(client: &reqwest::Client, settings: &Settings) -> Catalog {
    if let Some(ids) = cached_catalog(&settings.base_url) {
        return Catalog {
            ids,
            rejected_key: false,
            error: None,
        };
    }
    let request = client
        .get(format!("{}/models", settings.base_url))
        .bearer_auth(&settings.key)
        .timeout(CATALOG_TIMEOUT);
    let response = match request.send().await {
        Ok(response) => response,
        Err(error) => {
            return Catalog {
                ids: Vec::new(),
                rejected_key: false,
                error: Some(format!("the plan did not answer: {error}")),
            }
        }
    };
    let status = response.status();
    if !status.is_success() {
        let body = response.text().await.unwrap_or_default();
        return Catalog {
            ids: Vec::new(),
            rejected_key: status == reqwest::StatusCode::UNAUTHORIZED
                || status == reqwest::StatusCode::FORBIDDEN,
            error: Some(format!(
                "the plan answered {status}: {}",
                api_error_message(&body)
            )),
        };
    }
    let body: Value = match response.json().await {
        Ok(body) => body,
        Err(error) => {
            return Catalog {
                ids: Vec::new(),
                rejected_key: false,
                error: Some(format!("the plan sent an unreadable model list: {error}")),
            }
        }
    };
    let ids = parse_model_list(&body);
    if ids.is_empty() {
        return Catalog {
            ids,
            rejected_key: false,
            error: Some("the plan listed no chat models".to_string()),
        };
    }
    store_catalog(&settings.base_url, ids.clone());
    Catalog {
        ids,
        rejected_key: false,
        error: None,
    }
}

/// Chat-capable ids out of an OpenAI-shaped `{"data":[{"id":…}]}` body, order
/// preserved and duplicates dropped.
fn parse_model_list(body: &Value) -> Vec<String> {
    let mut seen = HashSet::new();
    body.get("data")
        .and_then(Value::as_array)
        .map(|models| {
            models
                .iter()
                .filter_map(|model| model.get("id").and_then(Value::as_str))
                .map(str::trim)
                .filter(|id| !id.is_empty() && is_chat_model(id))
                .filter(|id| seen.insert(id.to_string()))
                .map(str::to_string)
                .collect()
        })
        .unwrap_or_default()
}

/// The provider's own words for a failed request. Both its error shapes are
/// seen in the wild: the OpenAI-compatible `{"error":{"message":…}}` and a bare
/// string body.
fn api_error_message(body: &str) -> String {
    let trimmed = body.trim();
    if let Ok(value) = serde_json::from_str::<Value>(trimmed) {
        if let Some(message) = value
            .get("error")
            .and_then(|error| error.get("message"))
            .and_then(Value::as_str)
        {
            return message.trim().to_string();
        }
        if let Some(message) = value.get("message").and_then(Value::as_str) {
            return message.trim().to_string();
        }
    }
    let flat: String = trimmed.chars().take(300).collect();
    if flat.is_empty() {
        "no detail".to_string()
    } else {
        flat
    }
}

// --- detection ------------------------------------------------------------------

async fn detection(with_catalog: bool) -> Option<HarnessInfo> {
    let mut info = HarnessInfo::new(HARNESS_ID, HARNESS_NAME);
    // Nothing to install: the harness IS the plan's HTTP API.
    info.installed = true;
    info.auth_method = Some("apiKey");
    info.plan = Some("Token Plan".to_string());
    let Some(settings) = settings() else {
        info.authenticated = false;
        info.auth_state = HarnessAuthState::NeedsLogin;
        info.needs_config_repair = true;
        info.agent_note = Some(missing_key_note());
        info.models = known_models();
        return Some(info);
    };
    info.authenticated = true;
    info.auth_state = HarnessAuthState::Ready;
    info.agent_ready = true;
    info.account = Some(provider_label(&settings.base_url));
    info.models = known_models();
    if !with_catalog {
        return Some(info);
    }
    let client = http_client().ok()?;
    let catalog = fetch_catalog(&client, &settings).await;
    if !catalog.ids.is_empty() {
        info.models = catalog.ids.iter().map(ModelInfo::new).collect();
    }
    if catalog.rejected_key {
        // A key that resolves but is refused is not a usable turn — say so
        // rather than letting the user pick a model that 401s.
        info.authenticated = false;
        info.auth_state = HarnessAuthState::NeedsLogin;
        info.agent_ready = false;
        info.needs_config_repair = true;
        info.agent_note = Some(format!(
            "The token plan refused this key ({}). Issue a new key for the plan and set it as {}.",
            catalog.error.as_deref().unwrap_or("rejected"),
            KEY_VARS[0]
        ));
    } else if let Some(error) = catalog.error {
        info.agent_note = Some(format!("{error}; showing the plan's known models."));
    }
    Some(info)
}

/// A short host label for the card — the region the plan answers from.
fn provider_label(base_url: &str) -> String {
    reqwest::Url::parse(base_url)
        .ok()
        .and_then(|url| url.host_str().map(str::to_string))
        .unwrap_or_else(|| "token plan".to_string())
}

/// One client for catalog probes and turns. No proxy bypass here (the API is
/// external): the process-wide client settings apply.
fn http_client() -> Result<reqwest::Client> {
    reqwest::Client::builder()
        .timeout(CHAT_TIMEOUT)
        .build()
        .map_err(|e| anyhow!("could not build the HTTP client: {e}"))
}

// --- the API request ------------------------------------------------------------

/// One model step's outcome: what the model said, and what it wants run.
#[derive(Debug, Default)]
struct Step {
    text: String,
    calls: Vec<PlannedCall>,
    usage: Option<Value>,
}

/// A tool call the model asked for, with its arguments already parsed.
#[derive(Debug, Clone, PartialEq)]
struct PlannedCall {
    id: String,
    name: String,
    input: Value,
    /// Set when the model's argument JSON could not be read — the call is
    /// reported back to the model instead of being run.
    broken: Option<String>,
}

/// Accumulator for one streamed `tool_calls` entry. The arguments arrive as a
/// fragment sequence, so they are concatenated and parsed once, at the end.
#[derive(Debug, Default, Clone)]
struct CallAccum {
    id: String,
    name: String,
    arguments: String,
}

impl CallAccum {
    fn merge(&mut self, delta: &ToolCallDelta) {
        if let Some(id) = delta.id.as_deref().filter(|id| !id.is_empty()) {
            self.id = id.to_string();
        }
        if let Some(name) = delta.name.as_deref().filter(|name| !name.is_empty()) {
            self.name.push_str(name);
        }
        if let Some(arguments) = delta.arguments.as_deref() {
            self.arguments.push_str(arguments);
        }
    }

    fn plan(self, index: usize) -> PlannedCall {
        let input = if self.arguments.trim().is_empty() {
            json!({})
        } else {
            serde_json::from_str::<Value>(&self.arguments).unwrap_or(Value::Null)
        };
        let broken = input.is_null().then(|| {
            format!(
                "the arguments were not valid JSON: {}",
                self.arguments.chars().take(200).collect::<String>()
            )
        });
        PlannedCall {
            id: if self.id.is_empty() {
                format!("call_{index}")
            } else {
                self.id
            },
            name: if self.name.is_empty() {
                "unknown".to_string()
            } else {
                self.name
            },
            input,
            broken,
        }
    }
}

/// One decoded streamed chunk.
#[derive(Debug, Default, PartialEq)]
struct Chunk {
    text: String,
    reasoning: String,
    tool_calls: Vec<ToolCallDelta>,
    usage: Option<Value>,
    done: bool,
}

#[derive(Debug, Default, PartialEq, Clone)]
struct ToolCallDelta {
    index: usize,
    id: Option<String>,
    name: Option<String>,
    arguments: Option<String>,
}

/// Decode one `data:` payload of the stream. `[DONE]` is a terminator, and any
/// unreadable frame is skipped rather than failing the turn — a provider that
/// appends a keep-alive must not kill a turn mid-sentence.
fn decode_chunk(data: &str) -> Option<Chunk> {
    let data = data.trim();
    if data.is_empty() {
        return None;
    }
    if data == "[DONE]" {
        return Some(Chunk {
            done: true,
            ..Chunk::default()
        });
    }
    let value: Value = serde_json::from_str(data).ok()?;
    let mut chunk = Chunk {
        usage: value.get("usage").filter(|u| !u.is_null()).cloned(),
        ..Chunk::default()
    };
    for choice in value
        .get("choices")
        .and_then(Value::as_array)
        .unwrap_or(&Vec::new())
    {
        let delta = choice.get("delta").unwrap_or(&Value::Null);
        if let Some(text) = delta.get("content").and_then(Value::as_str) {
            chunk.text.push_str(text);
        }
        if let Some(reasoning) = delta.get("reasoning_content").and_then(Value::as_str) {
            chunk.reasoning.push_str(reasoning);
        }
        for call in delta
            .get("tool_calls")
            .and_then(Value::as_array)
            .unwrap_or(&Vec::new())
        {
            let function = call.get("function").unwrap_or(&Value::Null);
            chunk.tool_calls.push(ToolCallDelta {
                index: call.get("index").and_then(Value::as_u64).unwrap_or(0) as usize,
                id: call.get("id").and_then(Value::as_str).map(str::to_string),
                name: function
                    .get("name")
                    .and_then(Value::as_str)
                    .map(str::to_string),
                arguments: function
                    .get("arguments")
                    .and_then(Value::as_str)
                    .map(str::to_string),
            });
        }
    }
    Some(chunk)
}

/// Split SSE frames out of the byte stream. Frames are blank-line separated —
/// with an LF **or** a CRLF terminator — and the split happens on **bytes**:
/// decoding each network chunk on its own turns a multibyte character straddling
/// two reads into U+FFFD, which corrupts streamed text and, worse, the streamed
/// JSON tool-call arguments (they parse as `broken` and the call is refused). A
/// frame's bytes are whole by construction, because a frame boundary is always a
/// line boundary.
#[derive(Default)]
struct SseBuffer {
    pending: Vec<u8>,
}

impl SseBuffer {
    fn push(&mut self, bytes: &[u8]) -> Vec<String> {
        self.pending.extend_from_slice(bytes);
        let mut frames = Vec::new();
        while let Some((at, width)) = frame_boundary(&self.pending) {
            let frame: Vec<u8> = self.pending.drain(..at + width).collect();
            frames.push(String::from_utf8_lossy(&frame[..at]).into_owned());
        }
        frames
    }

    /// The tail after the last terminator. Nothing consumes it today: a stream
    /// that ends without `[DONE]` has still delivered every frame it sent.
    #[cfg(test)]
    fn rest(&self) -> String {
        String::from_utf8_lossy(&self.pending).into_owned()
    }
}

/// The earliest blank-line terminator, as `(offset, width)`: `\n\n`, or the CRLF
/// form `\r\n\r\n`. Both are searched for, because a CRLF stream contains no
/// `\n\n` at all — and a terminator split across two reads (a trailing `\r`, or
/// `\r\n\r`) stays buffered until the rest of it arrives, so it is still one
/// boundary rather than a merged frame.
fn frame_boundary(buffer: &[u8]) -> Option<(usize, usize)> {
    let lf = buffer
        .windows(2)
        .position(|window| window == b"\n\n")
        .map(|at| (at, 2));
    let crlf = buffer
        .windows(4)
        .position(|window| window == b"\r\n\r\n")
        .map(|at| (at, 4));
    match (lf, crlf) {
        (Some(a), Some(b)) => Some(if a.0 <= b.0 { a } else { b }),
        (found, None) | (None, found) => found,
    }
}

/// The `data:` lines of one frame, joined. Comments (`: ping`) and other fields
/// are ignored.
fn frame_data(frame: &str) -> Option<String> {
    let mut data: Vec<&str> = Vec::new();
    for line in frame.lines() {
        let line = line.trim_end();
        if let Some(rest) = line.strip_prefix("data:") {
            data.push(rest.trim_start());
        }
    }
    (!data.is_empty()).then(|| data.join("\n"))
}

/// Build the tool schemas the agent loop offers. Kept in one place because the
/// names here are load-bearing: the permission gate and the executor both match
/// on them.
fn tool_schemas() -> Value {
    json!([
        {
            "type": "function",
            "function": {
                "name": "bash",
                "description": "Run one shell command in the session worktree with `bash -lc`. \
                                Reports exit code, stdout and stderr.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "command": {"type": "string", "description": "The command line to run."}
                    },
                    "required": ["command"]
                }
            }
        },
        {
            "type": "function",
            "function": {
                "name": "read_file",
                "description": "Read a text file from the session worktree, with line numbers.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "path": {"type": "string", "description": "Worktree-relative path."},
                        "offset": {"type": "integer", "description": "First line (1-based)."},
                        "limit": {"type": "integer", "description": "How many lines."}
                    },
                    "required": ["path"]
                }
            }
        },
        {
            "type": "function",
            "function": {
                "name": "write_file",
                "description": "Write a text file in the session worktree, replacing any existing \
                                content and creating parent directories.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "path": {"type": "string", "description": "Worktree-relative path."},
                        "content": {"type": "string", "description": "The full new content."}
                    },
                    "required": ["path", "content"]
                }
            }
        },
        {
            "type": "function",
            "function": {
                "name": "edit_file",
                "description": "Replace an exact string in a text file in the worktree.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "path": {"type": "string"},
                        "old_string": {"type": "string", "description": "Exact text to replace."},
                        "new_string": {"type": "string", "description": "Replacement text."},
                        "replace_all": {
                            "type": "boolean",
                            "description": "Replace every occurrence (default: only if unique)."
                        }
                    },
                    "required": ["path", "old_string", "new_string"]
                }
            }
        }
    ])
}

// --- permission gating ----------------------------------------------------------

/// What the harness should do with one planned tool call.
#[derive(Debug, PartialEq)]
enum Gate {
    Run,
    /// Ask the user, through the shared bridge card. Blocks until answered.
    Ask,
    /// Refuse without a card, telling the model why.
    Refuse(String),
}

/// Reads never need approval; mutations depend on the session's mode.
fn gate(mode: PermissionMode, tool: &str, input: &Value) -> Gate {
    let readonly_command = tool == "bash"
        && input
            .get("command")
            .and_then(Value::as_str)
            .is_some_and(super::command_is_readonly);
    match mode {
        PermissionMode::Plan => match tool {
            "read_file" => Gate::Run,
            "bash" if readonly_command => Gate::Run,
            // A plan needs to look around; a command that only reads may run,
            // anything else is the user's call rather than an assumption.
            "bash" => Gate::Ask,
            _ => Gate::Refuse(
                "Plan mode is read-only: nothing is written. Present the plan in your reply so \
                 the user can approve it, then implement it once the mode changes."
                    .to_string(),
            ),
        },
        PermissionMode::Ask => {
            // The mode's own label is "ask before every command or file change",
            // so no command is exempt here — not even one the shared classifier
            // calls read-only. This harness runs no sandbox, so a cardless
            // `cat /root/.ssh/…` is exactly the call the user asked to see.
            if tool == "read_file" {
                Gate::Run
            } else {
                Gate::Ask
            }
        }
        PermissionMode::AcceptEdits => {
            // File tools are the mode's whole point; a command still has to be
            // provably read-only, and anything else is the user's call.
            if matches!(tool, "read_file" | "write_file" | "edit_file") || readonly_command {
                Gate::Run
            } else {
                Gate::Ask
            }
        }
        PermissionMode::Auto | PermissionMode::Bypass => Gate::Run,
    }
}

// --- worktree tools -------------------------------------------------------------

/// Resolve a model-supplied path inside the worktree, refusing both a lexical
/// escape and a link escape.
fn resolve_in_worktree(workdir: &Path, raw: &str) -> Result<PathBuf> {
    let raw = raw.trim();
    if raw.is_empty() {
        return Err(anyhow!("give a file path"));
    }
    let candidate = Path::new(raw);
    if candidate
        .components()
        .any(|component| matches!(component, Component::ParentDir))
    {
        return Err(anyhow!(
            "{raw} leaves the session worktree; paths are relative to it and may not contain `..`"
        ));
    }
    let joined = if candidate.is_absolute() {
        candidate.to_path_buf()
    } else {
        workdir.join(candidate)
    };
    let root = workdir
        .canonicalize()
        .unwrap_or_else(|_| workdir.to_path_buf());
    let anchor = if joined.exists() {
        joined.canonicalize().unwrap_or_else(|_| joined.clone())
    } else {
        joined
            .parent()
            .map(|parent| {
                parent
                    .canonicalize()
                    .unwrap_or_else(|_| parent.to_path_buf())
            })
            .map(|parent| parent.join(joined.file_name().unwrap_or_default()))
            .unwrap_or_else(|| joined.clone())
    };
    if !anchor.starts_with(&root) {
        return Err(anyhow!(
            "{} is outside the session worktree ({})",
            joined.display(),
            root.display()
        ));
    }
    Ok(joined)
}

/// The jailed resolver every file tool uses: lexical confinement from
/// [`resolve_in_worktree`], then a link check, because the path it returns can
/// still be a symlink whose target is somewhere else entirely.
fn resolve_confined(workdir: &Path, raw: &str) -> Result<PathBuf> {
    let path = resolve_in_worktree(workdir, raw)?;
    let root = workdir
        .canonicalize()
        .unwrap_or_else(|_| workdir.to_path_buf());
    assert_link_stays_inside(&root, &path)?;
    Ok(path)
}

/// Refuse a path whose final component is a link pointing outside `root`.
///
/// `resolve_in_worktree` confines the path it returns, but the read or write
/// that follows would follow the link: a symlink named `notes.md` → `/root/evil`
/// passes an `exists()` check when its target is missing, and only its *parent*
/// gets canonicalized, so `fs::write` would create that file outside the
/// worktree — in the default Ask mode as much as any other. Links are followed
/// here, bounded, and every hop's target parent must stay inside.
fn assert_link_stays_inside(root: &Path, path: &Path) -> Result<()> {
    let mut current = path.to_path_buf();
    for _ in 0..8 {
        let Ok(metadata) = std::fs::symlink_metadata(&current) else {
            // Nothing at the end of the chain: the write creates a real file in
            // a directory already proven to be inside.
            return Ok(());
        };
        if !metadata.file_type().is_symlink() {
            return Ok(());
        }
        let target = std::fs::read_link(&current)
            .map_err(|error| anyhow!("could not read the link {}: {error}", current.display()))?;
        let next = if target.is_absolute() {
            target
        } else {
            current
                .parent()
                .map(|parent| parent.join(&target))
                .unwrap_or(target)
        };
        let anchor = next
            .parent()
            .and_then(|parent| parent.canonicalize().ok())
            .unwrap_or_else(|| next.clone());
        if !anchor.starts_with(root) {
            return Err(anyhow!(
                "{} is a link to {}, which is outside the session worktree",
                path.display(),
                next.display()
            ));
        }
        current = next;
    }
    Err(anyhow!(
        "{} is a link chain that never resolves to a file",
        path.display()
    ))
}

/// One tool's result: what the model reads back, and the card's one-line title.
struct ToolResult {
    title: String,
    body: String,
    failed: bool,
}

fn tool_title(tool: &str, input: &Value) -> String {
    let detail = match tool {
        "bash" => input.get("command").and_then(Value::as_str).unwrap_or(""),
        _ => input.get("path").and_then(Value::as_str).unwrap_or(""),
    };
    let detail: String = detail
        .lines()
        .next()
        .unwrap_or("")
        .chars()
        .take(120)
        .collect();
    if detail.is_empty() {
        tool.to_string()
    } else {
        detail
    }
}

/// Keep the head and tail of a long output, marking the omitted middle — the
/// same shape the chat layer uses for tool text.
fn cap_output(text: String) -> String {
    if text.len() <= MAX_TOOL_OUTPUT {
        return text;
    }
    let head: String = text.chars().take(MAX_TOOL_OUTPUT / 2).collect();
    let tail: Vec<char> = text.chars().rev().take(MAX_TOOL_OUTPUT / 2).collect();
    let tail: String = tail.into_iter().rev().collect();
    format!("{head}\n… [output truncated]\n{tail}")
}

async fn run_bash(workdir: &Path, command: &str) -> ToolResult {
    let mut builder = Command::new("bash");
    builder
        .arg("-lc")
        .arg(command)
        .current_dir(workdir)
        .stdin(std::process::Stdio::null())
        .stdout(std::process::Stdio::piped())
        .stderr(std::process::Stdio::piped())
        .kill_on_drop(true);
    // Its own process group, so a timeout can take the whole tree with it:
    // killing the shell alone leaves a pipeline's other stages — and anything a
    // command backgrounded — running inside the worktree.
    #[cfg(unix)]
    builder.process_group(0);
    let child = match builder.spawn() {
        Ok(child) => child,
        Err(error) => {
            return ToolResult {
                title: tool_title("bash", &json!({"command": command})),
                body: format!("could not start bash: {error}"),
                failed: true,
            }
        }
    };
    // Taken before the child is consumed by `wait_with_output`: a timeout has to
    // signal the group by id.
    let group = child.id();
    let output = match tokio::time::timeout(COMMAND_TIMEOUT, child.wait_with_output()).await {
        Ok(Ok(output)) => output,
        Ok(Err(error)) => {
            return ToolResult {
                title: tool_title("bash", &json!({"command": command})),
                body: format!("bash failed: {error}"),
                failed: true,
            }
        }
        Err(_) => {
            // The shell itself dies with the dropped child (`kill_on_drop`); its
            // group does not, and a surviving stage would keep writing inside the
            // worktree long after the turn moved on.
            #[cfg(unix)]
            if let Some(pid) = group {
                // SAFETY: `pid` leads the group set on the builder above, so this
                // signals that command's tree and nothing else.
                unsafe {
                    libc::killpg(pid as libc::pid_t, libc::SIGKILL);
                }
            }
            return ToolResult {
                title: tool_title("bash", &json!({"command": command})),
                body: format!(
                    "the command was still running after {} minutes and was killed",
                    COMMAND_TIMEOUT.as_secs() / 60
                ),
                failed: true,
            };
        }
    };
    let stdout = String::from_utf8_lossy(&output.stdout).to_string();
    let stderr = String::from_utf8_lossy(&output.stderr).to_string();
    let code = output.status.code();
    let mut body = String::new();
    if !stdout.trim().is_empty() {
        body.push_str(&stdout);
        if !body.ends_with('\n') {
            body.push('\n');
        }
    }
    if !stderr.trim().is_empty() {
        body.push_str("[stderr]\n");
        body.push_str(&stderr);
    }
    if body.trim().is_empty() {
        body.push_str("(no output)");
    }
    let failed = code != Some(0);
    body.push_str(&format!(
        "\n[exit code: {}]",
        code.map(|c| c.to_string())
            .unwrap_or_else(|| "killed by signal".into())
    ));
    ToolResult {
        title: tool_title("bash", &json!({"command": command})),
        body: cap_output(body),
        failed,
    }
}

fn read_file(workdir: &Path, input: &Value) -> ToolResult {
    let raw = input.get("path").and_then(Value::as_str).unwrap_or("");
    let title = tool_title("read_file", input);
    let path = match resolve_confined(workdir, raw) {
        Ok(path) => path,
        Err(error) => {
            return ToolResult {
                title,
                body: error.to_string(),
                failed: true,
            }
        }
    };
    let bytes = match std::fs::read(&path) {
        Ok(bytes) => bytes,
        Err(error) => {
            return ToolResult {
                title,
                body: format!("could not read {}: {error}", path.display()),
                failed: true,
            }
        }
    };
    if bytes.len() > MAX_READ_BYTES {
        return ToolResult {
            title,
            body: format!(
                "{} is {} bytes; read it with a range instead (e.g. `sed -n '1,200p'`)",
                path.display(),
                bytes.len()
            ),
            failed: true,
        };
    }
    let Ok(text) = String::from_utf8(bytes) else {
        return ToolResult {
            title,
            body: format!("{} is not UTF-8 text", path.display()),
            failed: true,
        };
    };
    let offset = input
        .get("offset")
        .and_then(Value::as_u64)
        .unwrap_or(1)
        .max(1) as usize;
    let limit = input
        .get("limit")
        .and_then(Value::as_u64)
        .map(|limit| limit as usize)
        .unwrap_or(MAX_FILE_LINES)
        .min(MAX_FILE_LINES);
    let total = text.lines().count();
    let body: String = text
        .lines()
        .enumerate()
        .filter(|(index, _)| *index + 1 >= offset && *index + 1 < offset + limit)
        .map(|(index, line)| {
            // Char-boundary safe: a byte slice here panicked on a long line with
            // a multibyte character straddling the cut.
            let line: String = line.chars().take(2_000).collect();
            format!("{:>6}|{line}\n", index + 1)
        })
        .collect();
    ToolResult {
        title,
        body: if body.is_empty() {
            format!("(no lines from {offset} of {total})")
        } else {
            body
        },
        failed: false,
    }
}

fn write_file(workdir: &Path, input: &Value) -> ToolResult {
    let title = tool_title("write_file", input);
    let raw = input.get("path").and_then(Value::as_str).unwrap_or("");
    let content = input.get("content").and_then(Value::as_str).unwrap_or("");
    if content.len() > MAX_WRITE_BYTES {
        return ToolResult {
            title,
            body: format!("refusing to write {} bytes in one call", content.len()),
            failed: true,
        };
    }
    let path = match resolve_confined(workdir, raw) {
        Ok(path) => path,
        Err(error) => {
            return ToolResult {
                title,
                body: error.to_string(),
                failed: true,
            }
        }
    };
    if let Some(parent) = path.parent() {
        if let Err(error) = std::fs::create_dir_all(parent) {
            return ToolResult {
                title,
                body: format!("could not create {}: {error}", parent.display()),
                failed: true,
            };
        }
        // Re-check after creating: the parent could have been a symlink out of
        // the worktree, which the lexical check above cannot see.
        if let Err(error) = resolve_confined(workdir, &path.to_string_lossy()) {
            return ToolResult {
                title,
                body: error.to_string(),
                failed: true,
            };
        }
    }
    match std::fs::write(&path, content) {
        Ok(()) => ToolResult {
            title,
            body: format!("wrote {} bytes to {}", content.len(), path.display()),
            failed: false,
        },
        Err(error) => ToolResult {
            title,
            body: format!("could not write {}: {error}", path.display()),
            failed: true,
        },
    }
}

fn edit_file(workdir: &Path, input: &Value) -> ToolResult {
    let title = tool_title("edit_file", input);
    let raw = input.get("path").and_then(Value::as_str).unwrap_or("");
    let old = input
        .get("old_string")
        .and_then(Value::as_str)
        .unwrap_or("");
    let new = input
        .get("new_string")
        .and_then(Value::as_str)
        .unwrap_or("");
    let replace_all = input
        .get("replace_all")
        .and_then(Value::as_bool)
        .unwrap_or(false);
    if old.is_empty() {
        return ToolResult {
            title,
            body: "old_string must not be empty".to_string(),
            failed: true,
        };
    }
    let path = match resolve_confined(workdir, raw) {
        Ok(path) => path,
        Err(error) => {
            return ToolResult {
                title,
                body: error.to_string(),
                failed: true,
            }
        }
    };
    let text = match std::fs::read_to_string(&path) {
        Ok(text) => text,
        Err(error) => {
            return ToolResult {
                title,
                body: format!("could not read {}: {error}", path.display()),
                failed: true,
            }
        }
    };
    let count = text.matches(old).count();
    if count == 0 {
        return ToolResult {
            title,
            body: format!("old_string was not found in {}", path.display()),
            failed: true,
        };
    }
    if count > 1 && !replace_all {
        return ToolResult {
            title,
            body: format!(
                "old_string appears {count} times in {}; add surrounding context to make it \
                 unique, or pass replace_all",
                path.display()
            ),
            failed: true,
        };
    }
    let updated = if replace_all {
        text.replace(old, new)
    } else {
        text.replacen(old, new, 1)
    };
    match std::fs::write(&path, updated) {
        Ok(()) => ToolResult {
            title,
            body: format!("replaced {count} occurrence(s) in {}", path.display()),
            failed: false,
        },
        Err(error) => ToolResult {
            title,
            body: format!("could not write {}: {error}", path.display()),
            failed: true,
        },
    }
}

fn run_tool(workdir: &Path, call: &PlannedCall) -> ToolResult {
    match call.name.as_str() {
        "bash" => {
            let command = call
                .input
                .get("command")
                .and_then(Value::as_str)
                .unwrap_or("");
            if command.trim().is_empty() {
                return ToolResult {
                    title: "bash".to_string(),
                    body: "command must not be empty".to_string(),
                    failed: true,
                };
            }
            // Blocking spawn of an async fn: run_bash is async, so callers use
            // `run_tool_async`; this arm exists for the sync test surface.
            ToolResult {
                title: tool_title("bash", &call.input),
                body: "internal: bash must run through run_tool_async".to_string(),
                failed: true,
            }
        }
        "read_file" => read_file(workdir, &call.input),
        "write_file" => write_file(workdir, &call.input),
        "edit_file" => edit_file(workdir, &call.input),
        other => ToolResult {
            title: other.to_string(),
            body: format!("unknown tool {other}"),
            failed: true,
        },
    }
}

async fn run_tool_async(workdir: &Path, call: &PlannedCall) -> ToolResult {
    if call.name == "bash" {
        let command = call
            .input
            .get("command")
            .and_then(Value::as_str)
            .unwrap_or("");
        if command.trim().is_empty() {
            return ToolResult {
                title: "bash".to_string(),
                body: "command must not be empty".to_string(),
                failed: true,
            };
        }
        return run_bash(workdir, command).await;
    }
    let workdir = workdir.to_path_buf();
    let call = call.clone();
    let name = call.name.clone();
    // File tools are blocking fs work; keep them off the async runtime's threads.
    tokio::task::spawn_blocking(move || run_tool(&workdir, &call))
        .await
        .unwrap_or_else(|error| ToolResult {
            title: name,
            body: format!("tool task failed: {error}"),
            failed: true,
        })
}

// --- prompt assembly ------------------------------------------------------------

/// The session's agent prompt: how to work here, the project playbook, and the
/// skill set that a CLI harness would have auto-loaded from disk.
fn system_prompt(
    workdir: &Path,
    playbook: &Path,
    skills: &[(String, String, PathBuf)],
    bootstrap: Option<&str>,
    plan_mode: bool,
) -> String {
    let mut prompt = String::new();
    prompt.push_str(
        "You are the OpenResearch session agent, running on an Alibaba Cloud token plan model.\n\
         \n\
         You work in one private git worktree of the project's repository, using the tools you \
         are given (bash, read_file, write_file, edit_file). Make the change in the worktree \
         and explain what you did in your final reply; the user reads that reply, not the tool \
         output.\n\
         \n\
         Rules:\n\
         - Work only inside the worktree. Never read or write outside it.\n\
         - Never push to a remote, publish, or delete anything you did not create in this \
         session unless the user asked for exactly that.\n\
         - Prefer small, verifiable changes: run the project's own checks when there are any.\n\
         - If a request is ambiguous in a way that changes what you build, say so plainly rather \
         than guessing.\n",
    );
    prompt.push_str(&format!("\nWorking directory: {}\n", workdir.display()));
    if plan_mode {
        prompt.push_str(
            "\nPLAN MODE is on for this session: only read-only tools run. Explore, then present \
             the plan in your reply — do not attempt to change anything.\n",
        );
    }
    if let Some(bootstrap) = bootstrap.filter(|text| !text.trim().is_empty()) {
        prompt.push_str("\nThe session carries this context:\n");
        prompt.push_str(bootstrap.trim());
        prompt.push('\n');
    }
    match std::fs::read_to_string(playbook) {
        Ok(text) if !text.trim().is_empty() => {
            prompt.push_str(&format!(
                "\nThe project's session playbook is reproduced below (also on disk at {}).\n\n",
                playbook.display()
            ));
            prompt.push_str(&text);
            prompt.push('\n');
        }
        _ => {}
    }
    if !skills.is_empty() {
        prompt.push_str(
            "\n<session-skills>\nThese are the skills this session has installed. The full text \
             of the shorter ones is included; longer ones list their file path — read it with \
             read_file before relying on that skill.\n",
        );
        let mut budget = SKILL_TOTAL_CAP;
        for (name, body, path) in skills {
            if body.len() <= SKILL_INLINE_CAP && body.len() <= budget {
                budget -= body.len();
                prompt.push_str(&format!("\n<skill name=\"{name}\">\n{body}\n</skill>\n"));
            } else {
                prompt.push_str(&format!("- {name}: {}\n", path.display()));
            }
        }
        prompt.push_str("</session-skills>\n");
    }
    prompt
}

/// Every skill in the session's skills dir, as `(name, body, path)`.
fn session_skills(workdir: &Path) -> Vec<(String, String, PathBuf)> {
    let dir = workdir.join(SESSION_SKILLS_DIR);
    let Ok(entries) = std::fs::read_dir(&dir) else {
        return Vec::new();
    };
    let mut skills: Vec<(String, String, PathBuf)> = entries
        .flatten()
        .filter(|entry| entry.path().is_dir())
        .filter_map(|entry| {
            let path = entry.path().join("SKILL.md");
            let body = std::fs::read_to_string(&path).ok()?;
            let name = entry.file_name().to_string_lossy().to_string();
            Some((name, body, path))
        })
        .collect();
    skills.sort_by(|a, b| a.0.cmp(&b.0));
    skills
}

// --- transcript replay ----------------------------------------------------------

/// The session's conversation, oldest first, as API messages — the active
/// branch only, so a forked session replays the branch it is on. The in-flight
/// turn's own assistant message (still empty) and its user message are dropped:
/// the user half is re-sent as this turn's input, with attachments intact.
fn transcript_messages(session_id: &str) -> Result<(Vec<Value>, Option<String>)> {
    let store = crate::store::Store::open()?;
    let session = store.get_chat_session(session_id)?;
    let rows = store.list_chat_messages(session_id)?;
    let leaf = session
        .as_ref()
        .and_then(|session| session.active_leaf_id.clone());
    let mut path: Vec<&crate::store::StoredChatMessage> = Vec::new();
    if let Some(mut current) = leaf
        .as_deref()
        .and_then(|leaf| rows.iter().find(|row| row.id == leaf))
    {
        path.push(current);
        while let Some(parent) = current
            .parent_id
            .as_deref()
            .and_then(|parent| rows.iter().find(|row| row.id == parent))
        {
            current = parent;
            path.push(current);
        }
        path.reverse();
    } else {
        path.extend(rows.iter());
    }
    // The tail is this turn: its user row is sent as the live input, and its
    // assistant row has nothing in it yet.
    while path
        .last()
        .is_some_and(|row| row.role == "assistant" && is_empty_row(row))
    {
        path.pop();
    }
    let bootstrap = session.and_then(|session| session.bootstrap_context);
    // A compaction marker says the session was summarized: `bootstrap_context`
    // carries that summary, so replaying the branch above the marker would send
    // summary *plus* the full history — the one thing compaction exists to stop,
    // growing the request every turn instead of shrinking it. Truncate at the
    // last marker, but only when the summary is actually there: without it the
    // history is the only context the turn has.
    let start = match bootstrap.as_deref().filter(|text| !text.trim().is_empty()) {
        Some(_) => path
            .iter()
            .rposition(|row| is_compaction_marker(row))
            .map(|at| at + 1)
            .unwrap_or(0),
        None => 0,
    };
    let replay = &path[start..];
    let mut messages = Vec::new();
    for row in replay.iter().take(replay.len().saturating_sub(1)) {
        if let Some(message) = replay_message(row) {
            messages.push(message);
        }
    }
    Ok((messages, bootstrap))
}

/// Whether this row is the synthetic `compacted` marker the chat layer writes
/// when a session is summarized. Its status may be `running`/`error` (a
/// cancelled compaction) — the caller only truncates when a summary exists, so
/// a failed marker cannot silently drop the history.
fn is_compaction_marker(row: &crate::store::StoredChatMessage) -> bool {
    parts_of(row).iter().any(|part| {
        part.tool.as_deref() == Some(chat::COMPACTED_TOOL)
            && part
                .state
                .as_ref()
                .is_some_and(|state| state.status == "completed")
    })
}

fn is_empty_row(row: &crate::store::StoredChatMessage) -> bool {
    parts_of(row)
        .iter()
        .all(|part| part.text.as_deref().unwrap_or("").trim().is_empty() && part.kind == "text")
}

fn parts_of(row: &crate::store::StoredChatMessage) -> Vec<WirePart> {
    serde_json::from_str::<Vec<WirePart>>(&row.parts_json).unwrap_or_default()
}

/// One stored message as an API message. Tool activity from a previous turn is
/// folded in as a `<tool-activity>` block: the transcript keeps each tool part's
/// input and output but no `tool_call_id`, so a real tool exchange cannot be
/// rebuilt — a readable digest is the honest replay.
fn replay_message(row: &crate::store::StoredChatMessage) -> Option<Value> {
    let parts = parts_of(row);
    let mut content = String::new();
    let mut images: Vec<Value> = Vec::new();
    for part in &parts {
        match part.kind.as_str() {
            "text" => {
                if let Some(text) = part.text.as_deref().filter(|text| !text.trim().is_empty()) {
                    if !content.is_empty() {
                        content.push_str("\n\n");
                    }
                    content.push_str(text);
                }
            }
            "tool" => {
                let name = part.tool.as_deref().unwrap_or("tool");
                let (status, detail) = match part.state.as_ref() {
                    Some(state) => (
                        state.status.clone(),
                        state
                            .output
                            .clone()
                            .or_else(|| state.error.clone())
                            .unwrap_or_default(),
                    ),
                    None => ("completed".to_string(), String::new()),
                };
                let detail = detail.chars().take(2_000).collect::<String>();
                content.push_str(&format!(
                    "\n\n<tool-activity name=\"{name}\" status=\"{status}\">\n{detail}\n</tool-activity>"
                ));
            }
            "image" if row.role == "user" => {
                if let Some(image) = attachment_image(part.text.as_deref().unwrap_or("")) {
                    images.push(image);
                }
            }
            _ => {}
        }
    }
    if content.trim().is_empty() && images.is_empty() {
        return None;
    }
    Some(if images.is_empty() {
        json!({"role": row.role, "content": content})
    } else {
        let mut parts = vec![json!({"type": "text", "text": content})];
        parts.extend(images);
        json!({"role": row.role, "content": parts})
    })
}

/// An image attachment as an OpenAI `image_url` part, from the file the chat
/// layer saved. Larger than [`MAX_IMAGE_BYTES`] is skipped rather than blowing
/// up the request; the model still sees the file name in the input text.
fn attachment_image(file_name: &str) -> Option<Value> {
    let path = chat::attachments_dir().ok()?.join(file_name);
    let metadata = std::fs::metadata(&path).ok()?;
    if metadata.len() > MAX_IMAGE_BYTES {
        return None;
    }
    let media_type = image_media_type(&path)?;
    let bytes = std::fs::read(&path).ok()?;
    use base64::Engine as _;
    let encoded = base64::engine::general_purpose::STANDARD.encode(bytes);
    Some(json!({
        "type": "image_url",
        "image_url": {"url": format!("data:{media_type};base64,{encoded}")}
    }))
}

/// The media type of an image attachment, by extension. Anything else (a PDF, a
/// text file, an unknown extension) is left to the model's `read_file`.
fn image_media_type(path: &Path) -> Option<&'static str> {
    match path.extension().and_then(|ext| ext.to_str())? {
        "png" => Some("image/png"),
        "jpg" | "jpeg" => Some("image/jpeg"),
        "gif" => Some("image/gif"),
        "webp" => Some("image/webp"),
        _ => None,
    }
}

// --- the turn -------------------------------------------------------------------

async fn run_turn(ctx: &mut TurnCtx) -> Result<()> {
    let Some(settings) = settings() else {
        ctx.push_error(missing_key_note());
        return Err(anyhow!(
            "the Alibaba Cloud token plan key is not configured"
        ));
    };
    let client = http_client()?;
    let project = ctx.project.clone();
    let session_id = ctx.session_id.clone();
    let (workdir, playbook) = tokio::task::spawn_blocking(move || {
        ensure_playbook(&project, &session_id, Some(SESSION_SKILLS_DIR))
    })
    .await
    .map_err(|error| anyhow!("session setup task failed: {error}"))??;

    let plan_mode = ctx.permission_mode == Some(PermissionMode::Plan);
    let mode = ctx.permission_mode.unwrap_or(PermissionMode::Auto);
    let (history, bootstrap) = transcript_messages(&ctx.session_id)?;
    let skills = session_skills(&workdir);
    let system = system_prompt(
        &workdir,
        &playbook,
        &skills,
        bootstrap.as_deref(),
        plan_mode,
    );
    let model = ctx
        .model
        .clone()
        .filter(|model| !model.trim().is_empty())
        .unwrap_or_else(|| DEFAULT_MODEL.to_string());

    let mut messages = vec![json!({"role": "system", "content": system})];
    messages.extend(history);
    messages.push(current_user_message(ctx));

    let token = ctx.host.mint_gate_token(
        &ctx.session_id,
        plan_mode,
        matches!(mode, PermissionMode::Auto | PermissionMode::Bypass),
    );

    let mut delivered = false;
    for step in 0..MAX_STEPS {
        let outcome = stream_step(
            ctx,
            &client,
            &settings,
            &model,
            &messages,
            step,
            &mut delivered,
        )
        .await?;
        if let Some(usage) = outcome.usage.as_ref() {
            report_usage(ctx, usage);
        }
        if outcome.calls.is_empty() {
            ctx.mark_final_text_tail();
            return Ok(());
        }
        // The assistant's own tool-call turn, then one result per call, in the
        // order the model asked for them.
        messages.push(assistant_tool_message(&outcome));
        for call in &outcome.calls {
            let body = execute_call(ctx, &token, &workdir, call).await;
            messages.push(json!({
                "role": "tool",
                "tool_call_id": call.id,
                "content": body,
            }));
        }
    }
    ctx.push_error(format!(
        "the model reached the {MAX_STEPS}-step limit for one turn without finishing; \
         send another message to continue."
    ));
    Ok(())
}

/// The live turn's input, with any images the user attached as data URLs.
fn current_user_message(ctx: &TurnCtx) -> Value {
    let images = current_turn_images(&ctx.session_id);
    if images.is_empty() {
        json!({"role": "user", "content": ctx.text})
    } else {
        let mut content = vec![json!({"type": "text", "text": ctx.text})];
        content.extend(images);
        json!({"role": "user", "content": content})
    }
}

/// Images on the user message being answered — the last user row of the session.
fn current_turn_images(session_id: &str) -> Vec<Value> {
    let Ok(store) = crate::store::Store::open() else {
        return Vec::new();
    };
    let Ok(rows) = store.list_chat_messages(session_id) else {
        return Vec::new();
    };
    let Some(row) = rows.iter().rev().find(|row| row.role == "user") else {
        return Vec::new();
    };
    parts_of(row)
        .iter()
        .filter(|part| part.kind == "image")
        .filter_map(|part| part.text.as_deref())
        .filter_map(attachment_image)
        .collect()
}

fn assistant_tool_message(step: &Step) -> Value {
    let calls: Vec<Value> = step
        .calls
        .iter()
        .map(|call| {
            json!({
                "id": call.id,
                "type": "function",
                "function": {
                    "name": call.name,
                    "arguments": call.input.to_string(),
                }
            })
        })
        .collect();
    json!({"role": "assistant", "content": step.text, "tool_calls": calls})
}

/// Gate, then run one tool call; returns what the model reads back. The wire
/// part is streamed either way, so the transcript shows refusals and denials the
/// same way it shows successes.
async fn execute_call(
    ctx: &mut TurnCtx,
    token: &str,
    workdir: &Path,
    call: &PlannedCall,
) -> String {
    let part_id = format!("{}-tool-{}", ctx.turn_id, call.id);
    let mut part = WirePart::tool(part_id.clone(), call.name.clone(), "running", None);
    if let Some(state) = part.state.as_mut() {
        state.input = Some(call.input.clone());
        state.title = Some(tool_title(&call.name, &call.input));
    }
    ctx.upsert_part(part);
    ctx.maybe_flush();

    let outcome = if let Some(broken) = call.broken.as_deref() {
        ToolResult {
            title: tool_title(&call.name, &call.input),
            body: format!("I could not read this call: {broken}"),
            failed: true,
        }
    } else {
        match gate(
            ctx.permission_mode.unwrap_or(PermissionMode::Auto),
            &call.name,
            &call.input,
        ) {
            Gate::Refuse(message) => ToolResult {
                title: tool_title(&call.name, &call.input),
                body: message,
                failed: true,
            },
            Gate::Run => run_tool_async(workdir, call).await,
            Gate::Ask => {
                let host = ctx.host.clone();
                let session_id = ctx.session_id.clone();
                let decision = host
                    .request_permission(&session_id, token, &call.name, call.input.clone())
                    .await
                    .unwrap_or_else(|error| PermissionDecision::Deny {
                        message: error.to_string(),
                    });
                match decision {
                    PermissionDecision::Allow { .. } => run_tool_async(workdir, call).await,
                    PermissionDecision::Deny { message } => ToolResult {
                        title: tool_title(&call.name, &call.input),
                        body: message,
                        failed: true,
                    },
                }
            }
        }
    };

    let mut part = WirePart::tool(
        part_id,
        call.name.clone(),
        if outcome.failed { "error" } else { "completed" },
        outcome.failed.then(|| outcome.body.clone()),
    );
    if let Some(state) = part.state.as_mut() {
        state.input = Some(call.input.clone());
        state.title = Some(outcome.title);
        if !outcome.failed {
            state.output = Some(outcome.body.clone());
        }
    }
    ctx.upsert_part(part);
    ctx.maybe_flush();
    outcome.body
}

/// Stream one model step into wire parts and return what it decided.
async fn stream_step(
    ctx: &mut TurnCtx,
    client: &reqwest::Client,
    settings: &Settings,
    model: &str,
    messages: &[Value],
    step_index: usize,
    delivered: &mut bool,
) -> Result<Step> {
    let body = json!({
        "model": model,
        "messages": messages,
        "tools": tool_schemas(),
        "tool_choice": "auto",
        "stream": true,
        "stream_options": {"include_usage": true},
    });
    let response = client
        .post(format!("{}/chat/completions", settings.base_url))
        .bearer_auth(&settings.key)
        .json(&body)
        .send()
        .await
        .map_err(|error| anyhow!("the token plan did not answer: {error}"))?;
    let status = response.status();
    if !status.is_success() {
        let text = response.text().await.unwrap_or_default();
        return Err(anyhow!(
            "the token plan refused the request ({status}): {}",
            api_error_message(&text)
        ));
    }
    *delivered = true;
    ctx.mark_delivery(DeliveryState::Accepted);

    // Part ids are stable per step so a streaming update replaces the part the
    // previous frame opened instead of stacking a new one per delta.
    let text_id = format!("{}-s{step_index}-text", ctx.turn_id);
    let reasoning_id = format!("{}-s{step_index}-reasoning", ctx.turn_id);
    let mut text_open = false;
    let mut reasoning_open = false;
    let mut step = Step::default();
    let mut calls: Vec<CallAccum> = Vec::new();
    let mut buffer = SseBuffer::default();
    let mut stream = response.bytes_stream();
    // `[DONE]` ends the step. Breaking only the frame loop would leave the
    // stream polled until the stall watchdog fired on a connection the server
    // holds open after its terminator — a completed step reported as a stall.
    let mut finished = false;

    loop {
        let chunk = match tokio::time::timeout(STALL_TIMEOUT, stream.next()).await {
            Ok(Some(Ok(bytes))) => bytes,
            Ok(Some(Err(error))) => return Err(anyhow!("the token plan's stream broke: {error}")),
            Ok(None) => break,
            Err(_) => {
                return Err(anyhow!(
                    "the token plan sent nothing for {} minutes; the turn was stopped",
                    STALL_TIMEOUT.as_secs() / 60
                ))
            }
        };
        for frame in buffer.push(&chunk) {
            let Some(data) = frame_data(&frame) else {
                continue;
            };
            let Some(decoded) = decode_chunk(&data) else {
                continue;
            };
            if decoded.done {
                finished = true;
                break;
            }
            if !decoded.reasoning.is_empty() {
                if !reasoning_open {
                    ctx.upsert_part(WirePart::reasoning(reasoning_id.clone(), ""));
                    reasoning_open = true;
                }
                ctx.append_part_text(&reasoning_id, &decoded.reasoning);
            }
            if !decoded.text.is_empty() {
                if !text_open {
                    ctx.upsert_part(WirePart::text(text_id.clone(), ""));
                    text_open = true;
                }
                ctx.append_part_text(&text_id, &decoded.text);
                step.text.push_str(&decoded.text);
            }
            for delta in &decoded.tool_calls {
                while calls.len() <= delta.index {
                    calls.push(CallAccum::default());
                }
                calls[delta.index].merge(delta);
            }
            if decoded.usage.is_some() {
                step.usage = decoded.usage;
            }
            ctx.maybe_flush();
        }
        if finished {
            break;
        }
    }
    step.calls = calls
        .into_iter()
        .enumerate()
        .map(|(index, call)| call.plan(index))
        .collect();
    Ok(step)
}

/// Report the model's own usage numbers as the session's context usage. The plan
/// reports no context-window size, so only the consumed side is known — the UI
/// then shows consumption without a percentage.
fn report_usage(ctx: &mut TurnCtx, usage: &Value) {
    let used = ["prompt_tokens", "completion_tokens"]
        .iter()
        .filter_map(|key| usage.get(*key).and_then(Value::as_u64))
        .sum::<u64>();
    if used == 0 {
        return;
    }
    ctx.report_usage(ContextUsage {
        used_tokens: used,
        context_window: None,
    });
}

// --- the harness ----------------------------------------------------------------

#[async_trait]
impl Harness for AlibabaTokenPlan {
    fn id(&self) -> &'static str {
        HARNESS_ID
    }

    fn name(&self) -> &'static str {
        HARNESS_NAME
    }

    fn supports_chat(&self) -> bool {
        true
    }

    async fn detect(&self) -> Option<HarnessInfo> {
        detection(true).await
    }

    async fn detect_snapshot(&self) -> Option<HarnessInfo> {
        // No subprocess to wait for: the snapshot is the env check plus the
        // known catalog, which is already cheaper than any CLI probe.
        detection(false).await
    }

    async fn run_turn(&self, ctx: &mut TurnCtx) -> TurnResult {
        match run_turn(ctx).await {
            Ok(()) => Ok(TurnOutcome::Completed),
            Err(error) => Err(TurnFailure::adapter(error, ctx.delivery_state())),
        }
    }

    fn options(&self) -> HarnessOptions {
        HarnessOptions::none().with_permission_choices(
            vec![
                OptionChoice::described(
                    "plan",
                    "Plan",
                    "Read-only: explore and propose, change nothing",
                ),
                OptionChoice::described("ask", "Ask", "Ask before every command or file change"),
                OptionChoice::described("auto", "Auto", "Run commands and edit files freely"),
            ],
            "auto",
            super::options::PlanActivation::Permission,
        )
    }

    /// The bridged approval card is answered here: the tool call is parked
    /// inside the running turn waiting on this decision, so the answer is
    /// settled into it and the turn continues in place.
    async fn resume_from_prompt(
        &self,
        ctx: &ResumeCtx,
        prompt: &WirePrompt,
        answer: &PromptAnswer,
    ) -> Result<ResumeAction> {
        let Some(native_id) = prompt.native_id.as_deref() else {
            return Ok(ResumeAction::Nothing);
        };
        if !ctx.is_busy().await {
            ctx.host
                .resolve_zombie_prompt(&ctx.session_id, &answer.prompt_id);
            return Err(anyhow!("this approval is no longer pending"));
        }
        let note = answer
            .note
            .as_deref()
            .filter(|note| !note.trim().is_empty());
        match (prompt.kind.as_str(), answer.approve) {
            ("permission", true) => {
                ctx.host.settle_permission(
                    native_id,
                    PermissionDecision::Allow {
                        updated_input: prompt.tool_input.clone(),
                    },
                )?;
                Ok(ResumeAction::Handled { plan_mode: None })
            }
            ("permission", false) => {
                let message = match note {
                    Some(note) => format!(
                        "The user denied this action: {note}. Do not retry it; adjust course."
                    ),
                    None => {
                        "The user denied this action. Do not retry it; adjust course.".to_string()
                    }
                };
                ctx.host
                    .settle_permission(native_id, PermissionDecision::Deny { message })?;
                Ok(ResumeAction::Handled { plan_mode: None })
            }
            _ => Ok(ResumeAction::Nothing),
        }
    }

    /// One non-streamed request, tools off — the cheap children (session
    /// titles, one-shot helpers) run on this.
    async fn one_shot(&self, request: OneShot<'_>) -> Result<String> {
        let Some(settings) = settings() else {
            return Err(anyhow!(missing_key_note()));
        };
        let client = http_client()?;
        let model = request
            .model
            .filter(|model| !model.trim().is_empty())
            .unwrap_or(DEFAULT_MODEL);
        let body = json!({
            "model": model,
            "messages": [
                {"role": "system", "content": request.system},
                {"role": "user", "content": request.prompt},
            ],
            "stream": false,
        });
        let response = client
            .post(format!("{}/chat/completions", settings.base_url))
            .bearer_auth(&settings.key)
            .timeout(request.timeout)
            .json(&body)
            .send()
            .await
            .map_err(|error| anyhow!("the token plan did not answer: {error}"))?;
        let status = response.status();
        let text = response.text().await.unwrap_or_default();
        if !status.is_success() {
            return Err(anyhow!(
                "the token plan refused the request ({status}): {}",
                api_error_message(&text)
            ));
        }
        let value: Value = serde_json::from_str(&text)
            .map_err(|error| anyhow!("the token plan sent an unreadable reply: {error}"))?;
        let message = value
            .get("choices")
            .and_then(Value::as_array)
            .and_then(|choices| choices.first())
            .and_then(|choice| choice.get("message"))
            .cloned()
            .unwrap_or(Value::Null);
        let content = message
            .get("content")
            .and_then(Value::as_str)
            .unwrap_or_default()
            .to_string();
        if !content.trim().is_empty() {
            return Ok(content);
        }
        // A thinking model can answer with nothing but its reasoning; a caller
        // that asked for one short line wants that rather than an error.
        Ok(message
            .get("reasoning_content")
            .and_then(Value::as_str)
            .unwrap_or_default()
            .to_string())
    }

    fn session_skills_dir(&self) -> Option<&'static str> {
        Some(SESSION_SKILLS_DIR)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn temp_dir(tag: &str) -> PathBuf {
        let dir = std::env::temp_dir().join(format!("orx-alibaba-{tag}-{}", uuid::Uuid::new_v4()));
        std::fs::create_dir_all(&dir).unwrap();
        dir
    }

    fn stored(
        role: &str,
        parts: Vec<WirePart>,
        parent: Option<&str>,
    ) -> crate::store::StoredChatMessage {
        crate::store::StoredChatMessage {
            id: format!("msg_{}", uuid::Uuid::new_v4()),
            session_id: "chat_test".to_string(),
            role: role.to_string(),
            parts_json: serde_json::to_string(&parts).unwrap(),
            created_at: 0,
            completed_at: None,
            parent_id: parent.map(str::to_string),
            base_native_session_id: None,
            result_native_session_id: None,
        }
    }

    /// The plan serves image, TTS, audio and realtime models from the same
    /// endpoint. Offering one as a chat backend would fail every turn, so the
    /// catalog is filtered — and the known list is held to the same rule.
    #[test]
    fn only_chat_capable_models_are_offered() {
        assert!(is_chat_model("qwen3.8-max"));
        assert!(is_chat_model("auto"));
        assert!(is_chat_model("deepseek-v4.1-flash"));
        for id in [
            "wan2.7-image",
            "wan2.7-image-pro",
            "qwen-audio-3.0-tts-plus",
            "qwen-audio-3.0-realtime-plus",
        ] {
            assert!(!is_chat_model(id), "{id} cannot hold a chat turn");
        }
        assert!(
            KNOWN_MODELS.iter().all(|id| is_chat_model(id)),
            "the fallback catalog must not advertise a non-chat model"
        );
    }

    #[test]
    fn catalog_parse_keeps_order_drops_blanks_and_duplicates() {
        let body = json!({"data": [
            {"id": "qwen3.8-max"},
            {"id": "wan2.7-image"},
            {"id": "  "},
            {"id": "qwen3.8-max"},
            {"id": "deepseek-v4.1-flash"},
            {"id": "qwen-audio-3.0-tts-plus"},
            {"id": "auto"},
            {"no_id": true}
        ]});
        assert_eq!(
            parse_model_list(&body),
            vec!["qwen3.8-max", "deepseek-v4.1-flash", "auto"]
        );
        assert!(parse_model_list(&json!({"data": "nonsense"})).is_empty());
        assert!(parse_model_list(&json!({})).is_empty());
    }

    /// A gateway in front of the plan is allowed, but only over TLS — or over
    /// plain http back to loopback. The key must never reach a stranger in the
    /// clear.
    #[test]
    fn the_endpoint_must_be_https_unless_it_is_loopback() {
        assert_eq!(base_url_from(None).unwrap(), DEFAULT_BASE_URL.to_string());
        assert_eq!(
            base_url_from(Some("https://plan.example.test/v1/".into())).unwrap(),
            "https://plan.example.test/v1"
        );
        assert_eq!(
            base_url_from(Some("http://127.0.0.1:8080/v1".into())).unwrap(),
            "http://127.0.0.1:8080/v1"
        );
        assert_eq!(
            base_url_from(Some("http://localhost:9999".into())).unwrap(),
            "http://localhost:9999"
        );
        assert_eq!(
            base_url_from(Some("http://plan.example.test/v1".into())),
            None
        );
        assert_eq!(base_url_from(Some("ftp://plan.example.test".into())), None);
        assert_eq!(base_url_from(Some("".into())), None);
    }

    #[test]
    fn sse_frames_split_on_blank_lines_and_survive_crlf() {
        let mut buffer = SseBuffer::default();
        assert!(buffer.push(b"data: {\"a\":1}\r\n\r\n").len() == 1);
        // A frame split across two network reads is delivered whole.
        let mut buffer = SseBuffer::default();
        assert!(buffer.push(b"data: {\"a\":").is_empty());
        let frames = buffer.push(b"1}\n\ndata: [DONE]\n\n");
        assert_eq!(frames.len(), 2);
        assert_eq!(frame_data(&frames[0]).unwrap(), "{\"a\":1}");
        assert_eq!(frame_data(&frames[1]).unwrap(), "[DONE]");
        assert_eq!(buffer.rest(), "");
    }

    /// The stream is split on bytes, not on decoded text: a multibyte character
    /// straddling two network reads must survive intact. Decoding each chunk on
    /// its own corrupted it into U+FFFD — in streamed text and, worse, inside a
    /// streamed tool-call argument, where the JSON then failed to parse.
    #[test]
    fn a_multibyte_character_split_across_reads_survives() {
        let payload = json!({"choices": [{"delta": {"content": "héllo → 世界"}}]}).to_string();
        let frame = format!("data: {payload}\n\n");
        // Split inside the "→" (3 bytes) — the character straddles the reads.
        let mut buffer = SseBuffer::default();
        let mut frames = Vec::new();
        let arrow = frame.find('→').unwrap();
        frames.extend(buffer.push(&frame.as_bytes()[..arrow + 1]));
        frames.extend(buffer.push(&frame.as_bytes()[arrow + 1..]));
        assert_eq!(frames.len(), 1);
        let decoded = decode_chunk(&frame_data(&frames[0]).unwrap()).unwrap();
        assert_eq!(decoded.text, "héllo → 世界");
        assert!(!decoded.text.contains('\u{fffd}'));
    }

    /// A CRLF terminator split across two reads is still one boundary. The old
    /// normalizer ran per chunk, so `\r` and `\n\r\n` arriving separately left a
    /// raw `\r\n` mid-buffer and merged the next frame into the previous one.
    #[test]
    fn a_crlf_terminator_split_across_reads_is_one_boundary() {
        let mut buffer = SseBuffer::default();
        assert!(buffer.push(b"data: {\"a\":1}\r").is_empty());
        assert!(buffer.push(b"\n\r").is_empty());
        let frames = buffer.push(b"\ndata: {\"b\":2}\r\n\r\n");
        assert_eq!(frames.len(), 2, "both frames, not one merged frame");
        assert_eq!(frame_data(&frames[0]).unwrap(), "{\"a\":1}");
        assert_eq!(frame_data(&frames[1]).unwrap(), "{\"b\":2}");
    }

    /// A CRLF stream has no `\n\n` in it at all, so both terminators are
    /// searched for.
    #[test]
    fn a_whole_crlf_stream_splits_into_frames() {
        let mut buffer = SseBuffer::default();
        let frames = buffer.push(b"data: {\"a\":1}\r\n\r\ndata: [DONE]\r\n\r\n");
        assert_eq!(frames.len(), 2);
        assert!(decode_chunk(&frame_data(&frames[1]).unwrap()).unwrap().done);
    }

    /// A long line used to be byte-sliced at 2000, which panicked whenever a
    /// multibyte character straddled the cut.
    #[test]
    fn reading_a_long_line_with_a_multibyte_character_does_not_panic() {
        let workdir = temp_dir("longline");
        // 1999 ASCII bytes, then a 3-byte character straddling the 2000 cut.
        let line = format!("{}→{}", "a".repeat(1_999), "b".repeat(50));
        std::fs::write(workdir.join("data.txt"), format!("{line}\n")).unwrap();
        let result = read_file(&workdir, &json!({"path": "data.txt"}));
        assert!(!result.failed);
        assert!(result.body.contains('a'));
        assert!(result.body.contains('→'));
        std::fs::remove_dir_all(&workdir).ok();
    }

    /// The escape the lexical check cannot see: a symlink inside the worktree
    /// whose target is elsewhere. A dangling one passes `exists()`, so the write
    /// that follows created the file outside the worktree.
    #[test]
    fn a_symlink_pointing_out_of_the_worktree_is_refused() {
        let workdir = temp_dir("symlink-out");
        let outside = temp_dir("symlink-target");
        #[cfg(unix)]
        {
            // Dangling link → a path outside: the write must not create it.
            let escaping = outside.join("evil.txt");
            std::os::unix::fs::symlink(&escaping, workdir.join("notes.md")).unwrap();
            let refused = write_file(&workdir, &json!({"path": "notes.md", "content": "pwned"}));
            assert!(refused.failed, "{}", refused.body);
            assert!(refused.body.contains("outside the session worktree"));
            assert!(!escaping.exists(), "the file was created outside");

            // Reading through it is refused for the same reason.
            let read = read_file(&workdir, &json!({"path": "notes.md"}));
            assert!(read.failed);
            assert!(std::fs::read(workdir.join("notes.md")).is_err());

            // A link that stays inside is still usable.
            std::fs::create_dir_all(workdir.join("src")).unwrap();
            std::fs::write(workdir.join("src/real.txt"), "inside\n").unwrap();
            std::os::unix::fs::symlink("src/real.txt", workdir.join("alias.txt")).unwrap();
            let allowed = read_file(&workdir, &json!({"path": "alias.txt"}));
            assert!(!allowed.failed, "{}", allowed.body);
            assert!(allowed.body.contains("inside"));
        }
        std::fs::remove_dir_all(&workdir).ok();
        std::fs::remove_dir_all(&outside).ok();
    }

    /// Compaction is only honoured when the summary it points at exists —
    /// otherwise dropping the history would leave the turn with no context.
    #[test]
    fn the_compaction_marker_is_recognized() {
        let marker = stored(
            "assistant",
            vec![WirePart::tool(
                chat::COMPACTED_TOOL,
                chat::COMPACTED_TOOL,
                "completed",
                None,
            )],
            None,
        );
        assert!(is_compaction_marker(&marker));
        // A cancelled compaction is not a summary.
        let failed = stored(
            "assistant",
            vec![WirePart::tool(
                chat::COMPACTED_TOOL,
                chat::COMPACTED_TOOL,
                "error",
                Some("cancelled".into()),
            )],
            None,
        );
        assert!(!is_compaction_marker(&failed));
        assert!(!is_compaction_marker(&stored(
            "assistant",
            vec![WirePart::text("p", "hi")],
            None
        )));
    }

    #[test]
    fn frame_data_ignores_comments_and_other_fields() {
        let frame = ": keep-alive\nevent: message\ndata: {\"x\": true}\n";
        assert_eq!(frame_data(frame).unwrap(), "{\"x\": true}");
        assert_eq!(frame_data(": ping\n"), None);
    }

    #[test]
    fn a_chunk_carries_text_reasoning_calls_and_usage() {
        let data = json!({
            "choices": [{"delta": {
                "content": "hello",
                "reasoning_content": "thinking",
                "tool_calls": [
                    {"index": 0, "id": "call_1", "function": {"name": "bash", "arguments": "{\"comm"}}
                ]
            }}],
            "usage": {"prompt_tokens": 10, "completion_tokens": 2}
        })
        .to_string();
        let chunk = decode_chunk(&data).unwrap();
        assert_eq!(chunk.text, "hello");
        assert_eq!(chunk.reasoning, "thinking");
        assert_eq!(chunk.usage.unwrap()["prompt_tokens"], 10);
        assert_eq!(
            chunk.tool_calls,
            vec![ToolCallDelta {
                index: 0,
                id: Some("call_1".to_string()),
                name: Some("bash".to_string()),
                arguments: Some("{\"comm".to_string()),
            }]
        );
    }

    #[test]
    fn unreadable_and_terminating_chunks_are_handled() {
        assert!(decode_chunk("not json").is_none());
        assert!(decode_chunk("").is_none());
        assert!(decode_chunk("[DONE]").unwrap().done);
        // A keep-alive frame with no choices is still a frame, not an error.
        let chunk = decode_chunk("{\"choices\": []}").unwrap();
        assert!(chunk.text.is_empty() && !chunk.done);
    }

    /// Tool arguments stream in fragments; only the joined whole is parsed.
    #[test]
    fn tool_call_fragments_are_joined_and_planned() {
        let mut calls = vec![CallAccum::default()];
        for delta in [
            ToolCallDelta {
                index: 0,
                id: Some("call_1".into()),
                name: Some("bash".into()),
                arguments: Some("{\"command\": \"echo ".into()),
            },
            ToolCallDelta {
                index: 0,
                id: None,
                name: None,
                arguments: Some("hi\"}".into()),
            },
        ] {
            calls[0].merge(&delta);
        }
        let planned = calls.remove(0).plan(0);
        assert_eq!(planned.id, "call_1");
        assert_eq!(planned.name, "bash");
        assert_eq!(planned.input["command"], "echo hi");
        assert!(planned.broken.is_none());
    }

    #[test]
    fn a_call_with_unreadable_arguments_is_reported_not_run() {
        let mut call = CallAccum {
            id: String::new(),
            name: "bash".to_string(),
            arguments: "{not json".to_string(),
        };
        let planned = call.clone().plan(3);
        // No id from the provider: a synthetic one keeps the tool result linked.
        assert_eq!(planned.id, "call_3");
        assert!(planned.broken.is_some());
        assert!(planned.input.is_null());
        // An empty argument list is a legitimate no-argument call.
        call.arguments = String::new();
        let empty_args = call.plan(0);
        assert_eq!(empty_args.input, json!({}));
        assert!(empty_args.broken.is_none());
        // A nameless call still reaches the model as a failure, not a panic.
        let unnamed = CallAccum::default().plan(0);
        assert_eq!(unnamed.name, "unknown");
    }

    #[test]
    fn tools_cannot_leave_the_worktree() {
        let workdir = temp_dir("paths");
        std::fs::create_dir_all(workdir.join("src")).unwrap();
        std::fs::write(workdir.join("src/main.rs"), "fn main() {}\n").unwrap();
        assert!(resolve_in_worktree(&workdir, "src/main.rs").is_ok());
        assert!(resolve_in_worktree(&workdir, "./src/../src/main.rs").is_err());
        assert!(resolve_in_worktree(&workdir, "../secrets").is_err());
        assert!(resolve_in_worktree(&workdir, "/etc/passwd").is_err());
        assert!(resolve_in_worktree(&workdir, "   ").is_err());
        std::fs::remove_dir_all(&workdir).ok();
    }

    #[test]
    fn plan_mode_reads_and_refuses_without_a_card() {
        let read = json!({"path": "README.md"});
        let write = json!({"path": "a.txt", "content": "x"});
        assert_eq!(gate(PermissionMode::Plan, "read_file", &read), Gate::Run);
        assert_eq!(
            gate(
                PermissionMode::Plan,
                "bash",
                &json!({"command": "git status"})
            ),
            Gate::Run
        );
        assert_eq!(
            gate(
                PermissionMode::Plan,
                "bash",
                &json!({"command": "rm -rf build"})
            ),
            Gate::Ask
        );
        assert!(matches!(
            gate(PermissionMode::Plan, "write_file", &write),
            Gate::Refuse(_)
        ));
        assert!(matches!(
            gate(PermissionMode::Plan, "edit_file", &write),
            Gate::Refuse(_)
        ));
    }

    #[test]
    fn ask_mode_cards_every_command_but_lets_plain_reads_run() {
        assert_eq!(
            gate(PermissionMode::Ask, "read_file", &json!({"path": "a"})),
            Gate::Run
        );
        // Ask means ask: the shared classifier's read-only set is NOT an
        // exemption here, because this harness runs no sandbox — `cat
        // /root/.ssh/…` is read-only to the classifier and machine-wide.
        assert_eq!(
            gate(
                PermissionMode::Ask,
                "bash",
                &json!({"command": "git status"})
            ),
            Gate::Ask
        );
        assert_eq!(
            gate(PermissionMode::Ask, "bash", &json!({"command": "ls -la"})),
            Gate::Ask
        );
        assert_eq!(
            gate(PermissionMode::Ask, "bash", &json!({"command": "npm test"})),
            Gate::Ask
        );
        assert_eq!(
            gate(
                PermissionMode::Ask,
                "write_file",
                &json!({"path": "a", "content": "b"})
            ),
            Gate::Ask
        );
    }

    #[test]
    fn auto_and_bypass_never_ask() {
        for mode in [PermissionMode::Auto, PermissionMode::Bypass] {
            assert_eq!(
                gate(mode, "bash", &json!({"command": "rm -rf build"})),
                Gate::Run
            );
            assert_eq!(
                gate(mode, "write_file", &json!({"path": "a", "content": "b"})),
                Gate::Run
            );
        }
    }

    #[test]
    fn editing_requires_a_unique_match() {
        let workdir = temp_dir("edit");
        let path = workdir.join("notes.md");
        std::fs::write(&path, "alpha beta beta\n").unwrap();

        let ambiguous = edit_file(
            &workdir,
            &json!({"path": "notes.md", "old_string": "beta", "new_string": "gamma"}),
        );
        assert!(ambiguous.failed);
        assert!(ambiguous.body.contains("2 times"));
        assert_eq!(std::fs::read_to_string(&path).unwrap(), "alpha beta beta\n");

        let unique = edit_file(
            &workdir,
            &json!({"path": "notes.md", "old_string": "alpha", "new_string": "omega"}),
        );
        assert!(!unique.failed);

        let missing = edit_file(
            &workdir,
            &json!({"path": "notes.md", "old_string": "nope", "new_string": "x"}),
        );
        assert!(missing.failed);
        assert_eq!(std::fs::read_to_string(&path).unwrap(), "omega beta beta\n");

        let all = edit_file(
            &workdir,
            &json!({"path": "notes.md", "old_string": "beta", "new_string": "gamma", "replace_all": true}),
        );
        assert!(!all.failed);
        assert_eq!(
            std::fs::read_to_string(&path).unwrap(),
            "omega gamma gamma\n"
        );
        std::fs::remove_dir_all(&workdir).ok();
    }

    #[test]
    fn writing_and_reading_stay_inside_the_worktree() {
        let workdir = temp_dir("files");
        let written = write_file(
            &workdir,
            &json!({"path": "deep/nested/file.txt", "content": "one\ntwo\n"}),
        );
        assert!(!written.failed);
        assert!(workdir.join("deep/nested/file.txt").exists());

        let escaped = write_file(
            &workdir,
            &json!({"path": "../outside.txt", "content": "nope"}),
        );
        assert!(escaped.failed);
        assert!(!workdir.parent().unwrap().join("outside.txt").exists());

        let numbered = read_file(&workdir, &json!({"path": "deep/nested/file.txt"}));
        assert!(!numbered.failed);
        assert!(numbered.body.contains("     1|one"));
        assert!(numbered.body.contains("     2|two"));

        let ranged = read_file(
            &workdir,
            &json!({"path": "deep/nested/file.txt", "offset": 2, "limit": 1}),
        );
        assert!(ranged.body.contains("     2|two"));
        assert!(!ranged.body.contains("one"));

        let missing = read_file(&workdir, &json!({"path": "deep/nope.txt"}));
        assert!(missing.failed);
        std::fs::remove_dir_all(&workdir).ok();
    }

    #[test]
    fn tool_output_is_capped_with_both_ends_kept() {
        let long = "x".repeat(MAX_TOOL_OUTPUT * 2);
        let capped = cap_output(long);
        assert!(capped.len() < MAX_TOOL_OUTPUT + 100);
        assert!(capped.contains("[output truncated]"));
        assert!(cap_output("short".to_string()) == "short");
    }

    #[test]
    fn skill_bodies_are_inlined_and_long_ones_point_at_their_file() {
        let workdir = temp_dir("skills");
        let base = workdir.join(SESSION_SKILLS_DIR);
        std::fs::create_dir_all(base.join("orx-run")).unwrap();
        std::fs::write(base.join("orx-run/SKILL.md"), "short skill body\n").unwrap();
        std::fs::create_dir_all(base.join("huge")).unwrap();
        std::fs::write(base.join("huge/SKILL.md"), "y".repeat(SKILL_INLINE_CAP + 1)).unwrap();

        let skills = session_skills(&workdir);
        assert_eq!(skills.len(), 2);
        assert_eq!(skills[0].0, "huge");
        let prompt = system_prompt(&workdir, &workdir.join("playbook.md"), &skills, None, false);
        assert!(prompt.contains("short skill body"));
        assert!(prompt.contains("huge/SKILL.md"));
        assert!(!prompt.contains(&"y".repeat(SKILL_INLINE_CAP + 1)));
        std::fs::remove_dir_all(&workdir).ok();
    }

    #[test]
    fn the_prompt_carries_the_playbook_bootstrap_and_plan_note() {
        let workdir = temp_dir("prompt");
        let playbook = workdir.join("playbook.md");
        std::fs::write(&playbook, "PLAYBOOK FACTS\n").unwrap();
        let plain = system_prompt(&workdir, &playbook, &[], None, false);
        assert!(plain.contains("PLAYBOOK FACTS"));
        assert!(plain.contains(&workdir.display().to_string()));
        assert!(!plain.contains("PLAN MODE"));

        let plan = system_prompt(&workdir, &playbook, &[], Some("SIDE CHAT CONTEXT"), true);
        assert!(plan.contains("PLAN MODE"));
        assert!(plan.contains("SIDE CHAT CONTEXT"));

        // A missing playbook is not fatal — the turn still has its rules.
        let absent = system_prompt(&workdir, &workdir.join("nope.md"), &[], None, false);
        assert!(absent.contains("bash, read_file, write_file, edit_file"));
        std::fs::remove_dir_all(&workdir).ok();
    }

    /// The wire transcript keeps a tool part's input and output but no
    /// `tool_call_id`, so a previous turn's activity replays as a digest rather
    /// than a fake tool exchange.
    #[test]
    fn transcript_replay_folds_tool_activity_and_skips_empties() {
        let mut tool_part = WirePart::tool("p1", "bash", "completed", None);
        tool_part.state.as_mut().unwrap().output = Some("probe-ok".to_string());
        let row = stored(
            "assistant",
            vec![
                WirePart::text("p0", "I checked the tree."),
                tool_part,
                WirePart::reasoning("p2", "hidden"),
            ],
            None,
        );
        let message = replay_message(&row).unwrap();
        let content = message["content"].as_str().unwrap();
        assert!(content.contains("I checked the tree."));
        assert!(content.contains("<tool-activity name=\"bash\" status=\"completed\">"));
        assert!(content.contains("probe-ok"));
        assert!(!content.contains("hidden"));

        let empty = stored("assistant", vec![WirePart::text("p0", "   ")], None);
        assert!(replay_message(&empty).is_none());
        assert!(is_empty_row(&empty));
        assert!(!is_empty_row(&row));
    }

    #[test]
    fn attachment_media_types_are_recognized_by_extension() {
        assert_eq!(
            image_media_type(Path::new("att-1__shot.png")),
            Some("image/png")
        );
        assert_eq!(
            image_media_type(Path::new("shot.JPG".to_lowercase().as_str())),
            Some("image/jpeg")
        );
        assert_eq!(image_media_type(Path::new("paper.pdf")), None);
        assert_eq!(image_media_type(Path::new("noextension")), None);
    }

    /// The harness is what the rest of the process expects: a chat backend with
    /// the three permission modes, no steering, and the shared skills dir.
    #[test]
    fn the_harness_registers_as_a_chat_backend() {
        let harness = crate::local::harness::chat_harness(HARNESS_ID).expect("registered");
        assert!(harness.supports_chat());
        assert_eq!(harness.name(), "Alibaba Cloud");
        assert!(!harness.supports_steering());
        assert_eq!(harness.session_skills_dir(), Some(SESSION_SKILLS_DIR));
        assert_eq!(
            crate::local::harness::permission_mode_for(HARNESS_ID, "plan"),
            Some(PermissionMode::Plan)
        );
        assert_eq!(
            crate::local::harness::permission_mode_for(HARNESS_ID, "ask"),
            Some(PermissionMode::Ask)
        );
        assert_eq!(
            crate::local::harness::permission_mode_for(HARNESS_ID, "bypass"),
            None
        );
        let options = harness.options();
        assert_eq!(options.default_permission_mode, Some("auto"));
        assert_eq!(options.permission_modes.len(), 3);
        // No reasoning control: `reasoning_effort` is accepted by the plan but
        // measured to change nothing, so the composer must not offer it.
        assert!(options.reasoning_levels.is_empty());
        assert_eq!(
            crate::local::harness::registry()
                .iter()
                .filter(|entry| entry.id() == HARNESS_ID)
                .count(),
            1
        );
    }

    #[test]
    fn every_tool_schema_is_a_function_with_an_object_schema() {
        let schemas = tool_schemas();
        let list = schemas.as_array().unwrap();
        assert_eq!(list.len(), 4);
        for schema in list {
            assert_eq!(schema["type"], "function");
            assert_eq!(schema["function"]["parameters"]["type"], "object");
            assert!(!schema["function"]["name"].as_str().unwrap().is_empty());
            // Every name the executor matches on must be advertised.
            let name = schema["function"]["name"].as_str().unwrap();
            assert!(matches!(
                name,
                "bash" | "read_file" | "write_file" | "edit_file"
            ));
        }
    }

    /// Usage comes from the provider's own numbers, summed over the two sides it
    /// reports; a chunk with no usage leaves the session's figure untouched.
    #[test]
    fn usage_sums_the_providers_own_numbers() {
        let usage = json!({"prompt_tokens": 339, "completion_tokens": 42, "total_tokens": 381});
        let summed = ["prompt_tokens", "completion_tokens"]
            .iter()
            .filter_map(|key| usage.get(*key).and_then(Value::as_u64))
            .sum::<u64>();
        assert_eq!(summed, 381);
        let empty = json!({});
        assert_eq!(
            ["prompt_tokens", "completion_tokens"]
                .iter()
                .filter_map(|key| empty.get(*key).and_then(Value::as_u64))
                .sum::<u64>(),
            0
        );
    }
}
