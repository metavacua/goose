//! Tool-call emulation for the larql backend — text-based, matching the same textual
//! conventions `llamacpp/inference_emulated_tools.rs` uses (`$ command` lines,
//! ```` ```execute_typescript ```` fences) plus a fenced-JSON convention, adapted for
//! `larql chat`'s whole-response-per-turn framing (one full response printed per turn,
//! not a per-token stream — see `larql.rs`'s own module doc) rather than a token
//! callback. Ungated (no `mlx` feature dependency) and crate-visible, unlike
//! `tool_emulation.rs`/`native_tool_parsing.rs` (both `#[cfg(feature = "mlx")]`) or
//! `llamacpp/inference_emulated_tools.rs` (`pub(super)`-scoped to `llamacpp::`) — see
//! `docs/specs/2026-07-16-larql-goose-toolcalling-design.md` ADR-2 in `larql-to-sparql`
//! for why neither existing implementation is directly reusable here.
//!
//! Two conventions are supported, selected per call (matrix legs 1-2 of that design):
//! - [`EmulatorConvention::ShellCommand`]: a line starting with `$` is a shell command.
//! - [`EmulatorConvention::FencedJson`]: a ` ```tool_call ` ... ` ``` ` fenced block
//!   contains `{"name": "...", "arguments": {...}}`.

use std::borrow::Cow;

use goose_provider_types::conversation::message::{Message, MessageContent};
use rmcp::model::{CallToolRequestParams, Tool};
use serde_json::json;
use uuid::Uuid;

pub(crate) const SHELL_TOOL: &str = "developer__shell";

/// Which textual convention the model was prompted to use for tool calls.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub(crate) enum EmulatorConvention {
    /// `$ command` on its own line (matrix leg `smol135.emulate-stream.shell-conv`).
    ShellCommand,
    /// ` ```tool_call\n{"name": "...", "arguments": {...}}\n``` ` fenced JSON
    /// (matrix leg `smol135.emulate-stream.fenced-tool`).
    FencedJson,
}

#[derive(Debug, PartialEq)]
pub(crate) enum EmulatedAction {
    Text(String),
    ShellCommand(String),
    ToolCallJson { name: String, arguments: serde_json::Value },
}

#[derive(Clone, Copy, PartialEq, Eq)]
enum ParserState {
    Normal,
    InFence,
}

/// Buffered, whole-line-at-a-time parser — `push_line` is called once per line of
/// `larql chat`'s response (never a partial line, since `run_chat` prints and flushes
/// the whole response before the next turn prompt appears — residual K39), so there is
/// no cross-chunk pattern-splitting concern the streaming original has to guard
/// against with a hold-back buffer.
pub(crate) struct LarqlEmulatorParser {
    convention: EmulatorConvention,
    state: ParserState,
    fence_buffer: String,
}

impl LarqlEmulatorParser {
    pub(crate) fn new(convention: EmulatorConvention) -> Self {
        Self {
            convention,
            state: ParserState::Normal,
            fence_buffer: String::new(),
        }
    }

    /// Feed one line (no trailing `\n`) of the model's response. Returns `None` for
    /// ordinary text (the caller should still surface it as plain assistant text) or
    /// `Some(action)` when a tool-call pattern completes on this line.
    pub(crate) fn push_line(&mut self, line: &str) -> Option<EmulatedAction> {
        match self.convention {
            EmulatorConvention::ShellCommand => self.push_line_shell(line),
            EmulatorConvention::FencedJson => self.push_line_fenced(line),
        }
    }

    /// Stateless: a whole-line-at-a-time contract (residual K39) means a `$` command
    /// is fully decidable from the current line alone, unlike the streaming original
    /// this is adapted from, which needs `InCommand` to accumulate a still-arriving
    /// partial line.
    fn push_line_shell(&self, line: &str) -> Option<EmulatedAction> {
        if let Some(command) = line.strip_prefix('$') {
            let command = command.trim();
            if command.is_empty() {
                None
            } else {
                Some(EmulatedAction::ShellCommand(command.to_string()))
            }
        } else {
            Some(EmulatedAction::Text(line.to_string()))
        }
    }

    fn push_line_fenced(&mut self, line: &str) -> Option<EmulatedAction> {
        match self.state {
            ParserState::Normal => {
                if line.trim() == "```tool_call" {
                    self.state = ParserState::InFence;
                    self.fence_buffer.clear();
                    None
                } else {
                    Some(EmulatedAction::Text(line.to_string()))
                }
            }
            ParserState::InFence => {
                if line.trim() == "```" {
                    self.state = ParserState::Normal;
                    let parsed: Option<EmulatedAction> =
                        serde_json::from_str::<serde_json::Value>(&self.fence_buffer)
                            .ok()
                            .and_then(|v| {
                                let name = v.get("name")?.as_str()?.to_string();
                                let arguments =
                                    v.get("arguments").cloned().unwrap_or_else(|| json!({}));
                                Some(EmulatedAction::ToolCallJson { name, arguments })
                            });
                    self.fence_buffer.clear();
                    parsed
                } else {
                    self.fence_buffer.push_str(line);
                    self.fence_buffer.push('\n');
                    None
                }
            }
        }
    }
}

/// Tool-list prompt text, reusing the already-ungated `compact_tools_json` (K41) rather
/// than duplicating a serialization format.
pub(crate) fn build_larql_emulator_tool_description(tools: &[Tool]) -> String {
    let mut desc = String::from("\n\n# Tools\n\nYou have access to the following tools:\n\n");
    if let Some(json) = crate::tool_parsing::compact_tools_json(tools) {
        desc.push_str(&json);
        desc.push('\n');
    }
    desc
}

/// Builds the `Message` Goose's existing `reply_parts::categorize_tool_requests` (K40)
/// already knows how to dispatch, for either emulated-action kind.
pub(crate) fn message_for_action(action: &EmulatedAction) -> Message {
    match action {
        EmulatedAction::Text(text) => Message::assistant().with_text(text.clone()),
        EmulatedAction::ShellCommand(command) => {
            let mut args = serde_json::Map::new();
            args.insert("command".to_string(), json!(command));
            let tool_call =
                CallToolRequestParams::new(Cow::Borrowed(SHELL_TOOL)).with_arguments(args);
            let mut message = Message::assistant();
            message
                .content
                .push(MessageContent::tool_request(Uuid::new_v4().to_string(), Ok(tool_call)));
            message
        }
        EmulatedAction::ToolCallJson { name, arguments } => {
            let args = arguments
                .as_object()
                .cloned()
                .unwrap_or_default();
            let tool_call =
                CallToolRequestParams::new(Cow::Owned(name.clone())).with_arguments(args);
            let mut message = Message::assistant();
            message
                .content
                .push(MessageContent::tool_request(Uuid::new_v4().to_string(), Ok(tool_call)));
            message
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn shell_command_line_is_detected() {
        let mut p = LarqlEmulatorParser::new(EmulatorConvention::ShellCommand);
        let action = p.push_line("$ ls -1 /tmp | wc -l");
        assert_eq!(
            action,
            Some(EmulatedAction::ShellCommand("ls -1 /tmp | wc -l".to_string()))
        );
    }

    #[test]
    fn ordinary_text_line_is_not_a_command() {
        let mut p = LarqlEmulatorParser::new(EmulatorConvention::ShellCommand);
        let action = p.push_line("Let me check that for you.");
        assert_eq!(
            action,
            Some(EmulatedAction::Text("Let me check that for you.".to_string()))
        );
    }

    #[test]
    fn empty_command_is_ignored() {
        let mut p = LarqlEmulatorParser::new(EmulatorConvention::ShellCommand);
        assert_eq!(p.push_line("$"), None);
        assert_eq!(p.push_line("$  "), None);
    }

    #[test]
    fn mid_sentence_dollar_is_not_a_command() {
        let mut p = LarqlEmulatorParser::new(EmulatorConvention::ShellCommand);
        let action = p.push_line("It costs $50 per month");
        assert_eq!(
            action,
            Some(EmulatedAction::Text("It costs $50 per month".to_string()))
        );
    }

    #[test]
    fn fenced_json_tool_call_completes_on_closing_fence() {
        let mut p = LarqlEmulatorParser::new(EmulatorConvention::FencedJson);
        assert_eq!(p.push_line("```tool_call"), None);
        assert_eq!(
            p.push_line(r#"{"name": "developer__shell", "arguments": {"command": "ls"}}"#),
            None
        );
        let action = p.push_line("```");
        assert_eq!(
            action,
            Some(EmulatedAction::ToolCallJson {
                name: "developer__shell".to_string(),
                arguments: json!({"command": "ls"}),
            })
        );
    }

    #[test]
    fn fenced_json_lines_before_open_fence_are_plain_text() {
        let mut p = LarqlEmulatorParser::new(EmulatorConvention::FencedJson);
        let action = p.push_line("Here's the tool call:");
        assert_eq!(
            action,
            Some(EmulatedAction::Text("Here's the tool call:".to_string()))
        );
    }

    #[test]
    fn fenced_json_malformed_body_yields_no_action() {
        let mut p = LarqlEmulatorParser::new(EmulatorConvention::FencedJson);
        assert_eq!(p.push_line("```tool_call"), None);
        assert_eq!(p.push_line("not json"), None);
        assert_eq!(p.push_line("```"), None);
    }

    #[test]
    fn message_for_shell_command_produces_tool_request() {
        let action = EmulatedAction::ShellCommand("whoami".to_string());
        let message = message_for_action(&action);
        assert!(matches!(
            message.content.first(),
            Some(MessageContent::ToolRequest(_))
        ));
    }
}
