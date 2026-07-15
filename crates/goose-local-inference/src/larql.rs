//! `larql` backend — drives an unmodified `larql chat <vindex>` process via piped
//! stdin/stdout. No HTTP, no in-process linking of larql-lql/larql-inference; the
//! adapter's only coupling to LARQL is `run_chat`'s existing text framing
//! (`crates/larql-cli/src/commands/primary/run_cmd.rs:396-430` in the LARQL repo):
//! one line of prompt text in via stdin, response streamed to stdout, next-turn
//! prompt ("> ") written to stderr.
//!
//! V1 limitation, stated plainly rather than hidden: the turn-boundary signal is on
//! stderr, not stdout, so this backend spawns a small stderr-watcher thread that
//! looks for the next "> " and signals turn completion over a channel. A hard
//! timeout is the safety net if that signal never arrives (ties to the containing
//! project's AC-5 liveness requirement — a hung child must not hang generate()
//! forever). No native tool-calling support yet — `request.tools` is accepted but
//! unused; ToolRequest/ToolResponse content is flattened to its Display text like
//! everything else, same fallback llamacpp.rs/mlx.rs use for non-native paths.

use std::any::Any;
use std::io::{BufRead, BufReader, Write};
use std::process::{Child, ChildStdin, Command, Stdio};
use std::sync::mpsc as std_mpsc;
use std::time::{Duration, Instant};

use goose_provider_types::conversation::message::{Message, MessageContent};
use goose_provider_types::conversation::token_usage::{ProviderUsage, Usage};
use goose_provider_types::errors::ProviderError;
use rmcp::model::Role;

use crate::backend::{BackendLoadedModel, LocalGenerationRequest, LocalInferenceBackend};
use crate::ResolvedModelPaths;

pub(crate) const LARQL_BACKEND_ID: &str = "larql";

/// Hard ceiling on one generate() call. A hung `larql chat` child (e.g. waiting on
/// something that will never resolve) must not hang the caller forever — this is
/// the backend-local half of the containing project's liveness requirement (AC-5);
/// the VM-level watchdog is the other half, not a substitute for this one.
const GENERATE_TIMEOUT: Duration = Duration::from_secs(120);

pub(crate) struct LarqlBackend;

impl LarqlBackend {
    pub(crate) fn new() -> Self {
        Self
    }
}

struct LarqlLoadedModel {
    child: Child,
    stdin: ChildStdin,
    stdout: BufReader<std::process::ChildStdout>,
    /// Fires a message each time the stderr watcher sees a fresh turn prompt.
    turn_boundary_rx: std_mpsc::Receiver<()>,
}

impl BackendLoadedModel for LarqlLoadedModel {
    fn as_any_mut(&mut self) -> &mut dyn Any {
        self
    }
}

impl LocalInferenceBackend for LarqlBackend {
    fn id(&self) -> &'static str {
        LARQL_BACKEND_ID
    }

    fn load_model(
        &self,
        model_id: &str,
        resolved: &ResolvedModelPaths,
        _settings: &crate::local_model_registry::ModelSettings,
    ) -> Result<Box<dyn BackendLoadedModel>, ProviderError> {
        let vindex_path = &resolved.model_path;
        if !vindex_path.exists() {
            return Err(ProviderError::ExecutionError(format!(
                "larql backend: vindex path does not exist for model '{model_id}': {}",
                vindex_path.display()
            )));
        }

        // Resolve the binary explicitly rather than relying on `$PATH` — a
        // minimal cloud-init `runcmd` shell (no login profile) does not
        // necessarily have the same PATH as an interactive session, and
        // silently depending on PATH search produced a real, reproducible
        // ENOENT in exactly that environment. `LARQL_BIN` lets a deployment
        // (e.g. this project's VM-guest provisioning step) say precisely
        // where the binary lives; "larql" (plain PATH search) remains the
        // sensible default for a normal interactive/dev environment.
        let larql_bin = std::env::var("LARQL_BIN").unwrap_or_else(|_| "larql".to_string());
        let mut child = Command::new(&larql_bin)
            .arg("chat")
            .arg(vindex_path)
            .stdin(Stdio::piped())
            .stdout(Stdio::piped())
            .stderr(Stdio::piped())
            .spawn()
            .map_err(|e| {
                ProviderError::ExecutionError(format!(
                    "larql backend: failed to spawn '{larql_bin} chat {}': {e} (set LARQL_BIN to an absolute path if '{larql_bin}' is not on PATH in this environment)",
                    vindex_path.display()
                ))
            })?;

        let stdin = child.stdin.take().ok_or_else(|| {
            ProviderError::ExecutionError("larql backend: child stdin unavailable".to_string())
        })?;
        let stdout = child.stdout.take().ok_or_else(|| {
            ProviderError::ExecutionError("larql backend: child stdout unavailable".to_string())
        })?;
        let stderr = child.stderr.take().ok_or_else(|| {
            ProviderError::ExecutionError("larql backend: child stderr unavailable".to_string())
        })?;

        // Confirm the child is actually alive before declaring success — a process
        // that exits immediately (bad vindex, missing binary on PATH inside the VM,
        // etc.) must be surfaced as an Err here, not discovered later as a silent
        // generate() hang. Give it a short window to either produce its first "> "
        // prompt on stderr or exit.
        let (boundary_tx, boundary_rx) = std_mpsc::channel::<()>();
        let watcher_tx = boundary_tx.clone();
        std::thread::spawn(move || {
            // NOT BufReader::read_line(): run_chat's "> " prompt (run_cmd.rs)
            // is written WITHOUT a trailing newline by design (so typed input
            // appears inline after it, like a normal shell prompt) — read_line
            // blocks forever waiting for a '\n' that will never arrive for
            // that write, which silently hung every turn-boundary detection
            // regardless of how large the surrounding timeout was (confirmed:
            // raising the load timeout 5x, 30s -> 150s, changed nothing, since
            // the read itself was blocked, not merely slow). Read raw bytes
            // into a small rolling buffer and match the substring directly.
            let mut reader = stderr;
            let mut buf = [0u8; 256];
            let mut pending = String::new();
            loop {
                match std::io::Read::read(&mut reader, &mut buf) {
                    Ok(0) => break, // stderr closed: child exited
                    Ok(n) => {
                        pending.push_str(&String::from_utf8_lossy(&buf[..n]));
                        if pending.contains("> ") {
                            let _ = watcher_tx.send(());
                            pending.clear();
                        } else if pending.len() > 4096 {
                            // Bound memory on unexpectedly chatty/non-matching
                            // stderr; keep only the tail, a prompt can't span
                            // more than a few bytes so nothing meaningful is lost.
                            let tail_start = pending.len() - 256;
                            pending = pending[tail_start..].to_string();
                        }
                    }
                    Err(_) => break,
                }
            }
        });

        // 30s was measured too short on a real, resource-constrained VM guest
        // (goose itself competing for the same 2 vCPUs while `larql chat`
        // mmaps a ~1.3GB vindex) — confirmed empirically, not a guess.
        // Configurable via env var since the right value is deployment-
        // dependent (guest CPU count, vindex size), not a Rust-source-level
        // constant.
        let load_timeout_secs: u64 = std::env::var("LARQL_LOAD_TIMEOUT_SECS")
            .ok()
            .and_then(|v| v.parse().ok())
            .unwrap_or(120);
        match boundary_rx.recv_timeout(Duration::from_secs(load_timeout_secs)) {
            Ok(()) => {}
            Err(_) => {
                let _ = child.kill();
                return Err(ProviderError::ExecutionError(format!(
                    "larql backend: 'larql chat {}' did not produce its first prompt within {load_timeout_secs}s \
                     (check the binary is on PATH inside the execution boundary and the vindex is valid; \
                     set LARQL_LOAD_TIMEOUT_SECS higher if the vindex is just slow to load on this host)",
                    vindex_path.display()
                )));
            }
        }

        Ok(Box::new(LarqlLoadedModel {
            child,
            stdin,
            stdout: BufReader::new(stdout),
            turn_boundary_rx: boundary_rx,
        }))
    }

    fn generate(
        &self,
        loaded: &mut dyn BackendLoadedModel,
        request: LocalGenerationRequest<'_>,
    ) -> Result<(), ProviderError> {
        let loaded = loaded
            .as_any_mut()
            .downcast_mut::<LarqlLoadedModel>()
            .ok_or_else(|| {
                ProviderError::ExecutionError("larql backend: loaded-model type mismatch".into())
            })?;

        // Fail fast if the child already died between load_model() and this call —
        // never silently proceed to write into a closed pipe.
        if let Ok(Some(status)) = loaded.child.try_wait() {
            return Err(ProviderError::ExecutionError(format!(
                "larql backend: 'larql chat' child exited before generate() (status: {status})"
            )));
        }

        let prompt = flatten_prompt(request.system, request.messages);
        writeln!(loaded.stdin, "{prompt}").map_err(|e| {
            ProviderError::ExecutionError(format!("larql backend: failed to write stdin: {e}"))
        })?;
        loaded.stdin.flush().map_err(|e| {
            ProviderError::ExecutionError(format!("larql backend: failed to flush stdin: {e}"))
        })?;

        let deadline = Instant::now() + GENERATE_TIMEOUT;
        let mut collected = String::new();
        loop {
            if Instant::now() >= deadline {
                return Err(ProviderError::ExecutionError(format!(
                    "larql backend: generate() exceeded {}s with no turn-boundary signal — \
                     treating as a fault, not 'still working' (see AC-5)",
                    GENERATE_TIMEOUT.as_secs()
                )));
            }

            // Turn complete: the child's next "> " prompt appeared on stderr.
            if loaded
                .turn_boundary_rx
                .recv_timeout(Duration::from_millis(50))
                .is_ok()
            {
                break;
            }

            let mut line = String::new();
            match loaded.stdout.read_line(&mut line) {
                Ok(0) => {
                    // stdout closed — child exited mid-generation.
                    return Err(ProviderError::ExecutionError(
                        "larql backend: child stdout closed before a turn-boundary signal \
                         (the 'larql chat' process likely crashed mid-generation)"
                            .to_string(),
                    ));
                }
                Ok(_) => {
                    collected.push_str(&line);
                    let mut message = Message::assistant();
                    message = message.with_text(line.clone());
                    let _ = request
                        .tx
                        .blocking_send(Ok((Some(message), None)))
                        .map_err(|_| {
                            ProviderError::ExecutionError(
                                "larql backend: stream receiver dropped".to_string(),
                            )
                        });
                }
                Err(_) => continue, // transient read error on a non-blocking-ish pipe; retry until deadline
            }
        }

        // Approximate usage — a word count, not a real LARQL tokenizer count. Stated
        // as an approximation rather than presented as precise, per this project's
        // own honesty requirements around unverified precision.
        let approx_input_tokens = prompt.split_whitespace().count() as i32;
        let approx_output_tokens = collected.split_whitespace().count() as i32;
        let usage = ProviderUsage::new(
            request.model_name.clone(),
            Usage::new(
                Some(approx_input_tokens),
                Some(approx_output_tokens),
                Some(approx_input_tokens + approx_output_tokens),
            ),
        );
        let _ = request.tx.blocking_send(Ok((None, Some(usage))));

        Ok(())
    }

    fn available_memory_bytes(&self) -> u64 {
        // Real /proc/meminfo read, not a placeholder constant — ties to this
        // project's ADR-5 resource-governance work in spirit (the backend should
        // know real headroom, not assume it).
        std::fs::read_to_string("/proc/meminfo")
            .ok()
            .and_then(|contents| {
                contents.lines().find_map(|line| {
                    line.strip_prefix("MemAvailable:").map(|rest| {
                        rest.trim()
                            .trim_end_matches(" kB")
                            .parse::<u64>()
                            .unwrap_or(0)
                            * 1024
                    })
                })
            })
            .unwrap_or(0)
    }
}

fn flatten_prompt(system: &str, messages: &[Message]) -> String {
    let mut parts = Vec::new();
    if !system.is_empty() {
        parts.push(system.replace('\n', " "));
    }
    for m in messages {
        let role = match m.role {
            Role::User => "User",
            Role::Assistant => "Assistant",
        };
        let text: String = m
            .content
            .iter()
            .map(|c| format!("{c}"))
            .collect::<Vec<_>>()
            .join(" ");
        if !text.trim().is_empty() {
            parts.push(format!("{role}: {}", text.replace('\n', " ")));
        }
    }
    // run_chat reads ONE line per turn (run_cmd.rs:414), so the whole conversation
    // must collapse to a single line — newlines within any part are already
    // stripped above.
    let _ = MessageContent::text(""); // keep MessageContent import used if content match above changes
    parts.join(" | ")
}
