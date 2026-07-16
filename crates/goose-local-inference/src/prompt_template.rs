use include_dir::{include_dir, Dir};
use minijinja::{Environment, Error as MiniJinjaError, Value as MJValue};
use serde::Serialize;

use crate::paths::Paths;

static CORE_PROMPTS_DIR: Dir = include_dir!("$CARGO_MANIFEST_DIR/src/prompts");

pub fn render_string<T: Serialize>(
    template_str: &str,
    context: &T,
) -> Result<String, MiniJinjaError> {
    let mut env = Environment::new();
    env.set_trim_blocks(true);
    env.set_lstrip_blocks(true);
    env.add_template("template", template_str)?;
    let tmpl = env.get_template("template")?;
    let ctx = MJValue::from_serialize(context);
    let rendered = tmpl.render(ctx)?;
    Ok(rendered.trim().to_string())
}

pub fn render_template<T: Serialize>(name: &str, context: &T) -> Result<String, MiniJinjaError> {
    let user_path = Paths::config_dir().join("prompts").join(name);
    let template_str = if user_path.exists() {
        std::fs::read_to_string(&user_path).map_err(|e| {
            MiniJinjaError::new(
                minijinja::ErrorKind::InvalidOperation,
                format!("Failed to read user template: {}", e),
            )
        })?
    } else {
        let file = CORE_PROMPTS_DIR.get_file(name).ok_or_else(|| {
            MiniJinjaError::new(
                minijinja::ErrorKind::TemplateNotFound,
                format!("Built-in template '{}' not found", name),
            )
        })?;
        String::from_utf8_lossy(file.contents()).to_string()
    };

    render_string(&template_str, context)
}

/// `{os, working_directory, shell}` context for a tiny-model system prompt template.
/// Shared by every backend's own tiny-model prompt (llamacpp, mlx, larql) — each
/// renders a different template file with this same context shape, so this is the
/// one piece worth factoring out rather than duplicating the OS/cwd/`$SHELL`
/// detection three times over.
pub fn tiny_model_context() -> serde_json::Value {
    let os = if cfg!(target_os = "macos") {
        "macos"
    } else if cfg!(target_os = "linux") {
        "linux"
    } else if cfg!(target_os = "windows") {
        "windows"
    } else {
        "unknown"
    };
    let working_directory = std::env::current_dir()
        .map(|p| p.display().to_string())
        .unwrap_or_else(|_| "unknown".to_string());
    let shell = std::env::var("SHELL").unwrap_or_else(|_| "/bin/sh".to_string());
    serde_json::json!({
        "os": os,
        "working_directory": working_directory,
        "shell": shell,
    })
}
