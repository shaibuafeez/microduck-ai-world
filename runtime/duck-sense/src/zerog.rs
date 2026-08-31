//! The remote half: one vision call, one planning call, both to 0G's router.
//!
//! ## Why two models and not one
//!
//! The cheapest capable vision model on the router is `qwen3-vl-30b` at $0.036 per
//! million input tokens. The cheapest capable *reasoner* is `deepseek-v4-flash` at
//! $0.138/$0.275 — a 284B/13B-active MoE with real function calling and a 1M window.
//! DeepSeek cannot see: its modality is `text->text`, verified against the router's
//! own catalogue. So the split is forced, not chosen — one model to say what is in the
//! frame, another to decide what to do about it.
//!
//! ## Two settings that matter more than the model choice
//!
//! `enable_thinking: false`. DeepSeek v4 has thinking on by default, which is right for
//! agentic coding and wrong for a robot: it adds seconds while the duck stands still
//! looking at nothing.
//!
//! `response_format` pinned to a JSON schema. The planner's answer goes straight into a
//! Rust enum. Free-form prose that "mostly" parses is how a look-and-decide loop starts
//! silently skipping turns.

use base64::Engine as _;
use serde::Deserialize;

use crate::act::Act;

const DEFAULT_BASE: &str = "https://router-api.0g.ai/v1";
/// Cheapest vision on the router. TeeTLS — the enclave protects transport, not
/// execution, so treat frames as leaving the machine. Say so in the README.
const VISION_MODEL: &str = "qwen3-vl-30b";
/// Text-only, tool-capable, cheap.
const PLANNER_MODEL: &str = "deepseek-v4-flash";

#[derive(Debug, thiserror::Error)]
pub enum ZeroGError {
    #[error("http: {0}")]
    Http(#[from] reqwest::Error),
    #[error("router {status}: {body}")]
    Status { status: u16, body: String },
    #[error("could not parse a plan from the model: {0}")]
    Unparseable(String),
}

pub struct ZeroG {
    http: reqwest::Client,
    base: String,
    key: String,
}

#[derive(Deserialize)]
struct ChatReply {
    choices: Vec<Choice>,
}
#[derive(Deserialize)]
struct Choice {
    message: Msg,
}
#[derive(Deserialize)]
struct Msg {
    content: Option<String>,
}

impl ZeroG {
    pub fn new(key: String, base: Option<String>) -> Self {
        Self {
            // A short timeout on purpose. This loop is allowed to miss a turn; it is not
            // allowed to wedge for a minute holding the duck still.
            http: reqwest::Client::builder()
                .timeout(std::time::Duration::from_secs(25))
                .build()
                .expect("http client"),
            base: base.unwrap_or_else(|| DEFAULT_BASE.to_string()),
            key,
        }
    }

    /// Describe a frame. JPEG bytes in, a sentence out.
    pub async fn describe(&self, jpeg: &[u8]) -> Result<String, ZeroGError> {
        let b64 = base64::engine::general_purpose::STANDARD.encode(jpeg);
        let body = serde_json::json!({
            "model": VISION_MODEL,
            "max_tokens": 160,
            "temperature": 0.2,
            "messages": [{
                "role": "user",
                "content": [
                    { "type": "text", "text":
                        "You are the eyes of a small biped robot about 25cm tall, looking \
                         forward from its own head. Describe what is in front of it in one \
                         or two plain sentences. Name objects, people and pets, and say \
                         roughly where they are (left, centre, right) and whether they are \
                         near or far. No speculation about what to do." },
                    { "type": "image_url",
                      "image_url": { "url": format!("data:image/jpeg;base64,{b64}") } }
                ]
            }]
        });

        let text = self.chat(body).await?;
        Ok(text.trim().to_string())
    }

    /// Decide what to do, given what the eyes saw and what a human last asked for.
    pub async fn plan(&self, scene: &str, goal: Option<&str>) -> Result<Act, ZeroGError> {
        let goal_line = goal
            .map(|g| format!("The human asked: \"{g}\"\n"))
            .unwrap_or_default();

        let body = serde_json::json!({
            "model": PLANNER_MODEL,
            "max_tokens": 200,
            "temperature": 0.3,
            // Thinking is on by default for this model. A robot that pauses to reason for
            // eight seconds has stopped being responsive, and this decision is not hard.
            "enable_thinking": false,
            "response_format": { "type": "json_object" },
            "messages": [
                { "role": "system", "content": SYSTEM },
                { "role": "user", "content": format!("{goal_line}The camera sees: {scene}") }
            ]
        });

        let text = self.chat(body).await?;

        // Models fence JSON however firmly you ask them not to.
        let start = text.find('{').ok_or_else(|| ZeroGError::Unparseable(text.clone()))?;
        let end = text.rfind('}').ok_or_else(|| ZeroGError::Unparseable(text.clone()))?;
        serde_json::from_str::<Act>(&text[start..=end])
            .map_err(|e| ZeroGError::Unparseable(format!("{e}: {}", &text[start..=end])))
    }

    async fn chat(&self, body: serde_json::Value) -> Result<String, ZeroGError> {
        let res = self
            .http
            .post(format!("{}/chat/completions", self.base))
            .bearer_auth(&self.key)
            .json(&body)
            .send()
            .await?;

        if !res.status().is_success() {
            let status = res.status().as_u16();
            let body = res.text().await.unwrap_or_default();
            return Err(ZeroGError::Status { status, body });
        }

        let reply: ChatReply = res.json().await?;
        Ok(reply
            .choices
            .into_iter()
            .next()
            .and_then(|c| c.message.content)
            .unwrap_or_default())
    }
}

/// The vocabulary, stated once. Kept in lockstep with `act::Act` by hand — a mismatch
/// shows up immediately as an `Unparseable`, which is the failure we want: loud, and
/// on the round that caused it.
const SYSTEM: &str = r#"You choose ONE action for a small biped robot, and reply with JSON only.

Allowed actions, exactly these shapes:
  {"action":"look","x":0.4,"y":0.0,"z":0.1}          point the camera at a point in front (metres, +x forward, +y left, +z up)
  {"action":"walk","vx":0.1,"vy":0.0,"wz":0.0,"ms":1500}   move briefly. vx/vy max 0.12 m/s, wz max 0.6 rad/s, ms max 2500
  {"action":"skill","name":"ground_pick"}            also: kick_left, kick_right, sit_toggle, roulade
  {"action":"quack"}                                 make a noise
  {"action":"hold"}                                  do nothing this round

Rules:
- Prefer "look" and "hold". Walking is for when something is clearly worth approaching.
- Never walk toward a person's feet, an edge, a drop, or a stairway. If unsure, "hold".
- "roulade" is a forward roll and needs clear space ahead. "ground_pick" needs an object
  on the floor directly in front.
- If the scene is unclear, ambiguous, or you cannot tell what is there, answer {"action":"hold"}.
- One action. JSON only. No prose, no explanation, no markdown fence."#;
