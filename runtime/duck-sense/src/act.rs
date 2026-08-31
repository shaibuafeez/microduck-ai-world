//! The action vocabulary, and the only way this daemon can move the robot.
//!
//! ## Why a closed enum rather than "let the model call the API"
//!
//! The obvious design is to hand the planner the JSON-RPC surface and let it emit
//! whatever it likes. That is wrong here for a reason specific to this robot: the
//! surface includes `robot.enable`, `robot.init`, `robot.relax` and `robot.shutdown`,
//! and three of those are decisions about whether fifteen servos hold a 800 g biped
//! upright. A model that hallucinates `robot.relax` drops the duck on its face.
//!
//! So the vocabulary here is a closed set that deliberately *excludes* power and
//! enablement. Those stay with the human and with `padd`. What is left — look, walk a
//! bounded distance, run one of the five scripted skills, quack — cannot put the robot
//! in a state a person did not already authorise by turning it on.
//!
//! ## Why bounded motion rather than a velocity the planner chooses
//!
//! `robot.move` is a *continuous* intent: a notification, last-writer-wins, and
//! expiring. `robotd` guards it with a deadman on twist age, because a stale velocity
//! walks the robot into a wall (`robotd/src/intents.rs`, `twist_age`). A planner
//! answering every few seconds is exactly the client that deadman exists to defend
//! against — so we never hold a twist open. A `Walk` is re-notified for its own
//! bounded duration and then explicitly zeroed, and the deadman remains the backstop
//! rather than the mechanism.

use std::time::Duration;

use serde::{Deserialize, Serialize};
use tokio::io::{AsyncBufReadExt, AsyncWriteExt, BufReader};
use tokio::net::UnixStream;

/// Hard ceilings. The planner picks within these; it cannot widen them.
pub const MAX_WALK: Duration = Duration::from_millis(2500);
/// Metres per second. Well under what the walking policy can do — this loop is for
/// deciding, not for driving.
pub const MAX_SPEED: f64 = 0.12;
/// Radians per second.
pub const MAX_YAW: f64 = 0.6;
/// Twist re-notify period. 20 Hz: inside the deadman, cheap, and matches `padd`.
const TWIST_TICK: Duration = Duration::from_millis(50);

/// Everything the planner is allowed to ask for.
///
/// Serialised as a tagged union so it round-trips through the model's tool call and
/// back into Rust with no free-form strings reaching the socket.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
#[serde(tag = "action", rename_all = "snake_case")]
pub enum Act {
    /// Point the camera at a trunk-frame point. Discrete, answered.
    Look { x: f64, y: f64, z: f64 },
    /// Walk for a bounded time, then stop. Forward is +x.
    Walk { vx: f64, vy: f64, wz: f64, ms: u64 },
    /// One of the five scripted moves.
    Skill { name: SkillName },
    /// Make a noise.
    Quack,
    /// Do nothing this round. An explicit choice, not an absence — see `main.rs`.
    Hold,
}

#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum SkillName {
    GroundPick,
    KickLeft,
    KickRight,
    SitToggle,
    Roulade,
}

impl SkillName {
    /// The wire name `robot.do` expects.
    fn wire(self) -> &'static str {
        match self {
            Self::GroundPick => "groundPick",
            Self::KickLeft => "kickLeft",
            Self::KickRight => "kickRight",
            Self::SitToggle => "sitToggle",
            Self::Roulade => "roulade",
        }
    }
}

impl Act {
    /// Clamp into the ceilings above. Called before anything reaches the socket, so a
    /// planner that asks for 3 m/s gets 0.12 rather than a refusal — a refusal would
    /// just make the loop skip a beat, and a slow duck is a better failure than a still one.
    pub fn clamped(self) -> Self {
        match self {
            Self::Walk { vx, vy, wz, ms } => Self::Walk {
                vx: vx.clamp(-MAX_SPEED, MAX_SPEED),
                vy: vy.clamp(-MAX_SPEED, MAX_SPEED),
                wz: wz.clamp(-MAX_YAW, MAX_YAW),
                ms: ms.min(MAX_WALK.as_millis() as u64),
            },
            other => other,
        }
    }
}

#[derive(Debug, thiserror::Error)]
pub enum ActError {
    #[error("robotd socket: {0}")]
    Io(#[from] std::io::Error),
    #[error("robotd refused: {0}")]
    Refused(String),
    #[error("bad reply from robotd: {0}")]
    Protocol(String),
}

/// A client of `robotd`, exactly like `padd` and `btd` are.
///
/// One connection per act rather than a held one: this loop speaks every few seconds,
/// and a socket held open across a `robotd` restart is a socket that silently stops
/// working. Reconnecting costs nothing at this rate.
pub struct Robot {
    path: String,
    next_id: u64,
}

impl Robot {
    pub fn new(path: impl Into<String>) -> Self {
        Self { path: path.into(), next_id: 1 }
    }

    pub async fn apply(&mut self, act: Act) -> Result<(), ActError> {
        match act.clamped() {
            Act::Hold => Ok(()),

            Act::Look { x, y, z } => {
                self.request("robot.look", serde_json::json!({ "x": x, "y": y, "z": z })).await?;
                Ok(())
            }

            Act::Skill { name } => {
                self.request("robot.do", serde_json::json!({ "skill": name.wire() })).await?;
                Ok(())
            }

            Act::Quack => {
                self.request("robot.sound", serde_json::json!({ "tag": "quack" })).await?;
                Ok(())
            }

            Act::Walk { vx, vy, wz, ms } => {
                // Re-notify for the duration, then zero. We never leave a twist standing:
                // the deadman would catch it, but relying on a timeout for something we
                // can state explicitly is how a robot ends up walking after its client died.
                let until = tokio::time::Instant::now() + Duration::from_millis(ms);
                while tokio::time::Instant::now() < until {
                    self.notify("robot.move", serde_json::json!({ "vx": vx, "vy": vy, "wz": wz }))
                        .await?;
                    tokio::time::sleep(TWIST_TICK).await;
                }
                self.request("robot.stop", serde_json::json!({})).await?;
                Ok(())
            }
        }
    }

    /// Best-effort stillness. Used on shutdown and whenever the planner errors — the
    /// duck should never keep moving because something upstream stopped answering.
    pub async fn halt(&mut self) {
        let _ = self.request("robot.stop", serde_json::json!({})).await;
    }

    async fn notify(&mut self, method: &str, params: serde_json::Value) -> Result<(), ActError> {
        let msg = serde_json::json!({ "jsonrpc": "2.0", "method": method, "params": params });
        let mut s = UnixStream::connect(&self.path).await?;
        s.write_all(format!("{msg}\n").as_bytes()).await?;
        s.flush().await?;
        Ok(())
    }

    async fn request(
        &mut self,
        method: &str,
        params: serde_json::Value,
    ) -> Result<serde_json::Value, ActError> {
        let id = self.next_id;
        self.next_id += 1;

        let msg = serde_json::json!({ "jsonrpc": "2.0", "id": id, "method": method, "params": params });
        let s = UnixStream::connect(&self.path).await?;
        let (rd, mut wr) = s.into_split();
        wr.write_all(format!("{msg}\n").as_bytes()).await?;
        wr.flush().await?;

        let mut line = String::new();
        BufReader::new(rd).read_line(&mut line).await?;
        let v: serde_json::Value =
            serde_json::from_str(&line).map_err(|e| ActError::Protocol(e.to_string()))?;

        if let Some(err) = v.get("error") {
            // `robot.do` refuses with a reason naming the scripted move already holding
            // the robot. Surfacing it is the whole point — a swallowed refusal looks
            // like a planner that decided to do nothing.
            return Err(ActError::Refused(err.to_string()));
        }
        Ok(v.get("result").cloned().unwrap_or(serde_json::Value::Null))
    }
}
