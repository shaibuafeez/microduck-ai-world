//! `duck-sense` — a look-and-decide loop that runs beside the robot, never inside it.
//!
//! ## What this is, and what it deliberately is not
//!
//! It is an ordinary `robotd` client, like `padd` and `btd`. It grabs a frame, asks a
//! remote model what is in it, asks a second model what to do about it, and emits the
//! result as the same intents a gamepad emits.
//!
//! It is **not** part of the control loop, and it must never become part of it. The
//! 50 Hz loop drives fifteen servos from ONNX policies on the board, with a 20 ms
//! budget per tick (`duck-control/src/io.rs`). A network round trip is hundreds of
//! milliseconds and occasionally infinite. Anything that puts a remote call between a
//! sensor reading and a servo command turns a robot into a robot-shaped Wi-Fi client.
//!
//! The split this crate assumes:
//!
//! | | where | budget |
//! |---|---|---|
//! | balance, gait, falls | on the board, ONNX | 20 ms |
//! | duck/pet detection | on the board, NPU | frame rate |
//! | "what am I looking at, and what should I do" | remote | seconds |
//!
//! ## Failure is stillness
//!
//! Every error path here ends in the duck not moving. Not stopping abruptly, not
//! retrying a stale plan — simply not acting this round. The router has no published
//! SLA and a shared, IP-keyed rate limit, so "the remote brain is unavailable" is an
//! ordinary Tuesday and has to be boring rather than dramatic.
//!
//! ## Privacy, stated plainly
//!
//! Frames leave the machine. The vision model used here is TeeTLS on 0G's router,
//! which means the enclave protects the connection and an upstream provider still runs
//! the model and sees the image. This is not on-device vision and must not be
//! described as private. `duck-detect` and `pet-detect` remain the local path.

mod act;
mod zerog;

use std::path::PathBuf;
use std::time::Duration;

use clap::Parser;
use tracing::{error, info, warn};

use act::{Act, Robot};
use zerog::ZeroG;

#[derive(Parser, Debug)]
#[command(name = "duck-sense", about = "Remote look-and-decide loop for Microduck")]
struct Args {
    /// robotd's socket.
    #[arg(long, default_value = "/run/robotd.sock")]
    socket: String,

    /// Where a fresh JPEG frame appears. `mediad` owns the camera; this reads what it
    /// leaves behind rather than opening the device itself, because two readers of one
    /// V4L2 node is a fight nobody wins.
    #[arg(long, default_value = "/run/mediad/frame.jpg")]
    frame: PathBuf,

    /// Seconds between decisions. Deliberately slow — this is deciding, not driving.
    #[arg(long, default_value_t = 6)]
    period: u64,

    /// A standing instruction, e.g. "find the red ball and kick it".
    #[arg(long)]
    goal: Option<String>,

    /// Describe and plan, but never touch the robot. The right way to meet this daemon.
    #[arg(long)]
    dry_run: bool,

    /// One decision, then exit.
    #[arg(long)]
    once: bool,
}

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    tracing_subscriber::fmt()
        .with_env_filter(
            tracing_subscriber::EnvFilter::try_from_default_env()
                .unwrap_or_else(|_| tracing_subscriber::EnvFilter::new("duck_sense=info")),
        )
        .init();

    let args = Args::parse();

    let key = std::env::var("OG_API_KEY")
        .or_else(|_| std::env::var("ZG_API_KEY"))
        .map_err(|_| "OG_API_KEY is not set — get one at https://pc.0g.ai")?;

    let brain = ZeroG::new(key, std::env::var("OG_BASE_URL").ok());
    let mut robot = Robot::new(&args.socket);

    if args.dry_run {
        info!("dry run — nothing will be sent to {}", args.socket);
    }
    info!(period = args.period, goal = ?args.goal, "sense loop up");

    let mut ticker = tokio::time::interval(Duration::from_secs(args.period));
    let mut shutdown = std::pin::pin!(tokio::signal::ctrl_c());

    loop {
        tokio::select! {
            _ = &mut shutdown => {
                info!("stopping; halting the robot");
                if !args.dry_run { robot.halt().await; }
                return Ok(());
            }
            _ = ticker.tick() => {}
        }

        // One round. Every failure inside logs and falls through to the next tick,
        // which is the same thing as choosing Hold.
        match round(&brain, &mut robot, &args).await {
            Ok(Some(a)) => info!(?a, "acted"),
            Ok(None) => {}
            Err(e) => warn!("round skipped: {e}"),
        }

        if args.once {
            return Ok(());
        }
    }
}

async fn round(
    brain: &ZeroG,
    robot: &mut Robot,
    args: &Args,
) -> Result<Option<Act>, Box<dyn std::error::Error>> {
    let jpeg = tokio::fs::read(&args.frame)
        .await
        .map_err(|e| format!("no frame at {}: {e}", args.frame.display()))?;

    let scene = brain.describe(&jpeg).await?;
    info!(%scene, "saw");

    let plan = brain.plan(&scene, args.goal.as_deref()).await?;
    let plan = plan.clamped();

    if matches!(plan, Act::Hold) {
        info!("holding");
        return Ok(None);
    }
    if args.dry_run {
        info!(?plan, "would act (dry run)");
        return Ok(None);
    }

    if let Err(e) = robot.apply(plan).await {
        // A refusal from robotd is information, not a fault — `robot.do` refuses while
        // another scripted move holds the robot, which is the system working.
        error!("robotd: {e}");
        return Ok(None);
    }
    Ok(Some(plan))
}
