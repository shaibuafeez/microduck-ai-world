#!/usr/bin/env python3
"""A slow 0G brain that never blocks the 50 Hz loop.

The control loop cannot wait on a network call — 20 ms budget against a router
with no SLA. So the brain runs in its own thread and always holds a *current
intent*; the loop reads whatever is there this tick and moves on. Latency turns
into staleness rather than lag, and a two-second-old intent ("wander left, you
are curious") is invisible in behaviour.

Two cadences, for the same reason a robot has reflexes and thoughts:
  - fast  (~2 s): short prompt, recent events only -> a velocity intent
  - deep (~60 s): the whole event history in one prompt -> a remembered note.
                  At this price re-reading everything is cheaper than building
                  retrieval, and a pattern across distant events is exactly what
                  retrieval over recent ticks cannot see.

With no OG_API_KEY the whole thing degrades to `_local_intent`, which is also
what runs whenever the network is down. The duck must never depend on this.
"""
from __future__ import annotations

import json
import math
import os
import threading
import time
import urllib.error
import urllib.request
from collections import deque
from dataclasses import dataclass, field

BASE_URL = os.environ.get("OG_BASE_URL", "https://router-api.0g.ai/v1")
API_KEY = os.environ.get("OG_API_KEY", "")
# Default is the measured-fastest model that returns usable JSON and takes
# images: qwen3-vl-30b ~1.7s vs deepseek-v4-flash 6.5-9.5s (2026-08-29).
MODEL = os.environ.get("OG_BRAIN_MODEL", "qwen3-vl-30b")

# Cadence. The first run burned a balance in ~45 minutes: 1368 calls, each carrying a
# fresh JPEG. An image is worth roughly a thousand text tokens, so vision was almost
# all of the cost. Two levers, both nearly free behaviourally:
#   - roam slowly (the intent is advisory; nobody can tell a 5 s intent from a 2 s one)
#   - do not re-send a frame every call when nothing has changed
# Searching is different: it is closed-loop on what the camera sees, so it keeps the
# fast cadence and a frame every time.
FAST_PERIOD = 5.0          # roaming
SEEK_PERIOD = 1.8          # actively looking for something
FRAME_EVERY = 3            # roaming: send a frame every Nth call
DEEP_PERIOD = 180.0
HTTP_TIMEOUT = 12.0
# How long a spoken order stays in force. Long enough to carry out, short
# enough that the duck stops obeying a command from five minutes ago.
ORDER_TTL = 45.0

SYSTEM = (
    "You are the mood and intent of a small bipedal robot duck exploring a yard "
    "with houses, trees, steps, ramps, scattered blocks, an ORANGE BALL and a "
    "WHITE GOAL. The image is YOUR OWN "
    "point of view. You do NOT control balance or footsteps — only where you want "
    "to go, what trick to do, and what you say.\n"
    "Reply with ONE json object, no prose, no markdown fence:\n"
    '{"vx":<-0.3..0.4>,"wz":<-1.2..1.2>,"mood":"<one word>",'
    '"action":"none|roulade|ground_pick|sit|kick_left|kick_right",'
    '"say":"<max 9 words, first person, spoken aloud>","why":"<max 8 words>"}\n'
    "vx is forward m/s, wz is turn rad/s (positive = left).\n"
    "React to what you SEE: head toward something interesting, turn away from what "
    "blocks you. Use action rarely — only when the moment earns it (roulade when "
    "excited or blocked, ground_pick at something on the floor, sit when content, "
    "kick_left/kick_right ONLY when the orange ball is close and just ahead of you). "
    "If asked to score, walk up to the ball with the goal beyond it, then kick. "
    "Most turns are action:none. Prefer curiosity over stillness."
)


SEARCH_SYSTEM = (
    "You are the eyes of a small robot duck. You are LOOKING FOR ONE SPECIFIC THING. "
    "The image is your own point of view, a 78-degree field of view, camera 12cm off "
    "the ground.\n"
    "Reply with ONE json object, no prose, no markdown fence:\n"
    '{"sighted":<true|false>,"bearing":<-40..40>,"arrived":<true|false>,'
    '"vx":<0.0..0.4>,"mood":"<one word>","say":"<max 9 words>","why":"<max 8 words>"}\n'
    "sighted: is the target VISIBLE in this frame right now.\n"
    "bearing: where it is, in degrees from straight ahead. NEGATIVE = to your right, "
    "POSITIVE = to your left, 0 = dead ahead. If not sighted, give the bearing you want "
    "to TURN to search — sweep, do not spin.\n"
    "arrived: true only when the target fills a good part of the lower frame, i.e. you "
    "are within about half a metre of it.\n"
    "vx: 0.3-0.4 when sighted and far, 0.1 when close, 0.0 when arrived or searching in "
    "place.\n"
    "Judge SEMANTICALLY, not by colour alone: a red pot is not a red wall. If several "
    "things loosely match, pick the one that best fits the description and say which."
)


# Tricks the brain may ask for. Anything outside this set is dropped rather
# than passed through — a hallucinated action name must never reach the policy
# state machine, and the model will invent them.
ACTIONS = {"none", "roulade", "ground_pick", "sit", "kick_left", "kick_right"}


@dataclass
class Intent:
    vx: float = 0.0
    wz: float = 0.0
    mood: str = "idle"
    why: str = "starting up"
    action: str = "none"
    say: str = ""
    # --- semantic search ---
    bearing: float = 0.0     # degrees, + = left, relative to where the duck faces
    target: str = ""         # what it is currently looking for
    sighted: bool = False    # is the target in frame right now
    arrived: bool = False    # is it there
    source: str = "local"
    stamp: float = field(default_factory=time.monotonic)

    @property
    def age(self) -> float:
        return time.monotonic() - self.stamp


class Eyes:
    """Renders the duck's point of view for the vision model.

    Uses a real <camera name="duck_eye"> mounted at the beak in
    robot_allcollisions_cam.xml. An earlier version aimed a free camera by
    inverting MuJoCo's azimuth/elevation by hand and got it backwards: the
    result was a third-person shot of the duck from behind, so the model spent
    every call describing its own beak as "that yellow thing". A named camera
    has no such failure mode — it is rigidly attached to the head and moves
    with it.

    Small frames on purpose: the upload is on the critical path and 320px is
    plenty for "ball ahead, goal to the left".
    """

    WIDTH, HEIGHT, QUALITY = 320, 240, 70
    CAMERA = "duck_eye"

    def __init__(self, model, data, body_id: int | None = None) -> None:
        import mujoco
        self.model, self.data = model, data
        self.ok = True
        self.error = ""
        self.cam_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_CAMERA, self.CAMERA)
        if self.cam_id < 0:
            self.ok = False
            self.error = f"no camera named {self.CAMERA!r} — scene must include " \
                         "robot_allcollisions_cam.xml"
            return
        try:
            self._r = mujoco.Renderer(model, height=self.HEIGHT, width=self.WIDTH)
        except Exception as exc:                 # no GL context, offscreen too small
            self.ok = False
            self.error = f"{type(exc).__name__}: {exc}"[:120]

    def frame_b64(self) -> str | None:
        """One JPEG of what the duck is looking at, base64'd. None if blind."""
        if not self.ok:
            return None
        import base64
        import io
        from PIL import Image
        try:
            self._r.update_scene(self.data, camera=self.cam_id)
            img = Image.fromarray(self._r.render())
        except Exception as exc:
            self.ok = False
            self.error = f"{type(exc).__name__}: {exc}"[:120]
            return None
        buf = io.BytesIO()
        img.save(buf, format="JPEG", quality=self.QUALITY)
        return base64.b64encode(buf.getvalue()).decode()


# Words that mean "go look for a thing", and the noise around the thing itself.
_SEEK = ("find", "look for", "go to", "walk to", "head to", "search for", "fetch",
         "go find", "seek", "locate", "bring me")
_STOP = ("stop", "forget it", "never mind", "cancel", "at ease", "stand down")
_STRIP = ("the ", "a ", "an ", "that ", "some ", "me ", "to ")


def route(brain: "DuckBrain", text: str) -> str:
    """Decide whether something said to the duck is a SEARCH or a plain instruction.

    Done in code, not by asking the model: the routing must work when the network is
    down or out of credit, and "find the blue pot" is not a sentence that needs a
    language model to classify.
    """
    low = text.strip().lower().rstrip(".!?")
    if any(low.startswith(w) or low == w for w in _STOP):
        brain.seek("")
        return "stop"
    for verb in sorted(_SEEK, key=len, reverse=True):
        if verb in low:
            tgt = low.split(verb, 1)[1].strip()
            for a in _STRIP:
                if tgt.startswith(a):
                    tgt = tgt[len(a):]
            tgt = tgt.strip(" ,.")
            if tgt:
                brain.seek(tgt)
                return f"seek:{tgt}"
    brain.instruct(text)
    return "instruct"



class Ears:
    """Push-to-talk: record a few seconds of mic, transcribe, hand it to the brain.

    Push-to-talk rather than always-on because continuous capture means the duck
    hears its own `say` output and answers itself, and because a background
    process asking for the microphone is a permission dialog nobody sees.

    Transcription goes to 0G's whisper-large-v3 — the one model on that router
    where in-enclave execution is real rather than proxied.
    """

    SECONDS = 4
    DEVICE = os.environ.get("DUCK_MIC", ":0")     # avfoundation audio index

    def __init__(self) -> None:
        import shutil
        # Local Whisper first. 0G's whisper-large-v3 spent this project returning 400
        # (stale price cache) and then 402 (no balance), and the earlier research was
        # right anyway: Whisper is 1.5B params and runs faster than realtime here, so
        # not sending the audio anywhere beats sending it somewhere trustworthy.
        # LAZY. Constructing WhisperModel here loads CTranslate2 into the viewer
        # process, and initialising it alongside mjpython's GL context segfaults the
        # sim at startup (exit 139, before a single frame). The model is built on the
        # first listen instead, on the worker thread, where it is harmless.
        self.local = None
        self._local_tried = False
        try:
            import faster_whisper  # noqa: F401  -- availability check only, no load
            self.backend = "faster-whisper base.en (local, loads on first use)"
            self._have_local = True
        except Exception:
            self.backend = "0G whisper-large-v3 (no local backend installed)"
            self._have_local = False
        self.ok = bool(shutil.which("ffmpeg")) and (self._have_local or bool(API_KEY))
        self.busy = False
        self.last = ""
        self.error = ""

    def listen_async(self, brain: "DuckBrain") -> None:
        """Never blocks the caller — the control loop is on this thread."""
        if not self.ok or self.busy:
            return
        self.busy = True
        threading.Thread(target=self._worker, args=(brain,), daemon=True).start()

    def _worker(self, brain: "DuckBrain") -> None:
        import subprocess
        import tempfile
        try:
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
                wav = f.name
            subprocess.run(
                ["ffmpeg", "-nostdin", "-y", "-f", "avfoundation", "-i", self.DEVICE,
                 "-t", str(self.SECONDS), "-ar", "16000", "-ac", "1", wav],
                stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, timeout=self.SECONDS + 15,
            )
            text = self._transcribe(wav).strip()
            os.unlink(wav)
            if text:
                self.last = text
                route(brain, text)
                print(f'[heard] "{text}"')
            else:
                print("[heard] nothing")
        except Exception as exc:
            self.error = f"{type(exc).__name__}: {exc}"[:140]
            print(f"[ears] {self.error}")
        finally:
            self.busy = False

    def _transcribe(self, wav_path: str) -> str:
        if self._have_local and not self._local_tried:
            self._local_tried = True
            try:
                from faster_whisper import WhisperModel
                print("[ears] loading local speech model (first use)...")
                self.local = WhisperModel("base.en", device="cpu", compute_type="int8")
            except Exception as exc:
                self.error = f"local load failed: {type(exc).__name__}"
                self.local = None
        if self.local is not None:
            segs, _ = self.local.transcribe(wav_path)
            return "".join(sg.text for sg in segs)
        return self._transcribe_remote(wav_path)

    def _transcribe_remote(self, wav_path: str) -> str:
        """multipart/form-data by hand — no requests dependency in this repo."""
        boundary = "----duckform7f3a"
        with open(wav_path, "rb") as f:
            audio = f.read()
        parts = [
            f'--{boundary}\r\nContent-Disposition: form-data; name="model"\r\n\r\n'
            f'whisper-large-v3\r\n'.encode(),
            f'--{boundary}\r\nContent-Disposition: form-data; name="file"; '
            f'filename="a.wav"\r\nContent-Type: audio/wav\r\n\r\n'.encode(),
            audio, f'\r\n--{boundary}--\r\n'.encode(),
        ]
        req = urllib.request.Request(
            f"{BASE_URL}/audio/transcriptions", data=b"".join(parts),
            headers={"Authorization": f"Bearer {API_KEY}",
                     "Content-Type": f"multipart/form-data; boundary={boundary}"},
        )
        try:
            with urllib.request.urlopen(req, timeout=45) as r:
                return json.loads(r.read()).get("text", "")
        except urllib.error.HTTPError as e:
            # The router's own error body is the only useful diagnostic here —
            # a bare "HTTP 400" hides that the outage is on their side.
            detail = e.read()[:200].decode(errors="replace")
            raise RuntimeError(f"whisper HTTP {e.code}: {detail}") from None


RACE_SYSTEM = (
    "You are a small robot duck ON ROLLER SKATES, racing a lap around an oval "
    "marked by orange cones and white gates. The image is YOUR OWN point of view.\n"
    "Reply with ONE json object, no prose, no markdown fence:\n"
    '{"vx":<0.0..0.6>,"wz":<-1.0..1.0>,"mood":"<one word>",'
    '"action":"none","say":"<max 8 words>","why":"<max 8 words>"}\n'
    "THIS IS A RACE. Keep vx at 0.55-0.6 unless you are about to hit something. "
    "Steer with wz to keep the next gate ahead of you and stay between the cones. "
    "Never use an action — tricks lose time. Do not stop. Do not sit."
)


class Race:
    """Lap timing over the oval in scene_race.xml.

    Checkpoints must be taken IN ORDER — a radius test alone would let a duck
    that wanders backwards over the line score a lap it never drove.
    """

    RADIUS = 0.55

    def __init__(self, checkpoints: list[tuple[float, float]]) -> None:
        self.cp = checkpoints
        self.next = 1                       # 0 is the start/finish line
        self.lap = 0
        self.started: float | None = None
        self.splits: list[float] = []
        self.best: float | None = None
        self.last_msg = ""

    def update(self, x: float, y: float, now: float) -> str | None:
        cx, cy = self.cp[self.next]
        if math.hypot(x - cx, y - cy) > self.RADIUS:
            return None
        idx = self.next
        self.next = (self.next + 1) % len(self.cp)
        if idx == 0:                        # crossed the line
            if self.started is None:
                self.started = now
                self.last_msg = "lap 1 started"
            else:
                t = now - self.started
                self.lap += 1
                self.splits.append(t)
                self.best = t if self.best is None else min(self.best, t)
                self.started = now
                self.last_msg = f"LAP {self.lap} in {t:.1f}s (best {self.best:.1f}s)"
            return self.last_msg
        if self.started is None:            # rolling before the first line crossing
            self.started = now
        self.last_msg = f"gate {idx}/{len(self.cp) - 1}"
        return self.last_msg

    def hint(self) -> str:
        cx, cy = self.cp[self.next]
        return f"next gate is at ({cx:+.1f},{cy:+.1f})"


class TextChannel:
    """A file you can talk to the duck through, from any other terminal:

        echo "kick the ball into the goal" >> microduck_rl/say_to_duck.txt

    Exists because the voice path depends on a vendor endpoint that can be —
    and on 2026-08-29 was — returning 400 for everyone. An instruction channel
    with no third party in it is the one that always works.
    """

    def __init__(self, path: str) -> None:
        self.path = path
        self._pos = 0
        # Truncate at startup: yesterday's orders are not today's.
        try:
            open(self.path, "w").close()
        except OSError:
            pass

    def poll(self, brain: "DuckBrain") -> str | None:
        try:
            with open(self.path) as f:
                f.seek(self._pos)
                line = f.readline()
                self._pos = f.tell()
        except OSError:
            return None
        line = line.strip()
        if line:
            route(brain, line)
            return line
        return None


def speak(text: str) -> None:
    """Say it out loud, without ever blocking the caller. Dropped silently if
    `say` is unavailable — speech is decoration, not a dependency."""
    import shutil
    import subprocess
    if not text or not shutil.which("say"):
        return
    try:
        subprocess.Popen(["say", "-r", "190", text[:120]],
                         stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    except Exception:
        pass


class DuckBrain:
    def __init__(self, log_size: int = 400) -> None:
        self.events: deque[str] = deque(maxlen=log_size)
        self._intent = Intent()
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._notes: list[str] = []
        self.calls = 0
        self.errors = 0
        self.frames = 0
        self.last_error = ""
        self.latency = 0.0
        self.online = bool(API_KEY)
        self.eyes: "Eyes | None" = None
        self.race: "Race | None" = None
        self._frame: str | None = None
        self.target: str = ""          # semantic search goal, "" = free roam
        self.found_at: float = 0.0
        self._order: str = ""
        self._order_at: float = 0.0
        self._thread: threading.Thread | None = None

    # ---------------------------------------------------------------- public
    def start(self) -> None:
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()

    def log(self, text: str) -> None:
        self.events.append(f"{time.strftime('%H:%M:%S')} {text}")

    def current(self) -> Intent:
        with self._lock:
            return self._intent

    def notes(self) -> list[str]:
        return list(self._notes)

    def seek(self, target: str) -> None:
        """Start (or clear) a semantic search. While a target is set the brain runs the
        SEARCH prompt instead of the roaming one: it reports where the thing is rather
        than deciding what it feels like doing."""
        with self._lock:
            self.target = target.strip()
            self.found_at = 0.0
        self.log(f"NOW LOOKING FOR: {target}" if target else "search cancelled")

    def instruct(self, text: str) -> None:
        """A spoken instruction. Held separately from the event log and given
        its own line in the prompt: 'someone told me to X' must outrank
        'I noticed a block', or the duck ignores you."""
        with self._lock:
            self._order = text
            self._order_at = time.monotonic()
        self.log(f'SOMEONE SAID TO ME: "{text}"')

    def offer_frame(self, b64: str | None) -> None:
        """Hand the brain the latest view. MUST be called from the thread that
        owns the GL context — MuJoCo's Renderer is not thread-safe, and
        rendering from the brain thread fails silently into a blind duck."""
        if b64:
            with self._lock:
                self._frame = b64

    # --------------------------------------------------------------- private
    def _set(self, i: Intent) -> None:
        with self._lock:
            self._intent = i

    def _run(self) -> None:
        last_deep = 0.0
        while not self._stop.is_set():
            t0 = time.monotonic()
            try:
                if not API_KEY:
                    self._set(self._local_intent())
                else:
                    if time.monotonic() - last_deep > DEEP_PERIOD:
                        self._deep_read()
                        last_deep = time.monotonic()
                    self._set(self._remote_intent())
            except Exception as exc:                     # never kill the thread
                self.errors += 1
                self.last_error = f"{type(exc).__name__}: {exc}"[:120]
                self._set(self._local_intent())
            with self._lock:
                seeking = bool(self.target)
            period = SEEK_PERIOD if seeking else FAST_PERIOD
            self._stop.wait(max(0.0, period - (time.monotonic() - t0)))

    def _local_intent(self) -> Intent:
        """Offline fallback: a slow lissajous wander. Deliberately dull — its job
        is to keep the duck alive and legible, not to be interesting."""
        t = time.monotonic()
        return Intent(
            vx=0.16 + 0.10 * math.sin(t / 7.0),
            wz=0.55 * math.sin(t / 4.3),
            mood="offline",
            why="no brain — local wander",
            source="local",
        )

    def _post(self, messages: list[dict], max_tokens: int) -> str:
        body = json.dumps({
            "model": MODEL,
            "messages": messages,
            "max_tokens": max_tokens,
            "temperature": 0.8,
        }).encode()
        req = urllib.request.Request(
            f"{BASE_URL}/chat/completions", data=body,
            headers={"Authorization": f"Bearer {API_KEY}",
                     "Content-Type": "application/json"},
        )
        t0 = time.monotonic()
        with urllib.request.urlopen(req, timeout=HTTP_TIMEOUT) as r:
            payload = json.loads(r.read())
        self.latency = time.monotonic() - t0
        self.calls += 1
        return payload["choices"][0]["message"]["content"]

    def _remote_intent(self) -> Intent:
        recent = list(self.events)[-28:]
        notes = " | ".join(self._notes[-3:]) or "none yet"
        with self._lock:
            order, order_at = self._order, self._order_at
        head = ""
        if order and time.monotonic() - order_at < ORDER_TTL:
            head = (f'YOUR HUMAN JUST TOLD YOU: "{order}"\n'
                    "Do what they asked. It outranks anything you noticed yourself.\n\n")
        race = ""
        if self.race is not None:
            race = (f"RACE: lap {self.race.lap}, {self.race.hint()}, "
                    f"{self.race.last_msg}\n\n")
        text = head + race + f"What you remember: {notes}\n\nJust happened:\n" + "\n".join(recent)

        # Read the goal FIRST: the frame-budget decision below depends on it.
        with self._lock:
            target = self.target
            frame = self._frame
        # While roaming, most ticks do not need fresh vision: the scene has not changed
        # and the duck is only picking a mood. While seeking, every tick needs it.
        if frame and not target and (self.calls % FRAME_EVERY) != 0:
            frame = None
        if frame:
            self.frames += 1
            content = [
                {"type": "text", "text": text + "\n\nThis is what you see right now:"},
                {"type": "image_url",
                 "image_url": {"url": f"data:image/jpeg;base64,{frame}"}},
            ]
        else:
            content = text

        if target:
            system = SEARCH_SYSTEM
            if isinstance(content, list):
                content[0]["text"] = f'YOU ARE LOOKING FOR: {target}\n\n' + content[0]["text"]
            else:
                content = f'YOU ARE LOOKING FOR: {target}\n\n' + content
        elif self.race is not None:
            system = RACE_SYSTEM
        else:
            system = SYSTEM
        msg = [{"role": "system", "content": system}, {"role": "user", "content": content}]
        obj = _first_json(self._post(msg, 160).strip())

        if target:
            # Bearing -> turn rate. The VLM is reliable about "it is over to your left"
            # and unreliable about picking a rad/s, so it never sees a turn rate: we
            # convert here, with a cap, so a confident wrong bearing cannot spin it.
            bear = _clamp(float(obj.get("bearing", 0.0)), -40.0, 40.0)
            sighted = bool(obj.get("sighted", False))
            arrived = bool(obj.get("arrived", False))
            wz = _clamp(math.radians(bear) * 1.6, -1.0, 1.0)
            vx = _clamp(float(obj.get("vx", 0.0)), 0.0, 0.40)
            if arrived:
                vx, wz = 0.0, 0.0
            elif not sighted:
                vx = min(vx, 0.12)          # sweep, do not charge off blind
            return Intent(
                vx=vx, wz=wz,
                mood=str(obj.get("mood", "seeking"))[:16],
                why=str(obj.get("why", ""))[:60],
                say=str(obj.get("say", ""))[:120],
                bearing=bear, target=target, sighted=sighted, arrived=arrived,
                source=MODEL,
            )

        action = str(obj.get("action", "none")).strip().lower()
        if action not in ACTIONS:                # models invent action names
            action = "none"
        if self.race is not None:                # tricks lose time
            action = "none"
        return Intent(
            vx=_clamp(float(obj.get("vx", 0.0)), -0.30,
                      0.60 if self.race is not None else 0.40),
            wz=_clamp(float(obj.get("wz", 0.0)), -1.2, 1.2),
            mood=str(obj.get("mood", "?"))[:16],
            why=str(obj.get("why", ""))[:60],
            action=action,
            say=str(obj.get("say", ""))[:120],
            source=MODEL,
        )

    def _deep_read(self) -> None:
        """The whole history in one prompt. The point of the big context: find a
        pattern ACROSS distant events, which no retrieval over recent ticks can
        see because the two facts never land in the same window."""
        if not self.events:
            return
        msg = [
            {"role": "system", "content":
                "You are the long-term memory of a robot duck. Read its entire "
                "history and reply with ONE sentence (max 15 words) naming a "
                "pattern across distant events. No preamble."},
            {"role": "user", "content": "\n".join(self.events)},
        ]
        note = self._post(msg, 60).strip().replace("\n", " ")
        if note:
            self._notes.append(note[:120])


class SimSensor:
    """Turns sim state into the event log the brain reads.

    Deliberately coarse: the brain is not a controller and does not need
    50 Hz telemetry. It needs the few things a duck would notice — I moved,
    I got stuck, I fell over, something is in front of me — at roughly the
    rate those things actually happen.
    """

    PERIOD = 1.5          # seconds between observations
    STUCK_M = 0.04        # progress below this while commanding motion = stuck
    RESET_AFTER = 6.0     # seconds face-down before we pick it back up

    def __init__(self, model, data, trunk_body_id: int) -> None:
        self.acted_stamp = None
        self.model = model
        self.data = data
        self.trunk = trunk_body_id
        self.last = 0.0
        self.last_print = 0.0
        self.prev_xy = None
        self.was_down = False
        self.down_since = None
        self.needs_reset = False
        self._obstacles = self._obstacle_geoms()

    def _obstacle_geoms(self) -> list[int]:
        """Everything the duck could bump into: scene props, not its own body
        and not the floor."""
        import mujoco
        out = []
        for g in range(self.model.ngeom):
            name = mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_GEOM, g) or ""
            if name.startswith(("step", "ramp", "deb", "wall_", "bump", "v", "t")):
                out.append(g)
        return out

    def poll(self, now: float, brain: "DuckBrain") -> None:
        if now - self.last < self.PERIOD:
            return
        self.last = now

        import numpy as np
        pos = self.data.xpos[self.trunk].copy()
        xy = pos[:2]
        z = float(pos[2])

        moved = 0.0 if self.prev_xy is None else float(np.linalg.norm(xy - self.prev_xy))
        self.prev_xy = xy.copy()

        # nearest scene object, measured in the plane
        near_d, near_g = 9e9, -1
        for g in self._obstacles:
            d = float(np.linalg.norm(self.model.geom_pos[g][:2] - xy))
            if d < near_d:
                near_d, near_g = d, g

        bits = [f"at ({xy[0]:+.2f},{xy[1]:+.2f})", f"moved {moved*100:.0f}cm"]

        down = z < 0.08
        if down and not self.was_down:
            bits.append("FELL OVER")
            self.down_since = now
        elif self.was_down and not down:
            bits.append("back on my feet")
            self.down_since = None
        self.was_down = down

        # No stand-up policy ships in microduck/policies/ — the repo has a
        # Mjlab-StandUp task but its ONNX was never vendored, so a fallen duck
        # stays fallen forever and the demo ends. Reset it in place after a
        # while and tell the brain it happened, rather than pretending it got
        # up on its own. Train StandUp and load it to delete this.
        if down and self.down_since and (now - self.down_since) > self.RESET_AFTER:
            self.down_since = None
            self.needs_reset = True
            bits.append("someone stood me back up (no recovery policy loaded)")

        if moved < self.STUCK_M and not down:
            bits.append("barely moved — possibly stuck")

        if near_g >= 0 and near_d < 0.45:
            import mujoco
            nm = mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_GEOM, near_g) or "?"
            bits.append(f"{nm} is {near_d*100:.0f}cm away")

        brain.log(", ".join(bits))

        # Capture here, not in the brain thread: this runs on the loop thread
        # that owns the GL context.
        if brain.eyes is not None:
            brain.offer_frame(brain.eyes.frame_b64())


def _clamp(v: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, v))


def _first_json(text: str) -> dict:
    """Models wrap JSON in fences and prose no matter how firmly you ask."""
    i, j = text.find("{"), text.rfind("}")
    if i < 0 or j <= i:
        raise ValueError(f"no json in reply: {text[:80]!r}")
    return json.loads(text[i:j + 1])
