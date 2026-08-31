#!/usr/bin/env python3
"""Drive the microduck's joints through the BAM actuator model, not an ideal PD.

WHY THIS EXISTS
---------------
The shipped policies were trained against BAM: a voltage-controlled XL330 with
back-EMF, a firmware current limiter, and load-dependent Coulomb/Stribeck/viscous
friction. ``scripts/infer_policy.py`` deploys them against the MJCF's
``<position kp="0.55" kv="0">`` — an ideal PD. Measured consequence: the policy
BALANCES fine through that swap (up-vector +1.00, holds 116 mm indefinitely at
zero command) but cannot WALK — any forward command tips it within ~1.5 s.

Ruled out first, by measurement, so nobody repeats the work:
  - control rate      50 Hz and 125 Hz both fall
  - actuator delay    the trained 3-6 step lag makes it WORSE (falls even at vx=0)
  - observation       verified correct: gravity (0,0,-1), joints at default,
                      command right, actions unsaturated
  - actuator gain     15-point kp/kv sweep, every combination falls
  - action scale      8 values from 1.0 down to 0.05, every one falls

The repo's own note is the explanation: "at this scale, actuator fidelity is most
of the sim2real gap, which is why the actuator is modeled down to its voltage
control law instead of an ideal PD."

WHAT IT DOES
------------
Replaces MuJoCo's position servos with the same computation training used:

    duty  = clamp(error_gain * kp_fw * (q_target - q + q_offset), -max_pwm, max_pwm)
            with a current-limit clamp on duty first
    V     = vin * duty
    tau   = kt*V/R - kt^2*dq/R                       (DC motor with back-EMF)
    tau  -= friction, clipped to the BAM friction budget

Torque is applied through ``qfrc_applied`` and the MJCF actuators are zeroed, so
the plant is BAM's and nothing else contributes.

    from bam_plant import BamPlant
    plant = BamPlant(model, data, joint_names)
    ...
    plant.apply(target_qpos)     # once per control step, before mj_step
"""
from __future__ import annotations

import json
import os

import mujoco
import numpy as np

# Fitted M6 parameters for the XL330, shipped with the bam package. Same file the
# training env loads, so the plant here and the plant the policy learned agree.
_PARAM_PATH = None
try:
    from bam import model as _bam_model
    _PARAM_PATH = os.path.join(os.path.dirname(_bam_model.__file__),
                               "params", "xl330", "m6.json")
except Exception:
    pass

# Firmware constants for the XL330 position loop (bam/dynamixel/actuator.py).
XL330_ENCODER_COUNTS_PER_REV = 4096.0
XL330_KP_DIVISOR = 128.0
XL330_PWM_LIMIT = 885.0
KP_FW = 200.0          # microduck's preserved firmware stiffness (constants.py)
VIN = 7.5              # nominal battery; training randomised 6.5-8.2
MAX_PWM = 1.0
MAX_CURRENT = 1.75     # A


class BamPlant:
    """BAM voltage-controlled actuation for a set of MuJoCo hinge joints."""

    def __init__(self, model, data, joint_names, kp_fw: float = KP_FW,
                 vin: float = VIN, params_path: str | None = None):
        self.m, self.d = model, data
        self.kp_fw, self.vin = kp_fw, vin

        p = params_path or _PARAM_PATH
        if not p or not os.path.exists(p):
            raise FileNotFoundError(
                "BAM m6 parameters for the xl330 not found; install "
                "better-actuator-models (pip install "
                "'better-actuator-models @ git+https://github.com/Rhoban/bam.git"
                "@mjlab_frictionloss')")
        with open(p) as f:
            self.P = json.load(f)

        self.error_gain = ((XL330_ENCODER_COUNTS_PER_REV / (2 * np.pi))
                           / (XL330_KP_DIVISOR * XL330_PWM_LIMIT))

        self.jids, self.qadr, self.dadr = [], [], []
        for n in joint_names:
            j = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, n)
            if j < 0:
                raise ValueError(f"joint {n!r} not in model")
            self.jids.append(j)
            self.qadr.append(int(model.jnt_qposadr[j]))
            self.dadr.append(int(model.jnt_dofadr[j]))
        self.qadr = np.array(self.qadr)
        self.dadr = np.array(self.dadr)

        # Silence the MJCF position servos: BAM is the only thing driving these
        # joints now, and leaving an ideal PD underneath would sum two plants.
        self._muted = []
        for a in range(model.nu):
            trn = int(model.actuator_trnid[a, 0])
            if trn in self.jids:
                self._muted.append((a, model.actuator_gainprm[a].copy(),
                                    model.actuator_biasprm[a].copy()))
                model.actuator_gainprm[a, :] = 0.0
                model.actuator_biasprm[a, :] = 0.0

        # BAM adds the motor's apparent inertia to the joint, which an ideal PD
        # model leaves out. Without it the legs are too easy to accelerate.
        self.m.dof_armature[self.dadr] += float(self.P["armature"])
        # BAM computes friction itself, so MuJoCo's own frictionloss must go or
        # it is counted twice (this mirrors BamActuator.edit_spec upstream).
        self.m.dof_frictionloss[self.dadr] = 0.0

        self.last_tau = np.zeros(len(self.jids))

    # ------------------------------------------------------------------ plant
    def _friction_budget(self, tau_motor, dq):
        """BAM m6 friction budget: base + Stribeck + load-dependent, directional."""
        P = self.P
        # Stribeck weight: high near zero velocity, decaying with |dq|
        st = np.exp(-np.abs(dq) / max(P["dtheta_stribeck"], 1e-6)) ** P["alpha"]

        base = P["friction_base"] + P["friction_stribeck"] * st

        # Load-dependent terms, split motor-side vs external-side (directional).
        tm = np.abs(tau_motor)
        load = (P["load_friction_motor"] + P["load_friction_motor_stribeck"] * st) * tm
        load += (P["load_friction_external"]
                 + P["load_friction_external_stribeck"] * st) * tm
        load += P["load_friction_motor_quad"] * tm ** 2
        load += P["load_friction_external_quad"] * tm ** 2

        return base + load

    def apply(self, q_target) -> np.ndarray:
        """One control step. `q_target` is the desired joint angle per joint [rad]."""
        P = self.P
        q = self.d.qpos[self.qadr]
        dq = self.d.qvel[self.dadr]
        err = np.asarray(q_target, dtype=float) - q + P["q_offset"]

        duty = self.error_gain * self.kp_fw * err

        # Firmware current limiter, expressed on the duty cycle exactly as BAM does:
        # bound duty so I = (duty*vin - kt*dq)/R stays inside +/- max_current.
        back_emf = P["kt"] * dq
        span = P["R"] * MAX_CURRENT / self.vin
        centre = back_emf / self.vin
        duty = np.clip(duty, centre - span, centre + span)
        # Physical PWM limit last: the battery cannot exceed its own voltage.
        duty = np.clip(duty, -MAX_PWM, MAX_PWM)

        volts = self.vin * duty
        tau = P["kt"] * volts / P["R"] - (P["kt"] ** 2) * dq / P["R"]

        # Friction opposes motion. Two regimes, and conflating them was the bug in
        # the first version of this file (it divided by the timestep and applied the
        # viscous term twice, which destabilised even standing still):
        #   moving  -> a Coulomb torque of the full budget, against the velocity
        #   stopped -> only enough to cancel the applied torque, never more than
        #              the budget, otherwise friction would drive the joint
        budget = self._friction_budget(tau, dq)
        tau -= P["friction_viscous"] * dq
        moving = np.abs(dq) > 1e-4
        tau += np.where(moving,
                        -np.sign(dq) * budget,
                        -np.clip(tau, -budget, budget))

        self.d.qfrc_applied[self.dadr] = tau
        self.last_tau = tau
        return tau

    def restore(self) -> None:
        """Put the MJCF actuators back (for A/B tests in one process)."""
        for a, gain, bias in self._muted:
            self.m.actuator_gainprm[a, :] = gain
            self.m.actuator_biasprm[a, :] = bias
        self.m.dof_armature[self.dadr] -= float(self.P["armature"])
