"""Bench PWM sweep -- WHEELS OFF THE GROUND. Calibrates the sim's PWM map to THIS car.

    # on the LAPTOP, against a board that is already running board_server.py:
    python3 bench_pwm.py --host 192.168.0.70 --failsafes-confirmed --steer
    python3 bench_pwm.py --host 192.168.0.70 --failsafes-confirmed --throttle

WHY THIS EXISTS. `mdp.AckermannPwmActionCfg` maps a policy action in [-1,1] onto
**1150-1850 us steering and 1000-2000 us throttle** (`mdp.py:1974-1975`). Those
are SIM constants. If this car's servo reaches full lock at 1100/1900, the policy
under-steers by 12% at every command and nothing in the control loop reports it:
the actions look perfect, the car just does not turn as hard as it meant to. Same
for throttle -- an ESC with a +/-40 us neutral deadband turns every small action
into exactly zero drive.

Ponytail: there is no automatic measurement here because THIS CAR HAS NO SENSORS
to measure with -- no wheel encoders, no steering feedback. So the script holds
each PWM value steady long enough for a human to read a protractor or count wheel
revolutions on video, and prints a table to fill in. That is the honest
instrument; a fitted curve from no measurement would be a decoration.

SAFETY: wheels off the ground, every time. `--failsafes-confirmed` is required
for the same reason `drive_policy.py --arm` requires it (Constitution VI:
UNKNOWN blocks actuation exactly as FAIL does). Neutral is written on every exit
path, and the board's own watchdog forces neutral after 0.4 s of silence.
"""
import argparse
import signal
import socket
import sys
import time

CONTROL_PORT = 6001
STEER_NEUTRAL = THROTTLE_NEUTRAL = 1500
SIM_STEER_US = (1150, 1850)          # mdp.py:1974
SIM_THROTTLE_US = (1000, 2000)       # mdp.py:1975


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--host", required=True)
    p.add_argument("--steer", action="store_true")
    p.add_argument("--throttle", action="store_true")
    p.add_argument("--hold", type=float, default=3.0, help="seconds per step")
    p.add_argument("--steps", type=int, default=9)
    p.add_argument("--throttle-max-us", type=int, default=1650,
                   help="ceiling for the throttle sweep. 1650 = +0.30 of the sim "
                        "range. Do NOT raise it with the car on the ground.")
    p.add_argument("--forward-only", action="store_true",
                   help="throttle sweep from neutral up only (an ESC that needs a sweep to arm)")
    p.add_argument("--failsafes-confirmed", action="store_true")
    a = p.parse_args()

    if not (a.steer ^ a.throttle):
        raise SystemExit("choose exactly one of --steer / --throttle")
    if not a.failsafes_confirmed:
        raise SystemExit(
            "REFUSED: --failsafes-confirmed is required. This writes PWM to a "
            "real ESC and servo. Wheels off the ground, failsafes confirmed live "
            "in session by a human (STATE.md trap 5), then pass the flag.")

    s = socket.create_connection((a.host, CONTROL_PORT), timeout=5.0)
    s.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)

    def send(su, tu):
        s.sendall(f"{int(su)},{int(tu)}\n".encode())

    def neutral(*_):
        for _i in range(10):
            try:
                send(STEER_NEUTRAL, THROTTLE_NEUTRAL)
            except OSError:
                break
            time.sleep(0.05)
        print("\n[exit] neutral written")
        sys.exit(0)

    signal.signal(signal.SIGINT, neutral)
    signal.signal(signal.SIGTERM, neutral)

    if a.steer:
        lo, hi, what, unit = SIM_STEER_US[0], SIM_STEER_US[1], "STEER", "degrees"
        head = "measure with a protractor against the chassis centreline"
    else:
        lo, hi = (THROTTLE_NEUTRAL if a.forward_only else 2 * THROTTLE_NEUTRAL - a.throttle_max_us), a.throttle_max_us
        what, unit = "THROTTLE", "wheel rev/s"
        head = "count wheel revolutions on video over the hold window"

    print(f"\n=== {what} sweep: {lo} -> {hi} us, {a.steps} steps, "
          f"{a.hold:.1f}s hold ===")
    print(f"WHEELS OFF THE GROUND. {head}.\n")
    print(f"{'us':>6}  {'action':>7}   measured {unit}")
    try:
        # neutral first: an ESC that has not seen neutral may not arm
        send(STEER_NEUTRAL, THROTTLE_NEUTRAL)
        time.sleep(1.0)
        for i in range(a.steps):
            us = lo + (hi - lo) * i / (a.steps - 1)
            mid = (SIM_STEER_US[0] + SIM_STEER_US[1]) / 2 if a.steer else THROTTLE_NEUTRAL
            span = (SIM_STEER_US[1] - SIM_STEER_US[0]) / 2 if a.steer else \
                (SIM_THROTTLE_US[1] - SIM_THROTTLE_US[0]) / 2
            action = (us - mid) / span
            print(f"{us:6.0f}  {action:+7.3f}   ________", flush=True)
            t_end = time.time() + a.hold
            while time.time() < t_end:      # keep feeding: watchdog is 0.4 s
                send(us, THROTTLE_NEUTRAL) if a.steer else send(STEER_NEUTRAL, us)
                time.sleep(0.05)
    finally:
        neutral()


if __name__ == "__main__":
    main()
