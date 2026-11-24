"""Runs on the LAPTOP: keyboard teleop for board_server.py on the Pi 5, no video.

    /usr/bin/python3 drive_tk.py 10.42.0.139 --failsafes-confirmed
    /usr/bin/python3 drive_tk.py --selftest

THIS ACTUATES THE CAR: run preflight.sh first; refuses without --failsafes-confirmed.
Tk, not pynput: pynput's global key hook does not see keys on Wayland; a Tk window
gets its own press/release events. The window must have focus -- losing it is a stop.

It speaks the SAME action space as onboard/avoid_drive.py (to_pwm: steer +1 = left ->
1150 us on this car; throttle through EscGate), so what you see here is what the
autonomous stack will do. Arrows/WASD drive, SPACE held = stop, slider = throttle cap.
"""
import socket
import sys
import time
import tkinter as tk

CONTROL_PORT = 6001
# onboard/avoid_drive.py STEER_US: (right, left) lock -- 1850 us turns RIGHT (2026-10-07)
STEER_US, THROTTLE_US, NEUTRAL = (1850, 1150), (1000, 2000), 1500
STEER_RATE, THROTTLE_RATE = 4.0, 2.0     # action units per second (full lock in 0.25 s)
TICK_S = 0.05                            # 20 Hz, well inside board_server's 0.4 s deadman


class EscGate:
    """ponytail: copy of onboard/avoid_drive.py EscGate -- importing it drags in the Hailo
    stack; keep the two in sync. XL-5HV Trail mode: the first reverse push only brakes, the
    second reverses, from rest too. So EVERY reverse: push BRAKE_S, neutral GAP_S, reverse.
    2026-11-17: the real car reverses through this gate (ESC LED solid red = armed)."""
    BRAKE_S, GAP_S = 0.30, 0.15

    def __init__(self):
        self.phase, self.t = "neutral", 0.0

    def __call__(self, throttle, now):
        if throttle > 0.02:
            self.phase = "fwd"
        elif throttle < -0.02:
            if self.phase in ("fwd", "neutral"):    # EVERY reverse entry, standstill too
                self.phase, self.t = "brake", now
            if self.phase == "brake" and now - self.t >= self.BRAKE_S:
                self.phase, self.t = "gap", now
            if self.phase == "gap":
                if now - self.t < self.GAP_S:
                    return 0.0, "gap"
                self.phase = "rev"
        elif self.phase == "rev":
            self.phase = "neutral"
        return throttle, self.phase


def ramp(cur, target, step):
    return min(cur + step, target) if cur < target else max(cur - step, target)


class Drive:
    """Held keys -> (steer_us, throttle_us, esc phase). Pure, so --selftest can drive it."""

    def __init__(self):
        self.steer = self.throttle = 0.0
        self.esc = EscGate()

    def step(self, held, cap, now, dt=TICK_S):
        if "stop" in held:
            self.steer = self.throttle = 0.0
        else:
            self.steer = ramp(self.steer, ("left" in held) - ("right" in held), STEER_RATE * dt)
            self.throttle = ramp(self.throttle, ("fwd" in held) - ("rev" in held), THROTTLE_RATE * dt)
        thr, phase = self.esc(self.throttle * cap, now)
        su = round(sum(STEER_US) / 2 + self.steer * (STEER_US[1] - STEER_US[0]) / 2)
        tu = round(sum(THROTTLE_US) / 2 + thr * (THROTTLE_US[1] - THROTTLE_US[0]) / 2)
        return su, tu, phase


KEYS = {"w": "fwd", "Up": "fwd", "s": "rev", "Down": "rev", "a": "left", "Left": "left",
        "d": "right", "Right": "right", "space": "stop"}


def main(host):
    sock = socket.create_connection((host, CONTROL_PORT), timeout=3)
    sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
    drive, held, pending = Drive(), set(), {}
    root = tk.Tk()
    root.title(f"Traxxas teleop -> {host}")
    big = ("monospace", 20, "bold")
    status = tk.Label(root, font=big, width=34)
    status.pack(padx=12, pady=(12, 4))
    bars = tk.Canvas(root, width=440, height=150, bg="#202020", highlightthickness=0)
    bars.pack(padx=12)
    cap = tk.Scale(root, from_=0.05, to=1.0, resolution=0.05, orient="horizontal", length=440,
                   label="throttle cap (fraction of full ESC range)")
    cap.set(0.25)
    cap.pack(padx=12)
    tk.Label(root, justify="left", font=("monospace", 11), text=(
        "W/Up forward   S/Down brake->reverse   A/Left left (1150 us)\n"
        "D/Right right (1850 us)   SPACE (hold) stop   window loses focus = stop\n"
        "Click the slider, then click back on the bars before driving.")).pack(padx=12, pady=8)

    def press(e):
        k = KEYS.get(e.keysym)
        if k:
            if k in pending:                       # X11 autorepeat: release+press pairs
                root.after_cancel(pending.pop(k))
            held.add(k)

    def release(e):
        k = KEYS.get(e.keysym)
        if k:                                      # wait one beat for an autorepeat press
            pending[k] = root.after(30, lambda: (held.discard(k), pending.pop(k, None)))

    root.bind("<KeyPress>", press)
    root.bind("<KeyRelease>", release)
    root.bind("<FocusOut>", lambda e: held.clear())
    link = {"ok": True}

    def tick():
        su, tu, phase = drive.step(held, cap.get(), time.monotonic())
        if link["ok"]:
            try:
                sock.sendall(f"{su},{tu}\n".encode())
            except OSError:
                link["ok"] = False                 # board_server neutrals after 0.4 s
        status.config(text=f"steer {su} us   throttle {tu} us\nESC {phase:7s}  "
                           + ("LINK OK" if link["ok"] else "LINK LOST - car neutral"),
                      fg="#1a7f37" if link["ok"] else "#c00")
        bars.delete("all")
        for y, us, (lo, hi), name in ((40, su, STEER_US, "steer"), (110, tu, THROTTLE_US, "throttle")):
            x = 20 + 400 * (us - min(lo, hi)) / abs(hi - lo)
            bars.create_line(220, y - 22, 220, y + 22, fill="#666")
            bars.create_rectangle(min(220, x), y - 14, max(220, x), y + 14,
                                  fill="#3b82f6" if name == "steer" else ("#22c55e" if us >= NEUTRAL else "#ef4444"))
            bars.create_text(24, y - 28, text=f"{name}  {min(lo, hi)}..{max(lo, hi)}", fill="#ccc", anchor="w")
        root.after(int(TICK_S * 1000), tick)

    def close():
        for _ in range(5):
            try:
                sock.sendall(f"{NEUTRAL},{NEUTRAL}\n".encode())
            except OSError:
                break
            time.sleep(0.02)
        sock.close()
        root.destroy()

    root.protocol("WM_DELETE_WINDOW", close)
    tick()
    try:
        root.mainloop()
    finally:
        try:
            sock.close()
        except OSError:
            pass


def _selftest():
    d, t = Drive(), 0.0

    def run(keys, n, cap=0.25):
        nonlocal t
        out = None
        for _ in range(n):
            t += TICK_S
            out = d.step(set(keys), cap, t)
        return out

    assert run([], 1) == (1500, 1500, "neutral")
    assert run(["left"], 20)[0] == 1150                         # left = 1150 us on this car
    assert run(["fwd"], 20) == (1500, 1625, "fwd")              # cap 0.25 -> +125 us
    su, tu, ph = run(["rev"], 2)
    assert ph == "fwd" and tu > 1500                            # ramping down, still forward
    seen = [run(["rev"], 1)[2] for _ in range(30)]
    assert seen.index("brake") < seen.index("gap") < seen.index("rev"), seen
    assert run(["rev"], 20)[1] == 1375                          # true reverse, capped
    assert run(["stop", "fwd"], 1)[:2] == (1500, 1500)          # SPACE wins
    run([], 30)                                                 # at rest, then reverse:
    seen = [run(["rev"], 1)[2] for _ in range(20)]              # still push -> neutral -> rev
    assert "rev" in seen and seen.index("brake") < seen.index("gap") < seen.index("rev"), seen
    print("drive_tk selftest: OK (cap, left = 1150 us, push -> neutral -> reverse from fwd AND rest, stop)")


if __name__ == "__main__":
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    if "--selftest" in sys.argv:
        _selftest()
    elif "--failsafes-confirmed" not in sys.argv:
        raise SystemExit("REFUSED: drive_tk.py writes PWM to the car. Run preflight.sh first;\n"
                         "when every failsafe row is PASS, re-run with --failsafes-confirmed.")
    else:
        main(args[0] if args else "10.42.0.139")
