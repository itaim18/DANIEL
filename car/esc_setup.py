"""Runs on the LAPTOP: calibrate the XL-5 ESC against THIS PCA9685, then function-test the car.

    /usr/bin/python3 esc_setup.py 10.42.0.139 --failsafes-confirmed

G = auto-calibrate: after you put the ESC into calibration (EZ-Set held from OFF, released on
solid red), this plays the transmitter's part -- neutral, FULL forward, FULL reverse, neutral.
Calibrating to the PCA, not the Traxxas remote, makes the ESC's neutral our 1500 us and
absorbs the PCA's oscillator error. Re-profiling an XL-5 drops its calibration: it then
blinks green and will not arm (2026-10-07), so calibrate after every profile change.
T = function test through drive_tk.Drive (the same steer map + reverse gate as avoid_drive):
left, right, forward, reverse. N / F / R = hold neutral / full fwd / full rev by hand.
Any other key = stop at neutral. Close = neutral.
"""
import socket
import sys
import time
import tkinter as tk

from drive_tk import Drive

STEPS = """XL-5 calibration (wheels off the ground):
 1. ESC OFF. Hold EZ-Set: LED green -> red. Release on SOLID red.
 2. Press G. This window plays neutral -> FULL fwd -> FULL rev -> neutral.
 3. LED ends solid green = calibrated. Then press T and watch the car.
Profile (if ever needed): hold EZ-Set past red into red BLINKS, release after
 1 blink = Sport (fwd/brake/REVERSE). Then calibrate again (G)."""
CALIBRATE = [(1500, 1.5, "neutral"), (2000, 3.0, "FULL FORWARD (LED: red x1)"),
             (1000, 3.0, "FULL REVERSE (LED: red x2)"), (1500, 3.0, "neutral (LED: green)")]
TEST = [(set(), 1.0, "ready"), ({"left"}, 2.0, "STEER LEFT"), ({"right"}, 2.0, "STEER RIGHT"),
        (set(), 1.0, "centre"), ({"fwd"}, 2.5, "FORWARD"), (set(), 1.5, "coast"),
        ({"rev"}, 3.5, "REVERSE (push, neutral, push)"), (set(), 1.0, "stop")]

host = next((a for a in sys.argv[1:] if not a.startswith("--")), "10.42.0.139")
if "--failsafes-confirmed" not in sys.argv:
    raise SystemExit("REFUSED: writes full-range PWM to the ESC. Run preflight.sh first; "
                     "then re-run with --failsafes-confirmed.")
sock = socket.create_connection((host, 6001), timeout=3)
root = tk.Tk()
root.title(f"ESC setup -> {host}")
big = tk.Label(root, font=("sans", 40, "bold"), width=24)
big.pack(padx=20, pady=(14, 0))
small = tk.Label(root, font=("monospace", 16))
small.pack()
tk.Label(root, text=STEPS, justify="left", font=("monospace", 11)).pack(padx=20, pady=12)
st = {"hold": 1500, "plan": None, "k": 0, "t0": 0.0, "drive": None}


def start(plan):
    st.update(plan=plan, k=0, t0=time.monotonic(), drive=Drive())


def hold(us):
    st.update(plan=None, hold=us)


for key, fn in (("g", lambda: start(CALIBRATE)), ("t", lambda: start(TEST)),
                ("n", lambda: hold(1500)), ("f", lambda: hold(2000)), ("r", lambda: hold(1000))):
    root.bind(key, lambda e, fn=fn: fn())
root.bind("<Key>", lambda e: hold(1500) if e.keysym.lower() not in "gtnfr" else None, add="+")


def tick():
    now, su, tu, label, phase = time.monotonic(), 1500, st["hold"], f"holding {st['hold']} us", ""
    plan = st["plan"]
    if plan is not None:
        while st["k"] < len(plan) and now - st["t0"] >= plan[st["k"]][1]:
            st["k"] += 1
            st["t0"] = now
        if st["k"] >= len(plan):
            hold(1500)
            label = "DONE - neutral"
        else:
            what, _, label = plan[st["k"]]
            if plan is CALIBRATE:
                tu = what
            else:
                su, tu, phase = st["drive"].step(what, 0.25, now)
    sock.sendall(f"{su},{tu}\n".encode())          # 20 Hz keeps board_server's deadman fed
    big.config(text=label, fg="red" if tu < 1480 else ("green" if tu > 1520 else "black"))
    small.config(text=f"steer {su} us   throttle {tu} us   {phase}")
    root.after(50, tick)


def close():
    try:
        sock.sendall(b"1500,1500\n")
    finally:
        sock.close()
        root.destroy()


root.protocol("WM_DELETE_WINDOW", close)
tick()
root.mainloop()
