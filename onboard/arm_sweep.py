"""ARM ESC from the command line: hold the cockpit's ARM key for one full sweep (1500 -> 1650 us, 3 s
steps, capped by the lowest open page's throttle slider), then release. Needed after EVERY Pi/ESC power
cycle: since its 2026-10-07 reprofile the XL-5 does not arm on a steady neutral. It drives FORWARD the
moment the ESC arms -- wheels up, failsafes confirmed. Dying = key stale in 0.3 s = neutral.

ESC LED SOLID RED = armed: forward AND reverse work (first seen 2026-11-17). If a sweep does not arm it,
replay the order that worked that day: this sweep (twice; the PCA was outputting 1631 us on ch3 and the wheels
still did not move), then RESTART THE PI (pwm-safe-boot -> board_server neutral), look for solid red, then drive.
Which step armed it is not isolated.

    ssh pi5 python3 DANIEL/onboard/arm_sweep.py
"""
import http.client
import json
import sys
import time

c = http.client.HTTPConnection(sys.argv[1] if sys.argv[1:] else "127.0.0.1", 8090, timeout=2)


def post(keys):
    c.request("POST", "/keys", json.dumps({"id": "claude-arm", "keys": keys, "cap": 0.30}),
              {"Content-Type": "application/json"})
    return json.loads(c.getresponse().read())


t0, last = time.monotonic(), -1
try:
    while time.monotonic() - t0 < 28.5:
        s = post(["arm"])
        sec = int(time.monotonic() - t0)
        if sec != last:
            last = sec
            print(f"{sec:2d}s throttle {s['throttle']} us  {s['phase']}  dry={s['dry']}", flush=True)
        time.sleep(0.05)
finally:
    for _ in range(5):
        s = post([])
        time.sleep(0.05)
    print("released ->", s["throttle"], "us", s["phase"])
