#!/usr/bin/env python3
"""Self-check for board_server's operator deadman (--operator-port, --cockpit-port).

    python3 selftest_operator_deadman.py     # assert-based, no board, no PCA9685

Onboard, the policy's commands come from localhost, so a WiFi loss no longer
stops the car. The deadman is what does. This drives the REAL control_server,
watchdog_loop and operator_server over loopback against a fake I2C bus and
asserts the throttle register, which is what the ESC actually sees.
"""
import socket
import sys
import tempfile
import threading
import time
import types
import urllib.request
from pathlib import Path

REGS = {}


class FakeBus:
    def __init__(self, _bus):
        pass

    def read_byte_data(self, _addr, reg):
        return REGS.get(reg, 0)

    def write_byte_data(self, _addr, reg, val):
        REGS[reg] = val


sys.modules["smbus2"] = types.SimpleNamespace(SMBus=FakeBus)
sys.path.insert(0, str(Path(__file__).resolve().parent))
import board_server as bs  # noqa: E402

THR_OFF_L = bs.LED0_ON_L + 4 * bs.THROTTLE_CH + 2
CMD = 1700


def throttle_us():
    off = REGS[THR_OFF_L] | REGS[THR_OFF_L + 1] << 8
    return off * 1e6 / (bs.FREQ_HZ * 4096)


def moving():
    return abs(throttle_us() - CMD) < 5


def neutral_for(seconds):
    """True iff the throttle reads neutral at EVERY sample over the window."""
    end = time.time() + seconds
    while time.time() < end:
        if abs(throttle_us() - bs.THROTTLE_NEUTRAL) > 5:
            return False
        time.sleep(0.01)
    return True


def free_port():
    s = socket.socket()
    s.bind(("127.0.0.1", 0))
    port = s.getsockname()[1]
    s.close()
    return port


def main():
    bs.CONTROL_PORT, op_port = free_port(), free_port()
    bs.operator_timeout_s = 0.4
    for target, args in ((bs.watchdog_loop, ()), (bs.control_server, ()),
                         (bs.operator_server, (op_port,))):
        threading.Thread(target=target, args=args, daemon=True).start()
    time.sleep(0.2)

    ctl = socket.create_connection(("127.0.0.1", bs.CONTROL_PORT))

    def commander():                       # the onboard policy, 20 Hz, forever
        while True:
            ctl.sendall(f"1500,{CMD}\n".encode())
            time.sleep(0.05)

    threading.Thread(target=commander, daemon=True).start()

    # 1. no operator has ever connected -> commands flow, nothing moves
    assert neutral_for(0.5), "moved before any operator heartbeat"

    # 2. heartbeat flowing -> commands pass
    op = socket.create_connection(("127.0.0.1", op_port))
    beating = threading.Event()
    beating.set()

    def heart():
        while True:
            if beating.is_set():
                try:
                    op.sendall(b"\n")
                except OSError:            # case 4 hangs up on purpose
                    return
            time.sleep(0.1)

    threading.Thread(target=heart, daemon=True).start()
    time.sleep(0.3)
    assert moving(), f"heartbeat live but throttle {throttle_us():.0f} us"

    # 3. heartbeat stops, socket stays open (out of WiFi range: no EOF)
    beating.clear()
    t0 = time.time()
    while moving() and time.time() - t0 < 1.0:
        time.sleep(0.005)
    stop_s = time.time() - t0
    assert stop_s < 0.5, f"took {stop_s:.3f} s to stop, limit 0.5"
    assert neutral_for(0.5), "commands leaked through a stale deadman"

    # 4. operator hangs up (Ctrl-C on the laptop) -> stays stopped
    beating.set()
    time.sleep(0.3)
    assert moving()
    op.close()
    time.sleep(0.1)
    assert neutral_for(0.5), "moved after the operator hung up"

    # 5. deadman off (--operator-port unset) -> tethered behaviour unchanged
    bs.operator_timeout_s = None
    time.sleep(0.2)
    assert moving(), "without --operator-port commands must pass as before"

    # 6. the phone cockpit drives the SAME deadman over HTTP
    bs.operator_timeout_s, bs.last_op_time = 0.4, 0.0
    web = free_port()
    with tempfile.TemporaryDirectory() as d:
        pov = str(Path(d) / "pov.jpg")
        Path(pov).write_bytes(b"\xff\xd8JPEG\xff\xd9")
        threading.Thread(target=bs.cockpit_server, args=(web, pov), daemon=True).start()
        time.sleep(0.2)
        url = f"http://127.0.0.1:{web}"
        assert b"HOLD TO DRIVE" in urllib.request.urlopen(url + "/").read()
        with urllib.request.urlopen(url + "/pov.mjpg", timeout=2) as r:
            assert b"\xff\xd8JPEG\xff\xd9" in r.read(200), "MJPEG did not carry the view"
        assert neutral_for(0.3), "cockpit open but nobody holding: must not move"

        holding = threading.Event()
        holding.set()

        def phone():
            while holding.is_set():
                urllib.request.urlopen(urllib.request.Request(url + "/hb", method="POST"))
                time.sleep(0.1)

        threading.Thread(target=phone, daemon=True).start()
        time.sleep(0.3)
        assert moving(), f"button held but throttle {throttle_us():.0f} us"
        holding.clear()                              # finger lifted
        t0 = time.time()
        while moving() and time.time() - t0 < 1.0:
            time.sleep(0.005)
        web_stop_s = time.time() - t0
        assert web_stop_s < 0.5, f"cockpit release took {web_stop_s:.3f} s, limit 0.5"
        assert neutral_for(0.5), "commands leaked after the button was released"

    print(f"selftest_operator_deadman: 6/6 PASS (stop latency tcp {stop_s*1e3:.0f} ms, "
          f"cockpit {web_stop_s*1e3:.0f} ms)")


if __name__ == "__main__":
    main()
