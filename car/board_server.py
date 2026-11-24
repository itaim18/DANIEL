"""Runs on the Radxa Q6A: streams RealSense color frames to the laptop over
TCP and applies steering/throttle commands received from it to the PCA9685.

    python3 board_server.py

    # Pi 5, policy ONBOARD (drive_policy.py --camera picam owns the camera):
    I2C_BUS=1 python3 board_server.py --no-video --operator-port 6002 --cockpit-port 8080
    # ...the operator's hand on the deadman: a phone at http://<pi>:8080 (HOLD TO
    # DRIVE), or from a laptop:
    while :; do echo; sleep 0.1; done | nc <pi> 6002

Two plain TCP servers (video push, control pull), each single-client.
Safety: a watchdog forces both channels to neutral if no control command
has arrived in WATCHDOG_TIMEOUT seconds -- a dropped network connection
must never leave the car driving. PCA9685 @ i2c-6/0x40, channel 0 =
steering, channel 3 = throttle (confirmed 2026-07-21, see
plans/DANIEL_contract.md).

--operator-port closes the hole that running the policy onboard opens. Tethered,
the control socket crosses WiFi, so losing the link stops the car. Onboard, the
commands come from localhost and losing WiFi would NOT stop anything. With the
flag, PWM passes only while a remote heartbeat is fresher than
--operator-timeout: Ctrl-C on the laptop, or driving out of range, is a stop.
"""
import argparse
import atexit
import os
import signal
import select
import socket
import struct
import sys
import threading
import time

import smbus2

# Q6A: i2c-6. Pi 5 header pins 3/5: i2c-1.
I2C_BUS, PCA_ADDR = int(os.environ.get("I2C_BUS", 6)), 0x40
FREQ_HZ = 50
LED0_ON_L = 0x06
STEER_CH, THROTTLE_CH = 0, 3
STEER_NEUTRAL, THROTTLE_NEUTRAL = 1500, 1500

VIDEO_PORT, CONTROL_PORT = 6000, 6001
WATCHDOG_TIMEOUT_S = 0.4


class PCA9685:
    def __init__(self, bus_num, addr):
        self.bus = smbus2.SMBus(bus_num)
        self.addr = addr
        old = self.bus.read_byte_data(addr, 0x00)
        self.bus.write_byte_data(addr, 0x00, old | 0x10)
        prescale = round(25_000_000 / (4096 * FREQ_HZ)) - 1
        self.bus.write_byte_data(addr, 0xFE, prescale)
        self.bus.write_byte_data(addr, 0x00, old & ~0x10 & 0xFF)
        time.sleep(0.001)

    def set_pulse_us(self, ch, us):
        off = max(0, min(4095, round(us * FREQ_HZ * 4096 / 1_000_000)))
        base = LED0_ON_L + 4 * ch
        self.bus.write_byte_data(self.addr, base + 0, 0)
        self.bus.write_byte_data(self.addr, base + 1, 0)
        self.bus.write_byte_data(self.addr, base + 2, off & 0xFF)
        self.bus.write_byte_data(self.addr, base + 3, off >> 8)

    def all_outputs_off(self):
        # ALL_LED_OFF_H bit4 = full-off: kills every PWM output in one write.
        # No pulses at all -> the ESC stops; strictly safer than armed-neutral
        # for any exit path.
        self.bus.write_byte_data(self.addr, 0xFD, 0x10)


pca = PCA9685(I2C_BUS, PCA_ADDR)
pca.set_pulse_us(STEER_CH, STEER_NEUTRAL)
pca.set_pulse_us(THROTTLE_CH, THROTTLE_NEUTRAL)
state_lock = threading.Lock()
last_cmd_time = time.time()
# Operator deadman: None = off (tethered Q6A behaviour, unchanged). Set by
# --operator-port; last_op_time starts at 0 so nothing moves before the first
# heartbeat arrives.
operator_timeout_s = None
last_op_time = 0.0


def _operator_ok():
    return operator_timeout_s is None or time.time() - last_op_time <= operator_timeout_s


def neutral():
    pca.set_pulse_us(STEER_CH, STEER_NEUTRAL)
    pca.set_pulse_us(THROTTLE_CH, THROTTLE_NEUTRAL)


def _safe_exit(*_):
    sys.exit(0)


def _outputs_off_at_exit():
    try:
        pca.all_outputs_off()
    except OSError:
        pass


atexit.register(_outputs_off_at_exit)
signal.signal(signal.SIGTERM, _safe_exit)
signal.signal(signal.SIGINT, _safe_exit)


def watchdog_loop():
    # The failsafe thread must survive anything: an I2C hiccup here must not
    # kill it, and its writes must hold state_lock so they can't interleave
    # with the control thread's 4-register channel updates.
    global last_cmd_time
    while True:
        time.sleep(0.05)
        try:
            with state_lock:
                stale = time.time() - last_cmd_time > WATCHDOG_TIMEOUT_S
                if stale or not _operator_ok():
                    neutral()
        except OSError:
            pass


def control_server():
    global last_cmd_time
    srv = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    srv.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    srv.bind(("0.0.0.0", CONTROL_PORT))
    srv.listen(1)
    print(f"control server listening on {CONTROL_PORT}")
    while True:
        conn, addr = srv.accept()
        print("control client connected:", addr)
        buf = b""
        try:
            while True:
                data = conn.recv(1024)
                if not data:
                    break
                buf += data
                while b"\n" in buf:
                    line, buf = buf.split(b"\n", 1)
                    try:
                        steer_us, throttle_us = (int(x) for x in line.decode().split(","))
                        with state_lock:
                            # Gate HERE, not only in the watchdog: a 20 Hz command
                            # stream against a 20 Hz watchdog would otherwise
                            # alternate command/neutral and the car would stutter on.
                            if not _operator_ok():
                                continue
                            pca.set_pulse_us(STEER_CH, steer_us)
                            pca.set_pulse_us(THROTTLE_CH, throttle_us)
                            last_cmd_time = time.time()
                    except ValueError:
                        pass
        except OSError:
            pass
        finally:
            conn.close()
            try:
                neutral()
            except OSError:
                pass
            print("control client disconnected -> neutral")


def operator_server(port):
    """Any byte from the operator refreshes the deadman; EOF drops it at once."""
    global last_op_time
    srv = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    srv.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    srv.bind(("0.0.0.0", port))
    srv.listen(1)
    print(f"operator deadman listening on {port}")
    while True:
        conn, addr = srv.accept()
        # A laptop that left WiFi never sends EOF; without this the single-client
        # accept loop sits on the dead socket and a restarted `nc` never gets in.
        conn.settimeout(2.0)
        print("operator connected:", addr)
        try:
            while conn.recv(64):
                last_op_time = time.time()
        except OSError:
            pass
        finally:
            conn.close()
            last_op_time = 0.0
            print("operator gone -> deadman open, outputs neutral")


COCKPIT_HTML = b"""<!doctype html><meta name=viewport content="width=device-width,initial-scale=1">
<title>DANIEL cockpit</title><style>
body{margin:0;background:#111;color:#eee;font:18px system-ui,sans-serif;text-align:center;
 -webkit-user-select:none;user-select:none;-webkit-touch-callout:none}
img{width:100%;max-width:768px;display:block;margin:auto}
#b{width:92%;height:34vh;margin:2vh 0 1vh;font-size:1.8em;border:0;border-radius:1em;
 background:#833;color:#fff;touch-action:none}#b.on{background:#1a7f37}
</style><img src="/pov.mjpg"><button id=b>HOLD TO DRIVE</button><div id=s>stopped</div>
<script>
// The car moves only while this button is held AND each 100 ms heartbeat arrives.
// Release, screen lock, tab switch or a dead link all simply stop the beats.
const b=document.getElementById('b'),s=document.getElementById('s');let t=null;
function beat(){fetch('/hb',{method:'POST',cache:'no-store'})
 .then(r=>r.text()).then(age=>{s.textContent='HOLDING - view '+age+' s old'})
 .catch(()=>{s.textContent='LINK LOST - car stopping'})}
function go(e){e.preventDefault();if(t)return;b.setPointerCapture(e.pointerId);
 beat();t=setInterval(beat,100);b.className='on'}
function stop(){if(!t)return;clearInterval(t);t=null;b.className='';s.textContent='stopped'}
b.addEventListener('pointerdown',go);
['pointerup','pointercancel','lostpointercapture'].forEach(n=>b.addEventListener(n,stop));
document.addEventListener('visibilitychange',()=>{if(document.hidden)stop()});
addEventListener('blur',stop);b.addEventListener('contextmenu',e=>e.preventDefault());
</script>"""


def cockpit_server(port, pov_path):
    """Phone front-end for the SAME deadman: POST /hb refreshes last_op_time exactly
    as a byte on --operator-port does. Also serves the policy's view as MJPEG."""
    from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

    def pov_age():
        try:
            return time.time() - os.path.getmtime(pov_path)
        except OSError:
            return float("inf")

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *_):
            pass

        def _reply(self, code, body, ctype):
            self.send_response(code)
            self.send_header("Content-Type", ctype)
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(body)

        def do_POST(self):
            global last_op_time
            if self.path != "/hb":
                return self._reply(404, b"", "text/plain")
            last_op_time = time.time()
            # the page shows this, so a frozen view is never mistaken for a live one
            self._reply(200, f"{pov_age():.1f}".encode(), "text/plain")

        def do_GET(self):
            if self.path != "/pov.mjpg":
                return self._reply(200, COCKPIT_HTML, "text/html")
            self.send_response(200)
            self.send_header("Content-Type", "multipart/x-mixed-replace; boundary=f")
            self.end_headers()
            try:
                while True:
                    try:
                        with open(pov_path, "rb") as f:
                            jpg = f.read()
                        self.wfile.write(b"--f\r\nContent-Type: image/jpeg\r\n"
                                         b"Content-Length: %d\r\n\r\n" % len(jpg)
                                         + jpg + b"\r\n")
                    except FileNotFoundError:
                        pass                      # policy not started yet
                    time.sleep(0.1)
            except OSError:
                pass                              # phone went away

    print(f"cockpit on http://0.0.0.0:{port}  (view: {pov_path})")
    ThreadingHTTPServer(("0.0.0.0", port), Handler).serve_forever()


def video_server():
    # video-only deps: the Pi has no RealSense and its --no-video PWM process
    # needs nothing but smbus2
    import cv2
    import numpy as np
    import pyrealsense2 as rs
    pipe = rs.pipeline()
    cfg = rs.config()
    # 640x360, NOT 640x480 -- this is a sim2real CORRECTNESS setting, not a
    # bandwidth one. The D435's colour sensor is natively 16:9 (1920x1080,
    # 69.4 x 42.5 deg). A 4:3 mode like 640x480 is a horizontally cropped view
    # with a NARROWER horizontal FOV, and `dav2_preproc.preprocess_numpy` then
    # resizes it to the model's 504x280 with `cv2.resize`, which SQUASHES rather
    # than crops. The policy trains on a 504x280 (1.8:1) sim camera at focal
    # 15.13 mm = 69.4 x 42.1 deg (arena_env_cfg.py:229-241) and learns an
    # angle->pixel mapping, so a squashed frame mis-scales every bearing it
    # reads. 640x360 is 1.778:1 and is a scaled -- not cropped -- view of the
    # full 16:9 sensor, so it carries the same FOV the sim is calibrated to.
    # VERIFY ON THE REAL CAMERA before the first policy run: print the intrinsics
    # of both modes and confirm 640x360 reports the 69.4 deg horizontal FOV.
    cfg.enable_stream(rs.stream.color, 640, 360, rs.format.bgr8, 30)
    pipe.start(cfg)
    srv = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    srv.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    srv.bind(("0.0.0.0", VIDEO_PORT))
    srv.listen(1)
    print(f"video server listening on {VIDEO_PORT}")
    while True:
        conn, addr = srv.accept()
        # A consumer that stops reading mid-frame must not wedge this loop: the
        # writability check above cannot guarantee a whole 73 KB frame fits.
        conn.settimeout(2.0)
        # Bound the queue to a few frames. `select` writability only means "a
        # write will not block": a socket with megabytes of free buffer is
        # writable even when nothing is reaching the air, so the drop rule below
        # was preventing BLOCKING, not QUEUING. Measured 2026-08-24 over WiFi:
        # 265 frames -- 8.8 s of backlog -- arriving in a single burst after a
        # stall. With a small send buffer, writability tracks the LINK, and the
        # drop rule actually drops.
        conn.setsockopt(socket.SOL_SOCKET, socket.SO_SNDBUF, 128 * 1024)
        print("video client connected:", addr)
        try:
            while True:
                try:
                    frames = pipe.wait_for_frames(1000)
                except RuntimeError:
                    # camera stall must not kill the process (this is the main
                    # thread -- dying here would strand the PWM at last value)
                    continue
                # Drain the SDK queue to the NEWEST frame. wait_for_frames hands
                # back the oldest queued one, so a consumer slower than 30 fps
                # reads progressively staler images while every per-hop latency
                # number still looks perfect.
                while True:
                    newer = pipe.poll_for_frames()
                    if not newer:
                        break
                    frames = newer
                color = frames.get_color_frame()
                if not color:
                    continue
                img = np.asanyarray(color.get_data())
                ok, jpg = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 80])
                if not ok:
                    continue
                payload = jpg.tobytes()
                # DROP, never queue. A 20 Hz policy against a 30 fps camera
                # accumulates ~10 frames/s in the kernel socket buffers; at the
                # measured 73 KB/frame that is seconds of backlog within a
                # minute, and the policy then acts on a stale image. If the
                # consumer is not ready to read right now, this frame is thrown
                # away rather than buffered -- freshness beats completeness for
                # a control loop.
                readable, writable, _ = select.select([conn], [conn], [], 0)
                if readable:
                    # A peer that closed while our send buffer was FULL never
                    # becomes writable again, so the drop rule below would spin
                    # here forever -- never writing, never noticing the client
                    # left -- and every later client would hang against a server
                    # that is still listening. Observed 2026-08-24: port 6000 held
                    # in CLOSE-WAIT, and the next connection timed out. We never
                    # otherwise read this socket, so EOF is the only close signal.
                    if not conn.recv(1):
                        break
                if not writable:
                    continue
                conn.sendall(struct.pack(">I", len(payload)) + payload)
        except OSError:
            pass
        finally:
            conn.close()
            print("video client disconnected")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--no-video", action="store_true",
                    help="PWM only; the camera belongs to the onboard policy")
    ap.add_argument("--operator-port", type=int, default=None,
                    help="require a remote heartbeat on this port before any PWM passes")
    ap.add_argument("--operator-timeout", type=float, default=WATCHDOG_TIMEOUT_S,
                    help="heartbeat age that stops the car (s). Calibration knob: "
                         "WiFi stalls stop the car too, which is the point")
    ap.add_argument("--cockpit-port", type=int, default=None,
                    help="phone cockpit: live policy view + HOLD TO DRIVE deadman")
    ap.add_argument("--pov", default="/dev/shm/pov.jpg",
                    help="the view drive_policy.py --pov writes")
    args = ap.parse_args()
    if args.operator_port is not None or args.cockpit_port is not None:
        operator_timeout_s = args.operator_timeout
    if args.operator_port is not None:
        threading.Thread(target=operator_server, args=(args.operator_port,),
                         daemon=True).start()
    if args.cockpit_port is not None:
        threading.Thread(target=cockpit_server, args=(args.cockpit_port, args.pov),
                         daemon=True).start()
    threading.Thread(target=watchdog_loop, daemon=True).start()
    if args.no_video:
        control_server()
    else:
        threading.Thread(target=control_server, daemon=True).start()
        video_server()
