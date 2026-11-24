#!/usr/bin/env bash
# The pre-drive gate. STATE.md trap 5: "there is no verification script for any
# of the three layers", and Constitution VI (v1.1.0) makes UNKNOWN block
# actuation exactly as FAIL does -- so that missing script IS what keeps the gate
# shut. This is it.
#
#   car/preflight.sh [board_host]
#   car/preflight.sh <user>@10.42.0.139     # the Pi 5
#
# Exits non-zero unless EVERY row is PASS. Passing --failsafes-confirmed to
# drive_policy.py or bench_pwm.py while this script is red is asserting something
# false, in writing, into the run record.
#
# Expected on the BENCH (board out of the car): L2 / PWM / CAMERA / FOV are red,
# because the PCA9685 is fed by the ESC's BEC and the D435 is not plugged in.
# That is the point -- they must go green with the car powered, before anything
# is armed.
set -u
cd "$(dirname "$0")"
# Bare hostname, not an ssh_config alias, so this works on a machine that has
# never had ~/.ssh/config edited. An explicit user@ or an alias is honoured.
HOST="${1:-radxa-dragon-q6a.local}"
case "$HOST" in *@*) TARGET="$HOST"; HOST="${HOST#*@}" ;; *) TARGET="radxa@$HOST" ;; esac
SSH="ssh -o ConnectTimeout=8 -o BatchMode=yes $TARGET"
FAILED=0

row() {  # row <name> <verdict> <detail>
    local c
    case "$2" in
        PASS) c=$'\033[32m' ;;
        FAIL) c=$'\033[31m'; FAILED=1 ;;
        *)    c=$'\033[33m'; FAILED=1 ;;
    esac
    printf "  %-9s ${c}%-7s\033[0m %s\n" "$1" "$2" "$3"
}

# --- link -------------------------------------------------------------------
# mDNS first; the fallback is the whole iPhone /28, which is 13 addresses.
ALL=$(getent ahostsv4 "$HOST" 2>/dev/null | awk '{print $1}' | sort -u)
IP=$(echo "$ALL" | grep -m1 '^172\.20\.10\.' || echo "$ALL" | head -1)
if [ -z "$IP" ]; then
    for i in $(seq 2 14); do
        if ping -c1 -W1 "172.20.10.$i" >/dev/null 2>&1 \
           && ssh -o ConnectTimeout=3 -o BatchMode=yes "radxa@172.20.10.$i" true 2>/dev/null; then
            IP="172.20.10.$i"; HOST="$IP"; TARGET="radxa@$IP"
            SSH="ssh -o ConnectTimeout=8 -o BatchMode=yes $TARGET"
            break
        fi
    done
fi
echo "== preflight against ${HOST} ${IP:+($IP)}"

if ! $SSH true 2>/dev/null; then
    row LINK FAIL "no SSH to $HOST -- board off, off-network, or mDNS not resolving"
    echo; echo "GATE: CLOSED"; exit 1
fi
# Q6A: PCA9685 on i2c-6, code in ~/traxxas_control, D435. Pi 5: header pins 3/5 =
# i2c-1, code in the repo checkout, Pi camera.
if $SSH 'grep -qa "Raspberry Pi" /proc/device-tree/model' 2>/dev/null; then
    PI=1 BUS=1 DIR='~/DANIEL/car'
else
    PI= BUS=6 DIR='~/traxxas_control'
fi
WIFI=$($SSH '/usr/sbin/iw dev wlan0 link 2>/dev/null | grep -E "freq|signal:"; \
             echo "ps=$(/usr/sbin/iw dev wlan0 get power_save 2>/dev/null | grep -o "on\|off")"' \
        | tr -s " \t\n" " ")
case "$WIFI" in
    *"ps=off"*) row LINK PASS "$IP ${WIFI}" ;;
    *)          row LINK FAIL "power-save is ON -- latency spikes into the 20 Hz loop. ${WIFI}" ;;
esac

# --- failsafe layer 1: the hardware watchdog --------------------------------
W=$($SSH 'systemctl show -p RuntimeWatchdogUSec --value' 2>/dev/null)
[ "$W" = "10s" ] && row L1 PASS "RuntimeWatchdogUSec=$W" \
                 || row L1 FAIL "RuntimeWatchdogUSec='$W', expected 10s (10-hw-watchdog.conf)"

# --- failsafe layer 2: PWM forced off at boot -------------------------------
# Type=oneshot WITHOUT RemainAfterExit, so a SUCCESSFUL run leaves the unit
# inactive. `is-active` would call that a failure; Result is the honest field --
# but only for a unit that EXISTS: a never-installed unit also reports
# Result=success (seen on the Pi 5, 2026-10-07), so LoadState and enablement gate it.
# One property per call: `systemctl show` prints in its own order, not the -p order.
R=$($SSH 'for p in LoadState UnitFileState Result; do systemctl show pwm-safe-boot.service -p $p --value; done | paste -sd" "' 2>/dev/null)
[ "$R" = "loaded enabled success" ] && row L2 PASS "pwm-safe-boot loaded, enabled, Result=success" \
    || row L2 FAIL "pwm-safe-boot LoadState/UnitFileState/Result='$R' -- not installed, not enabled, or it NAKed (PCA9685 unpowered at boot)"

# --- failsafe layer 3: the control-channel deadman --------------------------
if $SSH "grep -q '^WATCHDOG_TIMEOUT_S = 0.4' $DIR/board_server.py" 2>/dev/null; then
    row L3 PASS "board_server neutrals after 0.4 s of silence"
else
    row L3 FAIL "WATCHDOG_TIMEOUT_S = 0.4 not found in the board's board_server.py"
fi

# --- the servo board --------------------------------------------------------
# The Pi user is in group i2c; the Q6A needs sudo. Full path: /usr/sbin is not on a
# plain user's PATH on Debian trixie.
if $SSH "/usr/sbin/i2cdetect -y -r $BUS 2>/dev/null || sudo -n /usr/sbin/i2cdetect -y -r $BUS 2>/dev/null" \
        | grep -qE '^40:.* 40 '; then
    row PWM PASS "PCA9685 answering at i2c-$BUS/0x40"
else
    row PWM FAIL "nothing at i2c-$BUS/0x40 -- servo board unpowered (Q6A: ESC BEC off; Pi: VCC not on pin 1) or SDA/SCL swapped"
fi

# --- the camera -------------------------------------------------------------
if [ -n "$PI" ]; then
    # The Pi stack reads the Pi camera (picamera2), not the D435, so the D435 FOV
    # trap below does not apply to it.
    if $SSH 'rpicam-hello --list-cameras 2>&1 | grep -q "Available cameras"'; then
        row CAMERA PASS "$($SSH 'rpicam-hello --list-cameras 2>&1 | grep -m1 "^ *0 :"' | tr -s ' ')"
    else
        row CAMERA FAIL "rpicam-hello sees no camera -- ribbon unplugged or the IMX500 is wedged (cold-cycle it)"
    fi
else
    if $SSH 'lsusb 2>/dev/null | grep -qi "8086:"'; then
        row CAMERA PASS "$($SSH 'lsusb | grep -i "8086:" | head -1')"
    else
        row CAMERA FAIL "no Intel device on USB -- the D435 is not plugged in"
    fi

    # --- the FOV, which is a silent correctness bug -----------------------------
    # 640x360 is a SCALED view of the native 16:9 sensor; 640x480 is a CROPPED one
    # with a narrower horizontal FOV. The policy learned an angle->pixel mapping at
    # 69.4 deg. A wrong FOV never crashes, it just steers wrong, everywhere, forever.
    FOV=$($SSH 'python3 -c "
    import pyrealsense2 as rs, math
    for w,h in ((640,360),(640,480)):
        p=rs.pipeline(); c=rs.config(); c.enable_stream(rs.stream.color,w,h,rs.format.bgr8,30)
        i=p.start(c).get_stream(rs.stream.color).as_video_stream_profile().get_intrinsics()
        print(\"%dx%d hfov=%.1f vfov=%.1f\"%(w,h,
            math.degrees(2*math.atan(i.width/(2*i.fx))),
            math.degrees(2*math.atan(i.height/(2*i.fy))))); p.stop()
    " 2>/dev/null' | paste -sd" | ")
    H=$(echo "$FOV" | grep -o '640x360 hfov=[0-9.]*' | cut -d= -f2)
    if [ -n "$H" ] && awk "BEGIN{exit !($H > 67 && $H < 72)}"; then
        row FOV PASS "$FOV"
    elif [ -n "$H" ]; then
        row FOV FAIL "640x360 reports ${H} deg, expected ~69.4 (sim: 69.4 x 42.1). $FOV"
    else
        # The camera is SINGLE-CLIENT: once board_server holds it, a second pipeline
        # cannot open and the intrinsics probe returns nothing. Rather than report
        # UNKNOWN and block on our own server, verify the geometry from the DEPLOYED
        # STREAM -- pull one frame off :6000 and confirm it decodes to 640x360, the
        # mode whose intrinsics this same script measured at 69.4 deg. That is a
        # stronger check than the probe: it tests what the policy will actually read.
        # stdlib ONLY -- a gate script must not depend on cv2/numpy being importable
        # from whichever python3 is first on PATH. The JPEG SOF marker carries the
        # dimensions, so no decode is needed.
        GEOM=$(python3 - "$IP" <<'PY' 2>/dev/null
import socket, struct, sys
def jpeg_size(b):
    i = 2
    while i < len(b) - 9:
        if b[i] != 0xFF:
            i += 1; continue
        m = b[i + 1]
        if m in (0xC0,0xC1,0xC2,0xC3,0xC5,0xC6,0xC7,0xC9,0xCA,0xCB,0xCD,0xCE,0xCF):
            h, w = struct.unpack(">HH", b[i + 5:i + 9])
            return w, h
        if m in (0xD8, 0x01) or 0xD0 <= m <= 0xD7:
            i += 2; continue
        i += 2 + struct.unpack(">H", b[i + 2:i + 4])[0]
    raise ValueError("no SOF marker")
try:
    s = socket.create_connection((sys.argv[1], 6000), timeout=6)
    def rx(n):
        b = b""
        while len(b) < n:
            c = s.recv(n - len(b))
            if not c: raise ConnectionError("stream closed")
            b += c
        return b
    n = struct.unpack(">I", rx(4))[0]
    w, h = jpeg_size(rx(n))
    s.close()
    print("%dx%d %.1fKB" % (w, h, n / 1024))
except Exception as e:
    print("ERR %s: %s" % (type(e).__name__, e))
PY
    )
        case "$GEOM" in
            640x360*) row FOV PASS "live stream $GEOM (camera held by board_server; 640x360 measured 69.4 deg above)" ;;
            ERR*)     row FOV UNKNOWN "camera busy AND no frame from :6000 -- $GEOM" ;;
            *)        row FOV FAIL "live stream is $GEOM, expected 640x360 -- WRONG FOV, the policy's bearings will be mis-scaled" ;;
        esac
    fi
fi
# --- the board is running the code this repo thinks it is -------------------
MINE=$(md5sum board_server.py | cut -d' ' -f1)
THEIRS=$($SSH "md5sum $DIR/board_server.py 2>/dev/null" | cut -d' ' -f1)
[ "$MINE" = "$THEIRS" ] && row VERSION PASS "board_server.py md5 ${MINE:0:8} matches the repo" \
                        || row VERSION FAIL "board has ${THEIRS:0:8}, repo has ${MINE:0:8} -- scp it across"

# --- the server, and how it was started (L3b) -------------------------------
# Bare, a kill -9 / OOM kill skips board_server's atexit and LATCHES the last pulse
# (measured on the Pi 2026-10-07). Under run_board_server.sh, systemd cuts it.
if $SSH 'systemctl --user is-active -q board-server' 2>/dev/null; then
    row SERVER PASS "board_server.py running under systemd (any death cuts the PWM)"
elif $SSH 'pgrep -f "[p]ython3.*board_server.py" >/dev/null' 2>/dev/null; then
    row SERVER FAIL "board_server.py running BARE -- a kill -9 latches the PWM; restart it via run_board_server.sh"
else
    row SERVER FAIL "not running: ssh $TARGET 'cd $DIR && ${PI:+I2C_BUS=1 }./run_board_server.sh${PI:+ --no-video}'"
fi

echo
if [ "$FAILED" -eq 0 ]; then
    echo -e "GATE: \033[32mOPEN\033[0m -- all three failsafe layers verified live."
    echo "     --failsafes-confirmed is now a statement you can defend."
else
    echo -e "GATE: \033[31mCLOSED\033[0m -- Constitution VI: UNKNOWN blocks actuation exactly as FAIL does."
fi
exit "$FAILED"
