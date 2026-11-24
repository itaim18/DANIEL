#!/usr/bin/env bash
# Start board_server.py under systemd so ANY death cuts the PWM -- failsafe L3b.
#
#   I2C_BUS=1 ./run_board_server.sh --no-video [--operator-port 6002 --cockpit-port 8080]
#   systemctl --user stop board-server          # stop (outputs cut)
#   journalctl --user -u board-server -f        # its log
#
# board_server's own atexit/SIGTERM handler cannot run on SIGKILL or an OOM kill.
# Measured on the Pi 5, 2026-10-07: kill -9 while steering left ch0 LATCHED at 1851 us
# with the board healthy, so the hardware watchdog never fires either. ExecStopPost runs
# after the main process exits for any reason, SIGKILL included.
# Needs the user in group i2c and linger on (install_failsafes.sh), else the unit
# stops -- outputs cut -- when the last SSH session closes.
set -euo pipefail
cd "$(dirname "$0")"
BUS=${I2C_BUS:-6}
exec systemd-run --user --unit=board-server --collect --same-dir \
    --setenv=I2C_BUS="$BUS" --setenv=PYTHONUNBUFFERED=1 \
    -p "ExecStopPost=/usr/sbin/i2cset -y $BUS 0x40 0xFD 0x10" \
    /usr/bin/python3 "$PWD/board_server.py" "$@"
