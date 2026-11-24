#!/usr/bin/env bash
# Failsafe layers 1+2 on the board (Q6A or Pi 5), from this directory. Run ON the board:
#
#   sudo bash install_failsafes.sh
#
# L1: systemd pets the hardware watchdog every few s; a frozen SoC reboots in <=10 s.
# L2: every boot force-clears all PCA9685 outputs before anything can drive.
# L3 is board_server.py itself (0.4 s deadman + outputs off on exit); start it with
# run_board_server.sh so a kill -9 cuts the outputs too (L3b).
# Then verify from the laptop: preflight.sh <user>@<board>.
set -euo pipefail
cd "$(dirname "$0")"
command -v i2cset >/dev/null || apt-get install -y i2c-tools iw
# 99-, not 10-: drop-ins apply in filename order across ALL dirs, and Raspberry Pi OS ships
# /usr/lib/systemd/system.conf.d/40-rpi-enable-watchdog.conf (1m), which beat a 10- file.
install -Dm644 10-hw-watchdog.conf /etc/systemd/system.conf.d/99-hw-watchdog.conf
install -Dm644 pwm-safe-boot.service /etc/systemd/system/pwm-safe-boot.service
if grep -qa "Raspberry Pi" /proc/device-tree/model; then
    # Pi 5: the PCA9685 sits on header pins 3/5 = i2c-1 (the unit defaults to the Q6A's 6)
    mkdir -p /etc/systemd/system/pwm-safe-boot.service.d
    printf '[Service]\nEnvironment=I2C_BUS=1\n' > /etc/systemd/system/pwm-safe-boot.service.d/bus.conf
fi
# L3b (run_board_server.sh) is a user unit: linger keeps it alive after SSH logs out
[ -n "${SUDO_USER:-}" ] && loginctl enable-linger "$SUDO_USER"
systemctl daemon-reexec           # re-reads system.conf.d -> RuntimeWatchdogSec
systemctl enable pwm-safe-boot.service
systemctl start pwm-safe-boot.service
systemctl show -p RuntimeWatchdogUSec
systemctl show pwm-safe-boot.service -p Environment -p Result
