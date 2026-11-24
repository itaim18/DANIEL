#!/usr/bin/env bash
# Rebuild daniel-pi5 (Pi 5 8 GB + Hailo-8L + D435 + PCA9685) on a fresh Raspberry Pi OS Lite (trixie, arm64).
# Written 2026-11-22 when the old SD card died and none of this was written down. Idempotent: re-run after a failure.
#
#   laptop:  card flashed with Raspberry Pi Imager (your user, SSH key, WiFi)
#   laptop:  git ls-files -z | rsync -a --from0 --files-from=- ./ <user>@<pi>:DANIEL/   (from the repo root)
#            rsync -a <HEFs, yard_terrain.npz> <user>@<pi>:models/
#   Pi:      bash ~/DANIEL/onboard/setup_pi.sh && sudo reboot
#   laptop:  car/preflight.sh <user>@<pi>
set -euxo pipefail
R=~/DANIEL
export DEBIAN_FRONTEND=noninteractive
sudo apt-get update
sudo -E apt-get full-upgrade -y
sudo -E apt-get install -y hailo-all i2c-tools iw v4l-utils ffmpeg rsync git curl tmux \
    python3-venv python3-pip python3-numpy python3-opencv python3-smbus2 python3-scipy python3-picamera2 rpicam-apps-lite \
    linux-headers-rpi-2712 tcpdump python3-tk
# hailort-pcie-driver builds hailo_pci only for the RUNNING kernel: after full-upgrade pulls a new kernel, the
# reboot has no /dev/hailo0 (HAILO_OUT_OF_PHYSICAL_DEVICES, 2026-11-22). Build it for every installed 2712 kernel.
for k in /lib/modules/*-rpi-2712; do sudo /usr/sbin/dkms autoinstall -k "$(basename "$k")" || true; done

# PCA9685 on header pins 3/5 = i2c-1; Hailo on PCIe Gen3 x1 (experiments/2026-09-28_pi5_edge_depth)
C=/boot/firmware/config.txt
for l in dtparam=i2c_arm=on dtparam=pciex1_gen=3; do grep -qx "$l" $C || echo "$l" | sudo tee -a $C; done
echo i2c-dev | sudo tee /etc/modules-load.d/i2c-dev.conf
# POWER_OFF_ON_HALT=1 makes `echo +20 > /sys/class/rtc/rtc0/wakealarm; halt` a remote cold power cycle (STATE.md)
if ! sudo rpi-eeprom-config | grep -qx 'POWER_OFF_ON_HALT=1'; then
    sudo rpi-eeprom-config | sed '/^POWER_OFF_ON_HALT/d' > /tmp/ee.txt
    echo POWER_OFF_ON_HALT=1 >> /tmp/ee.txt
    sudo rpi-eeprom-config --apply /tmp/ee.txt
fi

# WiFi power save OFF: preflight's LINK row fails on it (latency spikes into the 20 Hz loop)
printf '[connection]\nwifi.powersave = 2\n' | sudo tee /etc/NetworkManager/conf.d/99-wifi-powersave-off.conf
# Persistent journal: Raspberry Pi OS ships Storage=volatile, so two unexplained drops on 2026-11-22 left no log
sudo mkdir -p /etc/systemd/journald.conf.d
printf '[Journal]\nStorage=persistent\n' | sudo tee /etc/systemd/journald.conf.d/99-persistent.conf
sudo systemd-tmpfiles --create --prefix /var/log/journal && sudo systemctl restart systemd-journald
# REC ALL: steady SD writeback, no 2-3 s flush stalls
sudo cp $R/onboard/90-rec-all-writeback.conf /etc/sysctl.d/ && sudo sysctl --system

# daniel-venv runs cockpit + recorder; --system-site-packages so it sees apt's hailo_platform, numpy, cv2.
[ -x ~/daniel-venv/bin/python ] || python3 -m venv --system-site-packages ~/daniel-venv
~/daniel-venv/bin/pip install onnxruntime
# ponytail: torch skipped (only edge_depth --torch / the CPU bench ladder); add with
#   ~/daniel-venv/bin/pip install torch --index-url https://download.pytorch.org/whl/cpu

# Failsafes L1 (hw watchdog 10 s) + L2 (PWM off at boot) + linger, then the boot services
# Its last step starts pwm-safe-boot, which fails until the PCA9685 is wired and i2c-1 is on (after the
# reboot). Everything is installed and enabled by then; preflight's L2 row re-checks it on the car.
sudo bash $R/car/install_failsafes.sh || echo "pwm-safe-boot: PCA9685 not answering yet (unwired / pre-reboot)"
bash $R/onboard/install_services.sh

command -v tailscale >/dev/null || curl -fsSL https://tailscale.com/install.sh | sh
~/daniel-venv/bin/python -c 'import hailo_platform, cv2, numpy, onnxruntime; print("venv OK")'
hailortcli fw-control identify || true

# System python3 runs board_server.py (video mode) and preflight's FOV probe: both import pyrealsense2.
# No aarch64 wheel for Python 3.13 on PyPI/piwheels and no Debian package (2026-11-22): build it (~35 min).
# NO_RS=1 skips it, so the rest can be rebooted into and tested while it builds on a later run.
if [ -z "${NO_RS:-}" ] && ! python3 -c 'import pyrealsense2' 2>/dev/null; then
    sudo -E apt-get install -y cmake build-essential pkg-config libusb-1.0-0-dev libssl-dev libudev-dev python3-dev
    [ -d ~/librealsense ] || git clone --depth 1 https://github.com/IntelRealSense/librealsense ~/librealsense
    cmake -S ~/librealsense -B ~/librealsense/build -DCMAKE_BUILD_TYPE=Release -DBUILD_PYTHON_BINDINGS=ON \
        -DPYTHON_EXECUTABLE=/usr/bin/python3 -DBUILD_EXAMPLES=OFF -DBUILD_GRAPHICAL_EXAMPLES=OFF -DBUILD_UNIT_TESTS=OFF
    nice -n 19 make -C ~/librealsense/build -j3
    sudo make -C ~/librealsense/build install && sudo ldconfig
    sudo cp ~/librealsense/config/99-realsense-libusb.rules /etc/udev/rules.d/ && sudo udevadm control --reload
    python3 -c 'import pyrealsense2, smbus2; print("system python OK")'
fi
echo "SETUP DONE -- sudo reboot, then: sudo tailscale up --hostname=daniel-pi5"
