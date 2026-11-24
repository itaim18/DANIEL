#!/usr/bin/env bash
# Boot services as systemd USER units (linger keeps them running with nobody logged in).
#   on the Pi:      onboard/install_services.sh           -> board-server + cockpit + pi-recorder, up at power-on
#   on the laptop:  onboard/install_services.sh --laptop  -> daniel-recorder + daniel-sensor-sync
# Logs: journalctl --user -u <unit> -f.  Stop: systemctl --user stop <unit> (board-server: PWM cut).
set -euo pipefail
cd "$(dirname "$0")/systemd"
if [ "${1:-}" = "--laptop" ]; then units="daniel-recorder.service daniel-sensor-sync.service"; else units="board-server.service cockpit.service pi-recorder.service"; fi
systemctl --user stop $units 2>/dev/null || true      # also clears transient systemd-run units of the same name
mkdir -p ~/.config/systemd/user
cp $units ~/.config/systemd/user/
systemctl --user daemon-reload
systemctl --user enable --now $units
loginctl show-user "$USER" -p Linger
for u in $units; do echo "$u: $(systemctl --user is-active $u) / $(systemctl --user is-enabled $u)"; done
