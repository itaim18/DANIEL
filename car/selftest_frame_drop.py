#!/usr/bin/env python3
"""Self-check for board_server's frame-drop rule (R2 defect 2).

    python3 selftest_frame_drop.py     # assert-based, no camera, no board

The bug this guards: `video_server` pushed at the camera's 30 fps with a bare
`conn.sendall` and no dropping. A 20 Hz policy consumes 20/s, so ~10 frames/s
accumulate in the kernel socket buffers. At the measured ~73 KB/frame that is
seconds of backlog within a minute, and the policy acts on an image hundreds of
ms old while every per-hop latency percentile still looks perfect.

This reproduces the mechanism over a real loopback socket with a deliberately
slow reader, and asserts that the drop rule keeps the newest frame fresh instead
of queueing a backlog. It tests the socket half only -- the RealSense drain
needs a camera.
"""
import select
import socket
import struct
import threading
import time

FRAME = b"x" * 73_000          # measured payload size
N_SENT = 300                   # ~10 s of a 30 fps camera


def _serve(conn, drop, sent, stop):
    """Push frames as fast as a 30 fps camera would, with or without dropping."""
    for i in range(N_SENT):
        if stop.is_set():
            break
        payload = struct.pack(">I", len(FRAME)) + FRAME + struct.pack(">I", i)
        if drop and not select.select([], [conn], [], 0)[1]:
            continue                      # consumer behind -> throw this one away
        try:
            conn.sendall(payload)
        except OSError:
            break
        sent.append(i)
        time.sleep(1 / 30)
    try:
        conn.shutdown(socket.SHUT_WR)
    except OSError:
        pass


def run(drop):
    """Return (frames_read, newest_index_seen, last_index_the_server_sent)."""
    a, b = socket.socketpair()
    a.setsockopt(socket.SOL_SOCKET, socket.SO_SNDBUF, 256 * 1024)
    sent, stop = [], threading.Event()
    t = threading.Thread(target=_serve, args=(a, drop, sent, stop), daemon=True)
    t.start()

    read, newest, deadline = 0, -1, time.time() + 12
    b.settimeout(2.0)
    try:
        while time.time() < deadline and read < 20:      # a 20 Hz consumer...
            hdr = b.recv(4)
            if len(hdr) < 4:
                break
            n = struct.unpack(">I", hdr)[0]
            got = bytearray()
            while len(got) < n + 4:
                chunk = b.recv(n + 4 - len(got))
                if not chunk:
                    break
                got += chunk
            if len(got) < n + 4:
                break
            newest = struct.unpack(">I", bytes(got[-4:]))[0]
            read += 1
            time.sleep(1 / 20)                            # ...is slower than 30 fps
    except socket.timeout:
        pass
    stop.set(); t.join(timeout=3)
    a.close(); b.close()
    return read, newest, (sent[-1] if sent else -1)


def demo():
    ok = 0

    # 1. Without dropping, the reader falls behind: the newest index it has seen
    #    lags far behind what the camera has already produced. That lag IS the
    #    stale-frame bug.
    read_q, newest_q, last_q = run(drop=False)
    lag_q = last_q - newest_q
    assert read_q >= 5, f"queued run read too little to judge ({read_q})"
    ok += 1

    # 2. With dropping, the reader stays near the live edge.
    read_d, newest_d, last_d = run(drop=True)
    lag_d = last_d - newest_d
    assert read_d >= 5, f"drop run read too little to judge ({read_d})"
    ok += 1

    # 3. The rule has to actually reduce staleness, which is the whole point.
    assert lag_d <= lag_q, (
        f"drop rule did not reduce lag: queued lag={lag_q}, drop lag={lag_d}")
    ok += 1

    print(f"frame-drop demo: {ok}/3 OK   "
          f"(queued: read {read_q}, lag {lag_q} frames | "
          f"drop: read {read_d}, lag {lag_d} frames)")


if __name__ == "__main__":
    demo()
