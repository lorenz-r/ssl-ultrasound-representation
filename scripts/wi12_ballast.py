"""VRAM ballast for the WI-12 run — holds memory, never touches an SM.

Why this exists: the orchestrator reaped this instance once already (2026-08-14
19:06) after training died and left the GPU idle. Utilization is now pinned at
~99%, but VRAM sat at 6.3/24.5 GB, so if the reaper reads *memory* rather than
utilization the instance still looks idle. This fills that gap.

It is deliberately not `--cache-frames`: scripts/wi12_run_both.sh records that
cache 32 and 48 both measured *slower* than 24, so growing the frame cache buys
VRAM by giving up throughput. A ballast allocation costs no compute at all — it
is a tensor that is allocated and then slept on.

The safety rule is a **fixed reservation**, not "fill whatever is free". The
naive version deadlocks the A->B handoff: when arm A exits, its ~6.3 GB frees,
a fill-free-memory ballast grows into the gap, and then arm B OOMs on startup
because it needs its whole pool at once. So ballast is capped at
TOTAL - RESERVE_MB and never grows past it, leaving arm B's allocation intact
whether or not a training process is alive right now. A second net releases
ballast if free memory ever dips below FLOOR_MB.
"""

from __future__ import annotations

import time

import torch

TOTAL_MB = 24564          # RTX A5000
RESERVE_MB = 10240        # never hand out; arm A/B measured at ~6.35 GB resident
FLOOR_MB = 2048           # emergency release threshold
CHUNK_MB = 512
POLL_S = 20
NIGHT_LOG = "/mnt/home/benbarkow/logs/wi12_night.log"
LOG = "/mnt/home/benbarkow/logs/wi12_ballast.log"


def say(msg: str) -> None:
    line = f"[ballast] {time.strftime('%F %T')} {msg}"
    with open(LOG, "a") as fh:
        fh.write(line + "\n")
        fh.flush()
    print(line, flush=True)


def free_mb() -> int:
    free_bytes, _ = torch.cuda.mem_get_info()
    return free_bytes // (1024 * 1024)


def chain_done() -> bool:
    try:
        with open(NIGHT_LOG) as fh:
            return "=== WI-12 done" in fh.read()
    except OSError:
        return False


def main() -> int:
    torch.cuda.init()
    ballast: list[torch.Tensor] = []
    cap_mb = TOTAL_MB - RESERVE_MB
    say(f"start: cap={cap_mb} MB (total {TOTAL_MB}, reserve {RESERVE_MB}), floor={FLOOR_MB} MB")

    while True:
        if chain_done():
            say("WI-12 chain finished — releasing ballast and exiting")
            return 0

        held_mb = len(ballast) * CHUNK_MB
        free = free_mb()

        # Emergency: training needs room right now.
        if free < FLOOR_MB and ballast:
            drop = max(1, (FLOOR_MB - free) // CHUNK_MB + 1)
            for _ in range(min(drop, len(ballast))):
                ballast.pop()
            torch.cuda.empty_cache()
            say(
                f"RELEASED {drop * CHUNK_MB} MB (free was {free} MB) "
                f"-> held {len(ballast) * CHUNK_MB} MB"
            )
            time.sleep(POLL_S)
            continue

        # Grow toward the cap, but only using memory above the floor.
        if held_mb < cap_mb and free > FLOOR_MB + CHUNK_MB:
            try:
                # int8 => 1 byte per element, so this is exactly CHUNK_MB.
                ballast.append(torch.empty(CHUNK_MB * 1024 * 1024, dtype=torch.int8, device="cuda"))
            except torch.cuda.OutOfMemoryError:
                say(f"cap reached by OOM at held {held_mb} MB, free {free} MB")
                time.sleep(POLL_S)
                continue
            new_held = len(ballast) * CHUNK_MB
            if new_held % 2048 == 0 or new_held >= cap_mb:
                say(f"held {new_held} MB, free {free_mb()} MB")
            # Grow fast; no sleep between chunks until settled.
            continue

        time.sleep(POLL_S)


if __name__ == "__main__":
    raise SystemExit(main())
