# Deployment and database cutover runbook

This is the one-time move from the old layout (`main`: `./trades.db` bind-mounted
as a single file, models inside the container) to the Phase 2 layout
(`./data/` and `./model_artifacts/` host directories, one-shot `migrate`
service, read-only dashboard).

**Nothing in this document has been executed by the authors.** It is written
for a human operator on the Oracle host. Read it to the end before starting.

## Why the order matters

* With the old single-file mount, SQLite's write-ahead log (`trades.db-wal`)
  lives **inside the old bot container**, not on the host. Committed trades
  that were never checkpointed exist only there. `docker compose down`, or
  `up` with the new compose file (which recreates the containers), deletes it.
* The old code ignores SIGTERM, so `docker compose stop` force-kills it after
  10 s. SQLite is crash-safe, so this loses nothing, **provided the log is
  copied out of the stopped container before the container is removed.**
* The final copy is taken only after **every writer is stopped and verified
  stopped**, using SQLite's backup API on copies (`tools/db_maintenance.py
  cutover-import`), with `integrity_check` and per-table row counts verified.
  The original files are never opened for writing.

## 0. Before you start

- [ ] Pick a window when US equity markets are closed (e.g. a weekend). risky2
      (crypto, 24/7) is disabled in `core/config.py`; keep it disabled.
- [ ] In the Alpaca paper dashboard, write down every open position and open
      order. You will compare against these after the cutover.
- [ ] Decide what happens to positions the old bots hold (the new ledger starts
      empty and does **not** manage positions it does not own):
      **(a)** close them in the Alpaca UI before step 2, or
      **(b)** adopt them on the first start (step 7). Each symbol is assigned to
      the strategy whose universe contains it.
- [ ] `docker compose version` works and there is at least 5 GB free (`df -h`).
- [ ] Note the current commit: `git rev-parse HEAD > ~/singhquant-pre-cutover-commit.txt`.

Run every command from the repository directory on the host.

## 1. Copy what is needed out of the RUNNING old stack (read-only)

```bash
mkdir -p cutover/old_models
docker compose cp trading_bots:/app/models/. cutover/old_models/     # .pkl / .zip model files
ls -la cutover/old_models
```

## 2. Stop ALL writers and verify

Still on the **old** checkout (the old compose file defines the old services):

```bash
docker compose stop              # NOT `down`: the stopped containers keep their filesystem (and the WAL)
docker compose ps -a             # trading_bots and dashboard must both show "Exited"
docker ps --format '{{.Names}}'  # nothing from this project may be running
sudo fuser -v trades.db || echo "no host process has trades.db open"
```

Also stop anything else that could open `trades.db` (cron jobs, a `python`
shell, a local dashboard). Do not continue while anything is running.

## 3. Collect the database and the old containers' logs

```bash
BOTS=$(docker compose ps -a -q trading_bots)
DASH=$(docker compose ps -a -q dashboard)
docker cp "$BOTS:/app/trades.db-wal" cutover/bots.trades.db-wal  || echo "bots: no WAL file (already checkpointed)"
docker cp "$DASH:/app/trades.db-wal" cutover/dash.trades.db-wal  || echo "dashboard: no WAL file"
ls -la cutover/
cp -p trades.db cutover/trades.db.orig
sha256sum cutover/trades.db.orig cutover/*.trades.db-wal 2>/dev/null | tee cutover/SHA256SUMS
```

**Stop here if `cutover/dash.trades.db-wal` exists and is larger than 0
bytes.** The dashboard should never have written; two non-empty logs cannot
both be applied. Restart the old stack (`docker compose start`) and investigate
before trying again.

## 4. Switch code and build (no container is started)

```bash
git fetch origin
git checkout claude-infrastructure        # or main, once this work is merged
docker compose build                      # the build runs tools/smoke_imports.py and fails on a broken image
```

## 5. Import the database

```bash
mkdir -p data
docker compose run --rm --no-deps -v "$PWD/cutover:/cutover:ro" migrate \
  python -m tools.db_maintenance cutover-import \
    --db /cutover/trades.db.orig --wal /cutover/bots.trades.db-wal --dest /app/data/trades.db
# (omit --wal if step 3 found no bots WAL file)
docker compose run --rm --no-deps migrate                                  # schema migration (transactional)
docker compose run --rm --no-deps migrate python -m tools.db_maintenance check
```

`cutover-import` prints the row count of every table and refuses to overwrite
an existing destination. `check` must report `quick_check ok` and
`schema 1 (code expects 1)`.

## 6. Register the models

The bots open **no new positions** until every model they need is listed in
`model_artifacts/manifest.json` with a matching SHA-256. Register the models
you copied in step 1. Each register command is a dry run until `--confirm`.

```bash
mkdir -p model_artifacts
for NAME in stable_model.pkl risky1_model.pkl; do
  docker compose run --rm --no-deps -v "$PWD/cutover:/cutover:ro" trading_bots \
    python -m tools.model_manifest register "$NAME" --file "/cutover/old_models/$NAME" \
      --version "prod-legacy-$(date +%Y%m%d)" --note "copied from the old container by <your name>"
done
# if both validate, repeat the loop with --confirm appended, then:
docker compose run --rm --no-deps trading_bots python -m tools.model_manifest status    # exit 0 = ready
```

A model that fails validation (for example the wrong feature width) must not be
registered. Its strategy then runs **degraded**: no new entries, existing
positions still managed, with an alert every 6 hours, until a retrained model
is published.

## 7. Start

If you chose **(b) adopt** in step 0, add `SINGHQUANT_ADOPT_BROKER_POSITIONS=1`
to `.env` for this first start only.

```bash
docker compose up -d          # migrate runs first; bots and dashboard start only if it succeeds
docker compose ps             # migrate: Exited (0); trading_bots/dashboard: Up, then (healthy) after start_period
docker compose logs -f trading_bots   # look for "startup reconciliation complete" and "[models] ... loaded and verified"
```

Then, within the first cycle:

```bash
docker compose exec trading_bots python -m tools.healthcheck report       # alive / ready / degraded per strategy
docker compose exec trading_bots python -m tools.reconcile_position       # ledger discrepancies (should be none)
docker compose exec trading_bots python -m tools.safe_mode status
ls -la data/                                                              # trades.db, trades.db-wal, trades.db-shm together
```

Compare positions and orders with the notes from step 0. If you used adopt,
remove the flag from `.env` and run `docker compose up -d trading_bots`.

## 8. Afterwards

* Keep `cutover/` and the old images (`docker image ls`) for at least two weeks.
* Online backup at any time (safe while the bots run; uses the backup API):
  `docker compose exec trading_bots python -m tools.db_maintenance backup /app/data/backups/trades-$(date +%F).db`
* Stop: `docker compose stop` sends SIGTERM. The bots finish their current
  step and exit within the 60 s grace period, recorded as a clean stop.

## Rollback

1. `docker compose down` (new stack; a clean SIGTERM stop).
2. With everything stopped: `docker compose run --rm --no-deps migrate python -m tools.db_maintenance checkpoint` (must print `busy=0`).
3. Choose the database to return to:
   * **Keep post-cutover history:**
     `docker compose run --rm --no-deps -v "$PWD:/host" migrate python -m tools.db_maintenance backup /host/trades.db.rollback`.
     The schema changes are additive, so `main` reads this file.
   * **Exactly the pre-cutover state:** rebuild a single self-contained file from the saved pair
     (the old single-file mount cannot see a separate host-side `-wal` file):
     `docker compose run --rm --no-deps -v "$PWD/cutover:/cutover:ro" -v "$PWD:/host" migrate python -m tools.db_maintenance cutover-import --db /cutover/trades.db.orig --wal /cutover/bots.trades.db-wal --dest /host/trades.db.rollback`
4. `mv trades.db trades.db.cutover-original && mv trades.db.rollback trades.db`. The result is one file with no `-wal`, which is what the old single-file mount needs.
5. `git checkout "$(cat ~/singhquant-pre-cutover-commit.txt)" && docker compose up -d`.

Rolling back to `main` brings back every issue fixed since (live-endpoint
flag, cross-strategy liquidation, unenforced budgets, ...).

## Operating notes

| Situation | What you see | What to do |
|---|---|---|
| Model missing or invalid | dashboard "Degraded: no new entries", `MODEL UNAVAILABLE` alert every 6 h | publish or register the model (`tools/model_manifest.py`) |
| Broker API outage | heartbeat `SUSPECT`, alert after 3 cycles | nothing is liquidated; wait or check connectivity |
| Ledger discrepancy | `LEDGER DISCREPANCY` alert | `python -m tools.reconcile_position` |
| Crash loop (3 unclean exits/hour) | `CRASH LOOP: SAFE MODE` alert; no new entries | find the cause, then `python -m tools.safe_mode clear --note ... --confirm` |
| Thread died, restart budget spent | `STRATEGY THREAD FAILED - NOT RESTARTING` | that strategy's positions are unmanaged: intervene now |
| Storage/schema error at start | `STARTUP FAILED` alert, exit 2 after 5 min | fix `./data` (see the message), `docker compose up -d` |
