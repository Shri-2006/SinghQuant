"""
Model artifact storage, manifest and validation (Phase 2 I-3).

Artifacts live in SINGHQUANT_MODEL_DIR (in Docker: /app/model_artifacts on
the ./model_artifacts host volume, so retrained models survive image
rebuilds). Each artifact has a LOGICAL name the strategies ask for
("stable_model.pkl", "risky2_BTCUSD.zip") and an entry in manifest.json:

    {"schema": 1,
     "artifacts": {
        "stable_model.pkl": {"file": "stable_model.3f9c2a1b44de.pkl",
                             "sha256": "...", "version": "...", "strategy": "stable",
                             "created_at": "...", "size": 123, "meta": {...},
                             "previous": {...the entry it replaced...}}}}

Writes are atomic: the model is written to a temporary file, fsynced and
renamed to a content-addressed name, and only then does the manifest (also
written temp + fsync + rename) point at it. A reader therefore sees either
the old artifact or the new one, never a half-written file and never a
manifest that disagrees with the file it names.

A model is only loaded after its SHA-256 matches the manifest. A missing,
unregistered, tampered or unreadable artifact is an explicit status, and
the strategy that needs it takes no new entries (it keeps managing its
existing positions).
"""
import hashlib
import json
import os
import tempfile
from contextlib import contextmanager
from datetime import datetime, timezone

MANIFEST_NAME = "manifest.json"
MANIFEST_SCHEMA = 1
_DEFAULT_DIR = os.path.dirname(os.path.abspath(__file__))

# Statuses an artifact can be in. Only "ok" allows new entries.
OK = "ok"
MISSING = "missing"                  # manifest names a file that is not there
UNREGISTERED = "unregistered"        # no manifest entry for the logical name
HASH_MISMATCH = "hash_mismatch"      # file contents differ from the manifest
MANIFEST_INVALID = "manifest_invalid"
LOAD_ERROR = "load_error"            # verified file could not be deserialised
INCOMPATIBLE = "incompatible"        # loaded, but fails the strategy's own check (e.g. feature width)


class ArtifactError(Exception):
    def __init__(self, status, detail):
        super().__init__(f"{status}: {detail}")
        self.status = status
        self.detail = detail


def model_dir():
    """The artifact directory, read at call time so tests and tools can override it."""
    return os.getenv("SINGHQUANT_MODEL_DIR") or _DEFAULT_DIR


def expected_artifacts(strategy):
    """Logical artifact names a strategy needs before it may open new entries."""
    from core.config import RISKY2_ASSETS
    if strategy == "stable":
        return ["stable_model.pkl"]
    if strategy == "risky1":
        return ["risky1_model.pkl"]
    if strategy == "risky2":
        return [rl_artifact_name(t) for t in RISKY2_ASSETS]
    raise ValueError(f"unknown strategy {strategy!r}")


def rl_artifact_name(ticker):
    return f"risky2_{ticker.replace('X:', '').replace('/', '')}.zip"


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def manifest_path(directory=None):
    return os.path.join(directory or model_dir(), MANIFEST_NAME)


def load_manifest(directory=None):
    path = manifest_path(directory)
    if not os.path.exists(path):
        return {"schema": MANIFEST_SCHEMA, "artifacts": {}}
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
    except (OSError, ValueError) as e:
        raise ArtifactError(MANIFEST_INVALID, f"{path} is unreadable: {e}")
    if not isinstance(data, dict) or not isinstance(data.get("artifacts"), dict):
        raise ArtifactError(MANIFEST_INVALID, f"{path} has no 'artifacts' mapping")
    if data.get("schema") != MANIFEST_SCHEMA:
        raise ArtifactError(MANIFEST_INVALID, f"{path} schema {data.get('schema')!r}, expected {MANIFEST_SCHEMA}")
    return data


def _fsync_dir(directory):
    try:
        fd = os.open(directory, os.O_RDONLY)
    except OSError:
        return
    try:
        os.fsync(fd)
    except OSError:
        pass
    finally:
        os.close(fd)


def _atomic_write_bytes(path, data):
    directory = os.path.dirname(path)
    fd, tmp = tempfile.mkstemp(prefix=".tmp-", dir=directory)
    try:
        with os.fdopen(fd, "wb") as f:
            f.write(data)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, path)
        _fsync_dir(directory)
    except BaseException:
        if os.path.exists(tmp):
            os.remove(tmp)
        raise


@contextmanager
def _manifest_lock(directory):
    """Serialises manifest read-modify-write between processes (retrain vs the register tool)."""
    import fcntl
    with open(os.path.join(directory, ".manifest.lock"), "a+") as lock:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(lock.fileno(), fcntl.LOCK_UN)


def resolve(name, directory=None, manifest=None):
    """(entry, absolute_path) for a logical name, or ArtifactError(UNREGISTERED / MISSING)."""
    directory = directory or model_dir()
    manifest = manifest if manifest is not None else load_manifest(directory)
    entry = manifest["artifacts"].get(name)
    if not entry or not entry.get("file") or not entry.get("sha256"):
        raise ArtifactError(UNREGISTERED, f"{name} is not registered in {manifest_path(directory)}; "
                                          f"register it with tools/model_manifest.py")
    path = os.path.join(directory, entry["file"])
    if os.path.basename(entry["file"]) != entry["file"]:
        raise ArtifactError(MANIFEST_INVALID, f"{name}: file {entry['file']!r} must be a plain file name")
    if not os.path.isfile(path):
        raise ArtifactError(MISSING, f"{name}: manifest names {entry['file']}, which is not in {directory}")
    return entry, path


def verify(name, directory=None):
    """(entry, path) when the file exists and its SHA-256 matches the manifest; ArtifactError otherwise."""
    entry, path = resolve(name, directory)
    actual = sha256_file(path)
    if actual != entry["sha256"]:
        raise ArtifactError(HASH_MISMATCH, f"{name}: {entry['file']} sha256 {actual[:12]}... "
                                           f"does not match the manifest ({entry['sha256'][:12]}...)")
    return entry, path


def _versioned_name(name, digest):
    stem, ext = os.path.splitext(name)
    return f"{stem}.{digest[:12]}{ext}"


def publish(name, write_fn, strategy, version, meta=None, directory=None):
    """
    Writes a new artifact atomically and registers it.
    write_fn(tmp_path) must write the complete artifact to tmp_path (which
    has the logical name's extension). Returns the new manifest entry.
    """
    directory = directory or model_dir()
    os.makedirs(directory, exist_ok=True)
    ext = os.path.splitext(name)[1]
    fd, tmp = tempfile.mkstemp(prefix=".tmp-", suffix=ext, dir=directory)
    os.close(fd)
    try:
        write_fn(tmp)
        with open(tmp, "rb") as f:
            os.fsync(f.fileno())
        digest = sha256_file(tmp)
        final = os.path.join(directory, _versioned_name(name, digest))
        os.replace(tmp, final)
        _fsync_dir(directory)
    except BaseException:
        if os.path.exists(tmp):
            os.remove(tmp)
        raise
    return _register_entry(name, os.path.basename(final), digest, strategy, version, meta, directory)


def register_existing(name, source_path, strategy, version, meta=None, directory=None):
    """
    Registers an artifact produced elsewhere (e.g. a model copied out of the
    old container). The source is COPIED into the artifact directory under a
    content-addressed name; the source file is never modified.
    """
    directory = directory or model_dir()
    with open(source_path, "rb") as f:
        data = f.read()

    def write(tmp):
        with open(tmp, "wb") as out:
            out.write(data)

    return publish(name, write, strategy, version, meta, directory)


def _register_entry(name, filename, digest, strategy, version, meta, directory):
    with _manifest_lock(directory):
        manifest = load_manifest(directory)
        previous = manifest["artifacts"].get(name)
        if previous:
            previous = {k: v for k, v in previous.items() if k != "previous"}
        entry = {
            "file": filename,
            "sha256": digest,
            "version": str(version),
            "strategy": strategy,
            "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "size": os.path.getsize(os.path.join(directory, filename)),
            "meta": meta or {},
        }
        if previous:
            entry["previous"] = previous
        manifest["artifacts"][name] = entry
        manifest["schema"] = MANIFEST_SCHEMA
        _atomic_write_bytes(manifest_path(directory),
                            (json.dumps(manifest, indent=2, sort_keys=True) + "\n").encode("utf-8"))
    print(f"[models] published {name} -> {filename} (sha256 {digest[:12]}..., version {version})")
    return entry


class ArtifactHandle:
    """
    Verified, hot-reloading access to one logical artifact.

    `.model` returns the loaded object ONLY while the artifact verifies
    (manifest entry present, file present, SHA-256 matches, loader and
    `check` succeed). Otherwise it returns None and `.status` / `.detail`
    say why. Verification happens BEFORE deserialisation, so a tampered
    pickle is never unpickled. Changes to the manifest or file are picked up
    on the next access (retraining publishes a new file + manifest).
    """

    def __init__(self, name, loader, check=None, directory=None):
        self.name = name
        self.loader = loader
        self.check = check
        self._directory = directory
        self._model = None
        self._key = None
        self.status = UNREGISTERED
        self.detail = "not checked yet"
        self.entry = None

    @property
    def directory(self):
        return self._directory or model_dir()

    def _fingerprint(self):
        stats = []
        for p in (manifest_path(self.directory),):
            try:
                st = os.stat(p)
                stats.append((st.st_mtime_ns, st.st_size, st.st_ino))
            except OSError:
                stats.append(None)
        if self.entry:
            try:
                st = os.stat(os.path.join(self.directory, self.entry["file"]))
                stats.append((st.st_mtime_ns, st.st_size, st.st_ino))
            except OSError:
                stats.append(None)
        return (self.directory, tuple(stats))

    def refresh(self):
        # A verified artifact is cached until the manifest or its file changes.
        # Anything else is re-checked on every access, so a model that appears
        # (or is repaired) is picked up on the next cycle.
        key = self._fingerprint()
        if self.status == OK and key == self._key:
            return self.status
        previous_status, previous_detail = self.status, self.detail
        try:
            entry, path = verify(self.name, self.directory)
            try:
                model = self.loader(path)
            except Exception as e:
                raise ArtifactError(LOAD_ERROR, f"{self.name}: {entry['file']} could not be loaded: {e}")
            if self.check is not None:
                problem = self.check(model)
                if problem:
                    raise ArtifactError(INCOMPATIBLE, f"{self.name}: {problem}")
            self._model, self.entry = model, entry
            self.status, self.detail = OK, f"{entry['file']} version {entry.get('version')}"
        except ArtifactError as e:
            self._model = None
            self.status, self.detail = e.status, e.detail
            self.entry = self.entry if e.status in (HASH_MISMATCH, LOAD_ERROR, INCOMPATIBLE) else None
        self._key = self._fingerprint()
        if (self.status, self.detail) != (previous_status, previous_detail):
            if self.status == OK:
                print(f"[models] {self.name}: loaded and verified ({self.detail})")
            else:
                print(f"[models] ERROR {self.name}: {self.status} - {self.detail}. New entries are blocked "
                      f"for this model; existing positions are still risk-managed.")
        return self.status

    @property
    def model(self):
        self.refresh()
        return self._model if self.status == OK else None

    @property
    def ready(self):
        return self.refresh() == OK

    def predict(self, X):
        m = self.model
        return None if m is None else m.predict(X)
