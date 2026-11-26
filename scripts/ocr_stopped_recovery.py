"""Explicit offline settlement of a removed source project's separate-GPU OCR owner."""

import base64
import hashlib
import json
import os
import re
import stat
import subprocess
import tempfile
import time
import zlib
from collections.abc import Callable
from contextlib import ExitStack
from pathlib import Path
from typing import Any
from uuid import NAMESPACE_URL, UUID, uuid4, uuid5

ENDPOINT = "http://paddleocr-service:8082"
OCR_RESOURCE = "paddleocr:" + hashlib.sha256(ENDPOINT.encode()).hexdigest()
MOUNTS = {
    "ocr-coordination": "/coord",
    "ocr-activation-request": "/activation/request",
    "ocr-activation-status": "/activation/status",
    "ocr-activation-lock": "/activation/lock",
    "ocr-state": "/var/lib/ocr",
    "ocr-cache": "/var/cache/paddlex",
}
SERVING = "paddleocr-service"
REFUSALS = {
    "locks": "An original controller, admission, handoff or installation lock is unavailable. Stop all owners and preserve the lockfiles.",
    "metadata": "The original activation metadata or handoff records could not be verified. Preserve the volumes and inspect the original deployment.",
    "markers": "The native marker is ambiguous or unsafe. Preserve it and reconcile the original managed container before recovery.",
    "topology": "Separate GPUs and a matching ready installation or evidenced OCR restoration failure are required.",
    "direct": "The direct-call directory contains a marker or has unsafe permissions. Reconcile the original LLM request before recovery.",
    "shared": "The shared handoff journal is unsettled or belongs to another OCR endpoint. Reconcile its recorded owners first.",
    "ocr_only": "The separate-GPU journal must identify only the original local OCR worker with no LLM work.",
    "pending_write": "An incomplete handoff write remains in the original volume. Preserve it for inspection.",
}


class RecoveryRefused(ValueError):
    """Only these stable categories can leave the private verification helper."""


def inspection_program(entrypoint: str, *, recovery: bool = False) -> str:
    """Use only the invoking checkout even with a remote Docker daemon / WSL Desktop."""
    service = Path(__file__).resolve().parents[1] / SERVING
    sources = {
        name: (service / name).read_text()
        for name in ("ocr_service/initialization_lock.py", "ocr_service/supervision.py")
    }
    if recovery:
        # The stdlib-only modules can use namespace packages without loading Pydantic.
        sources["ocr_contract/activation.py"] = (service / "ocr_contract/activation.py").read_text()
        sources["ocr_stopped_recovery.py"] = Path(__file__).read_text()
    payload = base64.b64encode(zlib.compress(json.dumps(sources).encode())).decode()
    return (
        "import base64,json,sys,tempfile,zlib\nfrom pathlib import Path\n"
        "scratch=tempfile.TemporaryDirectory()\nroot=Path(scratch.name)\n"
        f"for name, code in json.loads(zlib.decompress(base64.b64decode({payload!r}))).items():\n"
        "    path=root/name\n    path.parent.mkdir(exist_ok=True)\n    path.write_text(code)\n"
        "sys.path.insert(0,str(root))\n" + entrypoint
    )


def _journal_bytes(path: Path) -> bytes:
    with os.fdopen(os.open(path, os.O_RDONLY | os.O_NONBLOCK | os.O_NOFOLLOW), "rb") as stream:
        info = os.fstat(stream.fileno())
        if (
            not stat.S_ISREG(info.st_mode)
            or info.st_uid != os.geteuid()
            or info.st_nlink != 1
            or info.st_mode & 0o177
            or info.st_size > 1024**2
        ):
            raise ValueError("Unsafe handoff journal")
        data = stream.read(1024**2 + 1)
        if len(data) != info.st_size:
            raise ValueError("Handoff journal changed")
        return data


def _native_marker(path: Path) -> bytes:
    """Exact v1 supervisor record; its run UUID is not a Docker container ID."""
    from ocr_service.initialization_lock import _unique_pairs

    try:
        raw = _journal_bytes(path)
        if len(raw) > 4096:
            raise ValueError("Oversized marker")
        value = json.loads(raw, object_pairs_hook=_unique_pairs)
        if (
            not isinstance(value, dict)
            or set(value) != {"schema", "role", "run_id"}
            or type(value["schema"]) is not int
            or value["schema"] != 1
            or value["role"] != "ocr"
            or not isinstance(value["run_id"], str)
            or not re.fullmatch(r"[0-9a-f]{32}", value["run_id"])
            or UUID(value["run_id"]).version != 4
        ):
            raise ValueError("Unknown marker")
        return raw
    except (OSError, ValueError) as exc:
        raise RecoveryRefused("markers") from exc


def _active_ocr_marker(coord: Path) -> bytes | None:
    names = [
        entry.name for entry in coord.iterdir() if entry.name.startswith("installation.active-")
    ]
    if not names:
        return None
    if names != ["installation.active-ocr.json"]:
        raise RecoveryRefused("markers")
    return _native_marker(coord / names[0])


def _publish_file(path: Path, data: bytes, directory: int) -> None:
    # Both backup and journal must be complete after any interrupted publication.
    fd, temporary = tempfile.mkstemp(dir=path.parent)
    try:
        with os.fdopen(fd, "wb") as output:
            output.write(data)
            output.flush()
            os.fsync(output.fileno())
        os.replace(temporary, path)
        os.fsync(directory)
    finally:
        Path(temporary).unlink(missing_ok=True)


def settle_stopped_owner(coord: Path, store: Any, confirm_stopped: Callable[[], None]) -> Path:
    """Caller proves daemon ownership while these original lock inodes are held."""
    from ocr_contract.activation import ActivationPhase, _lock, _private_directory
    from ocr_service.initialization_lock import OutstandingHandoff, _require_clean_journal
    from ocr_service.supervision import role_lock_path

    with ExitStack() as stack:
        for directory, name in (
            (store.lock_dir, "controller.lock"),
            (store.lock_dir, "admission.lock"),
            (coord, "gpu-handoff.lock"),
            (coord, "installation.lock"),
        ):
            try:
                stack.enter_context(
                    _lock(directory, name, exclusive=True, blocking=False, create=False)
                )
            except (OSError, RuntimeError) as exc:
                raise RecoveryRefused("locks") from exc
        marker = _active_ocr_marker(coord)
        request = store.read_request()
        status = store.read_status(request) if request else None
        restoration_failed = status is not None and (
            status.phase == ActivationPhase.FAILED
            and status.reason_code == "native_activation_recovery_required"
        )
        if (
            store.shared_gpu()
            or status is None
            or (status.phase != ActivationPhase.READY and not restoration_failed)
        ):
            raise RecoveryRefused("topology")
        markers = store.request_dir / "direct-calls"
        if os.path.lexists(markers):
            with _private_directory(markers, create=False) as descriptor:
                if os.fstat(descriptor).st_mode & 0o077 or os.listdir(descriptor):
                    raise RecoveryRefused("direct")

        peer = coord / "gpu-handoff.json"
        try:
            _require_clean_journal(peer)
        except RuntimeError as exc:
            raise RecoveryRefused("shared") from exc
        if (
            os.path.lexists(peer)
            and json.loads(_journal_bytes(peer))["resources"]["ocr"] != OCR_RESOURCE
        ):
            raise RecoveryRefused("shared")
        journal = coord / "gpu-handoff.ocr-only.json"
        try:
            _require_clean_journal(journal, ocr_only=True)
        except OutstandingHandoff:
            pass
        else:
            raise RecoveryRefused("ocr_only")
        before = _journal_bytes(journal)
        state = json.loads(before)
        if (
            state["resources"] != {"llm": "external-unmanaged", "ocr": OCR_RESOURCE}
            or state["models"] != []
            or state["unsettled"] is not False
            or state["ocr_pending"] is not True
        ):
            raise RecoveryRefused("ocr_only")
        if any(os.path.lexists(p.with_suffix(".pending")) for p in (journal, peer)):
            raise RecoveryRefused("pending_write")
        # Correlate recovery evidence with this activation generation and exact pending
        # journal. A crash after retiring the marker must leave a verifiable retry.
        evidence_id = uuid5(
            NAMESPACE_URL,
            f"vision-bill/ocr/{request.generation}/{hashlib.sha256(before).hexdigest()}",
        ).hex
        archived_marker = coord / f"installation.ocr.recovered-{evidence_id}.json"
        backup = coord / f"gpu-handoff.ocr-only.recovered-{evidence_id}.json"
        archived = os.path.lexists(archived_marker)
        if archived:
            _native_marker(archived_marker)
            if (
                marker is not None
                or not os.path.lexists(backup)
                or _journal_bytes(backup) != before
            ):
                raise RecoveryRefused("markers")
        if marker is not None or archived:
            # The original PID1 owns this lock for its whole native OCR scope. Never
            # create a replacement inode as evidence of a stopped container.
            role_lock = role_lock_path(coord / "installation.lock", "ocr")
            try:
                stack.enter_context(
                    _lock(coord, role_lock.name, exclusive=True, blocking=False, create=False)
                )
            except (OSError, RuntimeError) as exc:
                raise RecoveryRefused("locks") from exc
        elif restoration_failed:
            raise RecoveryRefused("topology")
        else:
            backup = coord / f"gpu-handoff.ocr-only.recovered-{uuid4().hex}.json"

        confirm_stopped()
        if _active_ocr_marker(coord) != marker or _journal_bytes(journal) != before:
            raise ValueError("Ownership evidence changed during stopped-owner verification")
        state["ocr_pending"] = False
        with _private_directory(coord, create=False) as directory:
            if os.path.lexists(backup):
                if _journal_bytes(backup) != before:
                    raise RecoveryRefused("markers")
            else:
                _publish_file(backup, before, directory)
            if marker is not None:
                if os.path.lexists(archived_marker):
                    raise RecoveryRefused("markers")
                # Atomically archive the original inode while the journal is still
                # pending. Never overwrite or clear an initializer or legacy fence.
                os.rename(coord / "installation.active-ocr.json", archived_marker)
            # A resumed rename may have survived an interrupted fsync. Establish
            # the archive barrier again before a clean journal can become visible.
            os.fsync(directory)
            _publish_file(journal, json.dumps(state).encode(), directory)
        return backup


def _await_file(path: Path, seconds: int = 30) -> None:
    deadline = time.monotonic() + seconds
    while not path.exists():
        if time.monotonic() >= deadline:
            raise ValueError("Stopped-owner verification timed out")
        time.sleep(0.05)


def _inside() -> None:
    from ocr_contract.activation import ActivationStore

    def handshake() -> None:
        Path("/tmp/locked").touch(exist_ok=False)
        _await_file(Path("/tmp/confirmed"))

    try:
        backup = settle_stopped_owner(Path("/coord"), ActivationStore(), handshake)
    except RecoveryRefused as exc:
        print(str(exc) if str(exc) in REFUSALS else "metadata", flush=True)
        raise SystemExit(1) from None
    except (ValueError, OSError, RuntimeError, KeyError):
        print("metadata", flush=True)
        raise SystemExit(1) from None
    print(backup.name)
    archived = backup.with_name(backup.name.replace("gpu-handoff.ocr-only.", "installation.ocr."))
    if archived.exists():
        print(archived.name)


def _volume_names(config: dict[str, Any], project: str) -> dict[str, str]:
    app, native = (config["services"][name] for name in ("vision_bill", SERVING))
    if (
        app["environment"].get("OCR_ACTIVATION_ENDPOINT") != ENDPOINT
        or app["environment"].get("VISION_BILL_OCR_ACTIVATION") != "true"
        or native.get("command") != ["python", "-m", "ocr_service.activation"]
    ):
        raise ValueError("Stopped OCR recovery requires the original local managed OCR endpoint")
    if (
        app["environment"].get("OCR_ACTIVATION_HANDOFF_LOCK_PATH") != "/coord/gpu-handoff.lock"
        or native["environment"].get("OCR_HANDOFF_LOCK_PATH") != "/coord/gpu-handoff.lock"
        or native["environment"].get("OCR_INSTALLATION_LOCK_PATH") != "/coord/installation.lock"
        or native["environment"].get("OCR_PORT") != "8082"
        or native["environment"].get("OCR_STATE_DIR") != "/var/lib/ocr"
    ):
        raise ValueError("Cannot prove original OCR lock and port configuration")
    networks = config.get("networks", {})
    if (
        set(networks) != {"default", "postgres_net"}
        or networks["default"].get("name") != f"{project}_default"
        or networks["postgres_net"].get("name") != "postgres_net"
        or any(
            set(net) - {"name", "driver", "ipam"}
            or net.get("ipam")
            or net.get("driver", "bridge") != "bridge"
            for net in networks.values()
        )
        or set(app.get("networks") or {}) != {"default", "postgres_net"}
        or set(native.get("networks") or {}) != {"default"}
        or any(
            attachment not in (None, {})
            for s in (app, native)
            for attachment in s["networks"].values()
        )
        or app.get("extra_hosts", [])
        not in ([], ["host.docker.internal=host-gateway"], ["host.docker.internal:host-gateway"])
        or native.get("extra_hosts")
        or any(
            s.get(k)
            for s in (app, native)
            for k in (
                "network_mode",
                "dns",
                "dns_search",
                "dns_opt",
                "links",
                "external_links",
                "hostname",
                "domainname",
                "container_name",
                "volumes_from",
                "configs",
                "entrypoint",
            )
        )
        or any(
            m.get("target") in ("/", "/etc", "/etc/hosts", "/etc/resolv.conf", "/etc/nsswitch.conf")
            for service in (app, native)
            for m in service["volumes"]
        )
    ):
        raise ValueError(
            "Stopped OCR recovery requires the original bridge networks and local service routing"
        )
    names = {}
    for logical, target in MOUNTS.items():
        services = (native,) if logical in ("ocr-state", "ocr-cache") else (app, native)
        for service in services:
            mounts = [m for m in service["volumes"] if m.get("target") == target]
            if (
                len(mounts) != 1
                or mounts[0].get("type") != "volume"
                or mounts[0].get("source") != logical
                or mounts[0].get("volume", {}).get("subpath")
            ):
                raise ValueError("Original source OCR volume mapping is required")
        spec = config["volumes"][logical]
        if (
            spec.get("name") != f"{project}_{logical}"
            or spec.get("external")
            or spec.get("driver_opts")
            or spec.get("driver", "local") != "local"
        ):
            raise ValueError("Original project-local OCR volumes are required")
        names[logical] = spec["name"]
    return names


def _verify_volume_owners(names: dict[str, str], project: str, run: Callable[..., str]) -> None:
    records = json.loads(run("docker", "volume", "inspect", *names.values(), capture=True))
    indexed = {v["Name"]: v for v in records}
    for logical, actual in names.items():
        volume = indexed.get(actual, {})
        labels = volume.get("Labels") or {}
        if (
            volume.get("Driver") != "local"
            or volume.get("Scope") != "local"
            or volume.get("Options")
            or labels.get("com.docker.compose.project") != project
            or labels.get("com.docker.compose.volume") != logical
        ):
            raise ValueError("Cannot prove ownership of the original source OCR volumes")


def _verify_stopped(
    names: dict[str, str], project: str, helper: str, run: Callable[..., str]
) -> None:
    # Inspect every container, including stopped, foreign and unlabeled ones.
    # Docker's volume filter can miss a mount that is specified as a bind.
    networks = run("docker", "network", "ls", "--format", "{{.Name}}", capture=True).splitlines()
    if {"postgres_net", f"{project}_default"}.intersection(networks):
        raise ValueError(
            "The original app networks still exist. Stop their participants and take the original Compose project down before recovery."
        )
    ids = run("docker", "ps", "--all", "--quiet", "--no-trunc", capture=True).split()
    if helper not in ids:
        raise ValueError("Stopped-owner inspector disappeared")
    others = [identifier for identifier in ids if identifier != helper]
    if not others:
        return
    mounts = {str(name) for name in names.values()}
    volumes = json.loads(run("docker", "volume", "inspect", *names.values(), capture=True))
    roots = [Path(v["Mountpoint"]) for v in volumes]
    # Inspect only routing and mount metadata. Container environments may contain credentials.
    template = '{"mounts":{{json .Mounts}},"labels":{{json .Config.Labels}}}'
    output = run("docker", "container", "inspect", "--format", template, *others, capture=True)
    bind_sources = []
    attached_volumes = set()
    for line in output.splitlines():
        container = json.loads(line)
        labels = container["labels"] or {}
        if labels.get("com.docker.compose.project") == project and labels.get(
            "com.docker.compose.service"
        ) in ("vision_bill", SERVING):
            raise ValueError("Remove the original app and OCR containers before recovery")
        for mount in container["mounts"]:
            if mount.get("Name") in mounts:
                raise ValueError("Another container still has access to the original OCR volumes")
            if mount.get("Type") == "bind":
                bind_sources.append(Path(mount["Source"]))
            elif mount.get("Type") == "volume":
                attached_volumes.add(mount["Name"])
    if attached_volumes:
        records = json.loads(
            run("docker", "volume", "inspect", *sorted(attached_volumes), capture=True)
        )
        for volume in records:
            options = volume.get("Options") or {}
            if volume.get("Driver") == "local" and {"bind", "rbind"}.intersection(
                options.get("o", "").split(",")
            ):
                bind_sources.append(Path(options["device"]))
    if any(
        path.is_relative_to(root) or root.is_relative_to(path)
        for path in bind_sources
        for root in roots
    ):
        raise ValueError("A container bind mount still exposes an original OCR volume")


def recover_stopped_ocr(
    config: dict[str, Any],
    previous: dict[str, Any],
    *,
    run: Callable[..., str],
    helper_image: str,
) -> None:
    project = previous["project"]
    names = _volume_names(config, project)
    _verify_volume_owners(names, project, run)
    args = []
    for logical, target in MOUNTS.items():
        if logical in ("ocr-state", "ocr-cache"):
            continue  # Only their Docker identities are needed; never expose model files.
        readonly = "" if logical == "ocr-coordination" else ",readonly"
        args.extend(
            [
                "--mount",
                f"type=volume,source={names[logical]},target={target},volume-nocopy{readonly}",
            ]
        )
    program = inspection_program(
        "from ocr_stopped_recovery import _inside\n_inside()", recovery=True
    )
    helper = run(
        "docker",
        "run",
        "--detach",
        "--pull=never",
        "--platform",
        "linux/amd64",
        "--network=none",
        "--read-only",
        "--cap-drop=ALL",
        "--security-opt=no-new-privileges",
        "--tmpfs=/tmp:rw,noexec,nosuid,nodev,size=1m",
        "--env",
        "NVIDIA_VISIBLE_DEVICES=void",
        *args,
        "--entrypoint",
        "python",
        helper_image,
        "-B",
        "-c",
        program,
        capture=True,
    ).strip()
    if not re.fullmatch(r"[0-9a-f]{64}", helper):
        raise ValueError("Cannot identify stopped-owner inspector")
    try:
        # The helper delays every durable write until the host confirms ownership under the locks.
        wait = "from pathlib import Path\nimport time\nfor _ in range(200):\n if Path('/tmp/locked').exists(): break\n time.sleep(.05)\nelse: raise SystemExit(1)"
        run("docker", "exec", helper, "python", "-c", wait, capture=True)
        _verify_volume_owners(names, project, run)
        _verify_stopped(names, project, helper, run)
        run(
            "docker",
            "exec",
            helper,
            "python",
            "-c",
            "open('/tmp/confirmed','x').close()",
            capture=True,
        )
        if run("docker", "wait", helper, capture=True).strip() != "0":
            raise ValueError("Stopped OCR ownership could not be settled")
        archives = run("docker", "logs", helper, capture=True).strip().splitlines()
        if (
            not archives
            or not re.fullmatch(r"gpu-handoff\.ocr-only\.recovered-[0-9a-f]{32}\.json", archives[0])
            or len(archives) > 1
            and archives[1:] != [archives[0].replace("gpu-handoff.ocr-only.", "installation.ocr.")]
        ):
            raise ValueError("Cannot confirm the saved OCR recovery evidence")
        print(
            f"Stopped local OCR ownership settled. Original journal saved in {names['ocr-coordination']}/{archives[0]}."
        )
        if len(archives) == 2:
            print(f"Native OCR crash marker archived in {names['ocr-coordination']}/{archives[1]}.")
    except subprocess.CalledProcessError:
        # Docker logs on stderr for a refused helper; read via explicit exit status text on stdout.
        failure = run("docker", "logs", helper, capture=True).strip()
        raise ValueError(
            REFUSALS.get(
                failure,
                "Stopped OCR recovery refused; verify original locks, topology, journals and container ownership.",
            )
        ) from None
    finally:
        run("docker", "rm", "--force", helper, capture=True)
