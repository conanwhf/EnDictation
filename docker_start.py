"""Container-only source update before starting the single Gunicorn worker."""

import logging
import os
from pathlib import Path
import signal
import subprocess
import tempfile


REPOSITORY = "https://github.com/conanwhf/EnDictation.git"
BRANCH = "main"
GIT_TIMEOUT_SECONDS = 60
IMAGE_FILES = ("requirements.txt", "Dockerfile", "docker_start.py")
logger = logging.getLogger("startup")


class UpdateError(RuntimeError):
    pass


def git(directory, *args):
    command = ["git", "-c", "credential.helper=", "-c", "core.hooksPath=/dev/null", *args]
    with subprocess.Popen(command, cwd=directory, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                          env={**os.environ, "GIT_TERMINAL_PROMPT": "0"},
                          start_new_session=True) as process:
        try:
            stdout, stderr = process.communicate(timeout=GIT_TIMEOUT_SECONDS)
        except subprocess.TimeoutExpired as exc:
            # Stop git's transport children too, before falling back to local code.
            os.killpg(process.pid, signal.SIGKILL)
            process.communicate()
            raise UpdateError(f"Git {args[0]} timed out after {GIT_TIMEOUT_SECONDS}s") from exc
    if process.returncode:
        raise UpdateError(f"Git {args[0]} failed: {stderr.decode(errors='replace').strip()}")
    return stdout


def check_source(image, source, ref):
    if git(source, "ls-tree", "--name-only", ref, "--", ".local-data").strip():
        raise UpdateError("Remote code must not track .local-data; private configuration is preserved")
    for name in IMAGE_FILES:
        if git(source, "show", f"{ref}:{name}") != (image / name).read_bytes():
            raise UpdateError(f"Image update required ({name} changed): Pull + Update Application")


def check_clean(source):
    if git(source, "status", "--porcelain", "--untracked-files=no").strip():
        raise UpdateError("Cached source has local changes; refusing to overwrite them")


def select_source(image):
    """Keep the private data volume outside Git; failed updates do not erase it."""
    data = image / ".local-data"
    data.mkdir(parents=True, exist_ok=True)
    source = data / "source"
    try:
        if source.exists():
            check_clean(source)
            git(source, "fetch", "--no-tags", REPOSITORY, BRANCH)
            check_source(image, source, "FETCH_HEAD")
            # A force-push or divergent checkout requires human attention, not a reset.
            git(source, "merge-base", "--is-ancestor", "HEAD", "FETCH_HEAD")
            git(source, "merge", "--ff-only", "FETCH_HEAD")
        else:
            with tempfile.TemporaryDirectory(prefix=".source-download-", dir=data) as directory:
                candidate = Path(directory) / "repo"
                git(data, "clone", "--single-branch", "--branch", BRANCH,
                    REPOSITORY, str(candidate))
                check_source(image, candidate, "HEAD")
                candidate.rename(source)
        logger.info("Source ready: %s", git(source, "rev-parse", "--short", "HEAD").decode().strip())
        return source
    except (UpdateError, OSError) as exc:
        logger.warning("Source update skipped: %s", exc)

    if source.exists():
        try:
            check_clean(source)
            check_source(image, source, "HEAD")
            logger.warning("Starting cached source: %s",
                           git(source, "rev-parse", "--short", "HEAD").decode().strip())
            return source
        except (UpdateError, OSError) as exc:
            logger.warning("Cached source unavailable: %s", exc)
    logger.warning("Starting code bundled in the image")
    return image


def main():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    image = Path("/app")
    source = select_source(image)
    if source != image:
        link = source / ".local-data"
        data = image / ".local-data"
        if not link.is_symlink() and not link.exists():
            link.symlink_to(data, target_is_directory=True)
        if not link.is_symlink() or link.resolve() != data.resolve():
            raise UpdateError("Cached .local-data is not linked to the private data volume")
    os.chdir(source)
    os.execvp("gunicorn", ["gunicorn", "--workers", "1", "--worker-class", "gthread",
                          "--threads", "4", "--bind", "0.0.0.0:5001", "app:app"])


if __name__ == "__main__":
    main()
