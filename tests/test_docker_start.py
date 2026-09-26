import logging
from pathlib import Path
import shutil
import subprocess

import pytest

import docker_start


def git(path, *args):
    return subprocess.check_output(
        ["git", "-c", "user.name=Test", "-c", "user.email=test@example.invalid", *args],
        cwd=path, stderr=subprocess.PIPE,
    ).decode().strip()


def commit(repo, name, content):
    (repo / name).write_text(content)
    git(repo, "add", "--", name)
    git(repo, "commit", "-m", "Test update")


@pytest.fixture
def deployment(tmp_path, monkeypatch):
    repo = tmp_path / "remote"
    repo.mkdir()
    git(repo, "init", "--initial-branch=main")
    image = tmp_path / "image"
    image.mkdir()
    for name in docker_start.IMAGE_FILES:
        (repo / name).write_text(f"original {name}\n")
        shutil.copyfile(repo / name, image / name)
    (repo / ".gitignore").write_text("/.local-data/\n")
    git(repo, "add", ".")
    git(repo, "commit", "-m", "Initial")
    commit(repo, "app.py", "version = 1\n")
    data = image / ".local-data"
    data.mkdir()
    (data / "config.json").write_text('{"private": "test-only"}')
    (data / ".session-key").write_bytes(b"x" * 32)
    monkeypatch.setattr(docker_start, "REPOSITORY", str(repo))
    return repo, image


def test_first_start_and_restart_update_without_touching_configuration(deployment):
    repo, image = deployment
    source = docker_start.select_source(image)
    assert source == image / ".local-data/source"
    assert (source / "app.py").read_text() == "version = 1\n"
    commit(repo, "app.py", "version = 2\n")
    assert docker_start.select_source(image) == source
    assert git(source, "rev-parse", "HEAD") == git(repo, "rev-parse", "HEAD")
    assert (source / "app.py").read_text() == "version = 2\n"
    assert (image / ".local-data/config.json").read_text() == '{"private": "test-only"}'
    assert (image / ".local-data/.session-key").read_bytes() == b"x" * 32


@pytest.mark.parametrize("cached", [False, True])
def test_offline_start_uses_cache_or_image(deployment, monkeypatch, caplog, cached):
    _, image = deployment
    expected = docker_start.select_source(image) if cached else image
    monkeypatch.setattr(docker_start, "REPOSITORY", str(image / "missing-remote"))
    with caplog.at_level(logging.WARNING):
        assert docker_start.select_source(image) == expected
    assert "Source update skipped" in caplog.text
    assert not list((image / ".local-data").glob(".source-download-*"))


@pytest.mark.parametrize("name", docker_start.IMAGE_FILES)
def test_image_changes_keep_old_code_until_image_replaced(deployment, caplog, name):
    repo, image = deployment
    source = docker_start.select_source(image)
    original = git(source, "rev-parse", "HEAD")
    commit(repo, name, "changed image requirement\n")
    commit(repo, "app.py", "version = 2\n")
    assert docker_start.select_source(image) == source
    assert git(source, "rev-parse", "HEAD") == original
    assert "Image update required" in caplog.text
    shutil.copyfile(repo / name, image / name)
    assert docker_start.select_source(image) == source
    assert (source / "app.py").read_text() == "version = 2\n"


def test_first_start_with_incompatible_remote_uses_image(deployment):
    repo, image = deployment
    commit(repo, "requirements.txt", "new dependency\n")
    assert docker_start.select_source(image) == image
    assert not (image / ".local-data/source").exists()


def test_replaced_image_does_not_use_incompatible_cache_offline(deployment, monkeypatch):
    _, image = deployment
    source = docker_start.select_source(image)
    (image / "requirements.txt").write_text("new image dependency\n")
    monkeypatch.setattr(docker_start, "REPOSITORY", str(image / "missing-remote"))
    assert docker_start.select_source(image) == image
    assert (source / "app.py").read_text() == "version = 1\n"


def test_dirty_cached_code_is_not_overwritten(deployment, caplog):
    repo, image = deployment
    source = docker_start.select_source(image)
    (source / "app.py").write_text("local edit\n")
    commit(repo, "app.py", "version = 2\n")
    assert docker_start.select_source(image) == image
    assert (source / "app.py").read_text() == "local edit\n"
    assert "local changes" in caplog.text


def test_diverged_history_is_not_reset(deployment, caplog):
    repo, image = deployment
    source = docker_start.select_source(image)
    commit(source, "app.py", "local committed edit\n")
    commit(repo, "app.py", "version = 2\n")
    assert docker_start.select_source(image) == source
    assert (source / "app.py").read_text() == "local committed edit\n"
    assert "Source update skipped" in caplog.text


def test_remote_cannot_track_private_data(deployment, caplog):
    repo, image = deployment
    source = docker_start.select_source(image)
    (repo / ".local-data").mkdir()
    (repo / ".local-data/config.json").write_text("must not apply")
    git(repo, "add", "--force", ".local-data")
    git(repo, "commit", "-m", "Bad private data")
    assert docker_start.select_source(image) == source
    assert "must not track .local-data" in caplog.text
    assert not (source / ".local-data").exists()
    assert (image / ".local-data/config.json").read_text() == '{"private": "test-only"}'


def test_timeout_falls_back_and_logs(deployment, monkeypatch, caplog):
    _, image = deployment
    monkeypatch.setattr(docker_start, "GIT_TIMEOUT_SECONDS", 0)
    assert docker_start.select_source(image) == image
    assert "timed out" in caplog.text


def test_main_links_private_data_and_execs_one_worker(deployment, monkeypatch):
    _, image = deployment
    source = docker_start.select_source(image)
    real_path = Path
    monkeypatch.setattr(docker_start, "Path", lambda value: image if value == "/app" else real_path(value))
    monkeypatch.setattr(docker_start.os, "chdir", lambda path: None)
    calls = []
    monkeypatch.setattr(docker_start.os, "execvp", lambda *args: calls.append(args))
    docker_start.main()
    assert (source / ".local-data").is_symlink()
    assert (source / ".local-data/config.json").read_text() == '{"private": "test-only"}'
    assert calls == [("gunicorn", ["gunicorn", "--workers", "1", "--worker-class", "gthread",
                                  "--threads", "4", "--bind", "0.0.0.0:5001", "app:app"])]
    # The symlink must not make the next start look like a dirty checkout.
    assert docker_start.select_source(image) == source
