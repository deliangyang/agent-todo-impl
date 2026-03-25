import subprocess
from pathlib import Path

from agent_todo_impl.git.git_manager import GitManager


def test_git_manager_branch_and_commit(tmp_path: Path):
    repo = tmp_path / "repo"
    repo.mkdir()
    subprocess.run(
        ["git", "init", "-b", "main"], cwd=repo, check=True, text=True, capture_output=True
    )
    (repo / "a.txt").write_text("a", encoding="utf-8")
    subprocess.run(["git", "add", "-A"], cwd=repo, check=True, text=True, capture_output=True)
    subprocess.run(
        ["git", "commit", "-m", "init"], cwd=repo, check=True, text=True, capture_output=True
    )

    gm = GitManager(repo)
    b = gm.create_run_branch("x")
    assert b.startswith("run/")
    (repo / "b.txt").write_text("b", encoding="utf-8")
    gm.add_all()
    gm.commit("add b")
    assert gm.status_porcelain() == ""
