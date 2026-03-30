import subprocess
from pathlib import Path

from agent_todo_impl.git.git_manager import GitManager, commit_changed_repos, discover_git_repos


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
    assert not gm.has_worktree_changes()
    (repo / "c.txt").write_text("c", encoding="utf-8")
    assert gm.has_worktree_changes()


def test_discover_git_repos_finds_nested_repos(tmp_path: Path):
    workspace = tmp_path / "workspace"
    workspace.mkdir()

    repo_a = workspace / "repo-a"
    repo_b = workspace / "nested" / "repo-b"
    repo_b.parent.mkdir(parents=True)
    repo_a.mkdir()
    repo_b.mkdir()

    subprocess.run(["git", "init"], cwd=repo_a, check=True, text=True, capture_output=True)
    subprocess.run(["git", "init"], cwd=repo_b, check=True, text=True, capture_output=True)

    repos = discover_git_repos(workspace)
    assert repos == sorted([repo_a.resolve(), repo_b.resolve()], key=str)


def test_commit_changed_repos_commits_each_repo_separately(tmp_path: Path):
    workspace = tmp_path / "workspace"
    workspace.mkdir()

    repo_a = workspace / "repo-a"
    repo_b = workspace / "repo-b"
    repo_a.mkdir()
    repo_b.mkdir()

    for repo in (repo_a, repo_b):
        subprocess.run(
            ["git", "init", "-b", "main"],
            cwd=repo,
            check=True,
            text=True,
            capture_output=True,
        )
        (repo / "init.txt").write_text("init", encoding="utf-8")
        subprocess.run(["git", "add", "-A"], cwd=repo, check=True, text=True, capture_output=True)
        subprocess.run(
            ["git", "commit", "-m", "init"],
            cwd=repo,
            check=True,
            text=True,
            capture_output=True,
        )

    (repo_a / "a.txt").write_text("a", encoding="utf-8")
    (repo_b / "b.txt").write_text("b", encoding="utf-8")

    commits = commit_changed_repos(workspace, "implement todo 1 update")
    assert {p for p, _sha in commits} == {repo_a.resolve(), repo_b.resolve()}

    for repo in (repo_a, repo_b):
        gm = GitManager(repo)
        assert not gm.has_worktree_changes()
