"""Tests for context management — directory tree, history, conversation."""

import os
import pytest

from terminator.context import (
    StaticContext,
    ExchangeRecord,
    ConversationHistory,
    DirectoryTree,
    ContextManager,
)


class TestStaticContext:
    def test_gather(self):
        ctx = StaticContext.gather()
        assert ctx.os_type  # Should be "Darwin", "Linux", etc.
        assert ctx.shell_type  # Should be "zsh", "bash", etc.
        assert ctx.username
        assert ctx.home_dir


class TestDirectoryTree:
    def test_basic_tree(self, tmp_path):
        (tmp_path / "file1.txt").write_text("hello")
        (tmp_path / "file2.py").write_text("world")
        subdir = tmp_path / "subdir"
        subdir.mkdir()
        (subdir / "nested.txt").write_text("nested")

        tree = DirectoryTree(max_depth=3, max_entries=100)
        result = tree.get(str(tmp_path))
        assert "file1.txt" in result
        assert "file2.py" in result
        assert "subdir/" in result
        assert "nested.txt" in result

    def test_skips_git(self, tmp_path):
        (tmp_path / ".git").mkdir()
        (tmp_path / ".git" / "HEAD").write_text("ref")
        (tmp_path / "src").mkdir()
        (tmp_path / "src" / "main.py").write_text("")

        tree = DirectoryTree()
        result = tree.get(str(tmp_path))
        assert ".git" not in result
        assert "main.py" in result

    def test_max_entries(self, tmp_path):
        for i in range(20):
            (tmp_path / f"file{i:03d}.txt").write_text("")

        tree = DirectoryTree(max_entries=10)
        result = tree.get(str(tmp_path))
        assert "... and more" in result

    def test_caching(self, tmp_path):
        (tmp_path / "file.txt").write_text("")
        tree = DirectoryTree()
        result1 = tree.get(str(tmp_path))
        result2 = tree.get(str(tmp_path))
        assert result1 == result2

    def test_invalidate(self, tmp_path):
        (tmp_path / "file.txt").write_text("")
        tree = DirectoryTree()
        tree.get(str(tmp_path))
        tree.invalidate(str(tmp_path))
        # After invalidation, cache should be empty
        assert str(tmp_path) not in tree._cache

    def test_empty_dir(self, tmp_path):
        tree = DirectoryTree()
        result = tree.get(str(tmp_path))
        assert "empty" in result.lower() or result.strip() == ""


class TestConversationHistory:
    def test_add_and_retrieve(self):
        history = ConversationHistory(max_length=5)
        record = ExchangeRecord(
            user_input="list files",
            generated_command="ls -la",
            stdout="file1.txt\nfile2.txt",
            exit_code=0,
        )
        history.add(record)
        assert len(history.records) == 1
        assert history.records[0].user_input == "list files"

    def test_eviction(self):
        history = ConversationHistory(max_length=3)
        for i in range(5):
            history.add(ExchangeRecord(
                user_input=f"query {i}",
                generated_command=f"cmd {i}",
            ))
        assert len(history.records) == 3
        assert history.records[0].user_input == "query 2"

    def test_clear(self):
        history = ConversationHistory()
        history.add(ExchangeRecord(user_input="test", generated_command="echo"))
        history.clear()
        assert len(history.records) == 0

    def test_to_messages(self):
        history = ConversationHistory()
        history.add(ExchangeRecord(
            user_input="list files",
            generated_command="ls -la",
            stdout="file1.txt",
            exit_code=0,
        ))
        messages = history.to_messages()
        assert len(messages) == 2
        assert messages[0]["role"] == "user"
        assert messages[0]["content"] == "list files"
        assert messages[1]["role"] == "assistant"
        assert "ls -la" in messages[1]["content"]

    def test_output_truncation(self):
        long_output = "\n".join(f"line {i}" for i in range(100))
        history = ConversationHistory(truncate_lines=20)
        history.add(ExchangeRecord(
            user_input="test",
            generated_command="cmd",
            stdout=long_output,
            exit_code=0,
        ))
        messages = history.to_messages()
        content = messages[1]["content"]
        assert "omitted" in content


class TestContextManager:
    def test_build_prompt_context(self):
        ctx_mgr = ContextManager()
        ctx = ctx_mgr.build_prompt_context()
        assert "os_type" in ctx
        assert "shell_type" in ctx
        assert "cwd" in ctx
        assert "dir_tree" in ctx
        assert "username" in ctx

    def test_get_chat_messages(self):
        ctx_mgr = ContextManager()
        messages = ctx_mgr.get_chat_messages("hello")
        assert len(messages) == 1
        assert messages[0]["role"] == "user"
        assert messages[0]["content"] == "hello"

    def test_get_chat_messages_with_history(self):
        ctx_mgr = ContextManager()
        ctx_mgr.record_exchange("list files", "ls", stdout="file.txt", exit_code=0)
        messages = ctx_mgr.get_chat_messages("now count them")
        assert len(messages) == 3  # 1 history user + 1 history assistant + 1 new user
        assert messages[-1]["content"] == "now count them"

    def test_update_cwd(self, tmp_path):
        ctx_mgr = ContextManager()
        old_cwd = os.getcwd()
        ctx_mgr.update_cwd(str(tmp_path))
        assert ctx_mgr.cwd == str(tmp_path)
        # Restore
        os.chdir(old_cwd)

    def test_update_cwd_nonexistent(self):
        ctx_mgr = ContextManager()
        old_cwd = ctx_mgr.cwd
        ctx_mgr.update_cwd("/nonexistent_xyz_123")
        assert ctx_mgr.cwd == old_cwd  # Should not change

    def test_record_exchange(self):
        ctx_mgr = ContextManager()
        ctx_mgr.record_exchange("test", "echo", stdout="hello", exit_code=0)
        assert len(ctx_mgr.history.records) == 1
