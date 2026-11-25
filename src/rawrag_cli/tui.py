from __future__ import annotations

import os
import subprocess
from pathlib import Path

import requests
from rich.console import Group
from rich.markdown import Markdown
from rich.panel import Panel
from rich.table import Table
from rich.text import Text
from textual import on, work
from textual.app import App, ComposeResult
from textual.binding import Binding
from textual.widgets import ContentSwitcher, Input, RichLog, Static

from rawrag_cli.cli import print_header
from rawrag_cli.core.pipeline import (
    IndexConsistencyError,
    IndexNotFoundError,
    NoRelevantChunksError,
    index_directory,
    query_codebase,
)
from rawrag_cli.core.config import version

TABS = ["Current", "Index", "History"]
version = version  # Use the version from config.py


def get_repo_context() -> str:
    r"""Return '~\path\to\repo [⎇ branch]' matching the Copilot CLI status line."""
    cwd = Path.cwd()
    home = Path.home()
    try:
        display_path = f"~\\{cwd.relative_to(home)}" if os.name == "nt" else f"~/{cwd.relative_to(home)}"
    except ValueError:
        display_path = str(cwd)

    try:
        branch = subprocess.check_output(
            ["git", "rev-parse", "--abbrev-ref", "HEAD"],
            stderr=subprocess.DEVNULL,
            text=True,
        ).strip()
        return f"{display_path} [⎇ {branch}]"
    except Exception:
        return display_path


def build_hero_banner(version: str = version) -> Panel:
    """Full-width ANSI Shadow header with concise startup help."""
    header = print_header(text="RawRAG", font="ansi_shadow", target=None)
    details = Text.from_markup(
        f"[#c9d1d9]RawRAG {version} · Local AI[/]\n"
        "[bold #c9d1d9]/index[/] index · "
        "[bold #c9d1d9]/model <name>[/] switch model · ask a question"
    )

    content = Table.grid(expand=True)
    content.add_column()
    content.add_row(header)
    content.add_row(details)

    return Panel(
        content,
        border_style="#30363d",
        style="on #0d1117",
        padding=(0, 0),
    )


class RawRAGApp(App):
    CSS = """
    Screen {
        background: #0d1117;
        color: #c9d1d9;
    }

    #top-tabs {
        height: 1;
        margin: 1 2 0 2;
        background: #0d1117;
    }

    #hero-box {
        height: auto;
        margin: 1 2 0 2;
    }

    #tab-content {
        height: 1fr;
        margin: 0 2;
    }

    #current-feed, #index-feed, #history-feed {
        height: 1fr;
        background: #0d1117;
        border: none;
        padding: 1 0;
    }

    #context-bar {
        height: 1;
        margin: 0 2;
        color: #8b949e;
    }

    #prompt-input {
        dock: bottom;
        margin: 0 2 2 2;
        background: #161b22;
        color: #f0f6fc;
        border-top: solid #30363d;
        border-bottom: solid #30363d;
        border-left: none;
        border-right: none;
        padding: 0 1;
    }

    #prompt-input:focus {
        border-top: solid #58a6ff;
        border-bottom: solid #58a6ff;
    }

    #bottom-bar {
        dock: bottom;
        height: 1;
        margin: 0 2 1 2;
        color: #8b949e;
    }
    """

    BINDINGS = [
        Binding("tab", "next_tab", "Next Tab", priority=True),
        Binding("ctrl+c", "quit", "Quit", priority=True),
    ]

    def __init__(self, model: str = "llama3") -> None:
        super().__init__()
        self.active_tab_idx = 0
        self.model = model
        self.history: list[tuple[str, str]] = []

    def compose(self) -> ComposeResult:
        yield Static(id="top-tabs")
        yield Static(build_hero_banner(), id="hero-box")
        with ContentSwitcher(id="tab-content", initial="current-feed"):
            yield RichLog(id="current-feed", wrap=True, markup=True)
            yield RichLog(id="index-feed", wrap=True, markup=True)
            yield RichLog(id="history-feed", wrap=True, markup=True)
        yield Static(id="context-bar")
        yield Input(placeholder="Ask a question about your repository or type /help...", id="prompt-input")
        yield Static(id="bottom-bar")

    def on_mount(self) -> None:
        self._render_tabs()
        self._render_status_bars()
        self._log_startup_checks()
        self.query_one("#prompt-input", Input).focus()

    def _render_tabs(self) -> None:
        bar = Text()
        for idx, name in enumerate(TABS):
            if idx == self.active_tab_idx:
                bar.append(f" {name} ", style="bold #ffffff on #1f6feb")
            else:
                bar.append(f" {name} ", style="#8b949e on #161b22")
            bar.append("  ")
        self.query_one("#top-tabs", Static).update(bar)

    def _render_status_bars(self) -> None:
        repo_str = get_repo_context()
        ctx = Text.from_markup(
            f"[#8b949e]{repo_str}[/]    [dim]Ollama model:[/] [#c9d1d9]{self.model}[/]"
        )
        self.query_one("#context-bar", Static).update(ctx)

        footer = Text.from_markup(
            "[#8b949e]Interactive · Local RAG · [bold #c9d1d9]/[/] commands · "
            "[bold #c9d1d9]?[/] help · [bold #c9d1d9]tab[/] next tab[/]"
        )
        self.query_one("#bottom-bar", Static).update(footer)

    def _log_startup_checks(self) -> None:
        feed = self.query_one("#current-feed", RichLog)
        db_exists = Path("rawrag_index.db").exists()
        vec_exists = Path("rawrag_vectors.npy").exists() and Path("vector_ids.json").exists()

        if db_exists and vec_exists:
            feed.write("[#c9d1d9]• Index files detected (rawrag_index.db + vector_ids.json)[/]")
        else:
            feed.write(
                "[#c9d1d9]• No index found in current directory. Run [bold #58a6ff]/index[/] to build it.[/]\n"
                "  [#3fb950]└ Ready to scan repository[/]"
            )

        feed.write(f"[#c9d1d9]• Local LLM target: Ollama ([bold]{self.model}[/])[/]")

    def action_next_tab(self) -> None:
        self._activate_tab((self.active_tab_idx + 1) % len(TABS))

    def _activate_tab(self, index: int) -> None:
        self.active_tab_idx = index
        self._render_tabs()
        tab_feeds = ("current-feed", "index-feed", "history-feed")
        self.query_one("#tab-content", ContentSwitcher).current = tab_feeds[index]

        if index == 1:
            self._render_index_overview()
        elif index == 2:
            self._render_history()

    def _render_index_overview(self) -> None:
        feed = self.query_one("#index-feed", RichLog)
        feed.clear()
        feed.write("[bold #58a6ff]• Index Overview[/]")
        for fname in ("rawrag_index.db", "rawrag_vectors.npy", "vector_ids.json"):
            path = Path(fname)
            status = f"{path.stat().st_size / 1024:.1f} KB" if path.exists() else "missing"
            feed.write(f"  [#8b949e]└ {fname}: {status}[/]")

    def _render_history(self) -> None:
        feed = self.query_one("#history-feed", RichLog)
        feed.clear()
        feed.write(f"[bold #58a6ff]• Session History ({len(self.history)} queries)[/]")
        for question, _ in self.history[-5:]:
            feed.write(f"  [#8b949e]└ ❯ {question}[/]")

    @on(Input.Submitted)
    def handle_submit(self, event: Input.Submitted) -> None:
        raw = event.value.strip()
        event.input.clear()
        if not raw:
            return

        self._activate_tab(0)
        self.query_one("#hero-box", Static).display = False
        feed = self.query_one("#current-feed", RichLog)

        if raw in {"/exit", "/quit"}:
            self.exit()
            return

        if raw == "/clear":
            feed.clear()
            return

        if raw in {"/help", "?"}:
            feed.write(
                "\n[#c9d1d9]• Available commands:[/]\n"
                "  [#8b949e]└ [bold #c9d1d9]/index[/]         Index the current directory[/]\n"
                "  [#8b949e]└ [bold #c9d1d9]/model <name>[/]  Switch Ollama model[/]\n"
                "  [#8b949e]└ [bold #c9d1d9]/clear[/]         Clear the output feed[/]\n"
                "  [#8b949e]└ [bold #c9d1d9]/exit[/]          Exit RawRAG[/]"
            )
            return

        if raw.startswith("/model"):
            parts = raw.split(maxsplit=1)
            if len(parts) == 2:
                self.model = parts[1].strip()
                self._render_status_bars()
                feed.write(f"\n[#c9d1d9]• Switched Ollama model to [bold #3fb950]{self.model}[/][/]")
            else:
                feed.write(f"\n[#c9d1d9]• Active model: [bold]{self.model}[/]. Usage: [bold]/model qwen2.5-coder[/][/]")
            return

        if raw == "/index":
            feed.write("\n[#c9d1d9]• Indexing repository...[/]")
            self.run_indexing_worker()
            return

        # Standard RAG question
        feed.write(f"\n[bold #58a6ff]❯ {raw}[/]")
        self.run_query_worker(raw, self.model)

    @work(thread=True)
    def run_indexing_worker(self) -> None:
        feed = self.query_one("#current-feed", RichLog)
        try:
            file_count = index_directory(Path.cwd())
        except (FileNotFoundError, OSError, RuntimeError, ValueError) as error:
            self.call_from_thread(feed.write, f"  [bold red]Indexing failed: {error}[/]")
            return
        self.call_from_thread(feed.write, f"  [#3fb950]└ Index updated: {file_count} files.[/]")
        self.call_from_thread(self._render_index_overview)

    @work(thread=True)
    def run_query_worker(self, question: str, model: str) -> None:
        feed = self.query_one("#current-feed", RichLog)
        self.call_from_thread(feed.write, "  [#8b949e]└ Retrieving chunks & querying Ollama...[/]")

        try:
            answer, chunks = query_codebase(question, model=model)
        except (IndexConsistencyError, IndexNotFoundError, NoRelevantChunksError) as error:
            self.call_from_thread(feed.write, f"  [bold red]Query failed: {error}[/]")
            return
        except requests.exceptions.ConnectionError as error:
            self.call_from_thread(
                feed.write,
                "  [bold red]Could not connect to Ollama. Make sure Ollama is running (`ollama serve`).[/]",
            )
            return
        except requests.exceptions.HTTPError as error:
            detail = error.response.text.strip() if error.response is not None else str(error)
            self.call_from_thread(feed.write, f"  [bold red]Ollama API error: {detail or error}[/]")
            return
        except requests.exceptions.Timeout:
            self.call_from_thread(feed.write, "  [bold red]Ollama timed out while generating the answer.[/]")
            return
        except requests.exceptions.RequestException as error:
            self.call_from_thread(feed.write, f"  [bold red]API error: {error}[/]")
            return

        self.call_from_thread(self.history.append, (question, answer))
        self.call_from_thread(self._render_history)
        block = Group(
            Markdown(answer),
            Text(
                f"\nSources: {', '.join(_source_from_chunk(chunk) for chunk in chunks)}",
                style="dim #8b949e",
            ),
        )
        self.call_from_thread(
            feed.write,
            Panel(block, border_style="#30363d", padding=(0, 1)),
        )


def main() -> None:
    RawRAGApp().run()


def _source_from_chunk(chunk: str) -> str:
    first_line, _, _ = chunk.partition("\n")
    return first_line.removeprefix("File: ")


if __name__ == "__main__":
    main()