import unittest
from pathlib import Path
from unittest.mock import patch

from rich.text import Text
from textual.widgets import ContentSwitcher, Input, RichLog
from typer.testing import CliRunner

from rawrag_cli.cli import app
from rawrag_cli.tui import RawRAGApp, build_hero_banner


class CliTuiWiringTests(unittest.TestCase):
    def test_cli_without_subcommand_launches_tui(self):
        with patch.object(RawRAGApp, "run") as run:
            result = CliRunner().invoke(app, [])

        self.assertEqual(result.exit_code, 0, result.output)
        run.assert_called_once_with()


class TuiQueryWiringTests(unittest.IsolatedAsyncioTestCase):
    def test_banner_uses_cli_header_without_printing_to_terminal(self):
        with patch("rawrag_cli.tui.print_header", return_value=Text("RawRAG CLI")) as print_header:
            build_hero_banner()

        print_header.assert_called_once_with(text="RawRAG", font="ansi_shadow", target=None)

    async def test_prompt_accepts_typed_characters(self):
        app = RawRAGApp()
        async with app.run_test() as pilot:
            await pilot.press("r", "a", "w")
            prompt = app.query_one("#prompt-input", Input)
            footer = app.query_one("#bottom-bar")

            self.assertIs(app.focused, prompt)
            self.assertLessEqual(prompt.region.y + prompt.region.height, footer.region.y)
            self.assertEqual(prompt.value, "raw")

    async def test_index_command_calls_shared_pipeline(self):
        with patch("rawrag_cli.tui.index_directory", return_value=2) as index_directory:
            app = RawRAGApp()
            async with app.run_test() as pilot:
                prompt = app.query_one("#prompt-input", Input)
                prompt.value = "/index"
                await pilot.press("enter")
                await pilot.pause(0.2)

            index_directory.assert_called_once_with(Path.cwd())

    async def test_tab_key_switches_visible_tab_content(self):
        app = RawRAGApp()
        async with app.run_test() as pilot:
            switcher = app.query_one("#tab-content", ContentSwitcher)
            self.assertEqual(switcher.current, "current-feed")

            await pilot.press("tab")
            self.assertEqual(switcher.current, "index-feed")
            self.assertEqual(app.active_tab_idx, 1)

            await pilot.press("tab")
            self.assertEqual(switcher.current, "history-feed")
            self.assertEqual(app.active_tab_idx, 2)

    async def test_commands_display_feedback_in_expanded_current_feed(self):
        app = RawRAGApp()
        async with app.run_test() as pilot:
            prompt = app.query_one("#prompt-input", Input)
            prompt.value = "/help"
            await pilot.press("enter")
            await pilot.pause()

            feed = app.query_one("#current-feed", RichLog)
            self.assertFalse(app.query_one("#hero-box").display)
            self.assertGreater(feed.region.height, 5)
            self.assertGreater(len(feed.lines), 2)

    async def test_query_uses_active_model_and_shared_pipeline(self):
        with patch("rawrag_cli.tui.query_codebase", return_value=("Found it.", ["File: src/example.py\nCode:\nexample"])) as query:
            app = RawRAGApp()
            async with app.run_test() as pilot:
                prompt = app.query_one("#prompt-input", Input)
                prompt.value = "/model qwen2.5-coder"
                await pilot.press("enter")
                prompt.value = "where is it?"
                await pilot.press("enter")
                await pilot.pause(0.2)

            query.assert_called_once_with("where is it?", model="qwen2.5-coder")
            self.assertIn(("where is it?", "Found it."), app.history)


if __name__ == "__main__":
    unittest.main()
