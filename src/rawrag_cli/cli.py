
from rich.markdown import Markdown
from rich.panel import Panel
import typer
import pyfiglet
from rich.console import Console
from rich.text import Text
import requests

from rawrag_cli.core.pipeline import (
    IndexConsistencyError,
    IndexNotFoundError,
    NoRelevantChunksError,
    index_directory,
    query_codebase,
)

app = typer.Typer()
console = Console()

def print_header(
    text: str = "RawRAG CLI",
    font: str = "ansi_shadow",
    target: Console | None = console,
) -> Text:
    header = Text(pyfiglet.figlet_format(text, font=font), style="bold #0045b5")
    if target is not None:
        target.print(header)
    return header

@app.command()
def index(directory: str = typer.Argument(".", help="Directory to index")):
    print_header()
    console.print(f"Indexing files in directory: [cyan]{directory}[/cyan]")

    try:
        with console.status("[bold yellow]Chunking and embedding files (this might take a minute)...[/bold yellow]", spinner="bouncingBar"):
            file_count = index_directory(directory)
    except FileNotFoundError as error:
        console.print(f"[bold red]{error}[/bold red]")
        raise typer.Exit(code=1) from error

    console.print(f"[bold green]✔ Successfully indexed {file_count} files![/bold green]")
    
@app.command()
def query(question: str):
    print_header()

    with console.status("[bold yellow]Processing your query...[/bold yellow]", spinner="bouncingBar"):
        try:
            answer, _ = query_codebase(question)
        except (IndexConsistencyError, IndexNotFoundError, NoRelevantChunksError) as error:
            console.print(f"[bold red]Index Error:[/bold red] {error}")
            raise typer.Exit(code=1)
        except requests.exceptions.ConnectionError as error:
            console.print("[bold red]Error:[/bold red] Could not connect to Ollama. Make sure Ollama is running in the background (`ollama serve`).")
            raise typer.Exit(code=1) from error
        except requests.exceptions.HTTPError as error:
            detail = error.response.text.strip() if error.response is not None else str(error)
            console.print(
                f"[bold red]Ollama API Error ({error.response.status_code if error.response is not None else 'unknown'}):[/bold red] "
                f"{detail or 'The server returned an internal error.'}"
            )
            raise typer.Exit(code=1) from error
        except requests.exceptions.Timeout as error:
            console.print("[bold red]Error:[/bold red] Ollama timed out while generating the answer.")
            raise typer.Exit(code=1) from error
        except requests.exceptions.RequestException as error:
            console.print(f"[bold red]API Error:[/bold red] {error}")
            raise typer.Exit(code=1) from error

    console.print(Panel(Markdown(answer), title="[bold green]RawRAG Answer[/bold green]", border_style="green"))


# Rag-logic

@app.callback(invoke_without_command=True)
def main(ctx: typer.Context) -> None:
    """RawRAG-CLI: Local codebase RAG."""
    if ctx.invoked_subcommand is None:
        from rawrag_cli.tui import main as tui_main

        tui_main()


if __name__ == "__main__":
    app()
