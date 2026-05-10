"""M.I.R.A. CLI – run analyses, monitor tickers, inspect jobs from your terminal.

Two execution modes:
  * --local   : run AgentCore in-process, no HTTP server required (fastest)
  * (default) : talk to a running M.I.R.A. API server (default http://localhost:8000)

Examples
--------
  mira analyze "Analyse Tesla (TSLA)" --local
  mira analyze "Analyse Apple (AAPL)" --watch
  mira status <job_id>
  mira logs <job_id> --follow
  mira monitor add TSLA --cadence 12
  mira monitor list
  mira monitor remove TSLA
  mira health
  mira config
  mira jobs --limit 10
"""

from __future__ import annotations

import asyncio
import json
import os
import sys
import time
import uuid
from pathlib import Path
from typing import Optional

# Ensure the project root is on PYTHONPATH whether run via `python cli.py` or
# the installed `mira` console script.
_ROOT = Path(__file__).resolve().parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import httpx
import typer
from rich.console import Console
from rich.json import JSON as RichJSON
from rich.live import Live
from rich.markdown import Markdown
from rich.panel import Panel
from rich.progress import (
    BarColumn,
    Progress,
    SpinnerColumn,
    TextColumn,
    TimeElapsedColumn,
)
from rich.table import Table
from rich.text import Text

app = typer.Typer(
    name="mira",
    help="M.I.R.A. — Market Intelligence & Research Agent CLI.",
    add_completion=False,
    no_args_is_help=True,
    rich_markup_mode="rich",
)
monitor_app = typer.Typer(help="Manage persistent ticker monitoring.", no_args_is_help=True)
app.add_typer(monitor_app, name="monitor")

console = Console()
err_console = Console(stderr=True, style="bold red")

# ── globals shared via callback ──────────────────────────────────────────────
_DEFAULT_API = os.environ.get("MIRA_API_URL", "http://localhost:8000")


class CLIState:
    api_url: str = _DEFAULT_API
    local: bool = False
    json_out: bool = False
    quiet: bool = False


state = CLIState()


@app.callback()
def _root(
    api: str = typer.Option(
        _DEFAULT_API,
        "--api",
        help="Base URL of a running M.I.R.A. API. Ignored with --local. "
             "Override via MIRA_API_URL env var.",
        envvar="MIRA_API_URL",
    ),
    local: bool = typer.Option(
        False,
        "--local",
        help="Run the agent in-process (no HTTP server). Requires Redis and full deps.",
    ),
    json_output: bool = typer.Option(
        False, "--json", help="Emit raw JSON output (good for piping)."
    ),
    quiet: bool = typer.Option(False, "--quiet", "-q", help="Suppress decorative output."),
) -> None:
    """Top-level options shared by every subcommand."""
    state.api_url = api.rstrip("/")
    state.local = local
    state.json_out = json_output
    state.quiet = quiet


# ╭─────────────────────────────────────────────────────────────────────────────╮
# │  helpers                                                                    │
# ╰─────────────────────────────────────────────────────────────────────────────╯
def _emit(payload: dict | list, fallback_renderer=None) -> None:
    """Emit either raw JSON or a pretty rendering."""
    if state.json_out:
        console.print_json(data=payload)
        return
    if fallback_renderer is not None:
        fallback_renderer(payload)
    else:
        console.print_json(data=payload)


def _http() -> httpx.Client:
    return httpx.Client(base_url=state.api_url, timeout=30.0)


def _async_http() -> httpx.AsyncClient:
    return httpx.AsyncClient(base_url=state.api_url, timeout=30.0)


def _die(msg: str, code: int = 1) -> None:
    err_console.print(f"✗ {msg}")
    raise typer.Exit(code=code)


def _short(text: str | None, n: int = 60) -> str:
    if not text:
        return ""
    return text if len(text) <= n else text[: n - 1] + "…"


# ╭─────────────────────────────────────────────────────────────────────────────╮
# │  banner                                                                     │
# ╰─────────────────────────────────────────────────────────────────────────────╯
_BANNER = r"""[bold magenta]
███╗   ███╗    ██╗    ██████╗    █████╗
████╗ ████║    ██║    ██╔══██╗  ██╔══██╗
██╔████╔██║    ██║    ██████╔╝  ███████║
██║╚██╔╝██║    ██║    ██╔══██╗  ██╔══██║
██║ ╚═╝ ██║    ██║    ██║  ██║  ██║  ██║
╚═╝     ╚═╝    ╚═╝    ╚═╝  ╚═╝  ╚═╝  ╚═╝[/bold magenta]
[dim]Market Intelligence & Research Agent — CLI[/dim]
"""


def _banner() -> None:
    if not state.quiet and not state.json_out:
        console.print(_BANNER)


# ╭─────────────────────────────────────────────────────────────────────────────╮
# │  analyze                                                                    │
# ╰─────────────────────────────────────────────────────────────────────────────╯
@app.command()
def analyze(
    query: str = typer.Argument(..., help="Natural-language analysis query."),
    no_watch: bool = typer.Option(
        False, "--no-watch",
        help="Submit-and-forget. Default is to poll until completion.",
    ),
    poll_interval: float = typer.Option(
        1.5, "--interval", help="Seconds between status polls.",
    ),
    timeout: int = typer.Option(
        300, "--timeout", help="Max seconds to wait for completion.",
    ),
    save: Optional[Path] = typer.Option(
        None, "--save", help="Write the final report JSON to this path.",
    ),
) -> None:
    """Submit an analysis query and (optionally) wait for the report."""
    watch = not no_watch
    _banner()

    if state.local:
        asyncio.run(_analyze_local(query, save=save))
        return

    try:
        with _http() as client:
            resp = client.post("/analyze", json={"query": query})
            resp.raise_for_status()
            job = resp.json()
    except httpx.HTTPError as exc:
        _die(f"Failed to submit analysis: {exc}")

    job_id: str = job["job_id"]
    if not state.quiet:
        console.print(
            Panel(
                f"[bold]Job ID[/bold]   [cyan]{job_id}[/cyan]\n"
                f"[bold]Query[/bold]    {query}\n"
                f"[bold]Server[/bold]   {state.api_url}",
                title="Analysis queued",
                border_style="magenta",
                expand=False,
            )
        )

    if not watch:
        _emit(job)
        return

    report = _watch_job(job_id, poll_interval=poll_interval, timeout=timeout)
    if report is None:
        return

    _render_report(report)
    if save:
        save.write_text(json.dumps(report, indent=2))
        console.print(f"[green]✓[/green] Report saved to [bold]{save}[/bold]")


async def _analyze_local(query: str, save: Optional[Path] = None) -> None:
    """Run AgentCore directly in-process (no HTTP server needed)."""
    from agent.core import AgentCore
    from storage.redis_client import RedisClient

    redis = RedisClient()
    if not redis.ping():
        _die("Redis is not reachable. Set REDIS_HOST / REDIS_PORT or start `redis-server`.")

    job_id = str(uuid.uuid4())
    redis.set_job_status(
        job_id,
        {"job_id": job_id, "status": "queued", "progress": 0, "tool_calls_used": 0},
    )

    if not state.quiet:
        console.print(
            Panel(
                f"[bold]Job ID[/bold]   [cyan]{job_id}[/cyan]\n"
                f"[bold]Query[/bold]    {query}\n"
                f"[bold]Mode[/bold]     [yellow]local (in-process)[/yellow]",
                title="Analysis starting",
                border_style="magenta",
                expand=False,
            )
        )

    agent = AgentCore()

    progress = Progress(
        SpinnerColumn(),
        TextColumn("[bold blue]{task.description}"),
        BarColumn(),
        TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
        TimeElapsedColumn(),
        console=console,
        transient=True,
        disable=state.quiet or state.json_out,
    )
    task_id = progress.add_task("queued", total=100)

    async def _poller() -> None:
        last_status = ""
        while True:
            await asyncio.sleep(0.5)
            s = redis.get_job_status(job_id) or {}
            status = s.get("status", "?")
            progress.update(task_id, completed=int(s.get("progress", 0) or 0), description=status)
            if status != last_status:
                last_status = status
            if status in ("completed", "failed"):
                return

    with progress:
        poller = asyncio.create_task(_poller())
        runner = asyncio.create_task(agent.run_analysis(job_id, query))
        await asyncio.gather(runner, poller, return_exceptions=True)

    s = redis.get_job_status(job_id) or {}
    if s.get("status") == "failed":
        _die(f"Analysis failed: {s.get('error', 'unknown error')}")

    report = redis.get_job_result(job_id) or {}
    _render_report(report)
    if save:
        save.write_text(json.dumps(report, indent=2))
        console.print(f"[green]✓[/green] Report saved to [bold]{save}[/bold]")


# ╭─────────────────────────────────────────────────────────────────────────────╮
# │  watch / poll                                                               │
# ╰─────────────────────────────────────────────────────────────────────────────╯
def _watch_job(job_id: str, poll_interval: float, timeout: int) -> dict | None:
    """Poll /status/{job_id} with a live progress bar until terminal state."""
    deadline = time.time() + timeout
    progress = Progress(
        SpinnerColumn(),
        TextColumn("[bold blue]{task.description}"),
        BarColumn(),
        TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
        TimeElapsedColumn(),
        console=console,
        transient=True,
        disable=state.quiet or state.json_out,
    )
    task_id = progress.add_task("queued", total=100)

    last_status = ""
    with progress, _http() as client:
        while time.time() < deadline:
            try:
                resp = client.get(f"/status/{job_id}")
                resp.raise_for_status()
                data = resp.json()
            except httpx.HTTPError as exc:
                _die(f"Status poll failed: {exc}")

            status = data.get("status") or "?"
            pct = int(data.get("progress") or 0)
            progress.update(task_id, completed=pct, description=status)
            last_status = status

            if status == "completed":
                report = data.get("result") or {}
                if not report:
                    # Fetch result via /status one more time to be safe.
                    resp = client.get(f"/status/{job_id}")
                    report = resp.json().get("result") or {}
                return report
            if status == "failed":
                _die(f"Analysis failed: {data.get('error', 'unknown error')}")

            time.sleep(poll_interval)

    _die(f"Timed out after {timeout}s waiting for job {job_id} (last status: {last_status})")
    return None


# ╭─────────────────────────────────────────────────────────────────────────────╮
# │  status                                                                     │
# ╰─────────────────────────────────────────────────────────────────────────────╯
@app.command()
def status(job_id: str = typer.Argument(..., help="UUID returned by `analyze`.")) -> None:
    """Fetch the current status of a job."""
    if state.local:
        from storage.redis_client import RedisClient
        data = RedisClient().get_job_status(job_id) or {}
        if not data:
            _die(f"Job {job_id} not found.")
    else:
        try:
            with _http() as client:
                resp = client.get(f"/status/{job_id}")
                resp.raise_for_status()
                data = resp.json()
        except httpx.HTTPError as exc:
            _die(f"Status request failed: {exc}")

    def _render(d: dict) -> None:
        table = Table(title=f"Job {job_id}", title_style="bold magenta")
        table.add_column("Field", style="cyan")
        table.add_column("Value", style="white")
        for key in ("status", "progress", "tool_calls_used", "budget_exceeded", "tag", "error"):
            if d.get(key) not in (None, ""):
                table.add_row(key, str(d.get(key)))
        console.print(table)
        if d.get("status") == "completed" and d.get("result"):
            _render_report(d["result"])

    _emit(data, _render)


# ╭─────────────────────────────────────────────────────────────────────────────╮
# │  logs                                                                       │
# ╰─────────────────────────────────────────────────────────────────────────────╯
@app.command()
def logs(
    job_id: str = typer.Argument(..., help="Job UUID."),
    follow: bool = typer.Option(False, "--follow", "-f", help="Stream new logs as they appear."),
    interval: float = typer.Option(1.0, "--interval", help="Seconds between polls when following."),
) -> None:
    """Show structured tool-invocation logs for a job."""
    if follow:
        _follow_logs(job_id, interval=interval)
        return

    data = _fetch_logs(job_id)

    def _render(d: dict) -> None:
        rows = d.get("logs", [])
        if not rows:
            console.print("[yellow]No logs yet.[/yellow]")
            return
        table = Table(title=f"Logs — {job_id}", title_style="bold magenta")
        table.add_column("#", style="dim")
        table.add_column("Tool", style="cyan")
        table.add_column("Latency (ms)", justify="right")
        table.add_column("Success", justify="center")
        table.add_column("Note")
        for i, row in enumerate(rows, 1):
            note = row.get("error") if not row.get("success", True) else _short(
                json.dumps(row.get("outputs"))[:80] if row.get("outputs") else ""
            )
            ok = "[green]✓[/green]" if row.get("success", True) else "[red]✗[/red]"
            table.add_row(
                str(i),
                row.get("tool_name", "?"),
                f"{row.get('latency_ms', 0):.0f}",
                ok,
                _short(str(note), 80),
            )
        console.print(table)
        tu = d.get("token_usage") or {}
        if tu:
            console.print(
                f"[dim]tokens: prompt={tu.get('prompt_tokens', 0)} "
                f"completion={tu.get('completion_tokens', 0)} "
                f"total={tu.get('total_tokens', 0)} "
                f"≈ ${tu.get('estimated_cost_usd', 0):.4f}[/dim]"
            )

    _emit(data, _render)


def _fetch_logs(job_id: str) -> dict:
    if state.local:
        from storage.redis_client import RedisClient
        r = RedisClient()
        return {
            "job_id": job_id,
            "logs": r.get_job_logs(job_id),
            "token_usage": r.get_job_token_usage(job_id),
        }
    try:
        with _http() as client:
            resp = client.get(f"/logs/{job_id}")
            resp.raise_for_status()
            return resp.json()
    except httpx.HTTPError as exc:
        _die(f"Log fetch failed: {exc}")
        return {}  # unreachable


def _follow_logs(job_id: str, interval: float) -> None:
    seen = 0
    try:
        while True:
            data = _fetch_logs(job_id)
            rows = data.get("logs", [])
            for row in rows[seen:]:
                ok = "[green]✓[/green]" if row.get("success", True) else "[red]✗[/red]"
                console.print(
                    f"{ok} [cyan]{row.get('tool_name', '?')}[/cyan] "
                    f"[dim]{row.get('latency_ms', 0):.0f}ms[/dim] "
                    f"{_short(json.dumps(row.get('outputs') or row.get('error') or ''), 100)}"
                )
            seen = len(rows)
            # Stop when the related job is in a terminal state.
            try:
                if state.local:
                    from storage.redis_client import RedisClient
                    s = (RedisClient().get_job_status(job_id) or {}).get("status")
                else:
                    with _http() as c:
                        s = c.get(f"/status/{job_id}").json().get("status")
                if s in ("completed", "failed"):
                    return
            except httpx.HTTPError:
                pass
            time.sleep(interval)
    except KeyboardInterrupt:
        return


# ╭─────────────────────────────────────────────────────────────────────────────╮
# │  monitor                                                                    │
# ╰─────────────────────────────────────────────────────────────────────────────╯
@monitor_app.command("add")
def monitor_add(
    ticker: str = typer.Argument(..., help="Ticker symbol, e.g. TSLA."),
    cadence: int = typer.Option(24, "--cadence", help="Hours between proactive checks (1-168)."),
) -> None:
    """Register a ticker for background monitoring."""
    if cadence < 1 or cadence > 168:
        _die("--cadence must be between 1 and 168 hours.")

    if state.local:
        from storage.redis_client import RedisClient
        RedisClient().register_monitored_ticker(ticker.upper(), cadence)
        result = {"status": "monitoring_started", "ticker": ticker.upper(), "cadence_hours": cadence}
    else:
        try:
            with _http() as client:
                resp = client.post(
                    "/monitor_start", json={"ticker": ticker.upper(), "cadence_hours": cadence}
                )
                resp.raise_for_status()
                result = resp.json()
        except httpx.HTTPError as exc:
            _die(f"Monitor add failed: {exc}")

    _emit(
        result,
        lambda r: console.print(
            f"[green]✓[/green] Monitoring [bold]{r['ticker']}[/bold] every "
            f"[bold]{r['cadence_hours']}h[/bold]."
        ),
    )


@monitor_app.command("list")
def monitor_list() -> None:
    """List all monitored tickers."""
    if state.local:
        from storage.redis_client import RedisClient
        r = RedisClient()
        tickers = r.get_all_monitored_tickers()
        rows = []
        for t in tickers:
            s = r.get_ticker_state(t) or {}
            rows.append({
                "ticker": t,
                "cadence_hours": s.get("cadence_hours"),
                "last_run": s.get("last_run"),
                "baseline_price": s.get("baseline_price"),
            })
    else:
        try:
            with _http() as client:
                resp = client.get("/monitor/list")
                resp.raise_for_status()
                rows = resp.json().get("monitored", [])
        except httpx.HTTPError as exc:
            _die(f"Monitor list failed: {exc}")

    def _render(items: list[dict]) -> None:
        if not items:
            console.print("[yellow]No tickers monitored.[/yellow]")
            return
        table = Table(title="Monitored tickers", title_style="bold magenta")
        for col in ("Ticker", "Cadence (h)", "Last run", "Baseline price"):
            table.add_column(col)
        for it in items:
            table.add_row(
                it["ticker"],
                str(it.get("cadence_hours") or ""),
                str(it.get("last_run") or "—"),
                f"{it.get('baseline_price') or '—'}",
            )
        console.print(table)

    _emit(rows, _render)


@monitor_app.command("remove")
def monitor_remove(ticker: str = typer.Argument(..., help="Ticker symbol to stop monitoring.")) -> None:
    """Remove a ticker from monitoring."""
    if state.local:
        from storage.redis_client import RedisClient
        removed = RedisClient().unregister_monitored_ticker(ticker.upper())
    else:
        try:
            with _http() as client:
                resp = client.delete(f"/monitor/{ticker.upper()}")
                resp.raise_for_status()
                removed = resp.json().get("status") == "removed"
        except httpx.HTTPError as exc:
            _die(f"Monitor remove failed: {exc}")
            return

    if removed:
        console.print(f"[green]✓[/green] Stopped monitoring [bold]{ticker.upper()}[/bold].")
    else:
        console.print(f"[yellow]![/yellow] {ticker.upper()} was not in the monitored set.")


# ╭─────────────────────────────────────────────────────────────────────────────╮
# │  jobs                                                                       │
# ╰─────────────────────────────────────────────────────────────────────────────╯
@app.command()
def jobs(limit: int = typer.Option(20, "--limit", help="Max jobs to show.")) -> None:
    """List recent jobs (scans Redis on the server side)."""
    if state.local:
        from storage.redis_client import RedisClient
        rows = RedisClient().list_recent_jobs(limit=limit)
    else:
        try:
            with _http() as client:
                resp = client.get("/jobs", params={"limit": limit})
                resp.raise_for_status()
                rows = resp.json().get("jobs", [])
        except httpx.HTTPError as exc:
            _die(f"Job list failed: {exc}")
            return

    def _render(items: list[dict]) -> None:
        if not items:
            console.print("[yellow]No jobs found.[/yellow]")
            return
        table = Table(title=f"Recent jobs (top {len(items)})", title_style="bold magenta")
        for col in ("Job ID", "Status", "Progress", "Tag", "Created"):
            table.add_column(col)
        for it in items:
            status = it.get("status", "?")
            colour = {
                "completed": "green",
                "failed": "red",
                "queued": "yellow",
            }.get(status, "cyan")
            table.add_row(
                _short(it.get("job_id", "?"), 36),
                f"[{colour}]{status}[/{colour}]",
                f"{it.get('progress', 0)}%",
                _short(it.get("tag") or "", 16),
                _short(it.get("created_at") or "", 25),
            )
        console.print(table)

    _emit(rows, _render)


# ╭─────────────────────────────────────────────────────────────────────────────╮
# │  health / config                                                            │
# ╰─────────────────────────────────────────────────────────────────────────────╯
@app.command()
def ui(
    no_open: bool = typer.Option(
        False, "--no-open", help="Just print the demo URL instead of opening a browser.",
    ),
) -> None:
    """Open the Neural Core demo in your browser (the API must be running).

    Equivalent to navigating to `--api` (defaults to http://localhost:8000/).
    """
    import webbrowser

    url = state.api_url + "/"
    # Quick health check so we fail fast if the server is down.
    try:
        with _http() as client:
            client.get("/health", timeout=2.0).raise_for_status()
        reachable = True
    except Exception:
        reachable = False

    if not reachable:
        err_console.print(
            f"[yellow]![/yellow] Could not reach {state.api_url}/health. "
            "Make sure the API is running:\n"
            "  uvicorn api.main:app --reload\n"
            "  # or\n"
            "  docker compose up --build\n"
        )

    if no_open:
        console.print(f"[cyan]→[/cyan] Demo URL: {url}")
    else:
        console.print(f"[cyan]→[/cyan] Opening {url}")
        webbrowser.open(url)


@app.command()
def serve(
    host: str = typer.Option("0.0.0.0", "--host", help="Bind address."),
    port: int = typer.Option(8000, "--port", help="Bind port."),
    reload: bool = typer.Option(False, "--reload", help="Enable auto-reload (dev only)."),
    open_browser: bool = typer.Option(False, "--open", help="Open the demo in your browser once the server is up."),
) -> None:
    """Start the FastAPI server (so `/` serves the demo HTML directly)."""
    try:
        import uvicorn
    except ImportError:
        _die("uvicorn is not installed. Run `pip install -e '.[full]'` or install uvicorn manually.")

    if open_browser:
        # Best-effort: open the browser shortly after the server boots.
        import threading
        import time as _time
        import webbrowser

        def _open_later() -> None:
            _time.sleep(1.5)
            webbrowser.open(f"http://localhost:{port}/")

        threading.Thread(target=_open_later, daemon=True).start()

    uvicorn.run("api.main:app", host=host, port=port, reload=reload)


@app.command()
def health() -> None:
    """Report API + Redis health."""
    if state.local:
        from storage.redis_client import RedisClient
        ok = RedisClient().ping()
        data = {"status": "ok" if ok else "degraded", "redis": ok, "mode": "local"}
    else:
        try:
            with _http() as client:
                resp = client.get("/health")
                resp.raise_for_status()
                data = resp.json()
        except httpx.HTTPError as exc:
            _die(f"Health check failed: {exc}")

    def _render(d: dict) -> None:
        colour = "green" if d.get("status") == "ok" else "yellow"
        console.print(
            Panel(
                f"status:  [{colour}]{d.get('status')}[/{colour}]\n"
                f"redis:   {d.get('redis')}\n"
                f"mode:    {d.get('mode', 'remote')}\n"
                f"target:  {state.api_url if not state.local else 'in-process'}",
                title="Health",
                border_style=colour,
                expand=False,
            )
        )

    _emit(data, _render)


@app.command()
def config() -> None:
    """Print active LLM configuration (LLM provider + models)."""
    if state.local:
        from utils.config import get_settings
        s = get_settings()
        data = {
            "llm_provider": s.llm_provider,
            "ollama_model": s.ollama_model,
            "hf_model_id": s.llm_hf_model_id,
            "hf_sentiment_model": s.hf_model_id,
        }
    else:
        try:
            with _http() as client:
                resp = client.get("/config")
                resp.raise_for_status()
                data = resp.json()
        except httpx.HTTPError as exc:
            _die(f"Config fetch failed: {exc}")

    def _render(d: dict) -> None:
        table = Table(title="Active LLM configuration", title_style="bold magenta")
        table.add_column("Key", style="cyan")
        table.add_column("Value", style="white")
        for k, v in d.items():
            table.add_row(k, str(v))
        console.print(table)

    _emit(data, _render)


# ╭─────────────────────────────────────────────────────────────────────────────╮
# │  report renderer                                                            │
# ╰─────────────────────────────────────────────────────────────────────────────╯
def _render_report(report: dict) -> None:
    if state.json_out:
        console.print_json(data=report)
        return
    if state.quiet:
        # Print only the summary on a single line so CI grep stays sane.
        print(report.get("analysis_summary", ""))
        return

    ticker = report.get("company_ticker", "?")
    name = report.get("company_name", "")
    summary = report.get("analysis_summary", "")
    local_s = report.get("sentiment_score")
    cloud_s = report.get("hf_sentiment_score")

    header = Text(f"{ticker} — {name}", style="bold magenta")
    console.print(Panel(header, expand=False, border_style="magenta"))

    if summary:
        console.print(Panel(Markdown(summary), title="Summary", border_style="cyan", expand=True))

    snap = report.get("market_snapshot", {}) or {}
    if snap:
        t = Table(title="Market snapshot", title_style="bold cyan")
        for col in ("Metric", "Value"):
            t.add_column(col)
        for k in ("price", "daily_change_pct", "market_cap", "pe_ratio", "52w_high", "52w_low"):
            v = snap.get(k)
            t.add_row(k, str(v) if v is not None else "—")
        console.print(t)

    corr = report.get("correlation_analysis", {}) or {}
    if corr.get("all_correlations"):
        t = Table(title="Correlations (1y, daily returns)", title_style="bold cyan")
        for col in ("Symbol", "ρ"):
            t.add_column(col)
        for sym, rho in corr.get("all_correlations", {}).items():
            colour = "green" if rho is not None and abs(rho) >= 0.7 else "white"
            t.add_row(sym, f"[{colour}]{rho}[/{colour}]")
        console.print(t)

    sent_table = Table(title="Sentiment", title_style="bold cyan")
    for col in ("Source", "Score"):
        sent_table.add_column(col)
    sent_table.add_row("Local FinBERT", _fmt_sentiment(local_s))
    sent_table.add_row("HF Cloud", _fmt_sentiment(cloud_s))
    console.print(sent_table)

    findings = report.get("key_findings") or []
    if findings:
        body = "\n".join(f"{i+1}. {f}" for i, f in enumerate(findings))
        console.print(Panel(body, title="Key findings", border_style="green", expand=True))

    refl = report.get("reflection") or {}
    triggers = refl.get("triggers_fired") or []
    if triggers:
        console.print(
            Panel(
                f"Triggers fired: [yellow]{', '.join(triggers)}[/yellow]\n"
                f"Reasoning: {refl.get('reasoning', '')}",
                title="Reflection",
                border_style="yellow",
                expand=False,
            )
        )

    cites = report.get("citation_sources") or []
    if cites:
        console.print(Panel("\n".join(cites[:10]), title="Citations", border_style="blue", expand=False))


def _fmt_sentiment(score) -> str:
    if score is None or score == "N/A":
        return "—"
    try:
        s = float(score)
    except (TypeError, ValueError):
        return str(score)
    if s > 0.05:
        return f"[green]+{s:+.3f}[/green]"
    if s < -0.05:
        return f"[red]{s:+.3f}[/red]"
    return f"[yellow]{s:+.3f}[/yellow]"


# ╭─────────────────────────────────────────────────────────────────────────────╮
# │  entrypoint                                                                 │
# ╰─────────────────────────────────────────────────────────────────────────────╯
def main() -> None:
    app()


if __name__ == "__main__":
    main()
