import io
from contextlib import redirect_stderr

from typer.testing import CliRunner

from otto_recsys.cli import app
from otto_recsys.progress import RichPipelineProgress


def test_rich_progress_shows_step_count_and_eta() -> None:
    output = io.StringIO()

    with redirect_stderr(output), RichPipelineProgress() as progress:
        progress.start_stage(5, 9, "Materialize training candidates", total=10, unit="sessions")
        progress.advance(4)
        progress.finish("completed")

    rendered = output.getvalue()
    assert "Step 5/9" in rendered
    assert "ETA" in rendered
    assert "completed" in rendered


def test_run_command_exposes_non_interactive_progress_option() -> None:
    result = CliRunner().invoke(app, ["run", "--help"])

    assert result.exit_code == 0, result.output
    assert "--no-progress" in result.output
    assert "--train" in result.output
    assert "--test" in result.output
    assert "--sample-submission" in result.output
    assert "--model-strategy" in result.output
