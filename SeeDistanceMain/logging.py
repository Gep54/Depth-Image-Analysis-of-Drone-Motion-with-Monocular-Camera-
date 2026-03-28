from pathlib import Path
from datetime import datetime


class Logger:
    """Simple logger for console or file output."""

    def __init__(self, log_file: str | Path | None = None) -> None:
        """Initialize the logger.

        Args:
            log_file: Path to the log file. If empty or None, logs are printed
                to the console instead.
        """
        self.log_file = Path(log_file) if log_file else None

    def _format_message(self, level: str, message: str) -> str:
        """Format a log message with timestamp and level."""
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        return f"[{timestamp}] [{level.upper()}] {message}"

    def _write(self, formatted_message: str) -> None:
        """Write a log message to file or console."""
        if self.log_file:
            self.log_file.parent.mkdir(parents=True, exist_ok=True)
            with self.log_file.open("a", encoding="utf-8") as file:
                file.write(formatted_message + "\n")
        else:
            print(formatted_message)

    def info(self, message: str) -> None:
        """Log an informational message."""
        self._write(self._format_message("INFO", message))

    def warning(self, message: str) -> None:
        """Log a warning message."""
        self._write(self._format_message("WARNING", message))

    def error(self, message: str) -> None:
        """Log an error message."""
        self._write(self._format_message("ERROR", message))