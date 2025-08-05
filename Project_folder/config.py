# config.py

# GUI
WINDOW_WIDTH = 1000
WINDOW_HEIGHT = 630
TITLE = "Payment Calculator"
CONSOLE_LINE_WIDTH = 80
TYPEWRITER_DELAY_MS = 5
TYPEWRITER_BLOCK_THRESHOLD = 2000  # characters; longer is inserted as a block
CONSOLE_QUEUE_MAXSIZE = 1000

# Layout: adjust row heights to include a progress bar row while keeping total 630px
ROW0 = 45
ROW1_TO_6 = 85  # each
ROW7 = 45
ROW8_PROGRESS = 30  # progress bar
# Sum: 45 + 6*85 + 45 + 30 = 630

# Files and logging
LOG_TIMESTAMP_FORMAT = "%Y%m%d_%H%M%S"

# CSV processing
CHUNK_SIZE = 100_000  # for big files
PROGRESS_EVERY_N = 10_000  # progress update frequency per rows processed

# SQLite
SQLITE_PRAGMAS = {
    "journal_mode": "WAL",
    "synchronous": "NORMAL",
    "temp_store": "MEMORY",
    "cache_size": -100000,  # ~100MB
}
SQLITE_INDEXES = [
    "CREATE INDEX IF NOT EXISTS idx_individuals_eligible ON individuals(eligible)",
    "CREATE INDEX IF NOT EXISTS idx_individuals_valver ON individuals(validation_version)",
    "CREATE INDEX IF NOT EXISTS idx_individuals_identver ON individuals(identification_version)",
    "CREATE INDEX IF NOT EXISTS idx_individuals_estver ON individuals(estimation_version)",
]
