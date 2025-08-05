# modules/db.py
import sqlite3, threading
from config import SQLITE_PRAGMAS, SQLITE_INDEXES

class LockedCursor:
    def __init__(self, cursor: sqlite3.Cursor, lock: threading.Lock):
        self._cursor = cursor
        self._lock = lock
    def execute(self, *a, **k): 
        with self._lock: 
            return self._cursor.execute(*a, **k)
    def executemany(self, *a, **k): 
        with self._lock: 
            return self._cursor.executemany(*a, **k)
    def fetchone(self, *a, **k): 
        with self._lock: 
            return self._cursor.fetchone(*a, **k)
    def fetchall(self, *a, **k): 
        with self._lock: 
            return self._cursor.fetchall(*a, **k)
    def __getattr__(self, name): 
        return getattr(self._cursor, name)

class LockedConnection:
    def __init__(self, conn: sqlite3.Connection, lock: threading.Lock):
        self._conn = conn
        self._lock = lock
    def commit(self): 
        with self._lock: 
            return self._conn.commit()
    def close(self): 
        with self._lock: 
            return self._conn.close()
    def cursor(self): 
        return LockedCursor(self._conn.cursor(), self._lock)
    def __getattr__(self, name): 
        return getattr(self._conn, name)

def connect(db_path: str):
    lock = threading.Lock()
    raw = sqlite3.connect(db_path, check_same_thread=False)
    cur = raw.cursor()
    for k, v in SQLITE_PRAGMAS.items():
        cur.execute(f"PRAGMA {k}={v}")
    raw.commit()
    conn = LockedConnection(raw, lock)
    cursor = LockedCursor(cur, lock)
    ensure_schema(conn, cursor)
    return conn, cursor, lock

def ensure_schema(conn: LockedConnection, cursor: LockedCursor):
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS individuals (
            individual_id TEXT PRIMARY KEY,
            data_hash TEXT,
            data_json TEXT,
            validation_version TEXT,
            validation_rule_hash TEXT,
            validation_date TEXT,
            validation_status TEXT,
            warnings TEXT,
            errors TEXT,
            identification_version TEXT,
            identification_rule_hash TEXT,
            identification_date TEXT,
            eligible INTEGER,
            estimation_version TEXT,
            estimation_rule_hash TEXT,
            estimation_date TEXT,
            payment_amount REAL
        )
    """)
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS datasets (
            dataset_hash TEXT PRIMARY KEY,
            validation_version TEXT,
            validation_rule_hash TEXT,
            validation_date TEXT,
            warnings_count INTEGER,
            errors_count INTEGER,
            record_count INTEGER
        )
    """)
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS runs (
            run_id INTEGER PRIMARY KEY AUTOINCREMENT,
            started_at TEXT,
            finished_at TEXT,
            file_path TEXT,
            dataset_hash TEXT,
            modules TEXT,
            status TEXT
        )
    """)
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS run_events (
            run_id INTEGER,
            ts TEXT,
            level TEXT,
            message TEXT
        )
    """)
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS individual_history (
            individual_id TEXT,
            module TEXT,
            data_hash TEXT,
            rule_hash TEXT,
            status TEXT,
            details TEXT,
            completed_at TEXT
        )
    """)
    for stmt in SQLITE_INDEXES:
        cursor.execute(stmt)
    conn.commit()
