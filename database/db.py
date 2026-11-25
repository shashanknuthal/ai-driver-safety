import sqlite3
import time


class SafetyDB:
    def __init__(self, path):
        self.conn = sqlite3.connect(path, check_same_thread=False)
        self.conn.execute("""
            CREATE TABLE IF NOT EXISTS events (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp REAL NOT NULL,
                event TEXT NOT NULL,
                drowsiness REAL,
                distraction REAL,
                status TEXT
            )
        """)
        self.conn.commit()
        self.last_event = None
        self.last_event_time = 0

    def log_if_event(self, risk):
        event = risk.get("event")
        if not event:
            return

        now = time.time()
        if event == self.last_event and now - self.last_event_time < 5:
            return

        self.conn.execute(
            "INSERT INTO events(timestamp,event,drowsiness,distraction,status) VALUES(?,?,?,?,?)",
            (
                now,
                event,
                risk["drowsiness_score"],
                risk["distraction_score"],
                risk["status"],
            ),
        )
        self.conn.commit()
        self.last_event = event
        self.last_event_time = now

    def fetch_recent(self, limit=1000):
        cur = self.conn.execute(
            "SELECT timestamp,event,drowsiness,distraction,status "
            "FROM events ORDER BY timestamp DESC LIMIT ?", (limit,)
        )
        return cur.fetchall()

    def close(self):
        self.conn.close()
