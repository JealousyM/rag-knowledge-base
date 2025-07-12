import os
import json
import psycopg2
from psycopg2 import sql
from psycopg2.extensions import ISOLATION_LEVEL_AUTOCOMMIT
from dotenv import load_dotenv

load_dotenv()

class MemoryStorage:
    def __init__(self):
        self.dbname = os.getenv("POSTGRES_DB")
        self.user = os.getenv("POSTGRES_USER")
        self.password = os.getenv("POSTGRES_PASSWORD")
        self.host = os.getenv("POSTGRES_HOST")
        self.port = os.getenv("POSTGRES_PORT")
        self.conn = None
        self.ensure_db_and_table_exist()

    def connect(self, dbname=None):
        """Connect to the PostgreSQL database."""
        try:
            self.conn = psycopg2.connect(
                dbname=dbname if dbname else self.dbname,
                user=self.user,
                password=self.password,
                host=self.host,
                port=self.port
            )
        except psycopg2.OperationalError as e:
            print(f"Could not connect to database '{dbname if dbname else self.dbname}'. Error: {e}")
            self.conn = None

    def ensure_db_and_table_exist(self):
        """Ensure the database and table exist."""
        try:
            self.connect('postgres')
            if not self.conn:
                return
            self.conn.set_isolation_level(ISOLATION_LEVEL_AUTOCOMMIT)
            
            with self.conn.cursor() as cur:
                cur.execute(sql.SQL("SELECT 1 FROM pg_database WHERE datname = %s"), [self.dbname])
                if not cur.fetchone():
                    cur.execute(sql.SQL("CREATE DATABASE {}").format(sql.Identifier(self.dbname)))
                    print(f"Database '{self.dbname}' created.")
            self.conn.close()

            self.connect()
            with self.conn.cursor() as cur:
                cur.execute("""
                    CREATE TABLE IF NOT EXISTS conversation_history (
                        id SERIAL PRIMARY KEY,
                        session_id VARCHAR(255) NOT NULL,
                        human_message TEXT NOT NULL,
                        ai_message TEXT NOT NULL,
                        sources JSONB,
                        timestamp TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
                    );
                """)
                
                cur.execute("""
                    CREATE TABLE IF NOT EXISTS llm_configurations (
                        id INT PRIMARY KEY,
                        config_data JSONB NOT NULL,
                        updated_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
                    );
                """)
                self.conn.commit()

                # Check if 'sources' column exists and add it if not
                cur.execute("""
                    SELECT column_name 
                    FROM information_schema.columns 
                    WHERE table_name='conversation_history' AND column_name='sources';
                """)
                if not cur.fetchone():
                    cur.execute("ALTER TABLE conversation_history ADD COLUMN sources JSONB;")
                    self.conn.commit()
                    print("Added 'sources' column to 'conversation_history' table.")

        except psycopg2.Error as e:
            print(f"Error during database/table setup: {e}")
        finally:
            if self.conn:
                self.conn.close()

    def add_message(self, session_id, human_message, ai_message, sources=None):
        """Add a message and its sources to the conversation history."""
        try:
            self.connect()
            with self.conn.cursor() as cur:
                cur.execute("""
                    INSERT INTO conversation_history (session_id, human_message, ai_message, sources)
                    VALUES (%s, %s, %s, %s);
                """, (session_id, human_message, ai_message, json.dumps(sources) if sources else None))
                self.conn.commit()
        except psycopg2.Error as e:
            print(f"Error adding message to history: {e}")
        finally:
            if self.conn:
                self.conn.close()

    def get_exact_answer(self, session_id, human_message):
        """Retrieve an exact answer for a given human_message in a session."""
        try:
            self.connect()
            with self.conn.cursor() as cur:
                cur.execute("""
                    SELECT ai_message, sources FROM conversation_history
                    WHERE session_id = %s AND human_message = %s
                    ORDER BY timestamp DESC
                    LIMIT 1;
                """, (session_id, human_message))
                result = cur.fetchone()
                if result:
                    # psycopg2 automatically converts JSONB to a Python dict/list
                    return result[0], result[1]
        except psycopg2.Error as e:
            print(f"Error getting exact answer: {e}")
        finally:
            if self.conn:
                self.conn.close()
        return None, None

    def get_conversation_history(self, session_id, limit=5):
        """Retrieve the last N messages for a given session_id."""
        history = []
        try:
            self.connect()
            with self.conn.cursor() as cur:
                cur.execute("""
                    SELECT human_message, ai_message FROM conversation_history
                    WHERE session_id = %s
                    ORDER BY timestamp DESC
                    LIMIT %s;
                """, (session_id, limit))
                rows = cur.fetchall()
                for row in reversed(rows):
                    history.append([row[0], row[1]])
        except psycopg2.Error as e:
            print(f"Error retrieving conversation history: {e}")
        finally:
            if self.conn:
                self.conn.close()
        return history

    def save_llm_configuration(self, config_data):
        """Save or update the LLM configuration."""
        try:
            self.connect()
            with self.conn.cursor() as cur:
                cur.execute("""
                    INSERT INTO llm_configurations (id, config_data, updated_at)
                    VALUES (1, %s, CURRENT_TIMESTAMP)
                    ON CONFLICT (id) DO UPDATE
                    SET config_data = EXCLUDED.config_data, updated_at = CURRENT_TIMESTAMP;
                """, (json.dumps(config_data),))
                self.conn.commit()
        except psycopg2.Error as e:
            print(f"Error saving LLM configuration: {e}")
        finally:
            if self.conn:
                self.conn.close()

    def load_llm_configuration(self):
        """Load the LLM configuration."""
        try:
            self.connect()
            with self.conn.cursor() as cur:
                cur.execute("SELECT config_data FROM llm_configurations WHERE id = 1;")
                result = cur.fetchone()
                if result:
                    return result[0]
        except psycopg2.Error as e:
            print(f"Error loading LLM configuration: {e}")
        finally:
            if self.conn:
                self.conn.close()
        return None
