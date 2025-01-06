import cx_Oracle
import pandas as pd
import pymssql
import pymysql

from common.log import Log

Log = Log()


class Database:
    """Abstract class for database interactions"""

    def __init__(self, server_dic: dict):
        self.server_dic = server_dic
        self.connection = None
        self.cursor = None

    def close(self):
        """Close the cursor and connection."""
        if self.cursor:
            self.cursor.close()
        if self.connection:
            self.connection.close()
        else:
            Log.error("Database connection closed.")

    def connect(self):
        """Abstract method for establishing a database connection."""
        raise NotImplementedError("Subclasses must implement the connect method.")

    def execute_query(self, query: str, params=None):
        """
        Executes a query without returning data (INSERT, UPDATE, DELETE).

        Args:
            query (str): The SQL query to execute.
            params (tuple or list, optional): Parameters to pass with the query.

        Raises:
            ValueError: If the query is a SELECT statement.
        """
        try:
            # Ensure the query is not a SELECT statement
            if query.strip().lower().startswith("select"):
                raise ValueError("SELECT statements are not allowed in execute_query. Use fetch_all or fetch_one instead.")

            self.connect()
            self.cursor.execute(query, params or ())
            self.connection.commit()
            Log.info("Query executed successfully.")

        except Exception as e:
            Log.error("Error executing query: %s", e)
            self.connection.rollback()
            raise

        finally:
            self.close()

    def fetch_all(self, query: str, params=None) -> pd.DataFrame | None:
        """
        Fetches all rows for a SELECT query and returns the dataframe.

        Args:
            query (str): The SELECT query to execute.
            params (tuple or list, optional): Parameters to pass with the query.

        Returns:
            tuple: A tuple containing:
                - rows (list): List of all rows returned by the query.
                - column_names (list): List of column names.

        Raises:
            ValueError: If the query is not a SELECT statement.
        """
        try:
            # Ensure the query is a SELECT statement
            if not query.strip().lower().startswith("select"):
                raise ValueError("fetch_all requires a SELECT statement.")

            self.connect()
            self.cursor.execute(query, params or ())
            rows = self.cursor.fetchall()
            column_names = [desc[0] for desc in self.cursor.description]
            # Flatten rows if they contain single-element tuples
            if len(column_names) == 1:
                rows = [row[0] for row in rows]
            return pd.DataFrame(rows, columns=column_names)

        except Exception as e:
            Log.error("Error fetching data: %s", e)
            return None

        finally:
            self.close()

    def fetch_one(self, query: str, params=None) -> pd.DataFrame:
        """
        Fetches a single row for a SELECT query and returns the dataframe.

        Args:
            query (str): The SELECT query to execute.
            params (tuple or list, optional): Parameters to pass with the query.

        Returns:
            tuple: A tuple containing:
                - row (tuple): The single row returned by the query.
                - column_names (list): List of column names.

        Raises:
            ValueError: If the query is not a SELECT statement.
        """
        try:
            # Ensure the query is a SELECT statement
            if not query.strip().lower().startswith("select"):
                raise ValueError("fetch_one requires a SELECT statement.")

            self.connect()
            self.cursor.execute(query, params or ())
            row = self.cursor.fetchone()
            column_names = [desc[0] for desc in self.cursor.description]
            return pd.DataFrame(row, columns=column_names)

        except Exception as e:
            Log.error("Error fetching data: ", e)

        finally:
            self.close()

    def insert_data(self, table: str, data: dict):
        """
        Inserts data into a specified table.

        Args:
            table (str): The name of the table.
            data (dict): A dictionary where keys are column names and values are the values to insert.
        """
        columns = ", ".join(data.keys())
        values = ", ".join(f":{key}" for key in data.keys())
        query = f"INSERT INTO {table} ({columns}) VALUES ({values})"

        try:
            self.execute_query(query, data)
            Log.info(f"Data successfully inserted into {table}.")
        except Exception as e:
            Log.error(f"Error inserting data into {table}: {e}")
            raise

    def update_data(self, table: str, data: dict, condition: str):
        """
        Updates data in a specified table.

        Args:
            table (str): The name of the table.
            data (dict): A dictionary where keys are column names and values are the new values.
            condition (str): A SQL condition (e.g., "WHERE id = :id") for the update.
        """
        set_clause = ", ".join([f"{key} = :{key}" for key in data.keys()])
        query = f"UPDATE {table} SET {set_clause} {condition}"

        try:
            self.execute_query(query, data)
            Log.info(f"Data successfully updated in {table}.")
        except Exception as e:
            Log.error(f"Error updating data in {table}: {e}")
            raise

    def upsert_data(self, table: str, data: dict, condition: str):
        """
        Inserts data into a specified table if it doesn't exist, or updates it if it does.

        Args:
            table (str): The name of the table.
            data (dict): A dictionary where keys are column names and values are the values to insert or update.
            condition (str): A SQL condition (e.g., "WHERE id = :id") to check for existing data.
        """
        # Construct the columns and values for the insert and update parts
        columns = ", ".join(data.keys())
        values = ", ".join(f":{key}" for key in data.keys())
        set_clause = ", ".join([f"{key} = :{key}" for key in data.keys()])

        # SQL query to check for existing data using the condition
        check_query = f"SELECT COUNT(*) FROM {table} {condition}"

        try:
            # Check if the record exists based on the provided condition
            self.connect()
            self.cursor.execute(check_query, data)
            exists = self.cursor.fetchone()[0] > 0

            if exists:
                # Record exists, perform UPDATE
                update_query = f"UPDATE {table} SET {set_clause} {condition}"
                self.execute_query(update_query, data)
                Log.info(f"Data successfully updated in {table}.")
            else:
                # Record doesn't exist, perform INSERT
                insert_query = f"INSERT INTO {table} ({columns}) VALUES ({values})"
                self.execute_query(insert_query, data)
                Log.info(f"Data successfully inserted into {table}.")

        except Exception as e:
            Log.error(f"Error upserting data in {table}: {e}")
            raise

        finally:
            self.close()


class MySqlDB(Database):
    """MySqlDB class for managing a connection to a MySQL database."""

    def __init__(self, server_dic: dict):
        super().__init__(server_dic)

    def connect(self) -> None:
        """Establish a MySQL database connection."""
        try:
            self.connection = pymysql.connect(
                host=self.server_dic["host"],
                port=self.server_dic["port"],
                user=self.server_dic["username"],
                passwd=self.server_dic["password"],
                db=self.server_dic["database"],
            )

            if self.connection.open:
                self.cursor = self.connection.cursor()
                Log.info("Connection to MySql database established.")

        except pymysql.MySQLError as e:
            Log.error("Error connecting to MySQL: ", e)
            raise


class MsSqlDB(Database):
    """MsSqlDB class for managing a connection to a MSSql database."""

    def __init__(self, server_dic: dict):
        super().__init__(server_dic)

    def connect(self) -> None:
        """Establish a MSSQL database connection."""
        try:
            self.connection = pymssql.connect(
                host=self.server_dic["host"],
                port=self.server_dic["port"],
                user=self.server_dic["username"],
                password=self.server_dic["password"],
                database=self.server_dic["database"],
                tds_version="7.0",  # 线上2.2.8版本加一个参数
            )

            self.cursor = self.connection.cursor()
            Log.info("Connection to MS database established.")

        except pymssql.Error as e:
            Log.error("Error connecting to MsSQL: ", e)
            raise

        except Exception as e:
            Log.error("General error during query execution: ", e)
            raise


class OracleDB(Database):
    """OracleDB class for managing a connection to an Oracle database."""

    def __init__(self, server_dic: dict):
        super().__init__(server_dic)

    def connect(self) -> None:
        """Establish a MSSQL database connection."""
        try:
            self.connection = cx_Oracle.connect(
                f"{self.server_dic['username']}/{self.server_dic['password']}@{self.server_dic['host']}/{self.server_dic['database']}"
            )

            self.cursor = self.connection.cursor()
            Log.info("Connection to Oracle database established.")

        except cx_Oracle.Error as e:
            Log.error("Error connecting to Oracle DB: ", e)
            raise

        except Exception as e:
            Log.error("General error during query execution: ", e)
            raise
