import mysql.connector

def get_connection():
    return mysql.connector.connect(
        host="localhost",
        user="root",
        password="YOUR_DB_PASSWORD",
        database="studytrack"
    )