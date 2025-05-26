from sqlalchemy import create_engine

env = {
    "username": "root",
    "password": "rootpass",
    "host": "mysql",
    "port": "3306",
    "database": "pfe"
}

engine = create_engine(
    f"mysql+mysqlconnector://{env['username']}:{env['password']}@{env['host']}:{env['port']}/{env['database']}"
)