import pandas as pd
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from mimo.data_managers.entities import Base


class Database:
    def __init__(self, user='evizuete', password='Ev1z43t3.00', host='10.1.21.25', port=3306, schema='bot_001'):
        self.engine = None
        self.SessionLocal = None
        self.user = user
        self.password = password
        self.host = host
        self.port = port
        self.schema = schema

    def connect(self):
        self.engine = create_engine(
            f'mysql+pymysql://{self.user}:{self.password}@{self.host}:{self.port}/{self.schema}',
            future=True, pool_pre_ping=True
        )

        Base.metadata.create_all(self.engine)

        self.SessionLocal = sessionmaker(bind=self.engine, autoflush=False, autocommit=False,future=True)
        return self.engine

    def session(self):
        if self.engine is None or self.SessionLocal is None:
            self.connect()

        return self.SessionLocal()

    def disconnect(self):
        self.engine.dispose()

    def save(self, df, table_name, method='append', index=False):
        if df is None:
            return

        if self.engine is None:
            self.connect()

        with self.engine.begin() as conn:
            df.to_sql(name=table_name, con=conn, if_exists=method, index=index)


    def save_massive_data(self, df, table_name, method='append', index=False, chunk_size=10000):
        if df is None:
            return

        if self.engine is None:
            self.connect()

        with self.engine.begin() as conn:
            df.to_sql(name=table_name, con=conn, if_exists=method, index=index, method='multi', chunksize=chunk_size)

    def read(self, query):
        if self.engine is None:
            self.connect()

        with self.engine.begin() as conn:
            df = pd.read_sql(query, con=conn)

        return df