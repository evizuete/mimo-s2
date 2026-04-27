import argparse
from datetime import datetime


def parse_date(s: str) -> datetime:
    return datetime.strptime(s, "%Y-%m-%d")



def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--release', required=True)
    ap.add_argument('--from', dest='from_date', required=True, help='Inicio de la ventana TRAIN')

    ap.add_argument('--artifacts_path', default='./artifacts')
    ap.add_argument('--n_trials', type=int, default=30)
    ap.add_argument('--train_months', type=int, default=2)
    ap.add_argument('--eval_months', type=int, default=1)

