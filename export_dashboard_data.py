#!/usr/bin/env python3
import json
import sqlite3
from pathlib import Path

DB_PATH = Path('/root/.openclaw/workspace/polymarket.sqlite')
OUT_DIR = Path('/root/.openclaw/workspace/quant_research/polymarket/dashboard/public/data')
SNAPSHOT_DATE = '2026-03-29'


def rows_to_dicts(cursor, rows):
    cols = [d[0] for d in cursor.description]
    return [dict(zip(cols, row)) for row in rows]


def fetch_all(conn, sql, params=()):
    cur = conn.execute(sql, params)
    return rows_to_dicts(cur, cur.fetchall())


def latest_signals(conn):
    sql = """
    WITH latest_pred AS (
      SELECT ticker, horizon, MAX(pred_date) AS pred_date
      FROM app_oos_predictions
      WHERE snapshot_date=?
      GROUP BY ticker, horizon
    ), latest_eval AS (
      SELECT e.*
      FROM app_model_eval e
      JOIN (
        SELECT ticker, horizon, MAX(eval_date) AS eval_date
        FROM app_model_eval
        WHERE snapshot_date=?
        GROUP BY ticker, horizon
      ) x
      ON e.ticker=x.ticker AND e.horizon=x.horizon AND e.eval_date=x.eval_date
      WHERE e.snapshot_date=?
    )
    SELECT p.ticker, p.horizon, p.pred_date, p.y_true_dir, p.y_pred_dir, p.y_prob,
           p.train_start, p.train_end, p.train_size,
           le.auc_all, le.accuracy_all, le.n_all, le.auc_recent, le.accuracy_recent, le.n_recent
    FROM app_oos_predictions p
    JOIN latest_pred lp
      ON p.ticker=lp.ticker AND p.horizon=lp.horizon AND p.pred_date=lp.pred_date
    LEFT JOIN latest_eval le
      ON p.ticker=le.ticker AND p.horizon=le.horizon
    WHERE p.snapshot_date=?
    ORDER BY p.ticker, p.horizon
    """
    return fetch_all(conn, sql, (SNAPSHOT_DATE, SNAPSHOT_DATE, SNAPSHOT_DATE, SNAPSHOT_DATE))


def summary(conn):
    latest = latest_signals(conn)
    recent_auc = [x['auc_recent'] for x in latest if x['auc_recent'] is not None]
    best = max([x for x in latest if x['auc_recent'] is not None], key=lambda x: x['auc_recent'], default=None)
    max_pred_date = max((x['pred_date'] for x in latest), default=None)
    return {
        'snapshot_date': SNAPSHOT_DATE,
        'latest_pred_date_max': max_pred_date,
        'asset_horizon_count': len(latest),
        'oos_record_count': conn.execute('SELECT COUNT(*) FROM app_oos_predictions WHERE snapshot_date=?', (SNAPSHOT_DATE,)).fetchone()[0],
        'eval_record_count': conn.execute('SELECT COUNT(*) FROM app_model_eval WHERE snapshot_date=?', (SNAPSHOT_DATE,)).fetchone()[0],
        'shap_record_count': conn.execute('SELECT COUNT(*) FROM app_model_shap WHERE snapshot_date=?', (SNAPSHOT_DATE,)).fetchone()[0],
        'avg_recent_auc': round(sum(recent_auc)/len(recent_auc), 4) if recent_auc else None,
        'best_signal': {
            'ticker': best['ticker'],
            'horizon': best['horizon'],
            'auc_recent': best['auc_recent']
        } if best else None,
        'notes': [
            'Historical predictions read from app_oos_predictions.',
            'pred_date is the latest valid trading day for each asset, not a unified date.',
            'OOS = Out-of-Sample (historical out-of-sample prediction).',
            'Current SHAP view shows latest-model aggregate feature importance.'
        ]
    }


def top_features(conn):
    sql = """
    SELECT ticker, target_col AS horizon, feature, mean_shap, mean_abs_shap, rank
    FROM app_model_shap
    WHERE snapshot_date=? AND rank <= 15
    ORDER BY ticker, target_col, rank
    """
    return fetch_all(conn, sql, (SNAPSHOT_DATE,))


def predictions(conn):
    sql = """
    SELECT pred_date, ticker, horizon, y_true_dir, y_pred_dir, y_prob, train_start, train_end, train_size
    FROM app_oos_predictions
    WHERE snapshot_date=?
    ORDER BY pred_date, ticker, horizon
    """
    return fetch_all(conn, sql, (SNAPSHOT_DATE,))


def evaluations(conn):
    sql = """
    SELECT eval_date, ticker, horizon, auc_all, accuracy_all, n_all, auc_recent, accuracy_recent, n_recent
    FROM app_model_eval
    WHERE snapshot_date=?
    ORDER BY eval_date, ticker, horizon
    """
    return fetch_all(conn, sql, (SNAPSHOT_DATE,))


def metadata():
    return {
        'title': 'Polymarket Quant Dashboard',
        'language_style': 'English first, plain-language labels, Chinese secondary notes',
        'tabs': ['Overview', 'Predictions', 'Model Evaluation', 'Top Features', 'Methodology & Notes'],
        'filters': {
            'prediction_trend': ['all', 'last_30', 'last_60', 'date_range'],
            'model_evaluation': ['all', 'last_30', 'last_60', 'date_range'],
            'top_features': ['latest_model', 'recent_10d_placeholder']
        },
        'pending_items': [
            'Top Features recent-10d aggregation logic is still to be finalized.',
            'Public deployment link not created yet; local/static MVP first.'
        ]
    }


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(DB_PATH)
    payload = {
        'summary': summary(conn),
        'latest_signals': latest_signals(conn),
        'predictions': predictions(conn),
        'evaluations': evaluations(conn),
        'top_features': top_features(conn),
        'metadata': metadata(),
    }
    (OUT_DIR / 'dashboard-data.json').write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding='utf-8')
    conn.close()
    print(f'Wrote {(OUT_DIR / "dashboard-data.json")}')


if __name__ == '__main__':
    main()
