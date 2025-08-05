# modules/estimation_module.py
import pandas as pd, json, os
from typing import Dict
from config import CHUNK_SIZE, PROGRESS_EVERY_N
from utils import now_str, rule_hash
from schemas import build_estimation_rules, build_validation_rules

class EstimationModule:
    def __init__(self, conn, cursor, rules_obj: dict, progress_cb=None, event_logger=None, validation_rules_obj: dict | None = None):
        self.conn = conn
        self.cursor = cursor
        self.rules = build_estimation_rules(rules_obj)
        self.rule_hash = rule_hash(rules_obj)
        self.progress_cb = progress_cb or (lambda cur, tot, msg=None: None)
        self.event = event_logger or (lambda level, msg: None)
        self.required_columns = ["individual_id"]
        if validation_rules_obj:
            v = build_validation_rules(validation_rules_obj)
            for col in v.columns:
                if col.name.lower()!="individual_id":
                    self.required_columns.append(col.name)
        self.output_dir = "estimation_outputs"

    def run(self, file_path: str, stop_event):
        total = 0
        ids_all = []
        for chunk in pd.read_csv(file_path, usecols=["individual_id"], dtype=str, chunksize=CHUNK_SIZE):
            ids = chunk["individual_id"].astype(str).tolist()
            ids_all.extend(ids); total += len(ids)
        if total==0:
            return {"status":"failed","message":"No data to process."}

        placeholders = ",".join("?" for _ in ids_all)
        self.cursor.execute(f"SELECT individual_id, validation_status, identification_version, eligible, estimation_version, estimation_rule_hash, payment_amount FROM individuals WHERE individual_id IN ({placeholders})", ids_all)
        recs = {row[0]: row for row in self.cursor.fetchall()}

        for iid in ids_all:
            r = recs.get(iid)
            if not r or r[1]!="validated":
                return {"status":"failed","message":"Cannot run estimation on unvalidated data."}
            if r[2] is None:
                return {"status":"failed","message":"Cannot run estimation on data that has not been identified."}

        needed = [iid for iid in ids_all if iid not in recs or recs[iid][4]!=self.rules.version or recs[iid][5]!=self.rule_hash]

        intercept = float(self.rules.intercept or 0.0)
        coefs: Dict[str,float] = {k: float(v) for k,v in (self.rules.coefficients or {}).items()}

        processed = 0
        if needed:
            placeholders_needed=",".join("?" for _ in needed)
            self.cursor.execute(f"SELECT individual_id, data_json, eligible FROM individuals WHERE individual_id IN ({placeholders_needed})", needed)
            data_records = {row[0]: (json.loads(row[1]) if row[1] else {}, row[2]) for row in self.cursor.fetchall()}

            for iid in needed:
                if stop_event.is_set():
                    return {"status":"failed","message":"Estimation cancelled by user."}
                data, _ = data_records.get(iid, ({}, None))
                pay = intercept
                for col, coef in coefs.items():
                    try:
                        val = float(data.get(col, 0))
                    except:
                        val = 0.0
                    pay += coef * val
                self.cursor.execute("UPDATE individuals SET estimation_version=?, estimation_rule_hash=?, estimation_date=?, payment_amount=? WHERE individual_id=?",
                                    (self.rules.version, self.rule_hash, now_str(), float(pay), iid))
                processed += 1
                if processed % PROGRESS_EVERY_N == 0:
                    self.progress_cb(processed, len(needed), f"Estimated {processed:,}/{len(needed):,}")
            self.conn.commit()

        self.cursor.execute(f"SELECT SUM(payment_amount) FROM individuals WHERE individual_id IN ({placeholders})", ids_all)
        total_payment = float(self.cursor.fetchone()[0] or 0.0)
        self.cursor.execute(f"SELECT SUM(payment_amount) FROM individuals WHERE individual_id IN ({placeholders}) AND eligible=1", ids_all)
        eligible_payment = float(self.cursor.fetchone()[0] or 0.0)
        ineligible_payment = total_payment - eligible_payment

        ts = pd.Timestamp.now().strftime("%Y%m%d_%H%M%S")
        df_full = pd.read_csv(file_path, dtype=str)
        self.cursor.execute(f"SELECT individual_id, payment_amount FROM individuals WHERE individual_id IN ({placeholders})", ids_all)
        pay_map = {row[0]: float(row[1] or 0.0) for row in self.cursor.fetchall()}
        df_full["payment_amount"] = df_full["individual_id"].astype(str).map(lambda x: pay_map.get(x, 0.0))
        df_full.to_csv(os.path.join(self.output_dir, f"estimation_output_{ts}.csv"), index=False)

        return {
            "status":"completed",
            "total_payment": total_payment,
            "eligible_payment": eligible_payment,
            "ineligible_payment": ineligible_payment
        }
