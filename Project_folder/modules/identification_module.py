# modules/identification_module.py
import pandas as pd, json, os
from config import CHUNK_SIZE, PROGRESS_EVERY_N
from utils import now_str, rule_hash
from schemas import build_identification_rules, build_validation_rules

class IdentificationModule:
    def __init__(self, conn, cursor, rules_obj: dict, progress_cb=None, event_logger=None, validation_rules_obj: dict | None = None):
        self.conn = conn
        self.cursor = cursor
        self.rules = build_identification_rules(rules_obj)
        self.rule_hash = rule_hash(rules_obj)
        self.progress_cb = progress_cb or (lambda cur, tot, msg=None: None)
        self.event = event_logger or (lambda level, msg: None)
        self.required_columns = ["individual_id"]
        if validation_rules_obj:
            v = build_validation_rules(validation_rules_obj)
            for col in v.columns:
                if col.name.lower()!="individual_id":
                    self.required_columns.append(col.name)
        self.output_dir = "identification_outputs"

    def _eligible(self, data: dict) -> int:
        if not self.rules.criteria:
            return 1
        for crit in self.rules.criteria:
            ok = True
            for cond in crit.conditions:
                col = cond.column
                if col not in data:
                    ok = False; break
                indiv_val = data[col]
                op = cond.operator; target = cond.value
                try:
                    a = float(indiv_val); b = float(target)
                except:
                    a = str(indiv_val); b = str(target)
                if op == "==" and not (a==b): ok=False
                if op == "!=" and not (a!=b): ok=False
                if op == ">"  and not (a>b):  ok=False
                if op == ">=" and not (a>=b): ok=False
                if op == "<"  and not (a<b):  ok=False
                if op == "<=" and not (a<=b): ok=False
                if not ok: break
            if ok: return 1
        return 0

    def run(self, file_path: str, stop_event):
        total = 0
        ids_all = []
        for chunk in pd.read_csv(file_path, usecols=["individual_id"], dtype=str, chunksize=CHUNK_SIZE):
            ids = chunk["individual_id"].astype(str).tolist()
            ids_all.extend(ids); total += len(ids)
        if total==0:
            return {"status":"failed","message":"No data to process."}

        placeholders = ",".join("?" for _ in ids_all)
        self.cursor.execute(f"SELECT individual_id, validation_status, validation_version, validation_rule_hash, data_json, identification_version, identification_rule_hash, eligible FROM individuals WHERE individual_id IN ({placeholders})", ids_all)
        recs = {row[0]: row for row in self.cursor.fetchall()}

        for iid in ids_all:
            r = recs.get(iid)
            if not r or r[1]!="validated":
                return {"status":"failed","message":"Cannot run identification on unvalidated data."}

        needed = [iid for iid in ids_all if iid not in recs or recs[iid][5]!=self.rules.version or recs[iid][6]!=self.rule_hash]

        processed = 0
        for iid in needed:
            if stop_event.is_set():
                return {"status":"failed","message":"Identification cancelled by user."}
            r = recs.get(iid)
            data = json.loads(r[4]) if r and r[4] else {}
            elig = self._eligible(data)
            self.cursor.execute("UPDATE individuals SET identification_version=?, identification_rule_hash=?, identification_date=?, eligible=? WHERE individual_id=?",
                                (self.rules.version, self.rule_hash, now_str(), int(elig), iid))
            processed += 1
            if processed % PROGRESS_EVERY_N == 0:
                self.progress_cb(processed, len(needed), f"Identified {processed:,}/{len(needed):,}")

        self.conn.commit()

        self.cursor.execute(f"SELECT COUNT(*) FROM individuals WHERE individual_id IN ({placeholders}) AND eligible=1", ids_all)
        in_scope = self.cursor.fetchone()[0] or 0
        out_scope = total - in_scope

        ts = pd.Timestamp.now().strftime("%Y%m%d_%H%M%S")
        df_all = pd.read_csv(file_path, usecols=self.required_columns, dtype=str)
        self.cursor.execute(f"SELECT individual_id, eligible FROM individuals WHERE individual_id IN ({placeholders})", ids_all)
        elig_map = {row[0]: int(row[1]) if row[1] is not None else 0 for row in self.cursor.fetchall()}
        df_all["__elig__"] = df_all["individual_id"].astype(str).map(lambda x: elig_map.get(x, 0))
        df_all[df_all["__elig__"]==1].drop(columns="__elig__").to_csv(os.path.join(self.output_dir, f"in_scope_obs_{ts}.csv"), index=False)
        df_all[df_all["__elig__"]==0].drop(columns="__elig__").to_csv(os.path.join(self.output_dir, f"out_of_scope_{ts}.csv"), index=False)

        return {
            "status":"completed",
            "in_scope_count": int(in_scope),
            "out_scope_count": int(out_scope),
            "in_scope_pct": round(in_scope/total*100, 2) if total>0 else 0.0,
            "out_scope_pct": round(out_scope/total*100, 2) if total>0 else 0.0
        }
