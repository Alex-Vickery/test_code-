# modules/validation_module.py
import pandas as pd
import numpy as np
import os, json, math
from config import CHUNK_SIZE, PROGRESS_EVERY_N, CONSOLE_LINE_WIDTH
from utils import file_sha256, row_hash, now_str, rule_hash, normalize_value
from schemas import build_validation_rules

class ValidationModule:
    def __init__(self, conn, cursor, rules_obj: dict, progress_cb=None, event_logger=None):
        self.conn = conn
        self.cursor = cursor
        self.rules = build_validation_rules(rules_obj)
        self.rule_hash = rule_hash(rules_obj)
        self.required_columns = []
        for col in self.rules.columns:
            if col.name.lower() != "individual_id":
                self.required_columns.append(col.name)
        if "individual_id" not in self.required_columns:
            self.required_columns.insert(0, "individual_id")
        self.output_dir = "validation_outputs"
        self.progress_cb = progress_cb or (lambda cur, tot, msg=None: None)
        self.event = event_logger or (lambda level, msg: None)

    def _stream_stats(self, path: str):
        num_cols = [c.name for c in self.rules.columns if c.type in ("numeric","integer","float","int")]
        stats = {c: {"n":0, "mean":0.0, "M2":0.0} for c in num_cols if c != "individual_id"}
        total = 0
        usecols = [c for c in self.required_columns if c in num_cols]
        if not usecols:
            return {}
        for chunk in pd.read_csv(path, usecols=usecols, dtype=str, chunksize=CHUNK_SIZE):
            for col in [c for c in chunk.columns if c in num_cols and c!="individual_id"]:
                vals = pd.to_numeric(chunk[col], errors="coerce").dropna().astype(float).values
                for x in vals:
                    s = stats[col]
                    s["n"] += 1
                    delta = x - s["mean"]
                    s["mean"] += delta / s["n"]
                    s["M2"] += delta * (x - s["mean"])
            total += len(chunk)
            if total % PROGRESS_EVERY_N == 0:
                self.progress_cb(total, -1, f"Computing stats… {total:,} rows")
        out = {}
        for col, s in stats.items():
            if s["n"] > 1:
                var = s["M2"] / (s["n"] - 1)
                out[col] = (s["mean"], math.sqrt(max(var, 0.0)))
            else:
                out[col] = (0.0, 0.0)
        return out

    def run(self, file_path: str, stop_event):
        start_ts = now_str()
        ds_hash = file_sha256(file_path)

        self.cursor.execute("SELECT validation_version, validation_rule_hash, validation_date, warnings_count, errors_count, record_count FROM datasets WHERE dataset_hash=?",
                            (ds_hash,))
        prev = self.cursor.fetchone()
        if prev and prev[0]==self.rules.version and prev[1]==self.rule_hash:
            return {
                "status":"completed",
                "dataset_valid": (prev[4] or 0)==0,
                "warnings_count": prev[3] or 0,
                "errors_count": prev[4] or 0,
                "record_count": prev[5] or 0,
                "message": f"Dataset previously validated on {prev[2]}."
            }

        try:
            header_df = pd.read_csv(file_path, nrows=0, dtype=str)
        except Exception as e:
            return {"status":"failed","message":f"Failed to read CSV: {e}"}
        for col in self.required_columns:
            if col not in header_df.columns:
                return {"status":"failed","message":f"Required column '{col}' is missing from the CSV."}

        seen_ids = set()
        for chunk in pd.read_csv(file_path, usecols=["individual_id"], dtype=str, chunksize=CHUNK_SIZE):
            ids = chunk["individual_id"].astype(str)
            dup = ids.duplicated().any() or any(i in seen_ids for i in ids)
            if dup:
                return {"status":"failed","message":"Validation failed: duplicate individual_id values detected."}
            seen_ids.update(ids)

        stats = self._stream_stats(file_path)

        issue_counts = {}
        total_rows = 0
        warnings_total = 0
        errors_total = 0
        batch_updates = []

        def count_issue(key):
            issue_counts[key] = issue_counts.get(key, 0) + 1

        for chunk in pd.read_csv(file_path, usecols=self.required_columns, dtype=str, chunksize=CHUNK_SIZE):
            if stop_event.is_set():
                return {"status":"failed","message":"Validation cancelled by user."}
            total_rows += len(chunk)
            warnings_col, errors_col = [], []

            for _, row in chunk.iterrows():
                w_list, e_list = [], []
                for col_rule in self.rules.columns:
                    col = col_rule.name
                    if col == "individual_id":
                        continue
                    val = row[col]

                    if pd.isna(val) or str(val).strip()=="":
                        key=f"Missing {col}"; w_list.append(key); count_issue(key); continue

                    t = col_rule.type
                    if t in ("numeric","integer","float","int"):
                        try:
                            num = float(val)
                        except:
                            key=f"Invalid type {col}"; e_list.append(key); count_issue(key); continue
                        if t in ("integer","int") and not float(num).is_integer():
                            key=f"Invalid type {col}"; e_list.append(key); count_issue(key); continue
                        if col_rule.min is not None and col_rule.max is not None:
                            if num < float(col_rule.min) or num > float(col_rule.max):
                                key=f"Out of range {col}"; e_list.append(key); count_issue(key)
                        mean,std = stats.get(col,(0.0,0.0))
                        thr = float(col_rule.outlier_threshold or 3.0)
                        if std>0 and abs(num-mean)>thr*std:
                            key=f"Outlier {col}"; w_list.append(key); count_issue(key)
                    elif t in ("string","object","categorical","categoric"):
                        allowed = col_rule.allowed_values
                        if allowed is not None and str(val) not in allowed:
                            key=f"Invalid value {col}"; e_list.append(key); count_issue(key)
                    elif t in ("date","datetime"):
                        try:
                            d = pd.to_datetime(str(val), errors="coerce")
                        except:
                            d = None
                        if d is None or pd.isna(d):
                            key=f"Invalid type {col}"; e_list.append(key); count_issue(key)

                w_text = "; ".join(w_list)
                e_text = "; ".join(e_list)
                warnings_col.append(w_text)
                errors_col.append(e_text)
                if w_text: warnings_total += 1
                if e_text: errors_total += 1

            colnames = self.required_columns
            chunk["__rowhash__"] = chunk.apply(lambda r: row_hash(r, colnames, self.rule_hash), axis=1)
            chunk_json = chunk.apply(lambda r: {c: ("" if pd.isna(r[c]) else str(r[c])) for c in self.required_columns}, axis=1)

            now = now_str()
            for i in range(len(chunk)):
                indiv_id = str(chunk.iloc[i]["individual_id"])
                status = "validated" if errors_col[i]=="" else "not validated"
                batch_updates.append((
                    indiv_id,
                    chunk.iloc[i]["__rowhash__"],
                    json.dumps(chunk_json.iloc[i]),
                    self.rules.version,
                    self.rule_hash,
                    now,
                    status,
                    warnings_col[i],
                    errors_col[i],
                    None, None, None,
                    None, None, None, None, None
                ))

            self.cursor.executemany("""
                INSERT OR REPLACE INTO individuals
                (individual_id, data_hash, data_json, 
                 validation_version, validation_rule_hash, validation_date, validation_status, warnings, errors,
                 identification_version, identification_rule_hash, identification_date, eligible,
                 estimation_version, estimation_rule_hash, estimation_date, payment_amount)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, batch_updates)
            self.conn.commit()
            batch_updates.clear()

            if total_rows % PROGRESS_EVERY_N == 0:
                self.progress_cb(total_rows, -1, f"Validated {total_rows:,} rows")

        end_ts = now_str()
        ts_tag = pd.Timestamp.now().strftime("%Y%m%d_%H%M%S")
        size_mb = os.path.getsize(file_path)/(1024*1024)
        possible_issues = []
        for col_rule in self.rules.columns:
            col = col_rule.name
            if col.lower()=="individual_id": continue
            possible_issues += [f"Missing {col}", f"Invalid type {col}"]
            if col_rule.min is not None and col_rule.max is not None:
                possible_issues.append(f"Out of range {col}")
            if col_rule.allowed_values:
                possible_issues.append(f"Invalid value {col}")
            if col_rule.type in ("numeric","integer","float","int"):
                possible_issues.append(f"Outlier {col}")
        seen=set(); possible_issues=[x for x in possible_issues if not (x in seen or seen.add(x))]
        lines = [
            "Validation Module Summary".ljust(CONSOLE_LINE_WIDTH, "="),
            f"Data file: {os.path.basename(file_path)}",
            f"File path: {file_path}",
            f"Start time: {start_ts}",
            f"End time: {end_ts}",
            f"File size: {size_mb:.2f} MB",
            f"Total individuals: {total_rows}",
            "="*CONSOLE_LINE_WIDTH
        ]
        for issue in possible_issues:
            cnt = issue_counts.get(issue, 0); pct = (cnt/total_rows*100) if total_rows>0 else 0.0
            lines.append(f"{issue}: {cnt} ({pct:.2f}%)")
        lines.append("="*CONSOLE_LINE_WIDTH)
        with open(os.path.join(self.output_dir, f"validation_summary_{ts_tag}.txt"), "w", encoding="utf-8") as f:
            f.write("\n".join(lines))

        df_all = pd.read_csv(file_path, dtype=str)
        ids = df_all["individual_id"].astype(str).tolist()
        placeholders=",".join("?" for _ in ids)
        self.cursor.execute(f"SELECT individual_id, warnings, errors FROM individuals WHERE individual_id IN ({placeholders})", ids)
        map_we = {row[0]:(row[1] or "", row[2] or "") for row in self.cursor.fetchall()}
        df_all["warnings"] = df_all["individual_id"].astype(str).map(lambda x: map_we.get(x, ("",""))[0])
        df_all["errors"] = df_all["individual_id"].astype(str).map(lambda x: map_we.get(x, ("",""))[1])
        clean_df = df_all[(df_all["warnings"]=="") & (df_all["errors"]=="")].drop(columns=["warnings","errors"])
        exceptions_df = df_all[(df_all["warnings"]!="") | (df_all["errors"]!="")]

        clean_df.to_csv(os.path.join(self.output_dir, f"validated_obs_{ts_tag}.csv"), index=False)
        exceptions_df.to_csv(os.path.join(self.output_dir, f"identified_exceptions_{ts_tag}.csv"), index=False)

        self.cursor.execute("""
            INSERT OR REPLACE INTO datasets
            (dataset_hash, validation_version, validation_rule_hash, validation_date, warnings_count, errors_count, record_count)
            VALUES (?, ?, ?, ?, ?, ?, ?)
        """, (ds_hash, self.rules.version, self.rule_hash, end_ts, int(warnings_total), int(errors_total), int(total_rows)))
        self.conn.commit()

        return {
            "status":"completed",
            "dataset_valid": (errors_total==0),
            "warnings_count": int(warnings_total),
            "errors_count": int(errors_total),
            "record_count": int(total_rows)
        }
