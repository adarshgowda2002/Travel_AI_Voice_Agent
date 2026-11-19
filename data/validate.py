"""Integrity + balance checks for dataset/out. Exit code 1 on any integrity failure."""
import csv
import json
import os
import sys
from collections import Counter

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "out")


def load(n):
    with open(os.path.join(OUT, n + ".csv"), encoding="utf-8") as f:
        return list(csv.DictReader(f))


users, sessions, convs = load("users"), load("sessions"), load("conversations")
trips, bookings, itins = load("trips"), load("bookings"), load("itineraries")
tasks, audio, audit, scen = load("agent_tasks"), load("audio_logs"), load("audit_logs"), load("scenarios")
rag = {json.loads(l)["doc_id"] for l in open(os.path.join(OUT, "rag_docs.jsonl"), encoding="utf-8")}
gold = [json.loads(l) for l in open(os.path.join(OUT, "eval_gold.jsonl"), encoding="utf-8")]

fails = []


def check(name, ok):
    print(("PASS  " if ok else "FAIL  ") + name)
    if not ok:
        fails.append(name)


uid = {u["id"] for u in users}
sid = {s["id"] for s in sessions}
tid = {t["id"] for t in trips}
print("== Referential integrity ==")
check("sessions.user_id -> users", all(s["user_id"] in uid for s in sessions))
check("sessions.trip_id -> trips", all(s["trip_id"] in tid for s in sessions))
check("trips.user_id -> users", all(t["user_id"] in uid for t in trips))
check("conversations.session_id -> sessions", all(c["session_id"] in sid for c in convs))
check("conversations.rag_doc_id -> rag_docs", all(c["rag_doc_id"] in rag for c in convs if c["rag_doc_id"]))
check("bookings.trip_id -> trips", all(b["trip_id"] in tid for b in bookings))
check("bookings.session_id -> sessions (voice only)", all((b["session_id"] in sid) if b["channel"] == "voice" else b["session_id"] == "" for b in bookings))
check("itineraries.trip_id -> trips", all(i["trip_id"] in tid for i in itins))
check("agent_tasks.session_id -> sessions", all(t["session_id"] in sid for t in tasks))
check("audio_logs.session_id -> sessions", all(a["session_id"] in sid for a in audio))
check("audit_logs.user_id -> users", all(a["user_id"] in uid for a in audit))
check("eval_gold rows == conversation rows", len(gold) == len(convs))
check("every conversation turn has 1 voice_in + 1 voice_out",
      len(audio) == 2 * len(convs))
check("unique ids in every table", all(len({r["id"] for r in t}) == len(t) for t in
                                       [users, sessions, convs, trips, bookings, tasks, audio, audit]))

print("\n== Timeline synchronization ==")
sess = {s["id"]: s for s in sessions}
check("session ended_at >= started_at", all(s["ended_at"] >= s["started_at"] for s in sessions))
check("turn timestamps inside session window",
      all(sess[c["session_id"]]["started_at"] <= c["ts"] <= sess[c["session_id"]]["ended_at"] for c in convs))
check("voice-booking timestamps inside session window",
      all(sess[b["session_id"]]["started_at"] <= b["created_at"] <= sess[b["session_id"]]["ended_at"]
          for b in bookings if b["channel"] == "voice"))
check("web bookings pre-date their session",
      all(b["created_at"] < next(s["started_at"] for s in sessions if s["trip_id"] == b["trip_id"])
          for b in bookings if b["channel"] == "web"))
check("task timestamps inside session window",
      all(sess[t["session_id"]]["started_at"] <= t["created_at"] <= sess[t["session_id"]]["ended_at"] for t in tasks))
check("sessions.turns == conversation rows",
      all(int(s["turns"]) == sum(1 for c in convs if c["session_id"] == s["id"]) for s in sessions[:500]))
tr = {t["id"]: t for t in trips}
check("trip end_date >= start_date", all(t["end_date"] >= t["start_date"] for t in trips))
check("booked trips have confirmed bookings",
      all(any(b["trip_id"] == t["id"] and b["status"] == "confirmed" for b in bookings)
          for t in trips if t["status"] == "booked"))
check("cancelled booking trips marked cancelled",
      all(tr[b["trip_id"]]["status"] == "cancelled" for b in bookings if b["status"] == "cancelled"))
check("latency NFR: >=90% of agent turns < 1.5s",
      sum(int(c["latency_ms"]) < 1500 for c in convs) / len(convs) >= 0.90)


def dist(title, key, rows=scen):
    c = Counter(r[key] for r in rows)
    lo, hi = min(c.values()), max(c.values())
    print(f"\n{title}  (min {lo}, max {hi}, spread {hi - lo})")
    for k, v in sorted(c.items()):
        print(f"   {k:<24}{v:>4}  {'#' * (v // 3)}")
    return hi - lo


print("\n== Balance ==")
spreads = {
    "intent": dist("Intent", "intent"),
    "outcome": dist("Outcome", "outcome"),
    "destination": dist("Destination", "destination"),
    "turn_bucket": dist("Turn-length bucket", "turn_bucket"),
    "accent": dist("Accent", "accent"),
    "noise_profile": dist("Noise profile", "noise_profile"),
}
print("\nSplit (stratified 80/10/10 by intent):", dict(Counter(s["split"] for s in scen)))
cells = Counter((s["intent"], s["outcome"]) for s in scen)
print(f"\nIntent x outcome cells: {len(cells)} cells, counts {min(cells.values())}..{max(cells.values())}")
check("all 30 intent x outcome cells populated", len(cells) == 30)
check("cell counts differ by <= 1", max(cells.values()) - min(cells.values()) <= 1)
check("intents balanced (spread <= 1)", spreads["intent"] <= 1)
check("outcomes balanced (spread <= 1)", spreads["outcome"] <= 1)
check("destinations balanced (spread <= 1)", spreads["destination"] <= 1)
check("accents/noise balanced (spread <= 1)", spreads["accent"] <= 1 and spreads["noise_profile"] <= 1)
for ph in ["disfluency", "correction", "barge_in"]:
    n = sum(int(s[ph]) for s in scen)
    print(f"   phenomenon {ph:<11}{n:>4} ({n / len(scen):.0%})")
by_split = Counter((s["split"], s["intent"]) for s in scen)
check("every intent present in train/val/test", all(by_split[(sp, i)] > 0 for sp in ["train", "val", "test"]
                                                    for i in {s["intent"] for s in scen}))
print(f"\nRows: users={len(users)} sessions={len(sessions)} conversations={len(convs)} trips={len(trips)} "
      f"bookings={len(bookings)} itineraries={len(itins)} agent_tasks={len(tasks)} audio={len(audio)} audit={len(audit)}")
print("\nRESULT:", "ALL CHECKS PASSED" if not fails else f"{len(fails)} FAILED: {fails}")
sys.exit(1 if fails else 0)
