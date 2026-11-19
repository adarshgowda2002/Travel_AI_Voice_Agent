"""
Synthetic, balanced, synchronized dataset for the Voice Travel Agent.

Mirrors the architecture poster (section 8 data model, section 5 agents/tools,
section 7 sequence flow). Deterministic: same SEED -> identical output.

    python dataset/generate.py            # writes dataset/out/*
    python dataset/validate.py            # integrity + balance report

Balance is enforced by stratified round-robin (NOT random sampling):
intent x outcome cells, destinations, turn-length, accents, noise profiles and
conversational phenomena are all assigned from shuffled-but-exact quotas.
Synchronization: one set of IDs/timeline runs through users, sessions,
conversations, trips, bookings, itineraries, agent_tasks, audio_logs,
audit_logs, rag_docs and the eval gold labels.
"""
import csv
import datetime as dt
import json
import os
import random

SEED = 42
N = 500
N_USERS = 120
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "out")
rnd = random.Random(SEED)

# --------------------------------------------------------------------------
# Seed catalog (frozen reference data; every layer samples from this)
# --------------------------------------------------------------------------
# city: (iata, country, base_flight_inr, flight_hours, hotel_factor, landmarks)
CITIES = {
    "Dubai": ("DXB", "UAE", 24000, 3.5, 1.1, ["Burj Khalifa", "Desert Safari", "Dubai Mall"]),
    "Singapore": ("SIN", "Singapore", 28000, 5.5, 1.3, ["Gardens by the Bay", "Sentosa Island", "Night Safari"]),
    "Bangkok": ("BKK", "Thailand", 19000, 4.0, 0.7, ["Grand Palace", "Chatuchak Market", "Chao Phraya Cruise"]),
    "Paris": ("CDG", "France", 52000, 9.0, 1.5, ["Eiffel Tower", "Louvre Museum", "Seine Cruise"]),
    "London": ("LHR", "UK", 55000, 10.0, 1.6, ["Tower of London", "British Museum", "Thames Walk"]),
    "Tokyo": ("NRT", "Japan", 48000, 8.0, 1.3, ["Senso-ji Temple", "Shibuya Crossing", "Tsukiji Food Tour"]),
    "Bali": ("DPS", "Indonesia", 30000, 7.0, 0.8, ["Ubud Rice Terraces", "Uluwatu Temple", "Nusa Penida Trip"]),
    "Rome": ("FCO", "Italy", 50000, 9.5, 1.3, ["Colosseum", "Vatican Museums", "Trevi Fountain"]),
    "Istanbul": ("IST", "Turkey", 38000, 7.0, 0.9, ["Hagia Sophia", "Grand Bazaar", "Bosphorus Cruise"]),
    "New York": ("JFK", "USA", 70000, 15.0, 1.9, ["Statue of Liberty", "Central Park", "Broadway Show"]),
    "Sydney": ("SYD", "Australia", 62000, 13.0, 1.5, ["Opera House", "Bondi Beach", "Harbour Bridge Climb"]),
    "Kuala Lumpur": ("KUL", "Malaysia", 17000, 4.5, 0.75, ["Petronas Towers", "Batu Caves", "Bukit Bintang Food Walk"]),
    "Maldives": ("MLE", "Maldives", 22000, 3.5, 2.2, ["Snorkelling Tour", "Sandbank Picnic", "Dolphin Cruise"]),
    "Goa": ("GOI", "India", 6000, 1.5, 0.8, ["Baga Beach", "Old Goa Churches", "Dudhsagar Falls"]),
    "Phuket": ("HKT", "Thailand", 21000, 4.0, 0.8, ["Phi Phi Islands", "Big Buddha", "Old Town Walk"]),
    "Barcelona": ("BCN", "Spain", 49000, 10.0, 1.3, ["Sagrada Familia", "Park Guell", "La Boqueria Food Tour"]),
    "Amsterdam": ("AMS", "Netherlands", 51000, 9.5, 1.4, ["Anne Frank House", "Canal Cruise", "Van Gogh Museum"]),
    "Seoul": ("ICN", "South Korea", 40000, 7.5, 1.1, ["Gyeongbokgung Palace", "Myeongdong Street Food", "N Seoul Tower"]),
    "Hong Kong": ("HKG", "Hong Kong", 33000, 5.5, 1.4, ["Victoria Peak", "Star Ferry", "Lantau Big Buddha"]),
    "Zurich": ("ZRH", "Switzerland", 56000, 9.0, 2.0, ["Lake Zurich Cruise", "Old Town Walk", "Uetliberg Hike"]),
}
CITY_LIST = list(CITIES)
ORIGINS = {"Bangalore": "BLR", "Delhi": "DEL", "Mumbai": "BOM", "Chennai": "MAA", "Hyderabad": "HYD"}
ORIGIN_LIST = list(ORIGINS)
AIRLINES = ["IndiGo", "Air India", "Emirates", "Singapore Airlines", "Qatar Airways",
            "Thai Airways", "Vistara", "Etihad", "Lufthansa", "Cathay Pacific"]
HOTEL_SUFFIX = [("Grand Plaza", 5), ("Skyline Suites", 4), ("Central Inn", 3), ("Harbour View", 4)]
STAR_BASE = {3: 3000, 4: 6000, 5: 12000}
HOTELS = {c: [(f"{c} {s}", st) for s, st in HOTEL_SUFFIX] for c in CITY_LIST}

FIRST = ["Aarav", "Diya", "Rohan", "Ananya", "Karthik", "Meera", "Vikram", "Priya", "Arjun", "Sneha",
         "Rahul", "Kavya", "Siddharth", "Isha", "Manoj", "Divya", "Nikhil", "Pooja", "Suresh", "Neha",
         "Akash", "Lakshmi", "Varun", "Tanvi"]
LAST = ["Sharma", "Gowda", "Reddy", "Iyer", "Nair", "Patel", "Singh", "Rao", "Menon", "Kapoor",
        "Verma", "Joshi", "Das", "Bhat", "Shetty"]

INTENTS = ["plan_trip", "search_book_flight", "search_book_hotel",
           "build_itinerary", "modify_refine", "trip_management"]
OUTCOMES = ["booked", "abandoned", "changed_mind", "tool_error", "cancelled"]
BUCKETS = ["short", "medium", "long"]
ACCENTS = ["indian_english", "us_english", "uk_english", "australian_english", "singapore_english"]
NOISES = ["clean", "cafe", "street", "car", "telephony_8k"]
NOISE_CONF = {"clean": 0.965, "cafe": 0.915, "street": 0.885, "car": 0.90, "telephony_8k": 0.895}
DISFLUENCY = ["um, ", "uh, ", "so, ", "okay so ", "hey, ", "hmm, "]


def money(x):
    return int(round(x / 50.0) * 50)


def fd(d):
    return f"{d.day} {d.strftime('%B')}"


def r(x):
    return f"₹{x:,}"


# --------------------------------------------------------------------------
# RAG corpus (Qdrant) - conversations cite these doc ids
# --------------------------------------------------------------------------
RAG_DOCS = []
POLICY_DOCS = {
    "baggage": "Economy fares include 15 kg check-in and 7 kg cabin baggage on most carriers; extra kilos can be added at booking.",
    "cancellation": "Bookings can be cancelled free of charge within 24 hours of purchase; after that the airline or hotel cancellation policy applies.",
    "refund": "Eligible refunds are returned to the original payment method within 5 to 7 working days.",
    "change_fee": "Date changes carry a fixed change fee of Rs 2,500 plus any fare difference, subject to availability.",
    "payment": "Payments are processed securely through Stripe or Razorpay; card details are never stored by the voice agent.",
    "documents": "Carry a passport valid for at least six months and check visa requirements for your destination before travel.",
}
for k, v in POLICY_DOCS.items():
    RAG_DOCS.append({"doc_id": f"KB-{len(RAG_DOCS)+1:03d}", "type": "policy", "topic": k, "text": v})
POLICY_ID = {d["topic"]: d["doc_id"] for d in RAG_DOCS}
SEASON = ["November to March", "April to June", "October to December", "March to May"]
CITY_DOC = {}
for i, (c, (code, ctry, *_rest)) in enumerate(CITIES.items()):
    lm = CITIES[c][5]
    doc = {"doc_id": f"KB-{len(RAG_DOCS)+1:03d}", "type": "destination", "topic": c,
           "text": f"{c}, {ctry} ({code}): best visited {SEASON[i % 4]}. Top experiences include {lm[0]}, {lm[1]} and {lm[2]}."}
    RAG_DOCS.append(doc)
    CITY_DOC[c] = doc["doc_id"]

# --------------------------------------------------------------------------
# Users
# --------------------------------------------------------------------------
ROLES = ["end_user"] * 118 + ["partner", "admin"]  # real users dominate
users = []
pairs = [(f, l) for f in FIRST for l in LAST]
rnd.shuffle(pairs)
for i in range(N_USERS):
    f, l = pairs[i]
    users.append({
        "id": f"U{i+1:04d}", "name": f"{f} {l}",
        "email": f"{f.lower()}.{l.lower()}{i+1}@example.com",
        "phone": f"+91 90000 {i+1:05d}", "home_city": ORIGIN_LIST[i % len(ORIGIN_LIST)],
        "role": ROLES[i] if i < len(ROLES) else "end_user",
    })


# --------------------------------------------------------------------------
# Stratified quota helpers
# --------------------------------------------------------------------------
def cycle(values, n):
    """Exactly balanced list of length n (counts differ by at most 1), shuffled
    in blocks so adjacent items are not correlated."""
    out = []
    while len(out) < n:
        block = list(values)
        rnd.shuffle(block)
        out.extend(block)
    return out[:n]


def exact_flags(n, frac):
    k = round(n * frac)
    flags = [True] * k + [False] * (n - k)
    rnd.shuffle(flags)
    return flags


cells = [(i, o) for i in INTENTS for o in OUTCOMES]  # 30 cells
# 500 = 16 full passes over 30 cells + 20 remainder slots. Spread the remainder so
# every outcome gets exactly 4 extras and intents get 3 or 4 (max spread 1).
_rem = [(INTENTS[(4 * k + j) % 6], OUTCOMES[k]) for k in range(5) for j in range(4)]
cell_seq = []
for _ in range(N // len(cells)):
    _blk = list(cells)
    rnd.shuffle(_blk)
    cell_seq.extend(_blk)
rnd.shuffle(_rem)
cell_seq.extend(_rem)
cell_seq = cell_seq[:N]
city_seq = cycle(CITY_LIST, N)
bucket_seq = cycle(BUCKETS, N)
accent_seq = cycle(ACCENTS, N)
noise_seq = cycle(NOISES, N)
user_seq = cycle([u["id"] for u in users], N)
dis_flags = exact_flags(N, 0.35)
corr_flags = exact_flags(N, 0.25)
barge_flags = exact_flags(N, 0.20)

# --------------------------------------------------------------------------
# Table buffers
# --------------------------------------------------------------------------
T = {k: [] for k in ["sessions", "conversations", "trips", "bookings", "itineraries",
                     "agent_tasks", "audio_logs", "audit_logs", "scenarios"]}
GOLD = []
counters = {k: 0 for k in ["C", "AT", "AU", "AL", "B", "IT"]}


def nid(prefix, width=6):
    counters[prefix] += 1
    return f"{prefix}{counters[prefix]:0{width}d}"


def iso(t):
    return t.strftime("%Y-%m-%dT%H:%M:%S")


# --------------------------------------------------------------------------
# Exchange builders (one exchange = user utterance + agent reply)
# --------------------------------------------------------------------------
def X(u, intent, ent, a, agent=None, tools=None, rag=None, event=None):
    return {"u": u, "intent": intent, "ent": ent, "a": a, "agent": agent or "conversation_agent",
            "tools": tools or [], "rag": rag, "event": event, "barge": False}


def tool(agent, name, args, res, status="success"):
    return {"agent": agent, "name": name, "args": args, "result": res, "status": status}


def pax_txt(p):
    return "1 traveller" if p == 1 else f"{p} travellers"


def flight_quote(sc, st, date, factor=1.0):
    code, ctry, base, hrs, hf, lm = CITIES[sc["city"]]
    p = money(base * rnd.uniform(0.85, 1.35) * sc["pax"] * factor)
    air = rnd.choice(AIRLINES)
    st["flight"] = {"airline": air, "price": p, "dur": f"{int(hrs)}h {int((hrs % 1) * 60):02d}m", "date": date}
    return st["flight"]


def hotel_quote(sc, st, factor=1.0):
    hotels = [h for h in HOTELS[sc["city"]] if h[1] == sc["stars"]]
    name, stars = rnd.choice(hotels)
    code, ctry, base, hrs, hf, lm = CITIES[sc["city"]]
    nightly = money(STAR_BASE[stars] * hf * rnd.uniform(0.9, 1.15) * factor)
    st["hotel"] = {"name": name, "stars": stars, "nightly": nightly, "total": nightly * sc["nights"]}
    return st["hotel"]


def ex_flight_search(sc, st, date, first=True, refine=False):
    c, o = sc["city"], sc["origin"]
    f = flight_quote(sc, st, date, 0.8 if refine else 1.0)
    f2p = money(f["price"] * rnd.uniform(1.1, 1.25))
    k = rnd.randint(4, 12)
    ent = {"origin": o, "destination": c, "depart_date": date.isoformat(), "travelers": sc["pax"]}
    if refine:
        u = rnd.choice(["Show me cheaper options", "Anything cheaper than that?", "Can you find a lower fare?"])
        a = (f"Sure. I found a cheaper fare on {f['airline']} at {r(f['price'])} total, {f['dur']}. "
             f"Would you like me to book it?")
        return X(u, "refine_search", {"price_pref": "cheaper"}, a, "flight_agent",
                 [tool("flight_agent", "flights.search", {**ent, "sort": "price"}, {"n": k, "cheapest": f["price"]})])
    u = rnd.choice([
        f"Find flights from {o} to {c} on {fd(date)}",
        f"I need a flight from {o} to {c} on {fd(date)} for {pax_txt(sc['pax'])}",
        f"Search flights {o} to {c}, {fd(date)}, {pax_txt(sc['pax'])}"])
    a = (f"I found {k} flights. Cheapest is {f['airline']} at {r(f['price'])} total, {f['dur']}. "
         f"Fastest option is {rnd.choice(AIRLINES)} at {r(f2p)}. Want me to book the cheapest?")
    return X(u, "search_flights", ent, a, "flight_agent",
             [tool("flight_agent", "flights.search", ent, {"n": k, "cheapest": f["price"]})])


def ex_hotel_search(sc, st, date, refine=False):
    c = sc["city"]
    h = hotel_quote(sc, st, 0.8 if refine else 1.0)
    ent = {"destination": c, "check_in": date.isoformat(), "nights": sc["nights"], "stars": sc["stars"]}
    if refine:
        u = rnd.choice(["Show me cheaper options", "Do you have something more affordable?", "Anything cheaper?"])
        a = (f"Yes. {h['name']}, {h['stars']} star, at {r(h['nightly'])} per night, {r(h['total'])} total. "
             f"Shall I book it?")
        return X(u, "refine_search", {"price_pref": "cheaper"}, a, "hotel_agent",
                 [tool("hotel_agent", "hotels.search", {**ent, "sort": "price"}, {"top": h["name"], "total": h["total"]})])
    u = rnd.choice([
        f"Book a {sc['stars']} star hotel in {c} for {sc['nights']} nights from {fd(date)}",
        f"I need a {sc['stars']} star hotel in {c}, {sc['nights']} nights starting {fd(date)}",
        f"Find me a {sc['stars']} star stay in {c} from {fd(date)} for {sc['nights']} nights"])
    a = (f"I found {h['name']}, {h['stars']} star, at {r(h['nightly'])} per night, "
         f"{r(h['total'])} total for {sc['nights']} nights. Shall I book it?")
    return X(u, "search_hotels", ent, a, "hotel_agent",
             [tool("hotel_agent", "hotels.search", ent, {"top": h["name"], "total": h["total"]})])


def itinerary_text(sc, days):
    lm = CITIES[sc["city"]][5]
    plan = [lm[d % 3] for d in range(days)]
    return plan, "; ".join(f"Day {d+1}: {plan[d]}" for d in range(min(days, 2))) + ("..." if days > 2 else "")


def ex_extras(sc, st, kind):
    c = sc["city"]
    lm = CITIES[c][5]
    if kind == "budget":
        total = st.get("flight", {}).get("price", money(CITIES[c][2] * sc["pax"])) + \
            st.get("hotel", {}).get("total", money(STAR_BASE[sc["stars"]] * CITIES[c][4] * sc["nights"]))
        act = money(total * 0.12)
        return X("What would the whole trip cost roughly?", "ask_budget", {"destination": c},
                 f"Roughly {r(total + act)} including about {r(act)} for activities and local transport.",
                 "budget_agent", [tool("budget_agent", "budget.estimate", {"destination": c, "nights": sc["nights"]},
                                       {"total": total + act})])
    if kind == "weather":
        return X(f"What's the weather like in {c} then?", "ask_weather", {"destination": c},
                 f"Expect pleasant conditions with highs around {rnd.randint(22, 34)} degrees and a low chance of rain.",
                 "conversation_agent", [tool("conversation_agent", "weather.forecast", {"destination": c}, {"ok": True})])
    if kind == "activities":
        return X(f"Suggest some things to do in {c}", "ask_activities", {"destination": c},
                 f"Popular picks are {lm[0]}, {lm[1]} and {lm[2]}. Want me to add any to your plan?",
                 "activity_agent", [tool("activity_agent", "activities.search", {"destination": c}, {"top": lm})])
    if kind == "policy":
        topic = rnd.choice(["baggage", "cancellation", "change_fee"])
        q = {"baggage": "What's the baggage allowance?", "cancellation": "What is the cancellation policy?",
             "change_fee": "Is there a fee if I change the date later?"}[topic]
        return X(q, "ask_policy", {"topic": topic}, POLICY_DOCS[topic], "conversation_agent",
                 [tool("conversation_agent", "rag.lookup", {"query": topic}, {"doc_id": POLICY_ID[topic]})],
                 rag=POLICY_ID[topic])
    return X(f"How far is the hotel from the airport in {c}?", "ask_location", {"destination": c},
             f"About {rnd.randint(12, 45)} minutes by taxi from {CITIES[c][0]}.",
             "conversation_agent", [tool("conversation_agent", "maps.distance", {"destination": c}, {"ok": True})])


def ex_correction(sc, st, kind):
    if kind == "days":
        d2 = sc["nights"] + 2
        st["days_override"] = d2
        _, txt = itinerary_text(sc, d2)
        return X(rnd.choice([f"Actually, make it {d2} days", f"Sorry, I meant {d2} days"]), "correct_slot",
                 {"days": d2}, f"Updated. {txt}", "itinerary_agent",
                 [tool("itinerary_agent", "itinerary.build", {"city": sc["city"], "days": d2}, {"days": d2})])
    d2 = sc["d1"] + dt.timedelta(days=rnd.randint(2, 6))
    st["date"] = d2
    u = rnd.choice([f"Actually, make it {fd(d2)} instead", f"Sorry, I meant {fd(d2)}", f"No wait, {fd(d2)} please"])
    if sc["kind"] == "hotel":
        e = ex_hotel_search(sc, st, d2)
    else:
        e = ex_flight_search(sc, st, d2)
    e["u"], e["intent"] = u, "correct_slot"
    e["ent"] = {"depart_date" if sc["kind"] != "hotel" else "check_in": d2.isoformat()}
    return e


# ---- closings -------------------------------------------------------------
def amount_for(sc, st):
    if sc["kind"] == "package":
        return [("flight", st["flight"]["price"], st["flight"]["airline"]),
                ("hotel", st["hotel"]["total"], st["hotel"]["name"])]
    if sc["kind"] == "flight":
        return [("flight", st["flight"]["price"], st["flight"]["airline"])]
    return [("hotel", st["hotel"]["total"], st["hotel"]["name"])]


def ref_code():
    return "TRV" + "".join(rnd.choice("ABCDEFGHJKLMNPQRSTUVWXYZ23456789") for _ in range(6))


def fail_pair(sc, agent, toolname, provider):
    """Provider timeout twice -> graceful degradation (poster NFR section 11)."""
    t1 = tool(agent, toolname, {"retry": 0}, {"error": "provider_timeout"}, "failed")
    t2 = tool(agent, toolname, {"retry": 1}, {"error": "provider_timeout"}, "failed")
    return [
        X("Yes, go ahead and book it", "confirm_booking", {"confirm": True},
          f"Sorry, the {provider} service timed out. Would you like me to try again?", agent, [t1]),
        X(rnd.choice(["Yes, try again", "Okay, retry please"]), "retry", {"retry": True},
          "It's still unavailable. I've queued the request and will notify you by email as soon as it goes through.",
          agent, [t2, tool("notification_service", "notify.queue", {"channel": "email"}, {"queued": True})]),
    ]


def closing(sc, st):
    o, k = sc["outcome"], sc["kind"]
    main_agent = {"flight": "flight_agent", "hotel": "hotel_agent", "package": "flight_agent",
                  "itinerary": "itinerary_agent"}[k]
    exs = []
    if o == "abandoned":
        return [X(rnd.choice(["Let me think about it and come back later", "I'll check with my family first"]),
                  "defer", {}, "No problem. I've saved your search so you can pick it up anytime.",
                  "conversation_agent", [tool("conversation_agent", "session.save_state", {}, {"saved": True})])]
    if o == "tool_error":
        provider = {"flight": "airline", "hotel": "hotel", "package": "airline", "itinerary": "itinerary"}[k]
        return fail_pair(sc, main_agent, "itinerary.save" if k == "itinerary" else "booking.create", provider)
    if o == "changed_mind":
        if k == "itinerary":
            d3 = (st.get("days_override") or sc["nights"] + 1) + 1
            st["days_override"] = d3
            exs.append(X(f"Hmm, actually let's do {d3} days", "change_mind", {"days": d3},
                         f"Sure, I've extended it to {d3} days.", "itinerary_agent",
                         [tool("itinerary_agent", "itinerary.build", {"city": sc["city"], "days": d3}, {"days": d3})]))
        else:
            exs.append(X("Hmm, actually can you find a different option?", "change_mind", {"want": "alternative"},
                         "Of course. Let me look for alternatives.", main_agent, []))
            if k in ("flight", "package"):
                alt = ex_flight_search(sc, st, st.get("date", sc["d1"]), refine=True)
                alt["u"], alt["intent"] = "Go with something lower priced", "refine_search"
                exs[-1]["a"] = f"Found a better fare on {st['flight']['airline']} at {r(st['flight']['price'])}."
                exs[-1]["tools"] = alt["tools"]
            if k in ("hotel", "package"):
                hotel_quote(sc, st, 0.85)
                if k == "hotel":
                    exs[-1]["a"] = f"{st['hotel']['name']} at {r(st['hotel']['total'])} total works out cheaper."
                    exs[-1]["tools"] = [tool("hotel_agent", "hotels.search", {"sort": "price"}, {"top": st["hotel"]["name"]})]
    # --- confirm + create (booked / changed_mind / cancelled) ---
    if k == "itinerary":
        exs.append(X("Yes, save it and send it to my email", "confirm_booking", {"confirm": True},
                     "Saved. I've emailed your itinerary and a shareable link.", "itinerary_agent",
                     [tool("itinerary_agent", "itinerary.save", {"trip": "current"}, {"saved": True}),
                      tool("notification_service", "notify.email", {"template": "itinerary"}, {"sent": True})],
                     event={"type": "itinerary_saved"}))
        if o == "cancelled":
            exs.append(X("Actually, please delete that itinerary", "cancel_booking", {"target": "itinerary"},
                         "Done, the itinerary has been deleted.", "itinerary_agent",
                         [tool("itinerary_agent", "itinerary.delete", {"trip": "current"}, {"deleted": True})],
                         event={"type": "itinerary_deleted"}))
        return exs
    ref = ref_code()
    st["ref"] = ref
    total = sum(a for _, a, _ in amount_for(sc, st))
    exs.append(X("Yes, book it", "confirm_booking", {"confirm": True},
                 f"Booked! Your reference is {ref}, total {r(total)}. A confirmation email is on its way.",
                 main_agent,
                 [tool("payment_service", "payment.charge", {"amount": total, "currency": "INR"}, {"status": "paid"}),
                  tool(main_agent, "booking.create", {"kind": k}, {"ref": ref}),
                  tool("notification_service", "notify.email", {"template": "booking_confirmation"}, {"sent": True})],
                 event={"type": "created", "ref": ref}))
    if o == "cancelled":
        refund = money(total * 0.9)
        exs.append(X("Sorry, please cancel that booking", "cancel_booking", {"ref": ref},
                     f"Booking {ref} is cancelled. A refund of {r(refund)} will reach you in 5 to 7 working days.",
                     main_agent,
                     [tool(main_agent, "booking.cancel", {"ref": ref}, {"refund": refund}),
                      tool("notification_service", "notify.email", {"template": "cancellation"}, {"sent": True})],
                     rag=POLICY_ID["refund"], event={"type": "cancelled", "refund": refund}))
    return exs


def manage_flow(sc, st):
    """trip_management: acts on a booking made earlier via the web app."""
    b = st["pre"]
    d_new = sc["d1"] + dt.timedelta(days=rnd.randint(3, 10))
    fee = 2500
    diff = money(b["amount"] * rnd.uniform(0.0, 0.1))
    o = sc["outcome"]
    cancel = o == "cancelled"
    if cancel:
        refund = money(b["amount"] * 0.9)
        first = X(rnd.choice([f"I need to cancel booking {b['ref']}", f"Please cancel my {b['type']} booking {b['ref']}"]),
                  "cancel_booking", {"ref": b["ref"]},
                  f"I found {b['ref']}: {sc['city']} {b['type']}, {r(b['amount'])}. Cancelling would refund {r(refund)}. Shall I proceed?",
                  "conversation_agent",
                  [tool("conversation_agent", "trips.lookup", {"ref": b["ref"]}, {"found": True})], rag=POLICY_ID["cancellation"])
    else:
        first = X(rnd.choice([f"Change my travel date to {fd(d_new)} for booking {b['ref']}",
                              f"I want to move booking {b['ref']} to {fd(d_new)}"]),
                  "modify_booking", {"ref": b["ref"], "new_date": d_new.isoformat()},
                  f"I found {b['ref']}: {sc['city']} {b['type']}. Moving it costs a {r(fee)} change fee plus {r(diff)} fare difference. Proceed?",
                  "conversation_agent", [tool("conversation_agent", "trips.lookup", {"ref": b["ref"]}, {"found": True})],
                  rag=POLICY_ID["change_fee"])
    exs = [first]
    return exs, d_new, fee, diff


def manage_closing(sc, st, d_new, fee, diff):
    b = st["pre"]
    o = sc["outcome"]
    agent = "flight_agent" if b["type"] == "flight" else "hotel_agent"
    if o == "abandoned":
        return [X("Let me check with my family and get back to you", "defer", {},
                  "Sure, nothing has been changed on your booking.", "conversation_agent", [])]
    if o == "tool_error":
        return fail_pair(sc, agent, "booking.modify", "airline" if b["type"] == "flight" else "hotel")
    if o == "cancelled":
        refund = money(b["amount"] * 0.9)
        return [X("Yes, cancel it", "confirm_booking", {"confirm": True},
                  f"Booking {b['ref']} is cancelled. Refund of {r(refund)} in 5 to 7 working days.", agent,
                  [tool(agent, "booking.cancel", {"ref": b["ref"]}, {"refund": refund}),
                   tool("notification_service", "notify.email", {"template": "cancellation"}, {"sent": True})],
                  event={"type": "cancelled", "refund": refund})]
    exs = [X("Yes, confirm the change", "confirm_booking", {"confirm": True},
             f"Done. Booking {b['ref']} now starts on {fd(d_new)}. Updated confirmation sent.", agent,
             [tool(agent, "booking.modify", {"ref": b["ref"], "new_date": d_new.isoformat()}, {"ok": True}),
              tool("notification_service", "notify.email", {"template": "booking_modified"}, {"sent": True})],
             event={"type": "modified", "delta": diff + fee})]
    if o == "changed_mind":
        d2 = d_new + dt.timedelta(days=7)
        exs.append(X("Actually, can we move it one more week?", "modify_booking", {"new_date": d2.isoformat()},
                     f"Sure, I've moved it again to {fd(d2)}. A second change fee of {r(fee)} applies.", agent,
                     [tool(agent, "booking.modify", {"ref": b["ref"], "new_date": d2.isoformat()}, {"ok": True})],
                     event={"type": "modified", "delta": fee}))
    return exs


def build_exchanges(sc, st):
    i, o = sc["intent"], sc["outcome"]
    c, d1 = sc["city"], sc["d1"]
    exs = []
    extras_n = {"short": 0, "medium": rnd.randint(1, 2), "long": rnd.randint(3, 4)}[sc["bucket"]]
    extra_kinds = rnd.sample(["budget", "weather", "activities", "policy", "location"], extras_n)
    corr = sc["corr"]

    if i == "trip_management":
        exs, d_new, fee, diff = manage_flow(sc, st)
        exs += [ex_extras(sc, st, k) for k in extra_kinds if k != "budget"][:extras_n]
        exs += manage_closing(sc, st, d_new, fee, diff)
        return exs

    if i == "plan_trip":
        flight_quote(sc, st, d1)
        hotel_quote(sc, st)
        budget = sc["budget"]
        est_a = money(st["hotel"]["total"] * 0.0 + budget * 0.15)
        exs.append(X(rnd.choice([f"Plan a {sc['nights']+1} day trip to {c}", f"I want to plan a {sc['nights']+1} day holiday in {c}"]),
                     "plan_trip", {"destination": c, "days": sc["nights"] + 1},
                     "Happy to help. What's your budget and when would you like to travel?", "conversation_agent", []))
        exs.append(X(f"Around {r(budget)}, leaving {fd(d1)} from {sc['origin']}, {pax_txt(sc['pax'])}", "provide_slots",
                     {"budget": budget, "depart_date": d1.isoformat(), "origin": sc["origin"], "travelers": sc["pax"]},
                     f"Estimate: flights {r(st['flight']['price'])}, hotel {r(st['hotel']['total'])}, activities {r(est_a)}. Shall I search options?",
                     "budget_agent", [tool("budget_agent", "budget.estimate", {"destination": c, "budget": budget}, {"total": st["flight"]["price"] + st["hotel"]["total"] + est_a})]))
        exs.append(X("Yes, show me options", "confirm_search", {"confirm": True},
                     f"Best match: {st['flight']['airline']} at {r(st['flight']['price'])} and {st['hotel']['name']} at {r(st['hotel']['total'])}. Book both?",
                     "flight_agent", [tool("flight_agent", "flights.search", {"destination": c}, {"cheapest": st["flight"]["price"]}),
                                      tool("hotel_agent", "hotels.search", {"destination": c}, {"top": st["hotel"]["name"]})]))
    elif i == "search_book_flight":
        exs.append(ex_flight_search(sc, st, d1))
    elif i == "search_book_hotel":
        exs.append(ex_hotel_search(sc, st, d1))
    elif i == "build_itinerary":
        days = sc["nights"] + 1
        plan, txt = itinerary_text(sc, days)
        exs.append(X(rnd.choice([f"Create a {days} day itinerary in {c}", f"Build me a {days} day plan for {c}"]),
                     "build_itinerary", {"destination": c, "days": days},
                     f"Here is a {days} day plan. {txt}", "itinerary_agent",
                     [tool("itinerary_agent", "itinerary.build", {"city": c, "days": days}, {"days": days})],
                     rag=CITY_DOC[c]))
        exs.append(X("Add more food experiences", "refine_itinerary", {"preference": "food"},
                     "Added a local food walk and a market visit.", "itinerary_agent",
                     [tool("itinerary_agent", "itinerary.modify", {"add": "food"}, {"ok": True})]))
    elif i == "modify_refine":
        if sc["kind"] == "hotel":
            exs.append(ex_hotel_search(sc, st, d1))
            exs.append(ex_hotel_search(sc, st, d1, refine=True))
        else:
            exs.append(ex_flight_search(sc, st, d1))
            exs.append(ex_flight_search(sc, st, d1, refine=True))

    if corr and i not in ("build_itinerary",):
        exs.append(ex_correction(sc, st, "date"))
    elif corr:
        exs.append(ex_correction(sc, st, "days"))

    for k in extra_kinds:
        exs.append(ex_extras(sc, st, k))
    exs += closing(sc, st)
    return exs


# --------------------------------------------------------------------------
# Materialize one scenario into every table with a single timeline
# --------------------------------------------------------------------------
def audit(user_id, ts, etype, details):
    T["audit_logs"].append({"id": nid("AL"), "event_type": etype, "user_id": user_id,
                            "details": json.dumps(details, ensure_ascii=False), "timestamp": iso(ts)})


def materialize(idx, cell, city, bucket, accent, noise, uid, dis, corr, barge):
    intent, outcome = cell
    user = users[int(uid[1:]) - 1]
    sid, tid = f"S{idx+1:04d}", f"T{idx+1:04d}"
    kind = {"plan_trip": "package", "search_book_flight": "flight", "search_book_hotel": "hotel",
            "build_itinerary": "itinerary", "modify_refine": "flight" if idx % 2 == 0 else "hotel",
            "trip_management": "flight" if idx % 2 == 0 else "hotel"}[intent]
    start = dt.datetime(2026, 1, 5, 0, 0, 0) + dt.timedelta(
        days=rnd.randint(0, 293), hours=rnd.randint(6, 22), minutes=rnd.randint(0, 59), seconds=rnd.randint(0, 59))
    nights = rnd.randint(3, 7)
    sc = {"idx": idx, "intent": intent, "outcome": outcome, "city": city, "origin": user["home_city"],
          "pax": rnd.choice([1, 1, 2, 2, 3, 4]), "nights": nights, "stars": rnd.choice([3, 4, 4, 5]),
          "d1": (start + dt.timedelta(days=rnd.randint(21, 150))).date(), "bucket": bucket, "kind": kind,
          "corr": corr, "budget": 0}
    st = {}
    # baseline budget for plan_trip
    code, ctry, base, hrs, hf, lm = CITIES[city]
    sc["budget"] = money((base * sc["pax"] + STAR_BASE[sc["stars"]] * hf * nights) * rnd.uniform(1.2, 1.6))

    if intent == "trip_management":
        amt = money(base * rnd.uniform(0.9, 1.3) * sc["pax"]) if kind == "flight" else \
            money(STAR_BASE[sc["stars"]] * hf * nights)
        st["pre"] = {"ref": ref_code(), "type": kind, "amount": amt,
                     "created": start - dt.timedelta(days=rnd.randint(5, 60))}

    exs = build_exchanges(sc, st)

    # phenomena: disfluency + barge-in (applied post-hoc, flagged for eval)
    if dis:
        for j in rnd.sample(range(len(exs)), k=min(len(exs), 2)):
            if not exs[j]["u"].startswith(tuple(DISFLUENCY)):
                exs[j]["u"] = rnd.choice(DISFLUENCY) + exs[j]["u"][0].lower() + exs[j]["u"][1:]
    if barge:
        cand = [j for j in range(len(exs) - 1) if len(exs[j]["a"].split()) > 12]
        if cand:
            j = rnd.choice(cand)
            words = exs[j]["a"].split()
            exs[j]["a"] = " ".join(words[: max(5, len(words) // 2)]) + "..."
            exs[j]["barge"] = True
            nxt = exs[j + 1]
            nxt["u"] = rnd.choice(["Sorry to cut in, ", "Wait, ", "Stop, "]) + nxt["u"][0].lower() + nxt["u"][1:]

    # ---- timeline ----
    t = start
    audit(user["id"], t, "session_started", {"session_id": sid, "channel": "voice", "accent": accent, "noise": noise})
    trip = {"id": tid, "user_id": user["id"], "destination": city,
            "start_date": None, "end_date": None, "budget": sc["budget"], "status": "draft",
            "created_at": iso(start if intent != "trip_management" else st["pre"]["created"])}
    bookings = []
    if intent == "trip_management":
        pb = st["pre"]
        brow = {"id": nid("B"), "trip_id": tid, "user_id": user["id"], "session_id": "", "channel": "web",
                "type": pb["type"], "provider": rnd.choice(AIRLINES) if pb["type"] == "flight" else f"{city} Grand Plaza",
                "reference": pb["ref"], "status": "confirmed", "amount": pb["amount"], "currency": "INR",
                "created_at": iso(pb["created"]), "updated_at": iso(pb["created"])}
        bookings.append(brow)
        trip["status"] = "booked"
        sc["d1"] = sc["d1"]
    final_date = sc["d1"]
    saved_itinerary = False
    gold_rows = []
    conv_rows = []
    for n, e in enumerate(exs, 1):
        u_words = len(e["u"].split())
        u_dur = round(u_words * 0.4 + rnd.uniform(0.4, 1.2), 2)
        conf = round(min(0.995, max(0.6, NOISE_CONF[noise] + rnd.uniform(-0.03, 0.03) - (0.015 if dis else 0))), 3)
        slow = outcome == "tool_error" and any(tl["status"] == "failed" for tl in e["tools"])
        latency = rnd.randint(1600, 2600) if slow else rnd.randint(700, 1450)
        a_words = len(e["a"].split())
        a_dur = round(a_words * 0.35, 2)
        if e["barge"]:
            a_dur = round(a_dur * 0.5, 2)
        ts_user = t
        ts_agent = t + dt.timedelta(seconds=u_dur + latency / 1000)
        cid = nid("C")
        conv_rows.append({"id": cid, "session_id": sid, "turn_no": n, "user_text": e["u"], "agent_text": e["a"],
                          "intent": e["intent"], "agent": e["agent"], "barge_in": int(e["barge"]),
                          "rag_doc_id": e["rag"] or "", "latency_ms": latency, "ts": iso(ts_user)})
        T["audio_logs"].append({"id": nid("AU"), "session_id": sid, "turn_no": n, "type": "voice_in",
                                "file_path": f"audio/{sid}/turn_{n:02d}_in.wav", "duration_sec": u_dur,
                                "sample_rate": 16000, "accent": accent, "noise_profile": noise,
                                "stt_model": "faster-whisper", "stt_confidence": conf, "source": "planned_synthetic"})
        T["audio_logs"].append({"id": nid("AU"), "session_id": sid, "turn_no": n, "type": "voice_out",
                                "file_path": f"audio/{sid}/turn_{n:02d}_out.wav", "duration_sec": a_dur,
                                "sample_rate": 22050, "accent": "piper_en", "noise_profile": "clean",
                                "stt_model": "", "stt_confidence": "", "source": "planned_synthetic"})
        gold_rows.append({"session_id": sid, "turn_no": n, "user_text": e["u"], "gold_intent": e["intent"],
                          "gold_entities": e["ent"], "gold_agent": e["agent"],
                          "gold_tools": [tl["name"] for tl in e["tools"]],
                          "phenomena": {"disfluency": bool(dis), "barge_in_after": e["barge"],
                                        "noise": noise, "accent": accent}})
        # agent tasks + audit
        for tl in e["tools"]:
            T["agent_tasks"].append({"id": nid("AT"), "session_id": sid, "agent": tl["agent"], "tool": tl["name"],
                                     "status": tl["status"], "request": json.dumps(tl["args"], ensure_ascii=False),
                                     "result": json.dumps(tl["result"], ensure_ascii=False), "created_at": iso(ts_agent)})
            audit(user["id"], ts_agent, "task_" + tl["status"], {"session_id": sid, "tool": tl["name"]})
        # booking events
        ev = e["event"]
        if ev:
            if ev["type"] == "created":
                for btype, amt, prov in amount_for(sc, st):
                    bookings.append({"id": nid("B"), "trip_id": tid, "user_id": user["id"], "session_id": sid,
                                     "channel": "voice", "type": btype, "provider": prov,
                                     "reference": ev["ref"] + ("" if len(amount_for(sc, st)) == 1 else btype[0].upper()),
                                     "status": "confirmed", "amount": amt, "currency": "INR",
                                     "created_at": iso(ts_agent), "updated_at": iso(ts_agent)})
                    audit(user["id"], ts_agent, "booking_created", {"session_id": sid, "type": btype, "amount": amt})
                trip["status"] = "booked"
            elif ev["type"] == "cancelled":
                for b in bookings:
                    b["status"], b["updated_at"] = "cancelled", iso(ts_agent)
                    audit(user["id"], ts_agent, "booking_cancelled", {"session_id": sid, "booking_id": b["id"]})
                trip["status"] = "cancelled"
            elif ev["type"] == "modified":
                for b in bookings:
                    b["amount"] = b["amount"] + ev["delta"]
                    b["updated_at"] = iso(ts_agent)
                    audit(user["id"], ts_agent, "booking_modified", {"session_id": sid, "booking_id": b["id"]})
            elif ev["type"] == "itinerary_saved":
                saved_itinerary = True
                trip["status"] = "planned"
                audit(user["id"], ts_agent, "itinerary_saved", {"session_id": sid})
            elif ev["type"] == "itinerary_deleted":
                saved_itinerary = False
                trip["status"] = "cancelled"
                audit(user["id"], ts_agent, "itinerary_deleted", {"session_id": sid})
        t = ts_agent + dt.timedelta(seconds=a_dur + rnd.uniform(1.0, 3.0))

    end = t
    audit(user["id"], end, "session_ended", {"session_id": sid, "turns": len(exs), "outcome": outcome})
    sd = st.get("date", sc["d1"])
    days = (st.get("days_override") or nights + 1) - 1 if intent in ("build_itinerary", "plan_trip") else nights
    trip["start_date"] = sd.isoformat()
    trip["end_date"] = (sd + dt.timedelta(days=days)).isoformat()
    T["trips"].append(trip)
    T["bookings"].extend(bookings)
    T["conversations"].extend(conv_rows)
    T["sessions"].append({"id": sid, "user_id": user["id"], "trip_id": tid, "status": "ended", "channel": "voice",
                          "started_at": iso(start), "ended_at": iso(end), "turns": len(exs),
                          "duration_sec": int((end - start).total_seconds())})
    if intent in ("plan_trip", "build_itinerary") and saved_itinerary or \
            (intent == "plan_trip" and outcome in ("booked", "changed_mind", "cancelled")):
        if outcome not in ("abandoned", "tool_error"):
            ndays = (st.get("days_override") or nights + 1)
            plan, _ = itinerary_text(sc, ndays)
            for d in range(ndays):
                T["itineraries"].append({"id": nid("IT", 5), "trip_id": tid, "day_no": d + 1,
                                         "activities": plan[d], "notes": f"Synthetic plan for {city}"})
    for g in gold_rows:
        g["scenario_id"] = sid
        GOLD.append(g)
    T["scenarios"].append({"session_id": sid, "intent": intent, "outcome": outcome, "destination": city,
                           "origin": sc["origin"], "turn_bucket": bucket, "n_turns": len(exs),
                           "accent": accent, "noise_profile": noise, "disfluency": int(dis),
                           "correction": int(corr), "barge_in": int(any(e["barge"] for e in exs)),
                           "travelers": sc["pax"], "nights": nights, "budget": sc["budget"]})


def write_csv(name, rows):
    if not rows:
        return
    with open(os.path.join(OUT, name + ".csv"), "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)


def main():
    os.makedirs(OUT, exist_ok=True)
    for i in range(N):
        materialize(i, cell_seq[i], city_seq[i], bucket_seq[i], accent_seq[i], noise_seq[i], user_seq[i],
                    dis_flags[i], corr_flags[i], barge_flags[i])

    # stratified 80/10/10 split on intent x outcome (by scenario, not turn: no leakage)
    split = {}
    by_cell = {}
    for s in T["scenarios"]:
        by_cell.setdefault((s["intent"], s["outcome"]), []).append(s["session_id"])
    for ids in by_cell.values():
        rnd.shuffle(ids)
    flat = []
    for ids in by_cell.values():
        flat.extend(ids)
    # interleave so split proportions hold per intent
    order = sorted(flat, key=lambda s: (T["scenarios"][int(s[1:]) - 1]["intent"], rnd.random()))
    per_intent = {}
    for sid in order:
        per_intent.setdefault(T["scenarios"][int(sid[1:]) - 1]["intent"], []).append(sid)
    for ids in per_intent.values():
        n = len(ids)
        for j, sid in enumerate(ids):
            split[sid] = "train" if j < round(n * 0.8) else ("val" if j < round(n * 0.9) else "test")
    for s in T["scenarios"]:
        s["split"] = split[s["session_id"]]
    for g in GOLD:
        g["split"] = split[g["session_id"]]

    write_csv("users", users)
    for name in ["sessions", "conversations", "trips", "bookings", "itineraries", "agent_tasks",
                 "audio_logs", "audit_logs", "scenarios"]:
        write_csv(name, T[name])
    with open(os.path.join(OUT, "rag_docs.jsonl"), "w", encoding="utf-8") as f:
        for d in RAG_DOCS:
            f.write(json.dumps(d, ensure_ascii=False) + "\n")
    with open(os.path.join(OUT, "eval_gold.jsonl"), "w", encoding="utf-8") as f:
        for g in GOLD:
            f.write(json.dumps(g, ensure_ascii=False) + "\n")
    with open(os.path.join(OUT, "conversations.jsonl"), "w", encoding="utf-8") as f:
        by_s = {}
        for c in T["conversations"]:
            by_s.setdefault(c["session_id"], []).append(c)
        meta = {s["session_id"]: s for s in T["scenarios"]}
        for sid, turns in by_s.items():
            f.write(json.dumps({"session_id": sid, "meta": meta[sid], "turns": turns}, ensure_ascii=False) + "\n")
    with open(os.path.join(OUT, "manifest.json"), "w", encoding="utf-8") as f:
        json.dump({"seed": SEED, "n_conversations": N, "n_users": N_USERS, "generated_for": "Voice Travel Agent LLD",
                   "tables": {k: len(v) for k, v in T.items()}, "currency": "INR",
                   "note": "Fully synthetic. Audio files are planned (metadata only); synthesize with Piper TTS + noise."},
                  f, indent=2)
    print({k: len(v) for k, v in T.items()})


if __name__ == "__main__":
    main()
