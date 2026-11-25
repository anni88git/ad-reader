from adreader import normalize as nz


def test_prices():
    cases = {
        ("₹ 27 99 000*", "Introductory price starts at"): (2799000, "price"),
        ("₹48/-", "old price (struck out)"): (48, "old_price"),
        ("र44/-", "मात्र"): (44, "offer_price"),
        ("Rs. 1.5 Lakh", None): (150000, "price"),
        ("₹2.5 Cr onwards", None): (25000000, "price"),
        ("₹10,000", "EMI transactions cashback*"): (10000, "cashback"),
        ("₹ 4,999/month", None): (4999, "emi"),
        ("रु ४४", None): (44, "price"),
        ("₹48/-", "Promotional Offer"): (48, "price"),       # 'Offer' must not match 'off'
    }
    for (text, label), (amount, kind) in cases.items():
        p = nz.parse_price(text, label)
        assert (p.amount, p.kind) == (amount, kind), (text, label, p)


def test_percentage_is_not_an_amount():
    p = nz.parse_price("Flat 20% off", None)
    assert p.amount is None and p.kind == "discount"


def test_phones():
    cases = {
        "Call 1800 108 5005": ("18001085005", "toll_free"),
        "Tel: 9289305692": ("9289305692", "mobile"),
        "+91-512-2240251": ("05122240251", "landline"),
        "0512-2240251": ("05122240251", "landline"),
        "०२२-६८२७६८२७": ("02268276827", "landline"),
        "(022) 68276827": ("02268276827", "landline"),
        "+91 98765 43210": ("9876543210", "mobile"),
        "मो0 नं0- 7988859406": ("7988859406", "mobile"),
    }
    for text, (num, kind) in cases.items():
        p = nz.parse_phone(text)
        assert (p.number, p.kind, p.valid) == (num, kind, True), (text, p)
    assert not nz.parse_phone("12345").valid


def test_websites():
    assert nz.parse_website("Visit www.kia.com")[0] == "www.kia.com"
    assert nz.parse_website("Website:www.namasteindiafoods.com")[0] == "www.namasteindiafoods.com"
    assert nz.parse_website("not a site")[0] is None


def test_fuel_and_ev():
    assert nz.parse_fuel_types("Hybrid & Diesel Powertrains") == ["Hybrid", "Diesel"]
    assert nz.ev_type(["Hybrid", "Diesel"]) == "Non-EV"
    assert nz.ev_type(nz.parse_fuel_types("100% Electric, 500 km range")) == "EV"
    assert nz.ev_type(nz.parse_fuel_types("Plug-in Hybrid")) == "Non-EV"
    assert nz.parse_fuel_types("Petrol / CNG") == ["Petrol", "CNG"]
    assert nz.ev_type([]) is None
