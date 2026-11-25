"""Claude-drafted labels for the HELD-OUT set (data/eval/holdout20) - written from the images
BEFORE looking at any system output for these ads. Same format/conventions as _make_labels.py."""
import json
import sys
from pathlib import Path

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE))
from _make_labels import ANY, CORE_SKIP, PROV  # noqa: E402

L = {}
OPT = lambda *a: {"required": [], "optional": list(a)}  # noqa: E731
ELEC = ANY("Retail & Consumer Electronics", "Retail & Consumer Electronics > Mobiles & Computers",
           "Retail & Consumer Electronics > Home Appliances")
FOOD = ANY("FMCG > Food & Beverages", "FMCG", "Food Delivery & Restaurants")


def lab(stem, skip=(), **fields):
    attrs = fields.pop("attrs", None)
    d = {**PROV, "_set": "holdout20", "_skip": CORE_SKIP + list(skip), **fields}
    if attrs:
        d["category_attributes"] = attrs
    L[stem] = d


lab("h01_One_Plus_mobiles_413", skip=["advertiser"], language=["English"], brand="OnePlus",
    product=ANY("OnePlus 5T", "5T"), model_variant=None, ad_type="manufacturer", category=ELEC,
    headline="A New View", sub_headline=ANY("Thank you for all the love.", "OnePlus 5T Pop ups around the world", None),
    price=None, phones=[], websites={"required": ["oneplusstore.in"], "optional": ["amazon.in"]}, emails=[],
    qr_code_present=False)

lab("h02_Titan_watches_423", skip=["advertiser", "product"], language=["English"], brand="Titan", model_variant=None,
    ad_type=ANY("manufacturer", "retailer"), category=ANY("Fashion & Apparel > Watches & Accessories", "Fashion & Apparel"),
    headline=ANY("Up to 40% Off*", "Up to 40% Off"),
    sub_headline=ANY("15th Dec Onwards", "Additional 5% off for Encircle members till 17th Dec", None),
    price=None, phones=["18002660123"], websites=["titan.co.in"], emails=[], qr_code_present=False)

lab("h03_Apple_laptops_and_computers_416", skip=["advertiser"], language=["English"], brand="Apple",
    product="iPhone XR", model_variant=None, ad_type=ANY("retailer", "manufacturer", "dealer"), category=ELEC,
    headline="Make the celebrations bigger with iPhone XR.",
    sub_headline=ANY(None, "10% Cashback on EMI only with Citi Credit Cards*",
                     "Offer valid at Apple Authorised Resellers and selected stores only."),
    price=None, phones=["7600030000"], websites=["www.indiaistore.com"], emails=[], qr_code_present=False)

SERVICE = ANY("service_provider", "manufacturer")

lab("h04_Sunfeast_402", skip=["advertiser", "product"], language=ANY(["English"], ["English", "Hindi"]),
    brand="Sunfeast", model_variant=None, ad_type="manufacturer", category=FOOD,
    headline="Celebrate this Diwali with three sweet delights",
    sub_headline=ANY("Spread the festive joy with Dark Fantasy Compliments, Mom's Magic Expressions and the Sunfeast Gift Box.", None),
    price=ANY(200, 150), phones=[], websites=[], emails=[], qr_code_present=False)

lab("h05_toyota_406", skip=["phones", "advertiser"], language=["English"], brand="Toyota",
    product=ANY("Yaris", "The Happenin' New Yaris", "Urban Cruiser", "Glanza"), model_variant=None,
    ad_type=ANY("dealer", "manufacturer"), category="Automobile > Cars",
    headline=ANY("This Fabulous February", "Drive to Prosperity with Toyota"),
    sub_headline=ANY("Drive to Prosperity with Toyota", "The Best Time to Buy Your Dream Toyota!",
                     "This Fabulous February", None),
    price=ANY(916000, 850000, 718000), websites=["www.toyotabharat.com"], emails=[], qr_code_present=False)

lab("h06_Club_Mahindra_408", skip=["advertiser"], language=["English"], brand="Club Mahindra",
    product=ANY("Club Mahindra Kumbhalgarh", "Club Mahindra membership", "Club Mahindra Holidays", None),
    model_variant=None, ad_type=SERVICE, category="Travel & Hospitality",
    headline=ANY("Everything for Nothing", "Experience everything at Club Mahindra Kumbhalgarh."),
    sub_headline=ANY("Experience everything at Club Mahindra Kumbhalgarh.", "Everything for Nothing",
                     "Only at Club Mahindra", None),
    price=None, phones={"required": ["18002092345"], "optional": ["56677"]},
    websites={"required": ["www.clubmahindra.com"], "optional": ["http://www.clubmahindra.com/offer_tc"]},
    emails=["Rahul.Malhotra@mahindraholidays.com"], qr_code_present=True)

lab("h07_tvs_416", skip=["advertiser", "product"], language=["English"], brand=ANY("Sundram Fasteners", "TVS"),
    model_variant=None, ad_type="manufacturer", category=ANY("Automobile > Tyres & Parts", "Automobile"),
    headline="When you put quality first, even the world's toughest examiners are impressed.",
    sub_headline=ANY("Sundram Fasteners wins the prestigious Deming Prize for a record 17 plants in one attempt.", None),
    price=None, phones=[], websites=[], emails=[], qr_code_present=False)

lab("h08_Nokia_mobiles_411", skip=["advertiser"], language=["English"], brand="Nokia",
    product=ANY("Nokia 6.1", "New Nokia 6.1"), model_variant=ANY(None, "3GB RAM | 32GB ROM", "4GB RAM | 64GB ROM"),
    ad_type="manufacturer", category=ELEC,
    headline=ANY("A phone you can rely on.", "New Nokia 6.1 A phone you can rely on."),
    sub_headline=ANY(None, "New Nokia 6.1"), price=ANY(16999, 18999), phones=[],
    websites=OPT("Nokia.com/phones", "www.nokia.com/phones", "amazon.in", "www.airtel.in/4gphone", "bit.ly/2pg0R2f",
                 "http://bit.ly/Servify811"),
    emails=[], qr_code_present=False)

lab("h09_renault_412", skip=["phones", "advertiser"], language=["English"], brand="Renault",
    product=ANY("Duster", "Triber", "Kwid", None), model_variant=None, ad_type=ANY("dealer", "manufacturer"),
    category="Automobile > Cars", headline=ANY("Reasons to Smile", "Renault Reasons to Smile"),
    sub_headline=ANY("make your new year bigger than ever", None), price=ANY(957000, 520000, 313000),
    websites={"required": ["renault.co.in"], "optional": ["www.renault.co.in/offers"]}, emails=["corpsales@renault.com"],
    qr_code_present=False)

lab("h10_renault_403", skip=["phones", "advertiser"], language=["English"], brand="Renault",
    product=ANY("Kiger", "Renault Kiger", "New Renault Kiger"), model_variant=ANY(None, "Petrol RXE"),
    ad_type=ANY("dealer", "manufacturer"), category="Automobile > Cars",
    headline=ANY("New Renault Kiger sporty smart stunning", "sporty smart stunning"),
    sub_headline=ANY("sporty smart stunning", "special loyalty benefits", None), price=545000,
    websites={"required": ["renault.co.in"], "optional": ["www.renault.co.in/offers"]}, emails=[], qr_code_present=False)

lab("h11_Sony_television_405", skip=["phones", "websites", "price", "advertiser"], language=["English"], brand="Sony",
    product=ANY("Bravia", "BRAVIA OLED", "Bravia OLED TV", "Bravia range", "Sony Bravia", None), model_variant=None,
    ad_type=ANY("manufacturer", "dealer", "retailer"), category=ELEC, headline="Live the stadium feel",
    sub_headline=ANY("Lifelike picture. Sound directly from the screen", None), emails=[], qr_code_present=False)

lab("h12_hyundai_415", skip=["phones", "websites", "advertiser"], language=["English"], brand="Hyundai",
    product=ANY("Aura", "Hyundai Aura", "Hyundai AURA"),
    model_variant=ANY(None, "1.2 l Eco TORQ", "Diesel", "1.2 l Eco TORQ Diesel", "1.2 l U2"),
    ad_type=ANY("dealer", "manufacturer"), category="Automobile > Cars", headline="Diesel fuels weekends.",
    sub_headline=ANY("Hyundai AURA fuelled by 1.2 l Eco TORQ engine delivers 75ps power and a superb mileage.",
                     "Hyundai presents the widest and the most powerful range of diesel engines.", None),
    price=None, emails=[], qr_code_present=False,
    attrs={"fuel_type": ANY(["Diesel"], None), "ev_type": ANY("Non-EV", None)})

lab("h13_Andhra_Jyothi_150", skip=["product", "advertiser"], language=ANY(["Telugu", "English"], ["Telugu"]),
    brand=ANY("Malabar Gold & Diamonds", "Malabar Gold and Diamonds", "Malabar"), model_variant=None,
    ad_type=ANY("retailer", "manufacturer"), category="Jewellery",
    headline="ఈ ఆషాడంలో, బంగారంలా మెరవండి",
    sub_headline=ANY("Celebrate the Beauty of Life", "Malabar Fair Price Promise", "ఆషాడంలో శుభం లాభం", None),
    price=None, phones=["18005720916", "04068138916"], websites=["malabargoldanddiamonds.com"], emails=[],
    qr_code_present=False)

lab("h14_Maharashtra_Times_692", skip=["advertiser"], language=["English"],
    brand=ANY("Online", "Online The Men's World", "ONLINE"), product=ANY("Wedding collection", None), model_variant=None,
    ad_type="retailer", category="Fashion & Apparel", headline="wedding collection",
    sub_headline=ANY("The Men's World", None), price=None, phones=["02512302408"], websites=[], emails=[],
    qr_code_present=False)

lab("h15_Mid_Day_Paper_624", skip=["phones", "advertiser"], language=["English"], brand="Maruti Suzuki",
    product=ANY("NEXA Black Edition", "Black Edition", None), model_variant=ANY(None, "Black Edition"),
    ad_type=ANY("dealer", "manufacturer"), category="Automobile > Cars", headline="Introducing NEXA Black Edition.",
    sub_headline=ANY("Drive towards exclusive experiences.", "Create. Inspire.", None), price=None,
    websites=["www.nexaexperience.com"], emails=[], qr_code_present=True)

lab("h16_Daily_Thanthi_148", skip=["advertiser"], language=ANY(["Tamil", "English"], ["Tamil"]), brand="Epson",
    product=ANY("EcoTank", "Epson EcoTank", "EcoTank printers", "Epson EcoTank printers", "L3260"), model_variant=ANY(None, "L3260"),
    ad_type="manufacturer", category=ELEC, headline="சூழல் புரட்சியில் ஒன்று சேர்ந்திடுவீர்",
    sub_headline=ANY("உங்கள் வீட்டிலும் வேலை செய்யும் இடத்திலும் சூழலுக்கேற்ற மலிவான பிரிண்டிங்கிற்கு மாறிடுங்கள்",
                     "Powered by Heat-Free Technology", None),
    price=None, phones=["9791652148"], websites=["epson.co.in/EcoTank"], emails=[], qr_code_present=True)

MR = ANY(["Marathi"], ["Marathi", "English"], ["Marathi", "Hindi"], ["Marathi", "Hindi", "English"])
HIEN = ANY(["Hindi", "English"], ["Hindi"])

lab("h17_Loksatta_1310", skip=["phones", "advertiser"], language=MR, brand=ANY("Runwal", "Runwal MyCity"),
    product=ANY("Runwal MyCity", "MyCity"), model_variant=ANY("2 BHK", "2 बीएचके", None),
    ad_type="real_estate_developer", category="Real Estate", headline="उन्हाळ्याचा पुरेपूर आनंद लुटा!!",
    sub_headline=ANY("आजच तुमचे घर बुक करा आणि रुणवाल मायसिटीमध्ये राहण्याचा आनंद स्वत: अनुभवा!", None),
    price=6500000, websites=OPT("https://maharera.mahaonline.gov.in"), emails=[], qr_code_present=False,
    reference_numbers={"required": ["P51700000528"],
                       "optional": ["P51700009168", "P51700008440", "P51700019085", "P51700032157", "P51700027171"]})

lab("h18_Anandabazaar_Patrika_278", language=ANY(["Bengali", "English"], ["Bengali"]), brand=ANY("Raja", "Raja Biscuits"),
    product=ANY("Raja Sugar Free Cream Cracker", "Sugar Free Cream Cracker", "Sugar Free Cream Cracker Biscuits",
                "Raja Sugar Free", "Sugar Free"), model_variant=None,
    advertiser=ANY("Raja Udyog Pvt. Ltd.", "Raja Udyog", None), ad_type="manufacturer", category=FOOD,
    headline="স্বাস্থ্য ও সম্পদের যুগলবন্দী",
    sub_headline=ANY("এসে গেল রাজা সুগার ফ্রী সোনা চাঁদি অফার", "সোনা চাঁদি অফার", None),
    price=None, phones=[], websites=[], emails=[], qr_code_present=False)

lab("h19_Dainik_Bhaskar_608", skip=["advertiser"], language=HIEN, brand="Parle",
    product=ANY("Happy Happy", "Happy Happy Choco-Chip Cookies", "Parle Happy Happy"), model_variant=None,
    ad_type="manufacturer", category=FOOD,
    headline=ANY("Happy Happy", "चोको-चिप कुकीज़", "Happy Happy चोको-चिप कुकीज़"),
    sub_headline=ANY("चोको-चिप कुकीज़", None), price=None, phones=[], websites=[], emails=[], qr_code_present=False)

lab("h20_Navbharat_Times_402", language=ANY(["English", "Hindi"], ["English"]),
    brand=ANY("Navdip Tundiya", "Kala Sangam"), product=ANY("Patola-Bandhani Exhibition", "Kala Sangam", None),
    model_variant=None, advertiser=ANY("Navdip Tundiya", None), ad_type=ANY("retailer", "other"),
    category="Fashion & Apparel",
    headline=ANY("कला Sangam", "Kala Sangam", "Patola-Bandhani Exhibition", "Navdip Tundiya Presents Patola-Bandhani Exhibition"),
    sub_headline=ANY("Patola-Bandhani Exhibition", "Today 1st Day", "Navdip Tundiya Presents", None),
    price=None, phones=[], websites=[], emails=[], qr_code_present=False)

if __name__ == "__main__":
    n = 0
    for stem, d in L.items():
        f = HERE / f"{stem}.json"
        if f.exists() and json.loads(f.read_text(encoding="utf-8")).get("_verified_by"):
            continue
        f.write_text(json.dumps(d, ensure_ascii=False, indent=1), encoding="utf-8")
        n += 1
    print(f"wrote {n} held-out labels")
