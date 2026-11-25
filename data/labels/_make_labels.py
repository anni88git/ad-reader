"""Writes Claude-drafted ground-truth labels (independent read of each ad image, NOT the system output).
Each label records its provenance; a human audit sets _audited_by / _audit_changes.
Run: python data/labels/_make_labels.py
"""
import json
from pathlib import Path

HERE = Path(__file__).parent
PROV = {"_label_source": "claude_independent_read",
        "_note": "Drafted by Claude from the ad image only (not from any system output). Core fields only. "
                 "Human audit pending unless _audited_by is set. Format: data/labels/FORMAT.md",
        "_verified_by": None}
CORE_SKIP = ["description", "body_copy", "ad_message", "features", "offers", "warranty", "disclaimer_present",
             "dealers", "addresses", "social_handles", "prices", "call_to_action"]
L = {}


def lab(stem, skip=(), **fields):
    attrs = fields.pop("attrs", None)
    d = {**PROV, "_skip": CORE_SKIP + list(skip), **fields}
    if attrs:
        d["category_attributes"] = attrs
    L[stem] = d


ANY = lambda *a: {"any_of": list(a)}  # noqa: E731

# ------------------------------------------------------------------ batch 1
lab("002_002_horlicks_hi", language=["Hindi", "English"], brand="Horlicks",
    product=ANY("Horlicks Chocolate Delight", "Chocolate Delight", "Horlicks", None), model_variant=ANY("500 g", "500g", None),
    advertiser=ANY("Horlicks", "Hindustan Unilever", None), ad_type="manufacturer", category=ANY("FMCG > Food & Beverages", "FMCG"),
    headline="हॉर्लिक्स दूध की शक्ति बढ़ाए", sub_headline=ANY(None, "हॉर्लिक्स में हैं ग्रोथ न्यूट्रीएंट्स जो बनाए बच्चों को टॉलर, स्ट्रॉन्गर और शार्पर"),
    price=None, phones=[], websites=[], emails=[], qr_code_present=False)

lab("003_003_namaste_butter_hi", language=["Hindi", "English"], brand="Namaste India",
    product=ANY("Butter", "Namaste India Butter"), model_variant=ANY("100 g", "100 gm", "100 ग्रा.", None),
    advertiser=ANY("Namaste India Foods", "Namaste India", None), ad_type="manufacturer",
    category=ANY("FMCG > Food & Beverages", "FMCG"),
    headline=ANY("नमक नहीं !", "नमक नहीं ! लो मक्खन का असली स्वाद"), sub_headline=ANY("लो मक्खन का असली स्वाद", None),
    price=44, phones=["7988859406", "18001230800", "05122240251"], websites=["www.namasteindiafoods.com"],
    emails=["crm@nirpl.com"], qr_code_present=False)

lab("004_car_brands_403_Volkswagen", skip=["phones", "advertiser"], language=["English"], brand="Volkswagen",
    product="Polo", model_variant=ANY("GT TSI", "Polo GT TSI", None), ad_type=ANY("dealer", "manufacturer"),
    category="Automobile > Cars",
    headline=ANY("Upgrade to feature packed Polo GT TSI at unbelievable prices of 2020",
                 "Upgrade to feature packed Polo GT TSI"),
    sub_headline=ANY("Offer valid for deliveries till 26th Jan", "at unbelievable prices of 2020", None),
    price=None, websites=["volkswagen.co.in"], emails=[], qr_code_present=False)

lab("005_hero_motocorp_429", skip=["phones", "advertiser"], language=["English"], brand="Hero MotoCorp",
    product=ANY("Destini 125", "Destini"), model_variant=ANY(None, "125"), ad_type=ANY("dealer", "manufacturer"),
    category="Automobile > Two Wheelers", headline="Socho Do Kadam Aagey",
    sub_headline=ANY("India's First Family Scooter with i3S Technology", None),
    price=55080, websites=ANY(["heromotocorp.com"], ["www.heromotocorp.com"]), emails=[], qr_code_present=False,
    attrs={"fuel_type": ANY(["Petrol"], None), "ev_type": ANY("Non-EV", None)})

lab("006_maruti_suzuki_410", skip=["phones", "advertiser"], language=["English"], brand="Maruti Suzuki",
    product="Baleno", model_variant=ANY(None, "Smart Hybrid", "1.2L VVT Petrol"), ad_type=ANY("dealer", "manufacturer"),
    category="Automobile > Cars", headline="Inspiring Design. Incredible Bold Appeal.",
    sub_headline=ANY("Make Bolder Moves.", "Baleno Make Bolder Moves.", None),
    price=None, websites=["www.nexaexperience.com"], emails=[], qr_code_present=True,
    attrs={"fuel_type": ANY(["Petrol"], ["Hybrid", "Petrol"], None), "ev_type": ANY("Non-EV", None)})

# ------------------------------------------------------------------ batch 2
OPT = lambda *a: {"required": [], "optional": list(a)}  # noqa: E731

lab("007_toyota_417", language=["English"], brand="Toyota", product=ANY("Yaris", None), model_variant=None,
    advertiser=ANY("Toyota Kirloskar Motor", "Toyota", None), ad_type="manufacturer", category="Automobile > Cars",
    headline=ANY("Arre Waah That's A Toyota!", "Arre Waah"),
    sub_headline=ANY("Service Cost at ₹3,500 a Year*", "A car with a yearly service cost lower than a full tank of petrol.",
                     "That's A Toyota!", None),
    price=ANY(None, 3500), phones={"required": ["18004250001"], "optional": ["04066293001"]},
    websites={"required": ["www.toyotabharat.com"], "optional": ["besafetyleaders.in"]}, emails=[],
    qr_code_present=False)

lab("008_tata_402", skip=["phones", "advertiser", "language"], brand="Tata Motors", product="Safari",
    model_variant=None, ad_type=ANY("dealer", "manufacturer"), category="Automobile > Cars",
    headline="Rule The Roads. Reclaim Your Life.", sub_headline=None, price=1499000,
    websites=OPT("cars.tatamotors.com/suv/safari", "cars.tatamotors.com"), emails=[], qr_code_present=True,
    attrs={"fuel_type": ANY(["Diesel"], None), "ev_type": ANY("Non-EV", None)})

lab("009_LG_television_409", skip=["phones", "price", "advertiser"], language=["English"],
    brand=ANY("LG Electronics", "LG"), product=ANY("OLED TV", "LG OLED TV", "LG OLED TV AI ThinQ", None),
    model_variant=ANY(None, "E8/C8/B8"),  # corrected during evaluation: printed model series
    ad_type=ANY("manufacturer", "retailer", "dealer"),
    category=ANY("Retail & Consumer Electronics", "Retail & Consumer Electronics > Home Appliances"),
    headline=ANY("LG OLED TV AI ThinQ", "This festive season, bring home the best of TV technology",
                 "World's Best Selling OLED Brand"),
    sub_headline=ANY("Life's Good", "This festive season, bring home the best of TV technology",
                     "World's Best Selling OLED Brand", None),
    websites=OPT("www.lg.com/in/support", "www.lg.com/in", "www.facebook.com/lgindiapage", "www.twitter.com/LGIndia"),
    emails=OPT("corp.enquiries@lge.com", "avhad.sukhdeo@lge.com"), qr_code_present=False)

# ------------------------------------------------------------------ batch 3
ELEC = ANY("Retail & Consumer Electronics", "Retail & Consumer Electronics > Mobiles & Computers",
           "Retail & Consumer Electronics > Home Appliances")

lab("010_Lenovo_laptops_and_computers_403", skip=["advertiser"], language=["English"], brand="Lenovo",
    product=ANY("K8 Series", "Lenovo K8 Series", "K8", "Lenovo K8", "K8 Note", "Lenovo K8 Note"), model_variant=ANY(None, "K8 Note", "K8"),
    ad_type="manufacturer", category=ELEC,
    headline=ANY("Killer Style Killer Performance", "Lenovo K8 Series Killer Style Killer Performance"),
    sub_headline=ANY(None, "Now Available in Stores", "Killer Performance"), price=ANY(10999, 11999),
    phones=[], websites=[], emails=[], qr_code_present=True)

lab("011_Xiaomi_mobiles_400", skip=["advertiser"], language=["English"], brand=ANY("Xiaomi", "Mi"),
    product=ANY("Mi 10T 5G", "Mi 10T"), model_variant=ANY(None, "5G"), ad_type="manufacturer", category=ELEC,
    headline="Built for Battle Royale", sub_headline=ANY(None, "Mi 10T 5G", "Additional No cost EMI offers available"),
    price=32999, phones=[], websites={"required": ["mi.com"], "optional": ["amazon.in", "flipkart.com"]},
    emails=[], qr_code_present=False)

lab("012_Hitachi_air_conditioners_404", skip=["phones"], language=["English"], brand="Hitachi",
    product=ANY("5-Star Expandable Inverter AC", "Expandable Inverter AC", "Hitachi Expandable Inverter AC", "Inverter AC", None),
    model_variant=ANY(None, "5-Star"),
    advertiser=ANY("Johnson Controls-Hitachi Air Conditioning India Limited", "Hitachi", None),
    ad_type=ANY("manufacturer", "dealer"), category=ELEC,
    headline="Get 5-Star offers with 5-Star Expandable Inverter AC.",
    sub_headline=ANY("Hitachi air Feel it to believe it", "Feel it to believe it", "Bring home Hitachi air at happy prices", None),
    price=None,
    websites={"required": ["www.hitachiaircon.in"],
              "optional": ["www.hitachiaircon.in/dealer-locator", "https://www.facebook.com/HitachiHomes/",
                           "https://twitter.com/HitachiHomes", "https://www.youtube.com/user/hitachihome",
                           "https://www.linkedin.com/company/johnson-controls-hitachi-air-conditioning"]},
    emails={"required": ["customercare@jci-hitachi.com"], "optional": ["sales@jci-hitachi.com"]},
    qr_code_present=False)

# ------------------------------------------------------------------ batch 4
SERVICE = ANY("service_provider", "manufacturer")      # bank / insurer / app brand advertising itself

lab("013_ICICI_Bank_402", skip=["advertiser"], language=ANY(["English"], ["English", "Hindi"]), brand="ICICI Bank",
    product=ANY("iMobile Pay", "ICICI Bank iMobile Pay"), model_variant=None, ad_type=SERVICE, category="Banking & Finance",
    headline="Poore India Ka Payment App",
    sub_headline=ANY("Link Savings Account of Any Bank", "Introducing iMobile Pay", "Pay Anyone on Any Payment App", None),
    price=None, phones=[], websites=["www.icicibank.com"], emails=[], qr_code_present=True)

lab("014_LIC_400", language=ANY(["English"], ["English", "Hindi"]),  # corrected: logo has Hindi
    brand="LIC", product=ANY("Dhan Vriddhi", "LIC's Dhan Vriddhi"), model_variant=None,
    advertiser=ANY("Life Insurance Corporation of India", "LIC"), ad_type=SERVICE, category=ANY("Insurance", "Banking & Finance"),
    headline="Grow Your Funds While Being Insured",
    sub_headline=ANY("A Single Premium Plan for steady growth with guaranteed additions", "A Single Premium Plan",
                     "Har Pal Aapke Saath", "Introducing LIC's Dhan Vriddhi", None),
    price=None, phones={"required": ["8976862090", "02268276827"], "optional": ["56767474"]},
    websites=["licindia.in"], emails=[], qr_code_present=False,
    reference_numbers={"required": ["UIN: 512N362V01", "Plan No.: 869"], "optional": ["IRDAI Regn No.: 512"]})

lab("015_State_Bank_Of_India_403", language=["English"], brand=ANY("State Bank of India", "SBI"), product=None,
    model_variant=None, advertiser=ANY("State Bank of India", "SBI", None), ad_type=SERVICE, category="Banking & Finance",
    headline=ANY("Celebrating 66th Bank Day",
                 'For us, every customer is important and each one of them is the "I" in SBI'),
    sub_headline=ANY("The banker to every Indian", "1st July, 2021",
                     'For us, every customer is important and each one of them is the "I" in SBI', None),
    price=None, phones=[], websites=[], emails=[], qr_code_present=False)

# ------------------------------------------------------------------ batch 5
BRANDSTORE = ANY("manufacturer", "retailer")

lab("016_Biba_411", skip=["phones", "advertiser"], language=["English"], brand="Biba", product=ANY("Anarkalis", None),
    model_variant=None, ad_type=BRANDSTORE, category="Fashion & Apparel",
    headline=ANY("Diwali Showstoppers",
                 "Karva Chauth Special - Shop at Biba this weekend and avail free mehandi at select stores"),
    sub_headline=ANY("Karva Chauth Special - Shop at Biba this weekend and avail free mehandi at select stores",
                     "Diwali Showstoppers", "Anarkalis", None),
    price=None, websites={"required": ["www.biba.in"], "optional": ["facebook.com/BibaIndia"]}, emails=[],
    qr_code_present=False)

lab("017_Malabar_gold_and_diamond_424", language=["English"], brand=ANY("Malabar Gold & Diamonds", "Forevermark"),
    product=ANY("Forevermark", "Forevermark diamonds", "Solitaire Diamond Show", "Forevermark diamond solitaires", None),
    model_variant=None, advertiser=ANY("Malabar Gold & Diamonds", "Malabar Gold and Diamonds", None),
    ad_type=ANY("retailer", "dealer", "manufacturer"), category="Jewellery", headline="Solitaire Diamond Show",
    sub_headline=ANY("A Diamond Is Forever", None), price=None,
    phones={"required": ["04428140910"], "optional": ["18002102121"]}, websites=["forevermark.com"], emails=[],
    qr_code_present=False)

lab("018_Fabindia_424", skip=["advertiser"], language=["English"], brand="Fabindia",
    product=ANY("Rajwada", "Rajwada by Fabindia"), model_variant=None, ad_type=BRANDSTORE, category="Fashion & Apparel",
    headline=ANY("Rajwada by fabindia", "#YoungRoyals", "Rajwada"),
    sub_headline=ANY("Celebrate India", "Now in Stores", "#YoungRoyals", None),
    price=None, phones=[], websites=["fabindia.com"], emails=[], qr_code_present=False)

# ------------------------------------------------------------------ batch 6
FOOD = ANY("FMCG > Food & Beverages", "FMCG", "Food Delivery & Restaurants")

lab("019_Haldiram_410", skip=["advertiser"], language=["English"], brand=ANY("Haldiram's", "Haldirams", "Haldiram"),
    product=ANY("Navratra Thali", "Navratra Feast", None), model_variant=None, ad_type=BRANDSTORE, category=FOOD,
    headline="Navratra Feast",
    sub_headline=ANY("Divine delicacies filled with wholesome flavors to lighten up blissful days of Navratra",
                     "6th - 12th April", None),
    price=None, phones=[], websites=["www.haldiramsonline.com"], emails=[], qr_code_present=False)

lab("020_Patanjali_402", skip=["advertiser"], language=ANY(["English"], ["English", "Hindi"]), brand="Patanjali",
    product=ANY("Special Chyawanprash", "Patanjali Special Chyawanprash", "Chyawanprash",
                "Special Chyawanprash with Saffron", "Chyawanprash with Saffron"),
    model_variant=ANY(None, "with Saffron", "Kesar"), ad_type="manufacturer",
    category=ANY("FMCG", "FMCG > Food & Beverages", "FMCG > Personal & Home Care", "Healthcare"),
    headline="Increase your immunity in this changing weather, acquire full strength to fight diseases.",
    sub_headline=ANY("Eat Patanjali Special Chyawanprash with the goodness of Kashmiri Kesar", "Prakriti ka Aashirwad", None),
    price=None, phones=["18001804108"], websites=["www.patanjaliayurved.net", "www.patanjaliayurved.org"],
    emails=["feedback@patanjaliayurved.org"], qr_code_present=False)

lab("021_Indigo_407", skip=["advertiser", "product"], language=["English"], brand="IndiGo", model_variant=None,
    ad_type=SERVICE, category="Travel & Hospitality", headline="IndiGo to Prayagraj",
    sub_headline=ANY("Now flying daily non-stop from Delhi to Prayagraj, starting at ₹2499.", None),
    price=2499, phones=[], websites=["goIndiGo.in"], emails=[], qr_code_present=False)

# ------------------------------------------------------------------ batch 7
lab("022_Make_MyTrip_405", skip=["advertiser", "product"], language=["English"], brand="MakeMyTrip", model_variant=None,
    ad_type=SERVICE, category="Travel & Hospitality", headline="Singapore is always happening!",
    sub_headline=ANY("Three reasons to plan your trip with MakeMyTrip.", None),
    price=None, phones=["18001028900", "18001033050", "01149052664", "01247177417"], websites=[], emails=[],
    qr_code_present=False)

lab("023_Pantene_shampoo_402", skip=["advertiser"], language=["English"], brand="Pantene",
    product=ANY("Pantene Shampoo", "Shampoo", "Pantene Pro-V Shampoo", "Pantene Pro-V Hair Fall Control Shampoo",
                "Hair Fall Control Shampoo", "New Pantene"),
    model_variant=ANY(None, "Hair Fall Control", "340ml", "180 ml"), ad_type="manufacturer",
    category=ANY("FMCG > Personal & Home Care", "FMCG"),
    headline="Thicker, Stronger Hair in 14 Days*",
    sub_headline=ANY("Strong is Beautiful", "New Pantene Strong is Beautiful", None),
    price=None, phones=[], websites=[], emails=[], qr_code_present=False)

lab("024_Dabur_Red_toothpaste_435", skip=["advertiser"], language=["Hindi", "English"], brand="Dabur",
    product=ANY("Red Paste", "Dabur Red Paste", "Red Paste Dental Brigade"), model_variant=None,
    ad_type="manufacturer", category=ANY("FMCG > Personal & Home Care", "FMCG"),
    headline="आयुर्वेद की शक्ति, साइंस का प्रूफ़.",
    sub_headline=ANY("करें डेंटल प्रॉब्लम्स को दूर, बनें रेड ब्रिगेड का हिस्सा!", "प्रूफ़ है",
                     "7 डेन्टल प्रॉब्लम्स पर काम करे 2 हफ़्तों में असर दिखाए", None),
    price=None, phones=["18001031644"], websites=["www.daburdentalcare.com"], emails=["dentalbrigade@nda.jagran.com"],
    qr_code_present=False)

# ------------------------------------------------------------------ batch 8
lab("025_Horlicks_nutrition_drink_411", skip=["phones"], language=["English"], brand="Horlicks",
    product=ANY("Horlicks Protein+", "Protein+", "Horlicks Protein Plus", "Protein Plus"), model_variant=ANY(None, "Chocolate", "Vanilla & Chocolate"),
    advertiser=ANY("GSK", "GlaxoSmithKline", "Horlicks", None), ad_type="manufacturer", category=FOOD,
    headline="Have you tried the better protein?", sub_headline=ANY(None, "Horlicks Protein+"),
    price=None, websites=ANY(["https://proteinplus.horlicks.in/"], ["proteinplus.horlicks.in"]), emails=[], qr_code_present=False)

lab("026_Jansatta_61", skip=["product", "advertiser"], language=ANY(["Hindi"], ["Hindi", "English"]), brand="Patanjali",
    model_variant=None, ad_type="manufacturer", category=ANY("FMCG", "FMCG > Food & Beverages", "FMCG > Personal & Home Care"),
    headline="स्वदेशी संकल्प एक मात्र विकल्प",
    sub_headline=ANY("पतंजलि के लिए देश एक परिवार है जबकि विदेशी कम्पनियों के लिए देश एक बाज़ार है",
                     "पतंजलि 100% प्योर प्रोडक्ट्स, 100% प्रॉस्पेरिटी फॉर चैरिटी", None),
    price=None, phones=[], websites=[], emails=[], qr_code_present=False)

lab("027_Navbharat_Times_878", skip=["advertiser"], language=["English"],
    brand=ANY("Rupa", "Rupa Frontline", "Colors", "Colors by Rupa Frontline", "Rupa & Co"),
    product=ANY("Colors", "Colors by Rupa Frontline", "Frontline", "Colors innerwear", None), model_variant=None,
    ad_type="manufacturer", category="Fashion & Apparel",
    headline=ANY("One India One Brand One Legacy", "Colors by Rupa Frontline", "Product Range", "One India Brand Legacy"),
    sub_headline=ANY(None, "Product Range", "One India One Brand One Legacy"),
    price=None, phones=["9810157101", "18001235001"],
    websites={"required": ["rupaonlinestore.com"], "optional": ["amazon.in"]}, emails=["sanjaymurarka@rupa.co.in"],
    qr_code_present=False)

# ------------------------------------------------------------------ batch 9
HI = ANY(["Hindi"], ["Hindi", "English"])

lab("028_Dainik_Jagran__5", skip=["advertiser"], language=["English"], brand="Nissan", product="Magnite",
    model_variant=ANY("Geza Special Edition", "Geza Edition", "Geza", None), ad_type=ANY("dealer", "manufacturer"),
    category="Automobile > Cars", headline="Magnite is Music",
    sub_headline=ANY("Introducing Nissan Magnite Geza Special Edition", "#BigBoldBeautiful", None),
    price=ANY(739000, 599900),
    phones={"required": ["8879509258"],
            "optional": ["9731169831", "9731170778", "9731170445", "9731169529", "9731169773"]},
    websites=["www.nissan.in"], emails=["dealer.management@email.nissan.in"], qr_code_present=True)

lab("029_Amar_Ujala__366", language=HI,
    brand=ANY("Delhi State Legal Services Authority", "Delhi Traffic Police", "Delhi Police", None),
    product=ANY("National Lok Adalat", None), model_variant=None,
    advertiser=ANY("Delhi State Legal Services Authority", "दिल्ली राज्य विधिक सेवाएं प्राधिकरण", "Delhi Traffic Police",
                   "Delhi State Legal Services Authority / Delhi Traffic Police", None),
    ad_type="public_notice", category="Government & Public Notices",
    headline=ANY("नेशनल लोक अदालत", "नेशनल लोक अदालत का आयोजन"),
    sub_headline=ANY("13 मई, 2023", "दिल्ली राज्य विधिक सेवाएं प्राधिकरण द्वारा दिल्ली यातायात पुलिस के सहयोग से",
                     "का आयोजन", None),
    price=None, phones={"required": ["01125844444"], "optional": []},
    websites={"required": ["https://traffic.delhipolice.gov.in/notice/lokadalat"],
              "optional": ["https://www.facebook.com/dtptraffic", "https://twitter.com/dtptraffic"]},
    emails=[], qr_code_present=True)

lab("030_Hindustan_29", language=HI, brand=ANY("Delhi Development Authority", "DDA", "दिल्ली विकास प्राधिकरण"),
    product=ANY("FCFS Scheme Phase 4", "एफसीएफएस योजना चरण 4", "DDA Housing Scheme", "FCFS Housing Scheme", None),
    model_variant=None, advertiser=ANY("Delhi Development Authority", "DDA", "दिल्ली विकास प्राधिकरण", None),
    ad_type=ANY("public_notice", "real_estate_developer"), category=ANY("Government & Public Notices", "Real Estate"),
    headline=ANY("घर खरीदने वाले हो जाएं तैयार !!!", "डीडीए की आवासीय योजना के लिए नरेला में मेगा कैंप का आयोजन"),
    sub_headline=ANY("डीडीए की आवासीय योजना के लिए नरेला में मेगा कैंप का आयोजन", "सार्वजनिक सूचना",
                     "घर खरीदने वाले हो जाएं तैयार !!!", None),
    price=None, phones=["1800110332"],
    websites={"required": ["www.dda.gov.in"], "optional": ["http://eservices.dda.org.in/public/uploads/Contact_Us_List.pdf"]},
    emails=[], qr_code_present=True)

# ------------------------------------------------------------------ batch 10
HIEN = ANY(["Hindi", "English"], ["Hindi"])

lab("031_Navbharat_Times_5", skip=["advertiser"], language=HI, brand="Kissan",
    product=ANY("Fresh Tomato Ketchup", "Kissan Fresh Tomato Ketchup", "Tomato Ketchup"),
    model_variant=ANY("850g", "850 g", None), ad_type="manufacturer", category=FOOD,
    headline=ANY("बड़ा पैक ₹100/- में", "बड़ा पैक"), sub_headline=ANY(None, "₹100/- में"),
    price=100, phones=[], websites=[], emails=[], qr_code_present=False)

lab("032_Hari_Bhoomi__130", language=HIEN, brand=ANY("Eye Mantra", "Divisa Herbal Care", "Dr. Juneja's"),
    product=ANY("Eye Mantra", "Eye Mantra Ayurvedic Eye Relief Drops", "आई मंत्रा", "Ayurvedic Eye Relief Drops",
                "Eye Mantra Eye Drops"),
    model_variant=ANY(None, "10ml", "10 ml"), advertiser=ANY("Divisa Herbal Care", "Eye Mantra", None),
    ad_type="manufacturer", category=ANY("Healthcare", "FMCG > Personal & Home Care", "FMCG"),
    headline=ANY("आए दूर तक नज़र!", "क्या आप सही 'आई' ड्रॉप की तलाश में है?"),
    sub_headline=ANY("क्या आप सही 'आई' ड्रॉप की तलाश में है?", "आँख की 5 समस्याओं से राहत!", None),
    price=None, phones=["8196822222"], websites=["www.eyemantra.com"], emails=[], qr_code_present=False)

lab("033_Dainik_Bhaskar_225", skip=["advertiser"], language=["Hindi"], brand="Dainik Bhaskar",
    product=ANY("Bhaskar Quiz League", "भास्कर क्विज लीग"), model_variant=None,
    ad_type=ANY("manufacturer", "service_provider", "other"), category="Media & Entertainment",
    headline=ANY("यदि हर रोज़ आप देख रहे हैं क्रिकेट, तो हर रोज़ आप जीत सकते हैं ढेरों इनाम",
                 "यदि हर रोज़ आप देख रहे हैं क्रिकेट, तो हर रोज़ आप जीत सकते हैं ढेरों", "भास्कर क्विज लीग"),
    sub_headline=ANY("तो अभी खोलिये स्पोर्ट्स पेज, और खेलिए भास्कर क्विज लीग", "पढ़ो और जीतो", "सच्ची बात, बेधड़क",
                     "भास्कर क्विज लीग पढ़ो और जीतो", None),
    price=None, phones=[], websites=[], emails=[], qr_code_present=False)

# ------------------------------------------------------------------ batch 11
MR = ANY(["Marathi"], ["Marathi", "English"], ["Marathi", "Hindi"], ["Marathi", "Hindi", "English"])

lab("034_Dainik_Bhaskar_625", skip=["advertiser", "phones", "websites", "emails"], language=["English"],
    brand=ANY("Rupa", "Rupa Frontline", "Colors", "Colors by Rupa Frontline", "Rupa & Co"),
    product=ANY("Colors", "Colors by Rupa Frontline", "Frontline", "Colors innerwear", None), model_variant=None,
    ad_type="manufacturer", category="Fashion & Apparel",
    headline=ANY("One India One Brand One Legacy", "Colors by Rupa Frontline", "Product Range", "One India Brand Legacy"),
    sub_headline=ANY(None, "Product Range", "One India One Brand One Legacy"), price=None, qr_code_present=False)

lab("035_Rashtriya_Sahara_128", language=HIEN, brand="LIC",
    product=ANY("Jeevan Akshay VII", "LIC's Jeevan Akshay VII", "Jeevan Akshay"), model_variant=ANY(None, "VII"),
    advertiser=ANY("Life Insurance Corporation of India", "LIC"), ad_type=SERVICE, category=ANY("Insurance", "Banking & Finance"),
    headline="हँसता खेलता रिटायरमेंट", sub_headline=ANY("हर पल आपके साथ", "हमारे बढ़े हुए वार्षिकी दरों के साथ", None),
    price=None, phones={"required": ["02268276827", "8976862090"], "optional": ["56767474"]},
    websites=["licindia.in"], emails=[], qr_code_present=False,
    reference_numbers={"required": ["UIN-512N337V04"],
                       "optional": ["प्लान नं. 857", "Plan No. 857", "IRDAI Regn No.: 512"]})

lab("036_Pudhari_180", skip=["phones", "websites", "advertiser"], language=MR,
    brand=ANY("Sachi Saheli", "सच्ची सहेली"),
    product=ANY("Sachi Saheli", "Sachi Saheli Ayurvedic Tonic", "सच्ची सहेली", "Ayurvedic Tonic"), model_variant=None,
    ad_type="manufacturer", category=ANY("Healthcare", "FMCG", "FMCG > Personal & Home Care"),
    headline="कठीण समस्या आता राहणार नाहीत",
    sub_headline=ANY("सच्ची सहेली आहे उत्तम आरोग्याकरिता सर्वात विश्वसनीय औषधी आणि टॉनिक", "पूर्ण स्वदेशी, पूर्ण आयुर्वेदिक", None),
    price=None, emails=[], qr_code_present=False)

lab("037_Loksatta_1673", skip=["product"], language=ANY(["Marathi", "English"], ["Marathi"]),
    brand=ANY("Fivebric", "Fivebric Realty", "Fivebric Realty Pvt Ltd"), model_variant=None,
    advertiser=ANY("Fivebric", "Fivebric Realty", "Fivebric Realty Pvt Ltd", None),
    ad_type=ANY("real_estate_developer", "dealer", "other"), category="Real Estate",
    headline=ANY("30 लाखात फ्लॅट घेण्यापेक्षा.. ₹3 लाखात प्लॉट घ्या..", "₹3 लाखात प्लॉट घ्या..", "3 लाखात प्लॉट घ्या.."),
    sub_headline=ANY("नवी मुंबई इंटर नॅशनल एअरपोर्ट परिसरात", "30 लाखात फ्लॅट घेण्यापेक्षा..", None),
    price=300000, phones=["8081300300"], websites=["www.fivebricrealty.in"], emails=[], qr_code_present=False)

# ------------------------------------------------------------------ batch 12
lab("038_Loksatta_1681", skip=["advertiser"], language=ANY(["English"], ["English", "Hindi"]), brand="Amul",
    product=ANY("Amul Kool", "Kool", None), model_variant=None, ad_type="manufacturer", category=FOOD,
    headline="Yeh Toh Maska Shuruat Hai",
    sub_headline=ANY("Amul For Every Cricket-Her!", "For Every Cricket-Her!",
                     "Amul Kool is the Official Beverage Partner - Women's Premier League", None),
    price=None, phones=[], websites=[], emails=[], qr_code_present=False)

lab("039_Daily_Thanthi_190", skip=["advertiser", "category", "product"],
    language=ANY(["Tamil", "English"], ["Tamil", "English", "Hindi"], ["Tamil"]),
    brand=ANY("IndianOil", "Indian Oil", "Indian Oil Corporation"), model_variant=None,
    ad_type=ANY("manufacturer", "service_provider"),
    headline=ANY("On Duty, Always!", "மத்தவங்களுக்கு இது வெறும் பிஸினஸ் எங்களுக்கு இது டூட்டி!"),
    sub_headline=ANY("IndianOil எப்போதும் ஓப்பன்!", "எப்போதும் ஓப்பன்!",
                     "மத்தவங்களுக்கு இது வெறும் பிஸினஸ் எங்களுக்கு இது டூட்டி!", "On Duty, Always!", None),
    price=None, phones=[], websites=[], emails=[], qr_code_present=False)

lab("040_Andhra_Jyothi_80", skip=["product", "phones"], language=ANY(["Telugu", "English"], ["Telugu"]),
    brand=ANY("The Chennai Shopping Mall", "Chennai Shopping Mall"), model_variant=None,
    advertiser=ANY("The Chennai Shopping Mall", "Chennai Shopping Mall", None), ad_type="retailer",
    category="Fashion & Apparel",
    headline=ANY("వెడ్డింగ్ కలెక్షన్", "వధూవరులకు అత్యద్భుతమైన శ్రేణిలో విలక్షణమైన వెడ్డింగ్ కలెక్షన్"),
    sub_headline=ANY(None, "పెట్టుబడి చీరలపై ప్రత్యేక కాంబో ఆఫర్లు",
                     "వధూవరులకు అత్యద్భుతమైన శ్రేణిలో విలక్షణమైన వెడ్డింగ్ కలెక్షన్", "Bridal Gowns & Lehengas"),
    price=None, websites=[], emails=[], qr_code_present=False)

# ------------------------------------------------------------------ batch 13
lab("041_V6_Velugu_30", skip=["product", "advertiser", "price", "sub_headline"],
    language=ANY(["Telugu"], ["Telugu", "English"]),
    brand=ANY("Kalyana Lakshmi Shopping Mall", "Kalyanalakshmi", "Kalyana Lakshmi", "Kalyanalakshmi Shopping Mall"),
    model_variant=None, ad_type="retailer", category="Fashion & Apparel",
    headline=ANY("గొప్ప పునః ప్రారంభం !", "ఈరోజు గొప్ప పునః ప్రారంభం !", "గొప్ప పునః ప్రారంభం"),
    phones=[], websites=[], emails=[], qr_code_present=False)

lab("042_Prajavani_61", skip=["advertiser"], language=ANY(["Kannada", "English"], ["Kannada"]),
    brand=ANY("Vibez Estates", "Vibez"), product=ANY("Vibez farm plots", "Farm plots", "Farm plot", None),
    model_variant=ANY(None, "6500 sqft"), ad_type=ANY("real_estate_developer", "other"), category="Real Estate",
    headline=ANY("ಒಡೆತನ ನಿಮ್ಮದು ದುಡಿಮೆ ನಮ್ಮದು", "ಸಕಲೇಶಪುರದಲ್ಲಿ ಏಕೆ ಹೋಟೆಲ್ ಬುಕ್ ಮಾಡುತ್ತೀರಾ...?"),
    sub_headline=ANY(None, "ಸಕಲೇಶಪುರದಲ್ಲಿ ಏಕೆ ಹೋಟೆಲ್ ಬುಕ್ ಮಾಡುತ್ತೀರಾ...?", "Vibez farm plots"),
    price=2150000, phones=["8152883388"], websites=["www.vibezestates.com"], emails=[], qr_code_present=False)

lab("043_Vijaya_Karnataka_350", skip=["product", "advertiser"], language=["English"],
    brand=ANY("Fashion Factory", "Reliance Fashion Factory", "Reliance"), model_variant=None, ad_type="retailer",
    category=ANY("Fashion & Apparel", "Retail & Consumer Electronics"),
    headline=ANY("Buy 2 Get 3 Free on almost everything", "Buy 2 Get 3 Free", "Fashion Factory Brands for Less"),
    sub_headline=ANY("Brands for Less", "Till 21st May", "On almost everything", "Fashion Factory Brands for Less", None),
    price=None, phones=["6360605247", "8104917753"], websites=[], emails=[], qr_code_present=True)

# ------------------------------------------------------------------ batch 14
lab("044_Mid_Day_Paper_405", skip=["advertiser"], language=["English"],
    brand=ANY("Shraddha", "Shraddha Prime Projects", "Shraddha Landmark"),
    product=ANY("Gold Crest", "Gold Crest by Shraddha"), model_variant=ANY("1 BHK | 2 BHK", "1 BHK / 2 BHK", "1 & 2 BHK", "1 BHK, 2 BHK", None),
    ad_type="real_estate_developer", category="Real Estate",
    headline=ANY("Gold Crest", "Gold Crest by Shraddha", "1 BHK | 2 BHK"),
    sub_headline=ANY("Kandivali West Near Station", "₹ 97 Lacs Onwards", "by Shraddha", "1 BHK | 2 BHK", None),
    price=9700000, phones=["8750039171", "8750042620"], websites=[], emails=[], qr_code_present=False,
    reference_numbers=["P51800050191"])

lab("045_Anandabazaar_Patrika_820", language=["English"],
    brand=ANY("United Colors of Benetton", "Benetton", "Lawrence & Mayo"),
    product=ANY("Benetton eyewear", "United Colors of Benetton eyewear", "Eyewear", None), model_variant=None,
    advertiser=ANY("Lawrence & Mayo", "L&M", None), ad_type=ANY("retailer", "dealer"),
    category=ANY("Fashion & Apparel", "Fashion & Apparel > Watches & Accessories", "Healthcare"),
    headline=ANY("Be Unique Vibrant Bold You", "BE Unique Vibrant Bold You", "Launching exclusively at L&M stores"),
    sub_headline=ANY("Launching exclusively at L&M stores", "United Colors of Benetton.", None),
    price=None, phones=["9154071877"], websites=[], emails=[], qr_code_present=False)

# ------------------------------------------------------------------ batch 15 (top-up regional ads)
lab("046_Namasthe_Telangana_8", skip=["phones", "product", "advertiser"], language=["English"], brand="Maruti Suzuki",
    model_variant=None, ad_type=ANY("dealer", "manufacturer"), category="Automobile > Cars",
    headline="Beat the Price Hike.", sub_headline=ANY("Buy a Maruti Suzuki Arena car now.", None),
    price=None, websites=["www.marutisuzuki.com"], emails=["nishant.vijayvergia@maruti.co.in"], qr_code_present=True)

lab("047_Anandabazaar_Patrika_929", skip=["advertiser"], language=ANY(["Bengali", "English"], ["Bengali"]),
    brand=ANY("Roop Mantra", "Dr. Juneja's", "রূপ মন্ত্রা"),
    product=ANY("Roop Mantra Ayurvedic Face Cream & Face Washes", "Ayurvedic Face Cream & Face Washes",
                "Roop Mantra Face Wash", "Ayurvedic Medicinal Facewash", "Roop Mantra Cucumber Face Wash", None),
    model_variant=None, ad_type="manufacturer", category=ANY("FMCG > Personal & Home Care", "FMCG", "Healthcare"),
    headline=ANY("স্বাস্থ্যোজ্জ্বল ও ঝলমলে ত্বকের জন্য", "রূপ মন্ত্রা"),
    sub_headline=ANY(None, "Ayurvedic Face Cream & Face Washes", "স্বাস্থ্যোজ্জ্বল ও ঝলমলে ত্বকের জন্য"),
    price=None, phones=["8725966666"], websites=["www.roopmantra.com"], emails=[], qr_code_present=False)

if __name__ == "__main__":
    written, kept = 0, []
    for stem, d in L.items():
        f = HERE / f"{stem}.json"
        if f.exists() and json.loads(f.read_text(encoding="utf-8")).get("_verified_by"):
            kept.append(stem)          # human-verified label: never overwritten by a draft
            continue
        f.write_text(json.dumps(d, ensure_ascii=False, indent=1), encoding="utf-8")
        written += 1
    print(f"wrote {written} draft labels; kept {len(kept)} human-verified: {kept}")
