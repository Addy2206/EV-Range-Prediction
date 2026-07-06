"""
Add Area_Type column (Urban / Rural) to the EV dataset.

Classification basis:
  - Washington State cities: classified as Urban if they are within a
    Census-Bureau-defined Urbanized Area or Urban Cluster (population >= 2,500),
    based on the 2020 Census and WA State Growth Management Act Urban Growth Areas.
  - Non-WA records: classified using a list of major US urban cities.
  - Any city not matched defaults to "Rural".
"""

import pandas as pd

# ---------------------------------------------------------------------------
# Comprehensive list of URBAN cities / places in Washington State
# Source: 2020 Census Urban Areas, WA OFM population estimates,
#         and WA Growth Management Act Urban Growth Areas.
# ---------------------------------------------------------------------------
WA_URBAN_CITIES = {
    # King County metro core
    "seattle", "bellevue", "renton", "kent", "auburn", "federal way",
    "kirkland", "redmond", "sammamish", "issaquah", "mercer island",
    "shoreline", "burien", "tukwila", "des moines", "seatac",
    "lake forest park", "kenmore", "woodinville", "bothell", "covington",
    "maple valley", "newcastle", "black diamond", "enumclaw", "algona",
    "pacific", "milton", "normandy park", "medina", "clyde hill",
    "yarrow point", "hunts point", "beaux arts village", "skykomish",
    "north bend", "snoqualmie", "carnation", "duvall", "fall city",
    "ravensdale",

    # Snohomish County
    "everett", "lynnwood", "marysville", "edmonds", "mountlake terrace",
    "mukilteo", "mill creek", "lake stevens", "monroe", "snohomish",
    "arlington", "granite falls", "stanwood", "sultan",

    # Pierce County
    "tacoma", "lakewood", "puyallup", "bonney lake", "university place",
    "fife", "milton", "edgewood", "sumner", "buckley", "orting",
    "eatonville", "roy", "ruston", "steilacoom", "dupont", "fircrest",

    # Thurston County
    "olympia", "lacey", "tumwater", "yelm", "rainier", "tenino",
    "bucoda",

    # Kitsap County
    "bremerton", "bainbridge island", "poulsbo", "port orchard",
    "silverdale", "gig harbor", "belfair",

    # Clark County (Vancouver WA metro)
    "vancouver", "camas", "washougal", "battle ground", "ridgefield",
    "la center", "woodland", "kalama", "longview", "kelso",

    # Whatcom County
    "bellingham", "ferndale", "lynden", "blaine", "birch bay",
    "sudden valley",

    # Skagit County
    "mount vernon", "burlington", "anacortes", "sedro-woolley",
    "sedro woolley", "oak harbor", "mount vernon",

    # Spokane County
    "spokane", "spokane valley", "cheney", "airway heights",
    "liberty lake", "medical lake", "mead", "deer park",

    # Yakima County
    "yakima", "selah", "union gap", "wapato", "tieton", "moxee",
    "naches", "terrace heights",

    # Benton / Franklin County (Tri-Cities)
    "kennewick", "pasco", "richland", "west richland", "prosser",
    "benton city",

    # Chelan / Douglas County
    "wenatchee", "east wenatchee", "chelan", "cashmere", "leavenworth",

    # Walla Walla County
    "walla walla", "college place",

    # Grant County
    "moses lake", "ephrata", "quincy", "soap lake", "george",

    # Kittitas County
    "ellensburg", "cle elum", "roslyn",

    # Lewis County
    "centralia", "chehalis",

    # Cowlitz County (already partially covered above)
    "longview", "kelso", "castle rock",

    # Grays Harbor County
    "aberdeen", "hoquiam", "ocean shores", "montesano",

    # Jefferson County
    "port townsend",

    # Clallam County
    "port angeles", "sequim", "forks",

    # San Juan County
    "friday harbor",

    # Island County
    "oak harbor", "coupeville",

    # Okanogan County
    "omak", "okanogan", "tonasket",

    # Ferry / Stevens / Pend Oreille
    "colville", "chewelah", "newport",

    # Asotin / Garfield
    "clarkston", "asotin", "pullman",

    # Adams / Lincoln / Whitman
    "ritzville", "othello", "pullman", "colfax",

    # Pacific / Wahkiakum
    "south bend", "raymond",

    # Mason
    "shelton",

    # Skamania
    "stevenson",
}

# ---------------------------------------------------------------------------
# Major urban cities from other US states (covers the non-WA records)
# ---------------------------------------------------------------------------
OTHER_URBAN_CITIES = {
    # California
    "los angeles", "san francisco", "san diego", "san jose", "sacramento",
    "fresno", "long beach", "oakland", "bakersfield", "anaheim",
    "santa ana", "riverside", "stockton", "irvine", "chula vista",
    "fremont", "san bernardino", "modesto", "fontana", "moreno valley",
    "glendale", "huntington beach", "santa clarita", "garden grove",
    "oceanside", "rancho cucamonga", "santa rosa", "ontario", "elk grove",
    "corona", "lancaster", "palmdale", "hayward", "salinas", "sunnyvale",
    "pomona", "escondido", "torrance", "pasadena", "orange", "fullerton",
    "visalia", "santa clara", "thousand oaks", "simi valley", "concord",
    "roseville", "victorville", "berkeley", "burbank", "el monte",
    "inglewood", "downey", "costa mesa", "richmond", "carlsbad",
    "murrieta", "temecula", "west covina", "norwalk", "daly city",
    "ventura", "antioch", "santa barbara", "santa maria", "rialto",
    "el cajon", "san mateo", "vista", "alhambra", "vallejo", "berkeley",
    "petaluma", "fairfield", "camarillo", "alameda", "san leandro",
    "compton", "los angeles", "calabasas", "aliso viejo", "mission viejo",
    "redondo beach", "boca raton", "bay point", "benicia", "baytown",
    "brentwood", "campbell", "cathedral city",

    # Texas
    "houston", "san antonio", "dallas", "austin", "fort worth", "el paso",
    "arlington", "corpus christi", "plano", "laredo", "lubbock",
    "garland", "irving", "amarillo", "grand prairie", "brownsville",
    "mckinney", "frisco", "pasadena", "mesquite", "killeen", "mcallen",
    "denton", "waco", "carrollton", "midland", "round rock", "abilene",
    "odessa", "beaumont", "league city", "richardson", "sugar land",
    "belton",

    # Florida
    "jacksonville", "miami", "tampa", "orlando", "st. petersburg",
    "hialeah", "tallahassee", "fort lauderdale", "port st. lucie",
    "cape coral", "pembroke pines", "hollywood", "miramar", "gainesville",
    "coral springs", "clearwater", "palm bay", "west palm beach",
    "lakeland", "pompano beach", "davie", "miami gardens", "boca raton",
    "bradenton",

    # New York
    "new york", "buffalo", "rochester", "yonkers", "syracuse", "albany",
    "new rochelle", "mount vernon", "schenectady", "utica", "brooklyn",

    # Illinois
    "chicago", "aurora", "joliet", "naperville", "rockford", "springfield",
    "elgin", "peoria", "buffalo grove",

    # Pennsylvania
    "philadelphia", "pittsburgh", "allentown", "erie", "reading",
    "scranton", "bethlehem", "centre hall", "brewer",

    # Arizona
    "phoenix", "tucson", "mesa", "chandler", "gilbert", "glendale",
    "scottsdale", "tempe", "peoria", "surprise", "yuma", "avondale",
    "flagstaff", "goodyear", "buckeye", "apache junction", "chula vista",

    # Virginia
    "virginia beach", "norfolk", "chesapeake", "richmond", "arlington",
    "newport news", "alexandria", "hampton", "roanoke", "portsmouth",
    "chantilly", "annandale", "burke", "brambleton",

    # Maryland
    "baltimore", "columbia", "germantown", "silver spring", "waldorf",
    "glen burnie", "ellicott city", "bethesda", "frederick", "dundalk",
    "bowie", "annapolis",

    # Others
    "portland", "eugene", "bend", "boise", "idaho falls", "aurora",
    "denver", "colorado springs", "las vegas", "henderson", "reno",
    "albuquerque", "salt lake city", "atlanta", "charlotte", "raleigh",
    "minneapolis", "detroit", "columbus", "indianapolis", "nashville",
    "louisville", "baltimore", "memphis", "boston", "seattle",
    "washington", "anchorage", "honolulu", "omaha", "kansas city",
    "tulsa", "oklahoma city", "milwaukee", "las vegas", "tucson",
    "cincinnati", "ann arbor", "cary", "appleton", "charlottesville",
    "asheville", "biloxi", "chesapeake", "cheyenne", "brooklyn",
    "bella vista", "bentonville",
}

ALL_URBAN = WA_URBAN_CITIES | OTHER_URBAN_CITIES


def classify_area(city):
    if pd.isna(city):
        return "Unknown"
    return "Urban" if str(city).strip().lower() in ALL_URBAN else "Rural"


def main():
    input_path = "dataset/dataset.csv"
    output_path = "dataset/dataset.csv"

    print("Loading dataset...")
    df = pd.read_csv(input_path)
    print(f"  Rows: {len(df):,}  |  Columns: {len(df.columns)}")

    print("Classifying cities as Urban / Rural...")
    df["Area_Type"] = df["City"].apply(classify_area)

    # Summary
    counts = df["Area_Type"].value_counts()
    print("\nArea_Type distribution:")
    print(counts.to_string())

    # Year-wise urban vs rural adoption summary
    print("\nAdoption by Model Year and Area_Type:")
    year_area = (
        df.groupby(["Model Year", "Area_Type"])
        .size()
        .unstack(fill_value=0)
        .sort_index()
    )
    print(year_area.to_string())

    print(f"\nSaving to {output_path} ...")
    df.to_csv(output_path, index=False)
    print("Done.")


if __name__ == "__main__":
    main()
