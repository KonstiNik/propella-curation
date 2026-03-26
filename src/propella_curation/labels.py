"""Propella annotation label definitions.

Contains all 18 propella annotation properties, split into:
- ORDINAL_LABELS: 12 single-value enum properties with a natural ordering (best → worst)
- CATEGORICAL_LABELS: 5 multi-label array properties (no natural ordering)
- FREE_TEXT_LABELS: 1 free-text property

SCORE_MAPS provides monotonic label-to-score mappings for ordinal properties,
following the convention s(ℓ) ∈ {1, …, N} with N = best.
NORM_SCORE_MAPS rescales these to [0, 1].
"""

from __future__ import annotations


# ============================================================
# Ordinal properties — single-value enums with natural ordering
# ============================================================

SCORE_MAPS: dict[str, dict[str, float]] = {
    "content_integrity": {
        "complete": 4.0,
        "mostly_complete": 3.0,
        "fragment": 2.0,
        "severely_degraded": 1.0,
    },
    "content_ratio": {
        "complete_content": 5.0,
        "mostly_content": 4.0,
        "mixed_content": 3.0,
        "mostly_navigation": 2.0,
        "minimal_content": 1.0,
    },
    "content_length": {
        "substantial": 4.0,
        "moderate": 3.0,
        "brief": 2.0,
        "minimal": 1.0,
    },
    "content_quality": {
        "excellent": 5.0,
        "good": 4.0,
        "adequate": 3.0,
        "poor": 2.0,
        "unacceptable": 1.0,
    },
    "information_density": {
        "dense": 5.0,
        "adequate": 4.0,
        "moderate": 3.0,
        "thin": 2.0,
        "empty": 1.0,
    },
    "educational_value": {
        "high": 5.0,
        "moderate": 4.0,
        "basic": 3.0,
        "minimal": 2.0,
        "none": 1.0,
    },
    "reasoning_indicators": {
        "analytical": 5.0,
        "explanatory": 4.0,
        "basic_reasoning": 3.0,
        "minimal": 2.0,
        "none": 1.0,
    },
    "audience_level": {
        "expert": 6.0,
        "advanced": 5.0,
        "general": 4.0,
        "beginner": 3.0,
        "youth": 2.0,
        "children": 1.0,
    },
    "commercial_bias": {
        "none": 5.0,
        "minimal": 4.0,
        "moderate": 3.0,
        "heavy": 2.0,
        "pure_marketing": 1.0,
    },
    "time_sensitivity": {
        "evergreen": 4.0,
        "slowly_changing": 3.0,
        "regularly_updating": 2.0,
        "time_sensitive": 1.0,
    },
    "content_safety": {
        "safe": 5.0,
        "mild_concerns": 4.0,
        "nsfw": 3.0,
        "harmful": 2.0,
        "illegal": 1.0,
    },
    "pii_presence": {
        "no_pii": 2.0,
        "contains_pii": 1.0,
    },
}

ORDINAL_LABELS: dict[str, list[str]] = {
    col: list(mapping.keys()) for col, mapping in SCORE_MAPS.items()
}


# ============================================================
# Categorical properties — multi-label arrays, no ordering
# ============================================================

CATEGORICAL_LABELS: dict[str, list[str]] = {
    "content_type": [
        "analytical",
        "instructional",
        "reference",
        "procedural",
        "qa_structured",
        "conversational",
        "creative",
        "transactional",
        "boilerplate",
        "news_report",
        "opinion_editorial",
        "review_critique",
        "technical_documentation",
        "specification_standard",
        "legal_document",
        "press_release",
        "structured_data",
        "source_code",
    ],
    "business_sector": [
        "academic_research",
        "education_sector",
        "technology_software",
        "hardware_electronics",
        "healthcare_medical",
        "pharmaceutical_biotech",
        "financial_services",
        "legal_services",
        "government_public",
        "manufacturing_industrial",
        "mining_resources",
        "chemicals_materials",
        "energy_utilities",
        "retail_commerce",
        "wholesale_distribution",
        "real_estate_construction",
        "transportation_logistics",
        "automotive_industry",
        "telecommunications",
        "media_entertainment",
        "advertising_marketing",
        "hospitality_tourism",
        "agriculture_food",
        "environmental_services",
        "aerospace_defense",
        "insurance_industry",
        "nonprofit_ngo",
        "consulting_professional",
        "human_resources",
        "security_cyber",
        "gaming_industry",
        "gambling_betting",
        "travel_aviation",
        "food_beverage_hospitality",
        "consumer_goods",
        "general_interest",
        "other",
    ],
    "technical_content": [
        "code_heavy",
        "math_heavy",
        "scientific",
        "data_heavy",
        "engineering",
        "basic_technical",
        "non_technical",
    ],
    "regional_relevance": [
        "european",
        "north_american",
        "east_asian",
        "south_asian",
        "southeast_asian",
        "middle_eastern",
        "sub_saharan_african",
        "latin_american",
        "oceanian",
        "central_asian",
        "russian_sphere",
        "global",
        "culturally_neutral",
        "indeterminate",
    ],
    "country_relevance": [
        "afghanistan", "albania", "algeria", "andorra", "angola",
        "antigua_and_barbuda", "argentina", "armenia", "australia", "austria",
        "azerbaijan", "bahamas", "bahrain", "bangladesh", "barbados",
        "belarus", "belgium", "belize", "benin", "bhutan",
        "bolivia", "bosnia_and_herzegovina", "botswana", "brazil", "brunei",
        "bulgaria", "burkina_faso", "burundi", "cabo_verde", "cambodia",
        "cameroon", "canada", "central_african_republic", "chad", "chile",
        "china", "colombia", "comoros", "congo", "congo_democratic_republic",
        "cook_islands", "costa_rica", "croatia", "cuba", "cyprus",
        "czech_republic", "denmark", "djibouti", "dominica", "dominican_republic",
        "ecuador", "egypt", "el_salvador", "equatorial_guinea", "eritrea",
        "estonia", "eswatini", "ethiopia", "fiji", "finland",
        "france", "gabon", "gambia", "georgia", "germany",
        "ghana", "greece", "grenada", "guatemala", "guinea",
        "guinea_bissau", "guyana", "haiti", "honduras", "hungary",
        "iceland", "india", "indonesia", "iran", "iraq",
        "ireland", "israel", "italy", "ivory_coast", "jamaica",
        "japan", "jordan", "kazakhstan", "kenya", "kiribati",
        "north_korea", "south_korea", "kosovo", "kuwait", "kyrgyzstan",
        "laos", "latvia", "lebanon", "lesotho", "liberia",
        "libya", "liechtenstein", "lithuania", "luxembourg", "madagascar",
        "malawi", "malaysia", "maldives", "mali", "malta",
        "marshall_islands", "mauritania", "mauritius", "mexico", "micronesia",
        "moldova", "monaco", "mongolia", "montenegro", "morocco",
        "mozambique", "myanmar", "namibia", "nauru", "nepal",
        "netherlands", "new_zealand", "nicaragua", "niger", "nigeria",
        "niue", "north_macedonia", "norway", "oman", "pakistan",
        "palau", "palestine", "panama", "papua_new_guinea", "paraguay",
        "peru", "philippines", "poland", "portugal", "qatar",
        "romania", "russia", "rwanda", "saint_kitts_and_nevis", "saint_lucia",
        "saint_vincent_and_the_grenadines", "samoa", "san_marino",
        "sao_tome_and_principe", "saudi_arabia", "senegal", "serbia",
        "seychelles", "sierra_leone", "singapore", "slovakia", "slovenia",
        "solomon_islands", "somalia", "south_africa", "south_sudan", "spain",
        "sri_lanka", "sudan", "suriname", "sweden", "switzerland",
        "syria", "tajikistan", "tanzania", "thailand", "timor_leste",
        "togo", "tonga", "trinidad_and_tobago", "tunisia", "turkey",
        "turkmenistan", "tuvalu", "uganda", "ukraine", "united_arab_emirates",
        "united_kingdom", "united_states", "uruguay", "uzbekistan", "vanuatu",
        "vatican_city", "venezuela", "vietnam", "yemen", "zambia", "zimbabwe",
        # Dependent territories and special areas
        "aland_islands", "american_samoa", "anguilla", "antarctica", "aruba",
        "ascension_island", "bermuda", "british_virgin_islands", "cayman_islands",
        "christmas_island", "cocos_islands", "curacao", "falkland_islands",
        "faroe_islands", "french_guiana", "french_polynesia", "gibraltar",
        "greenland", "guadeloupe", "guam", "guernsey", "hong_kong",
        "isle_of_man", "jersey", "macau", "martinique", "mayotte",
        "montserrat", "new_caledonia", "norfolk_island", "northern_mariana_islands",
        "pitcairn_islands", "puerto_rico", "reunion", "saint_barthelemy",
        "saint_helena", "saint_martin", "saint_pierre_and_miquelon",
        "sint_maarten", "svalbard_and_jan_mayen", "taiwan", "tokelau",
        "tristan_da_cunha", "turks_and_caicos_islands", "us_virgin_islands",
        "wallis_and_futuna", "western_sahara",
        # Special values
        "supranational", "none",
    ],
}


# ============================================================
# Free-text property
# ============================================================

FREE_TEXT_LABELS: list[str] = ["one_sentence_description"]


# ============================================================
# Derived maps
# ============================================================

ALL_PROPERTIES: list[str] = (
    list(ORDINAL_LABELS.keys())
    + list(CATEGORICAL_LABELS.keys())
    + FREE_TEXT_LABELS
)


def _normalize(mapping: dict[str, float]) -> dict[str, float]:
    """Linearly rescale values to [0, 1] where 0 = worst and 1 = best."""
    lo, hi = min(mapping.values()), max(mapping.values())
    if lo == hi:
        return {k: 1.0 for k in mapping}
    return {k: (v - lo) / (hi - lo) for k, v in mapping.items()}


NORM_SCORE_MAPS: dict[str, dict[str, float]] = {
    col: _normalize(m) for col, m in SCORE_MAPS.items()
}
