"""
Query expansion: map colloquial scenario phrasings to formal legal terminology
BEFORE the keyword half of hybrid re-rank.

Users describe situations in plain language ("on m'a volé ma voiture") rather
than with legal terms ("vol"). The embedding half of retrieval already handles
paraphrase, but the keyword half is keyed to formal vocabulary — this expansion
injects the formal terms so the right articles win the keyword-overlap signal.

Applied to the KEYWORD term set only; the query embedding is left untouched.
"""
from __future__ import annotations

import re
import unicodedata

_TOKEN_RE = re.compile(r"[a-zà-ÿ]{4,}")


def _deaccent(s: str) -> str:
    return "".join(
        c for c in unicodedata.normalize("NFKD", s) if not unicodedata.combining(c)
    )


# colloquial phrase (deaccented, lowercased substring) -> formal legal terms to add.
# Triggers are matched as substrings against the deaccented query, so "volé",
# "voler", "cambriolé" etc. all fire the theft expansion.
_EXPANSIONS: list[tuple[str, list[str]]] = [
    # Theft / burglary ------------------------------------------------------
    ("vole", ["vol", "cambriolage", "soustraction", "détournement", "recel"]),
    ("cambriol", ["vol", "cambriolage", "soustraction", "effraction"]),
    ("derob", ["vol", "soustraction", "détournement"]),
    ("derobe", ["vol", "soustraction", "détournement"]),
    # Aggravated theft — the Code Pénal aggravated-penalty articles (260-262)
    # say "le vol commis avec la réunion des circonstances…" and the circumstance
    # terms (escalade/effraction/armes) are DEFINED in arts 270-272. Inject those
    # distinctive terms + "circonstances" so the penalty + definition articles win
    # the keyword half of re-rank, not just art. 264 ("autres vols").
    ("escalade", ["effraction", "escalade", "circonstances", "enclos"]),
    ("effraction", ["escalade", "effraction", "circonstances"]),
    ("cloture", ["escalade", "enclos", "effraction"]),
    ("clôture", ["escalade", "enclos", "effraction"]),
    ("couteau", ["armes", "arme", "circonstances"]),
    ("arme", ["armes", "arme", "circonstances"]),
    ("a main armee", ["armes", "arme", "circonstances"]),
    ("poignard", ["armes", "arme", "circonstances"]),
    ("nuit", ["circonstances", "effraction", "escalade"]),
    # Embezzlement / misappropriation --------------------------------------
    ("detourn", ["abus de confiance", "détournement", "escroquerie", "abus", "confiance"]),
    ("argent de la caisse", ["abus de confiance", "détournement", "escroquerie"]),
    # Fraud / scams ---------------------------------------------------------
    ("arnaque", ["escroquerie", "escroquer", "fraude", "tromperie", "faux"]),
    ("escroqu", ["escroquerie", "escroquer", "fraude", "tromperie"]),
    ("fraud", ["fraude", "escroquerie", "faux", "faux en écriture"]),
    # Assault / battery -----------------------------------------------------
    ("agress", ["agression", "coups et blessures", "violences", "voies de fait"]),
    ("tabass", ["coups et blessures", "violences", "agression"]),
    ("frappe", ["coups et blessures", "violences", "agression"]),
    # Employment dismissal --------------------------------------------------
    ("vire", ["licenciement", "congédiement", "préavis", "rupture", "travail"]),
    ("renvoy", ["licenciement", "congédiement", "préavis", "rupture"]),
    ("licenci", ["licenciement", "congédiement", "préavis", "rupture"]),
    ("a la porte", ["licenciement", "congédiement", "rupture", "préavis"]),
    # Tenancy / eviction ----------------------------------------------------
    ("expuls", ["expulsion", "résiliation", "bail", "locataire", "loyer"]),
    ("loyer", ["bail", "locataire", "bailleur", "loyer", "expulsion"]),
    ("proprietaire", ["bailleur", "bail", "locataire", "loyer"]),
    # Traffic / accidents ---------------------------------------------------
    ("accident de voiture", ["accident", "route", "circulation", "responsabilité", "assurance"]),
    ("accident de la route", ["accident", "route", "circulation", "responsabilité", "assurance"]),
    ("accident", ["accident", "route", "circulation", "responsabilité", "assurance"]),
    ("ivre", ["ivresse", "alcool", "conduite", "route"]),
    # Drunk driving — the sanction article (art. 87) lists "conduite sous l'empire
    # d'un état alcoolique" as délit n°1, but shares "alcoolique" with the
    # procedural articles (dépistage art. 3, immobilisation art. 105) and with the
    # involuntary-homicide articles (89/90, where alcohol is only an aggravator),
    # so it loses the keyword half to them. "délit"/"obtempérer"/"quiconque" come
    # from 87's penalty list and separate it from that cluster.
    ("ivresse", ["alcoolique", "délit", "obtempérer", "quiconque"]),
    ("alcool", ["ivresse", "alcoolémie", "conduite", "route"]),
    # Succession ------------------------------------------------------------
    ("herite", ["succession", "héritage", "héritier", "partage", "réserve"]),
    ("heritage", ["succession", "héritage", "héritier", "partage"]),
    # Unpaid / debt ---------------------------------------------------------
    ("pas paye", ["impayé", "recouvrement", "inexécution", "créance", "obligation"]),
    ("impaye", ["impayé", "recouvrement", "inexécution", "créance"]),
    ("non paye", ["impayé", "recouvrement", "inexécution", "créance"]),
    # Contract breach -------------------------------------------------------
    ("rompu le contrat", ["résiliation", "inexécution", "manquement", "contrat"]),
    ("pas respecte le contrat", ["inexécution", "manquement", "résiliation", "contrat"]),
]


def expand_terms(query: str) -> list[str]:
    """Return deaccented-lowercased query tokens PLUS expansion legal terms.

    Expansion terms are added when their colloquial trigger appears in the query.
    Duplicates are removed; order is preserved (base tokens first).
    """
    base = _TOKEN_RE.findall(_deaccent(query).lower())
    out: list[str] = list(base)
    seen = set(_deaccent(t).lower() for t in base)
    dq = _deaccent(query).lower()
    for trigger, terms in _EXPANSIONS:
        if trigger in dq:
            for t in terms:
                dt = _deaccent(t).lower()
                if dt not in seen:
                    seen.add(dt)
                    out.append(dt)
    return out
