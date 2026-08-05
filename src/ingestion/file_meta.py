"""
Per-file metadata: the SINGLE SOURCE OF TRUTH for a source PDF's
`code_name`, primary `domain`, and optional `domain_secondary`.

Replaces the old scheme where domain was derived from the folder
(pdfs/Finance & Banking vs pdfs/General legal). With 17 fine-grained domains
spread across three folders, domain must be assigned per-file explicitly.

Used by:
- ingestion (`pipeline.ingest_file`) to tag each chunk's domain + secondary,
- `scripts/migrate_domains.py` to re-tag the existing 15,116 chunks in place,
- `code_names.CODE_NAME_OVERRIDES` (derived from this) for citation code_names.

`law_reference` is intentionally NOT stored here — it is detected at ingestion
time from the text (`pipeline._detect_law_reference`) and, for already-indexed
files, is already present in the chunk metadata (migration does not touch it).

The Moroccan "Code des juridictions financières" is deliberately absent
(not Tunisian law — excluded from the corpus).
"""
from __future__ import annotations

from src.routing.domains import DOMAIN_SET

# filename -> {"code_name": str, "domain": <primary>, "domain_secondary": <code|"">}
FILE_META: dict[str, dict[str, str]] = {
    # ===================== EXISTING 45 (Finance & Banking folder) =====================
    "08-Circulaire-aux-banques-et-aux-etablissements-financiers-n°2022-08-du-20-Octobre-2022.pdf": {
        "code_name": "Circulaire BCT n° 2022-08 du 20 octobre 2022 (traitement des réclamations de la clientèle)",
        "domain": "droit_bancaire_financier", "domain_secondary": ""},
    "11.pdf": {
        "code_name": "Décret-loi n° 2011-117 du 5 novembre 2011 portant organisation des institutions de microfinance",
        "domain": "droit_bancaire_financier", "domain_secondary": ""},
    "code-de-la-tva-2024-30.pdf": {
        "code_name": "Code de la Taxe sur la Valeur Ajoutée (TVA), version 2024",
        "domain": "droit_fiscal_douanier", "domain_secondary": ""},
    "Code-de-l’impôt-sur-le-Revenu-des-personnes-physiques-et-de-limpot-sur-les-sociétés-2020.pdf": {
        "code_name": "Code de l'Impôt sur le Revenu (IRPP et IS), version 2020",
        "domain": "droit_fiscal_douanier", "domain_secondary": ""},
    "Code-des-Droits-dEnregistrement-et-de-Timbre-2025.pdf": {
        "code_name": "Code des Droits d'Enregistrement et de Timbre, version 2025",
        "domain": "droit_fiscal_douanier", "domain_secondary": ""},
    "code_opc_fr_v2020.pdf": {
        "code_name": "Code des Organismes de Placement Collectif (OPC), version 2020",
        "domain": "droit_bancaire_financier", "domain_secondary": ""},
    "code_procedures_fiscaux.pdf": {
        "code_name": "Code des Droits et Procédures Fiscales (Loi n° 2000-82)",
        "domain": "droit_fiscal_douanier", "domain_secondary": ""},
    "Décret2006_1881.pdf": {
        "code_name": "Décret n° 2006-1881 du 10 juillet 2006 (services bancaires de base)",
        "domain": "droit_bancaire_financier", "domain_secondary": ""},
    "Jo06094.pdf": {
        "code_name": "Journal Officiel de la République Tunisienne — lois n° 94-87 et 94-88 de 1994",
        "domain": "droit_bancaire_financier", "domain_secondary": ""},
    "Loi-de-Finances-2022.pdf": {
        "code_name": "Loi de Finances 2022 (Décret-loi n° 2021-21)",
        "domain": "droit_fiscal_douanier", "domain_secondary": ""},
    "Loi-lutte-contre-le-blanchiment-des-capitaux-et-financement-du-Terrorisme.pdf": {
        "code_name": "Loi n° 2016-33 du 31 octobre 2016 relative à la lutte contre le blanchiment des capitaux et le financement du terrorisme",
        "domain": "droit_bancaire_financier", "domain_secondary": "droit_securite_conformite"},
    "Loi2009-64.pdf": {
        "code_name": "Loi n° 2009-64 du 12 août 2009 portant promulgation du code de prestation des services financiers aux non-résidents",
        "domain": "droit_bancaire_financier", "domain_secondary": ""},
    "LOI71-2016_fr.pdf": {
        "code_name": "Loi n° 2016-71 du 30 septembre 2016 portant loi de l'investissement",
        "domain": "droit_investissement", "domain_secondary": "droit_fiscal_douanier"},
    "loi7618.pdf": {
        "code_name": "Loi n° 76-18 du 21 janvier 1976 portant codification de la législation des changes et du commerce extérieur",
        "domain": "droit_bancaire_financier", "domain_secondary": "droit_fiscal_douanier"},
    "loi_2016-35_250416_fr.pdf": {
        "code_name": "Loi n° 2016-35 du 25 avril 2016 portant statut de la Banque Centrale de Tunisie",
        "domain": "droit_bancaire_financier", "domain_secondary": ""},
    "Loi_2016_48_fr.pdf": {
        "code_name": "Loi n° 2016-48 du 10 novembre 2016 relative aux banques et aux établissements financiers",
        "domain": "droit_bancaire_financier", "domain_secondary": ""},
    "loi_94117_141194_fr.pdf": {
        "code_name": "Loi n° 94-117 du 14 novembre 1994 portant réorganisation du marché financier",
        "domain": "droit_bancaire_financier", "domain_secondary": ""},
    "loi_crowfunding_37_06082020_fr.pdf": {
        "code_name": "Loi n° 2020-37 du 6 août 2020 relative au crowdfunding (financement participatif)",
        "domain": "droit_bancaire_financier", "domain_secondary": ""},
    "reg_bancaire.pdf": {
        "code_name": "Réglementation Bancaire — Recueil de textes (Banque Centrale de Tunisie, 2017)",
        "domain": "droit_bancaire_financier", "domain_secondary": ""},
    "reg_blanchiment_fr.pdf": {
        "code_name": "Réglementation relative à la lutte contre le blanchiment de capitaux (recueil)",
        "domain": "droit_bancaire_financier", "domain_secondary": "droit_securite_conformite"},
    "statut_ib_fr.pdf": {
        "code_name": "Décret n° 99-2478 du 1er novembre 1999 portant statut des intermédiaires en bourse",
        "domain": "droit_bancaire_financier", "domain_secondary": ""},

    # ===================== EXISTING 45 (General legal folder) =====================
    "18.pdf": {
        "code_name": "Code de l'Aménagement du Territoire et de l'Urbanisme (2011)",
        "domain": "droit_administratif_collectivites", "domain_secondary": ""},
    "3bafee2a-7964-4756-bcd4-7eb52ce0b0fe.pdf": {
        "code_name": "Code de Droit International Privé (2024)",
        "domain": "procedure_civile_dip", "domain_secondary": ""},
    "57598e7c-dd33-40f3-883b-8c2b747fc6a8.pdf": {
        "code_name": "Code de la Protection de l'Enfant (2024)",
        "domain": "droit_famille_statut_personnel", "domain_secondary": "droit_penal"},
    "7edf1fbb-d498-438e-83e6-7ffb146a3fb3.pdf": {
        "code_name": "Code de la Nationalité Tunisienne (2024)",
        "domain": "droit_famille_statut_personnel", "domain_secondary": ""},
    "8968c19c-9ef2-4929-aa8c-5689f93198cc.pdf": {
        "code_name": "Code des Droits Réels (2024)",
        "domain": "droit_civil_obligations", "domain_secondary": ""},
    "c5c21867-2bd7-4932-8a86-347d9162edb1.pdf": {
        "code_name": "Code de l'Arbitrage (2024)",
        "domain": "droit_commercial_societes", "domain_secondary": "procedure_civile_dip"},
    "CODE COC.pdf": {
        "code_name": "Code des Obligations et des Contrats (COC, 2015)",
        "domain": "droit_civil_obligations", "domain_secondary": ""},
    "CODE DE COMMERCE.pdf": {
        "code_name": "Code de Commerce (loi n° 2010-39, révision 2014)",
        "domain": "droit_commercial_societes", "domain_secondary": ""},
    "CODE DES PROCEDURES PENALES.pdf": {
        "code_name": "Code de Procédure Pénale (2013)",
        "domain": "droit_penal", "domain_secondary": ""},
    "code du statut personnel_fr.pdf": {
        "code_name": "Code du Statut Personnel (Décret du 13 août 1956)",
        "domain": "droit_famille_statut_personnel", "domain_secondary": ""},
    "code-de-travail-2016-6.pdf": {
        "code_name": "Code du Travail (2016)",
        "domain": "droit_travail_fonction_publique", "domain_secondary": ""},
    "Code_Assurance_Version_FR.pdf": {
        "code_name": "Code des Assurances (2020)",
        "domain": "droit_bancaire_financier", "domain_secondary": ""},
    "code_societes_fr.pdf": {
        "code_name": "Code des Sociétés Commerciales (2022)",
        "domain": "droit_commercial_societes", "domain_secondary": ""},
    "Decloi54.pdf": {
        "code_name": "Décret-loi n° 2022-54 du 13 septembre 2022 (lutte contre les infractions aux systèmes d'information — cybercriminalité)",
        "domain": "droit_penal", "domain_secondary": "droit_numerique_donnees"},
    "F777739995_TUN-65196.pdf": {
        "code_name": "Code de Procédure Civile et Commerciale (2010)",
        "domain": "procedure_civile_dip", "domain_secondary": "droit_commercial_societes"},
    "Loi 63-2004 Fr.pdf": {
        "code_name": "Loi organique n° 2004-63 du 27 juillet 2004 relative à la protection des données à caractère personnel",
        "domain": "droit_numerique_donnees", "domain_secondary": ""},
    "Loi-organique-n°-2013-13-du-2-Mai-2013-Fr.pdf": {
        "code_name": "Loi organique n° 2013-13 du 2 mai 2013 (instance de supervision de la justice judiciaire)",
        "domain": "droit_constitutionnel_institutionnel", "domain_secondary": ""},
    "Loi-organique-n°-2017-58-du-11-août-2017.pdf": {
        "code_name": "Loi organique n° 2017-58 du 11 août 2017 relative à l'élimination de la violence à l'égard des femmes",
        "domain": "droit_penal", "domain_secondary": "droit_famille_statut_personnel"},
    "Recueil-Textes-Bailleur-Locataire.pdf": {
        "code_name": "Recueil des textes régissant les rapports entre bailleurs et locataires (2013)",
        "domain": "droit_civil_obligations", "domain_secondary": ""},
    "Route.pdf": {
        "code_name": "Code de la Route (2012)",
        "domain": "droit_route_transport", "domain_secondary": ""},
    "tun128749.pdf": {
        "code_name": "Recueil — La législation du secteur de la sécurité en Tunisie (incl. Constitution 2022)",
        "domain": "droit_securite_conformite", "domain_secondary": "droit_constitutionnel_institutionnel"},
    "Tun190410.pdf": {
        "code_name": "Loi n° 92-117 du 7 décembre 1992 relative à la protection du consommateur",
        "domain": "droit_consommation_concurrence", "domain_secondary": ""},
    "tun202128.pdf": {
        "code_name": "Loi organique n° 2018-29 du 9 mai 2018 relative au code des collectivités locales",
        "domain": "droit_administratif_collectivites", "domain_secondary": ""},
    "Tunisia-Penal-Code-2012.pdf": {
        "code_name": "Code Pénal (2012)",
        "domain": "droit_penal", "domain_secondary": ""},

    # ===================== NEW (pdfs/new) — 14 files (Moroccan doc excluded) =====================
    "023.pdf": {
        "code_name": "Loi n° 94-36 du 24 février 1994 relative à la propriété littéraire et artistique (droit d'auteur)",
        "domain": "droit_propriete_intellectuelle", "domain_secondary": ""},
    "1410083987.pdf": {
        "code_name": "Décret-loi n° 2011-88 du 24 septembre 2011 portant organisation des associations",
        "domain": "droit_constitutionnel_institutionnel", "domain_secondary": ""},
    "Arrete2020_3040.pdf": {
        "code_name": "Arrêté n° 2020-3040 (statut général des personnels de l'État — Loi n° 83-112)",
        "domain": "droit_travail_fonction_publique", "domain_secondary": ""},
    "CODE DES DOUANES.pdf": {
        "code_name": "Code des Douanes (2016)",
        "domain": "droit_fiscal_douanier", "domain_secondary": ""},
    "codedeont.pdf": {
        "code_name": "Code de Déontologie Médicale (édition février 2021)",
        "domain": "droit_sante", "domain_secondary": ""},
    "dec2019_54_130622_fr.pdf": {
        "code_name": "Décret n° 2019-54 (relatif au registre national des entreprises — Loi n° 2018-52)",
        "domain": "droit_commercial_societes", "domain_secondary": "droit_administratif_collectivites"},
    "Loi n° 2001-21 du 6 février 2001, relative à la protection des dessins et modèles industriels.pdf": {
        "code_name": "Loi n° 2001-21 du 6 février 2001 relative à la protection des dessins et modèles industriels",
        "domain": "droit_propriete_intellectuelle", "domain_secondary": ""},
    "loi_193.pdf": {
        "code_name": "Loi n° 91-64 du 29 juillet 1991 relative à la concurrence et aux prix",
        "domain": "droit_consommation_concurrence", "domain_secondary": ""},
    "loi2001_1.pdf": {
        "code_name": "Loi n° 2001-1 du 15 janvier 2001 portant promulgation du Code des Télécommunications",
        "domain": "droit_numerique_donnees", "domain_secondary": ""},
    "loi2001-36fr.pdf": {
        "code_name": "Loi n° 2001-36 relative aux marques de fabrique et de commerce",
        "domain": "droit_propriete_intellectuelle", "domain_secondary": ""},
    "telechargement26 (3).pdf": {
        "code_name": "Loi n° 2000-84 du 24 août 2000 relative aux brevets d'invention",
        "domain": "droit_propriete_intellectuelle", "domain_secondary": ""},
    "tun_RecAcFreeLi.pdf": {
        "code_name": "Constitution de la République Tunisienne du 25 juillet 2022",
        "domain": "droit_constitutionnel_institutionnel", "domain_secondary": ""},
    "Tunisia-Loi-72-40-FRA.pdf": {
        "code_name": "Loi n° 72-40 du 1er juin 1972 relative au Tribunal Administratif",
        "domain": "droit_administratif_collectivites", "domain_secondary": ""},
    "tunisie-loi-organique-nb0-2014-16-du-26-mai-2014.pdf": {
        "code_name": "Loi organique n° 2014-16 du 26 mai 2014 relative aux élections et aux référendums",
        "domain": "droit_constitutionnel_institutionnel", "domain_secondary": ""},
}


def get_file_meta(filename: str) -> dict[str, str]:
    """Return {code_name, domain, domain_secondary} for a source file.

    Falls back to a neutral default (domain="", code_name="") if the file is not
    in FILE_META — callers (ingestion) should then derive code_name from the
    first-page heuristic and refuse to ingest without a known domain.
    """
    m = FILE_META.get(filename)
    if m:
        return {
            "code_name": m["code_name"],
            "domain": m["domain"],
            "domain_secondary": m.get("domain_secondary", ""),
        }
    return {"code_name": "", "domain": "", "domain_secondary": ""}


def _validate() -> None:
    bad = []
    for fn, m in FILE_META.items():
        if m["domain"] not in DOMAIN_SET:
            bad.append((fn, m["domain"], "primary"))
        sec = m.get("domain_secondary", "")
        if sec and sec not in DOMAIN_SET:
            bad.append((fn, sec, "secondary"))
    if bad:
        raise ValueError(f"invalid domain codes in FILE_META: {bad}")


_validate()
