"""
Multi-domain taxonomy for the Tunisian legal RAG (replaces the old binary
finance_banking / general_legal split).

This module is the SINGLE SOURCE OF TRUTH for the domain codes. It is imported
by the router (keyword + embedding scoring), ingestion/migration (per-file domain
tagging), config (validation), and eval (expected_domain values).

Each domain has:
- a human French `LABEL` (for display / API responses),
- `KEYWORDS`: distinctive French legal (and some colloquial) phrases — drives the
  keyword half of routing + retrieval re-rank,
- `EXEMPLARS`: 3-5 representative questions — the router embeds these and compares
  a query to each domain's centroid (best-matching exemplar) for the embedding
  half of routing.

The set is deliberately a subdivision of the previous two buckets; several files
legitimately span two domains and carry a `domain_secondary` tag (see
src/ingestion/file_meta.py).
"""
from __future__ import annotations

# 17 legal sub-domain codes. `droit_sante` covers the Code de Déontologie Médicale.
DOMAINS: tuple[str, ...] = (
    "droit_penal",
    "droit_civil_obligations",
    "droit_commercial_societes",
    "droit_travail_fonction_publique",
    "droit_fiscal_douanier",
    "droit_bancaire_financier",
    "droit_famille_statut_personnel",
    "droit_administratif_collectivites",
    "droit_propriete_intellectuelle",
    "droit_consommation_concurrence",
    "procedure_civile_dip",
    "droit_constitutionnel_institutionnel",
    "droit_route_transport",
    "droit_numerique_donnees",
    "droit_investissement",
    "droit_securite_conformite",
    "droit_sante",
)

DOMAIN_SET: frozenset[str] = frozenset(DOMAINS)

LABELS: dict[str, str] = {
    "droit_penal": "Droit pénal",
    "droit_civil_obligations": "Droit civil / obligations",
    "droit_commercial_societes": "Droit commercial / sociétés",
    "droit_travail_fonction_publique": "Droit du travail / fonction publique",
    "droit_fiscal_douanier": "Droit fiscal / douanier",
    "droit_bancaire_financier": "Droit bancaire / financier",
    "droit_famille_statut_personnel": "Droit de la famille / statut personnel",
    "droit_administratif_collectivites": "Droit administratif / collectivités",
    "droit_propriete_intellectuelle": "Propriété intellectuelle",
    "droit_consommation_concurrence": "Consommation / concurrence",
    "procedure_civile_dip": "Procédure civile / droit international privé",
    "droit_constitutionnel_institutionnel": "Droit constitutionnel / institutionnel",
    "droit_route_transport": "Code de la route / transport",
    "droit_numerique_donnees": "Numérique / données personnelles",
    "droit_investissement": "Investissement",
    "droit_securite_conformite": "Sécurité / conformité (LBC-FT)",
    "droit_sante": "Santé / déontologie médicale",
}

KEYWORDS: dict[str, list[str]] = {
    "droit_penal": [
        "pénal", "peine", "peines", "infraction", "délit", "crime", "contravention",
        "prison", "emprisonnement", "amende", "incarcération", "réclusion", "vol",
        "volé", "cambriolage", "cambriolé", "escroquerie", "escroquer", "abus de confiance",
        "détournement", "détourné", "détourner", "soustraction", "soustraire", "recel",
        "meurtre", "homicide", "assassinat", "viol", "agression", "coups et blessures",
        "vol à main armée", "complicité", "tentative", "faux", "usage de faux", "fraude",
        "garde à vue", "détention préventive", "instruction", "mandat d'arrêt",
        "arrestation", "poursuite", "poursuivi", "tribunal correctionnel", "code pénal",
        "procédure pénale", "cybercriminalité", "responsabilité pénale",
    ],
    "droit_civil_obligations": [
        "contrat", "contrats", "contractuel", "obligation", "obligations",
        "responsabilité civile", "dommage", "dommages-intérêts", "validité du contrat",
        "validité", "nullité", "résiliation", "résilier", "inexécution", "manquement",
        "vice du consentement", "consentement", "dol", "erreur", "violence", "lésion",
        "cause licite", "objet du contrat", "capacité", "prescription", "forclusion",
        "propriété", "droit réel", "droits réels", "servitude", "hypothèque", "gage",
        "sûreté", "sûretés", "responsabilité délictuelle", "quasi-délit",
        "enrichissement sans cause", "indemnisation", "réparation",
    ],
    "droit_commercial_societes": [
        "société", "sociétés", "sarl", "suarl", "société anonyme", "société en nom collectif",
        "société en commandite", "actionnaire", "actions", "parts sociales", "partage",
        "fonds de commerce", "commerce", "commerçant", "immatriculation",
        "registre de commerce", "registre national des entreprises", "bail commercial",
        "effet de commerce", "lettre de change", "billet à ordre", "faillite",
        "redressement", "liquidation", "liquidateur", "procédures collectives",
        "arbitrage", "arbitral", "sentence arbitrale", "tribunal de commerce",
        "gérance", "assemblée générale", "statuts d'une société",
    ],
    "droit_travail_fonction_publique": [
        "travail", "travailleur", "travailleurs", "employeur", "salarié", "salariés",
        "licenciement", "licencié", "licencier", "congédiement", "viré", "renvoyé",
        "renvoyer", "préavis", "salaire", "rémunération", "contrat de travail",
        "congé", "congés payés", "période d'essai", "heures supplémentaires",
        "syndicat", "inspection du travail", "rupture du contrat", "négociation collective",
        "fonction publique", "fonctionnaire", "agent de l'état", "agent public",
        "statut général", "statut des personnels", "carrière administrative",
        "administration publique", "agent de l'administration",
    ],
    "droit_fiscal_douanier": [
        "impôt", "impôts", "fiscal", "fiscalité", "irpp", "impôt sur les sociétés",
        "impôt sur le revenu", "tva", "taxe sur la valeur ajoutée", "valeur ajoutée",
        "droits d'enregistrement", "timbre", "droit de timbre", "douane", "douanes",
        "tarif douanier", "recouvrement", "procédures fiscales", "contrôle fiscal",
        "bénéfice imposable", "revenu imposable", "loi de finances", "assiette",
        "taux", "exonération", "déclaration fiscale", "importation", "exportation",
        "taxe", "taxes", "redevance", "reddition", "liquidation d'impôt",
    ],
    "droit_bancaire_financier": [
        "banque", "banques", "bancaire", "établissement financier", "établissements financiers",
        "crédit", "crédits", "prêt", "prêts", "emprunt", "emprunter", "taux d'intérêt",
        "agios", "compte bancaire", "chèque", "chèques", "sans provision", "provision",
        "carte bancaire", "monétique", "paiement", "microfinance", "opc", "opcvm",
        "bourse", "valeurs mobilières", "crowdfunding", "financement participatif",
        "intermédiaire en bourse", "banque centrale", "bct", "crédit-bail", "leasing",
        "assurance", "assureur", "assuré", "sinistre", "police d'assurance", "prime",
        "indemnisation", "garantie", "compagnie d'assurance",
    ],
    "droit_famille_statut_personnel": [
        "mariage", "marié", "divorce", "divorcé", "filiation", "succession", "héritage",
        "héritier", "testament", "légataire", "statut personnel", "tutelle", "curatelle",
        "pension", "pension alimentaire", "garde des enfants", "garde",
        "kafala", "dot", "mahr", "répudiation", "répudier", "nationalité",
        "naturalisation", "tunisien", "citoyenneté", "mineur", "majeur", "capacité civile",
        "partage successoral", "réserve héréditaire",
    ],
    "droit_administratif_collectivites": [
        "collectivités locales", "collectivité locale", "commune", "municipalité",
        "aménagement du territoire", "urbanisme", "permis de construire",
        "plan d'aménagement", "lotissement", "expropriation", "tribunal administratif",
        "contentieux administratif", "recours administratif", "acte administratif",
        "décentralisation", "conseil municipal", "région", "marchés publics",
        "domaine public", "service public", "excessif de pouvoir", "recours pour excès",
        "juridictions financières", "cour des comptes",
    ],
    "droit_propriete_intellectuelle": [
        "propriété intellectuelle", "droit d'auteur", "propriété littéraire",
        "propriété littéraire et artistique", "brevet", "brevets", "brevet d'invention",
        "marque", "marques", "marques de fabrique", "dessins et modèles", "dessin",
        "modèle industriel", "indications géographiques", "copyright", "contrefaçon",
        "contrefaire", "licence", "royalties", "propriété industrielle", "œuvre",
        "créateur", "inventeur", "invention", "dépôt de marque",
    ],
    "droit_consommation_concurrence": [
        "consommateur", "consommateurs", "consommation", "concurrence", "prix",
        "entente", "abus de position dominante", "position dominante",
        "pratique anticoncurrentielle", "anticoncurrentiel", "publicité mensongère",
        "publicité trompeuse", "vente", "garantie", "vice caché", "vices cachés",
        "clauses abusives", "clause abusive", "tarif", "régulation des prix",
        "hausse des prix", "concurrence déloyale", "regroupement", "pratiques commerciales",
    ],
    "procedure_civile_dip": [
        "procédure civile", "procédure commerciale", "instance", "assignation",
        "requête", "citation", "appel", "pourvoi en cassation", "jugement",
        "exécution forcée", "injonction", "saisie", "saisie-exécution", "saisie-arrêt",
        "voies de recours", "délai de recours", "arbitrage international",
        "droit international privé", "conflit de lois", "conflit de juridictions",
        "reconnaissance de jugement", "exequatur", "compétence juridictionnelle",
        "tribunal de grande instance", "litige transfrontière", "constitution de partie civile",
    ],
    "droit_constitutionnel_institutionnel": [
        "constitution", "constitutionnel", "droits de l'homme", "libertés fondamentales",
        "liberté", "élection", "élections", "référendum", "vote", "suffrage",
        "parti politique", "associations", "association", "instance",
        "parlement", "député", "assemblée", "président de la république",
        "cour constitutionnelle", "décret présidentiel", "garantie des droits",
        "vie publique", "institution", "institutions",
    ],
    "droit_route_transport": [
        "code de la route", "circulation", "circuler", "permis de conduire",
        "conduite", "conduire", "véhicule", "véhicules", "accident de la route",
        "accident", "ivresse", "état d'ivresse", "excès de vitesse", "vitesse",
        "immatriculation", "assurance auto", "infraction routière", "retrait de permis",
        "suspension de permis", "transport", "transporteur", "transport de marchandises",
        "conducteur", "poids lourd",
    ],
    "droit_numerique_donnees": [
        "données personnelles", "données à caractère personnel", "numérique",
        "télécommunications", "communications électroniques", "cybercriminalité",
        "signature électronique", "commerce électronique", "internet", "opérateur télécom",
        "vie privée", "traitement de données", "protection des données",
        "sécurité informatique", "systèmes d'information", "système d'information",
        "réseau", "infractions informatiques", "cryptographie", "plateforme",
    ],
    "droit_investissement": [
        "investissement", "investir", "investisseur", "incitation", "agrément",
        "projet d'investissement", "agence de promotion", "avantages fiscaux",
        "capital étranger", "implantation", "zone franche", "promotion des investissements",
        "loi de l'investissement", "investissement étranger", "cession d'actifs",
    ],
    "droit_securite_conformite": [
        "sécurité", "force de sécurité", "forces de sécurité", "police", "renseignement",
        "sécurité publique", "défense", "défense nationale", "conformité", "lbc-ft",
        "lutte contre le blanchiment", "blanchiment de capitaux", "financement du terrorisme",
        "vigilance", "kyc", "déclaration de soupçon", "client à risque", "secteur de la sécurité",
        "gardiens de la paix", "sécurité intérieure",
    ],
    "droit_sante": [
        "santé", "médical", "médecin", "déontologie", "déontologie médicale", "patient",
        "établissement de santé", "pharmaceutique", "médicament", "ordonnance",
        "responsabilité médicale", "secret médical", "hôpital", "clinique", "santé publique",
        "soins", "profession de santé", "conseil de l'ordre des médecins",
    ],
}

EXEMPLARS: dict[str, list[str]] = {
    "droit_penal": [
        "Quelle est la peine prévue par le code pénal pour le meurtre ?",
        "À partir de quel âge une personne est-elle pénalement responsable ?",
        "On m'a volé ma voiture, que risque le voleur ?",
        "Quelles sont les règles de la garde à vue et de la détention préventive ?",
        "Qu'est-ce que l'abus de confiance et quelle peine encourt-on ?",
    ],
    "droit_civil_obligations": [
        "Quelles sont les conditions de fond requises pour la validité d'un contrat ?",
        "Comment se répare un dommage en responsabilité civile délictuelle ?",
        "Quels sont les cas de résiliation d'un contrat pour inexécution ?",
        "Quel est le délai de prescription d'une action en responsabilité ?",
        "Qu'est-ce qu'un vice du consentement comme le dol ou l'erreur ?",
    ],
    "droit_commercial_societes": [
        "Quelles sont les différentes formes de sociétés commerciales reconnues par la loi ?",
        "Qu'est-ce qu'un fonds de commerce et comment se transmet-il ?",
        "Quelle est la procédure de redressement ou de faillite d'une entreprise ?",
        "Comment fonctionne l'arbitrage commercial en Tunisie ?",
        "Quelles sont les obligations d'immatriculation au registre national des entreprises ?",
    ],
    "droit_travail_fonction_publique": [
        "Quel est le délai de préavis en cas de licenciement du travailleur ?",
        "Mon employeur m'a licencié sans préavis, quels sont mes droits ?",
        "Comment sont payées les heures supplémentaires ?",
        "Quel est le statut général des fonctionnaires de l'État ?",
        "Quelles sont les règles du congé payé annuel ?",
    ],
    "droit_fiscal_douanier": [
        "Quel est le taux normal de la taxe sur la valeur ajoutée (TVA) applicable en Tunisie ?",
        "Comment est déterminé le bénéfice imposable à l'impôt sur les sociétés ?",
        "Quels sont les droits de douane à l'importation de marchandises ?",
        "Quelles sont les procédures de contrôle et de recouvrement fiscal ?",
        "Que prévoit la dernière loi de finances ?",
    ],
    "droit_bancaire_financier": [
        "Quelles sont les obligations des banques en matière de déclaration de soupçon ?",
        "Quelle est la sanction pénale pour l'émission d'un chèque sans provision ?",
        "Qu'est-ce que le financement participatif (crowdfunding) et comment est-il encadré ?",
        "Quelles sont les missions et les pouvoirs de la Banque Centrale de Tunisie ?",
        "Quelles sont les obligations de l'assureur à l'égard de l'assuré ?",
    ],
    "droit_famille_statut_personnel": [
        "Quelles sont les causes légales de divorce prévues par le code du statut personnel ?",
        "Quelles sont les règles relatives à la garde des enfants après le divorce ?",
        "Comment s'opère le partage d'une succession ?",
        "Comment acquiert-on la nationalité tunisienne ?",
        "Quelle pension alimentaire est due après le divorce ?",
    ],
    "droit_administratif_collectivites": [
        "Comment obtenir un permis de construire et quel recours en cas de refus ?",
        "Quelles sont les compétences des collectivités locales ?",
        "Comment contester une décision devant le tribunal administratif ?",
        "Quelle est la procédure d'expropriation pour cause d'utilité publique ?",
        "Quelles sont les règles des marchés publics ?",
    ],
    "droit_propriete_intellectuelle": [
        "Que protège le droit d'auteur et pendant combien de temps ?",
        "Comment déposer une marque de fabrique en Tunisie ?",
        "Qu'est-ce qu'un brevet d'invention et comment le protéger ?",
        "Quelles sont les sanctions de la contrefaçon ?",
        "Comment protéger un dessin ou modèle industriel ?",
    ],
    "droit_consommation_concurrence": [
        "Quels sont les droits du consommateur face aux clauses abusives ?",
        "Qu'est-ce qu'un vice caché dans la vente et quel recours ?",
        "Quelles pratiques sont sanctionnées comme concurrence déloyale ?",
        "Comment est réglementé l'abus de position dominante ?",
        "Quelles obligations pèsent sur la publicité commerciale ?",
    ],
    "procedure_civile_dip": [
        "Quelles sont les voies de recours contre un jugement civil ?",
        "Comment saisir le tribunal en procédure civile ?",
        "Quelles sont les règles de la saisie-exécution ?",
        "Quelles sont les règles de conflit de lois en droit international privé ?",
        "Comment obtenir l'exequatur d'un jugement étranger ?",
    ],
    "droit_constitutionnel_institutionnel": [
        "Quels sont les droits et libertés garantis par la Constitution ?",
        "Comment sont organisées les élections et les référendums ?",
        "Quel est le cadre légal des associations en Tunisie ?",
        "Comment se constituent les partis politiques ?",
        "Quel est le rôle de la cour constitutionnelle ?",
    ],
    "droit_route_transport": [
        "Quelle est la sanction prévue pour la conduite en état d'ivresse ?",
        "Que faire en cas d'accident de la route ?",
        "Quelles sont les conditions d'obtention du permis de conduire ?",
        "Quelle amende pour excès de vitesse ?",
        "Dans quels cas y a-t-il retrait ou suspension du permis ?",
    ],
    "droit_numerique_donnees": [
        "Comment sont protégées les données personnelles en Tunisie ?",
        "Quelles sont les infractions de cybercriminalité et leurs sanctions ?",
        "Quelles sont les règles applicables aux communications électroniques ?",
        "Quelle valeur juridique a la signature électronique ?",
        "Quelles obligations pour le commerce électronique ?",
    ],
    "droit_investissement": [
        "Quels sont les avantages prévus par la loi de l'investissement ?",
        "Comment obtenir l'agrément pour un projet d'investissement ?",
        "Quels incitations pour l'investissement étranger ?",
        "Qu'est-ce qu'une zone franche et comment s'y implanter ?",
        "Quel est le rôle de l'agence de promotion des investissements ?",
    ],
    "droit_securite_conformite": [
        "Quels sont les textes régissant le secteur de la sécurité en Tunisie ?",
        "Quelles obligations de conformité pour la lutte contre le blanchiment ?",
        "Qu'est-ce que la déclaration de soupçon (LBC-FT) ?",
        "Quelles sont les mesures de vigilance à l'égard des clients à risque ?",
        "Comment est organisée la sécurité publique ?",
    ],
    "droit_sante": [
        "Quelles sont les règles déontologiques applicables aux médecins ?",
        "Quelle est la responsabilité civile du médecin en cas d'erreur ?",
        "Quelles obligations tiennent au secret médical ?",
        "Comment sont régis les établissements de santé ?",
        "Quelles règles pour la prescription et la délivrance de médicaments ?",
    ],
}


def is_valid_domain(code: str) -> bool:
    return code in DOMAIN_SET
