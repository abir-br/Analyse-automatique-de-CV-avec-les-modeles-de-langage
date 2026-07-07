"""
APPLICATION RECRUIT AI
"""

import os
import re
import numpy as np
from flask import Flask, request, jsonify, render_template
from werkzeug.utils import secure_filename
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import fitz  # PyMuPDF pour l'extraction des PDFs
import unicodedata
from PIL import Image
import io

# ========== CONFIGURATION FLASK ==========

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'                    # Dossier pour les fichiers temporaires
app.config['MAX_CONTENT_LENGTH'] = 50 * 1024 * 1024        # Limite de 50 Mo par fichier
ALLOWED_EXTENSIONS = {'pdf'}                               # Seuls les PDFs sont acceptes

# Creation des dossiers necessaires
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
os.makedirs('templates', exist_ok=True)

# ========== CHARGEMENT DU MODELE ==========
# Modele multilingue pour les embeddings (francais/anglais)
print("Chargement du modele sentence-transformers...")
model = SentenceTransformer("paraphrase-multilingual-MiniLM-L12-v2")
print("Modele charge avec succes")


# ========== FONCTIONS D'EXTRACTION TEXTE CV ==========
# Ces fonctions sont utilisees pour extraire et nettoyer le texte des PDFs
# Elles gerent les colonnes, les sauts de ligne et les caracteres speciaux

def nettoyer_bloc(txt):
    """Nettoie un bloc de texte en supprimant les espaces superflus"""
    return txt.strip()


def supprimer_emojis_et_symboles_inutiles(txt):
    """
    Supprime les emojis, icones et caracteres non imprimables.
    Garde uniquement: lettres, chiffres, ponctuation, espaces et retours a la ligne.
    """
    resultat = []
    for ch in txt:
        cat = unicodedata.category(ch)
        if ch in "\n\t ":
            resultat.append(ch)
        elif cat.startswith("L") or cat.startswith("N") or cat.startswith("P"):
            resultat.append(ch)
        elif cat.startswith("Z"):
            resultat.append(" ")
    return "".join(resultat)


def transformer_barres_progression(ligne):
    """
    Convertit une barre de progression (ex: ███░░) en texte lisible.
    Exemple: "Python ███░░" -> "Python niveau 3/5"
    """
    m = re.match(r"^(.*?)([█▓▒░#=\-●○◯■□⬛⬜]{2,})\s*$", ligne)
    if not m:
        return ligne
    competence = m.group(1).strip()
    barres = m.group(2)
    vides = set("░-○◯□⬜ ")
    total = len(barres)
    pleins = sum(1 for c in barres if c not in vides)
    if total == 0:
        return ligne
    niveau = round((pleins / total) * 5)
    niveau = max(1, min(niveau, 5))
    return f"{competence} niveau {niveau}/5"


def nettoyer_cv_final(texte):
    """
    Nettoie le texte final du CV:
    - Supprime les lignes vides en trop
    - Remplace les puces par des tirets
    - Convertit les barres de progression
    """
    lignes = texte.split("\n")
    nouvelles_lignes = []
    for ligne in lignes:
        ligne = ligne.strip()
        if not ligne:
            nouvelles_lignes.append("")
            continue
        # Remplacement des differents types de puces
        ligne = ligne.replace("▪", "- ").replace("•", "- ").replace("▸", "- ")
        ligne = ligne.replace("►", "- ").replace("→", "- ").replace("‣", "- ")
        ligne = transformer_barres_progression(ligne)
        # Suppression des lignes de separation (ex: -----)
        if re.fullmatch(r"[-_=─]{4,}", ligne):
            continue
        ligne = supprimer_emojis_et_symboles_inutiles(ligne)
        ligne = re.sub(r"[ \t]+", " ", ligne)
        nouvelles_lignes.append(ligne)
    texte = "\n".join(nouvelles_lignes)
    texte = re.sub(r"\n{3,}", "\n\n", texte)
    return texte.strip()


def texte_ligne_dict(line):
    """Extrait le texte d'une ligne a partir de la structure PyMuPDF"""
    morceaux = []
    for span in line.get("spans", []):
        t = span.get("text", "")
        if t and t.strip():
            morceaux.append(t.strip())
    return nettoyer_bloc(" ".join(morceaux))


def extraire_lignes_page(page):
    """
    Extrait toutes les lignes d'une page PDF avec leurs coordonnees.
    Retourne une liste de dictionnaires contenant le texte et la position (x0,y0,x1,y1)
    """
    data = page.get_text("dict")
    lignes = []
    for block in data.get("blocks", []):
        if block.get("type") != 0:  # Ignorer les blocs non-textuels (images, etc.)
            continue
        for line in block.get("lines", []):
            x0, y0, x1, y1 = line["bbox"]
            texte = texte_ligne_dict(line)
            if not texte:
                continue
            lignes.append({
                "x0": x0,
                "y0": y0,
                "x1": x1,
                "y1": y1,
                "width": x1 - x0,
                "height": y1 - y0,
                "text": texte
            })
    return lignes


def clusteriser_colonnes(lignes, largeur_page, gap_x_ratio=0.08):
    """
    Regroupe les lignes en clusters (colonnes) selon leur position horizontale.
    Deux lignes sont dans la meme colonne si leur position x est proche.
    """
    if not lignes:
        return []
    lignes = sorted(lignes, key=lambda l: l["x0"])
    gap_x = largeur_page * gap_x_ratio
    clusters = [[lignes[0]]]
    for l in lignes[1:]:
        moyenne_x = sum(x["x0"] for x in clusters[-1]) / len(clusters[-1])
        if abs(l["x0"] - moyenne_x) <= gap_x:
            clusters[-1].append(l)
        else:
            clusters.append([l])
    return clusters


def overlap_vertical(y0a, y1a, y0b, y1b):
    """Calcule le chevauchement vertical entre deux zones"""
    return max(0, min(y1a, y1b) - max(y0a, y0b))


def est_vraie_mise_en_colonnes(clusters):
    """
    Determine si le texte est vraiment organise en colonnes.
    Retourne True si plusieurs clusters ont un chevauchement vertical important.
    """
    if len(clusters) < 2:
        return False
    infos = []
    for cluster in clusters:
        y0 = min(l["y0"] for l in cluster)
        y1 = max(l["y1"] for l in cluster)
        total_chars = sum(len(l["text"]) for l in cluster)
        mean_x = sum(l["x0"] for l in cluster) / len(cluster)
        infos.append({
            "cluster": cluster,
            "y0": y0,
            "y1": y1,
            "total_chars": total_chars,
            "mean_x": mean_x
        })
    infos = sorted(infos, key=lambda x: x["mean_x"])
    gros_clusters = [c for c in infos if c["total_chars"] >= 80]
    if len(gros_clusters) < 2:
        return False
    for i in range(len(gros_clusters)):
        for j in range(i + 1, len(gros_clusters)):
            ov = overlap_vertical(
                gros_clusters[i]["y0"], gros_clusters[i]["y1"],
                gros_clusters[j]["y0"], gros_clusters[j]["y1"]
            )
            if ov >= 80:  # Chevauchement significatif
                return True
    return False


def ordonner_clusters(clusters):
    """
    Ordonne les clusters pour la lecture:
    - Le cluster principal (le plus gros) en premier
    - Puis les autres de gauche a droite
    """
    infos = []
    for cluster in clusters:
        total_chars = sum(len(l["text"]) for l in cluster)
        mean_x = sum(l["x0"] for l in cluster) / len(cluster)
        infos.append({
            "cluster": cluster,
            "total_chars": total_chars,
            "mean_x": mean_x
        })
    principal = max(infos, key=lambda x: x["total_chars"])
    autres = [c for c in infos if c is not principal]
    autres = sorted(autres, key=lambda x: x["mean_x"])
    return [principal["cluster"]] + [x["cluster"] for x in autres]


def reconstruire_lignes_ordonnee(lignes_ordonnee, seuil_ligne_vide=18, seuil_meme_ligne=2):
    """
    Reconstruit le texte a partir de lignes ordonnees.
    - Tres proches: meme ligne (espace)
    - Ecart normal: retour a la ligne
    - Grand ecart: nouveau paragraphe
    """
    if not lignes_ordonnee:
        return ""
    resultat = lignes_ordonnee[0]["text"]
    precedente = lignes_ordonnee[0]
    for ligne in lignes_ordonnee[1:]:
        gap_y = ligne["y0"] - precedente["y1"]
        if abs(ligne["y0"] - precedente["y0"]) < seuil_meme_ligne:
            resultat += " " + ligne["text"]
        elif gap_y < seuil_ligne_vide:
            resultat += "\n" + ligne["text"]
        else:
            resultat += "\n\n" + ligne["text"]
        precedente = ligne
    return resultat.strip()


def reconstruire_page(page):
    """
    Reconstruit le texte d'une page en gerant la mise en colonnes.
    C'est la fonction principale d'extraction de texte.
    """
    largeur_page = page.rect.width
    lignes = extraire_lignes_page(page)
    if not lignes:
        return ""
    clusters = clusteriser_colonnes(lignes, largeur_page)
    
    # Pas de mise en colonnes: lecture normale haut->bas
    if not est_vraie_mise_en_colonnes(clusters):
        lignes_ordonnee = sorted(lignes, key=lambda l: (l["y0"], l["x0"]))
        texte = reconstruire_lignes_ordonnee(lignes_ordonnee)
        return nettoyer_cv_final(texte)
    
    # Mise en colonnes: lire colonne par colonne
    clusters_ordonnes = ordonner_clusters(clusters)
    morceaux = []
    for cluster in clusters_ordonnes:
        cluster = sorted(cluster, key=lambda l: (l["y0"], l["x0"]))
        morceaux.append(reconstruire_lignes_ordonnee(cluster))
    texte = "\n".join(m for m in morceaux if m.strip())
    return nettoyer_cv_final(texte)


def extraire_texte_cv(pdf_path, activer_ocr=False, seuil_min_caracteres=80):
    """
    Extrait le texte complet d'un CV PDF.
    Parcourt toutes les pages et applique l'OCR si le texte est insuffisant.
    """
    doc = fitz.open(pdf_path)
    pages_textes = []
    for page in doc:
        texte_page = reconstruire_page(page)
        # OCR de secours si le texte est trop court (PDF scannes)
        if activer_ocr and len(texte_page.strip()) < seuil_min_caracteres:
            try:
                pix = page.get_pixmap(matrix=fitz.Matrix(2, 2), alpha=False)
                img_bytes = pix.tobytes("png")
                image = Image.open(io.BytesIO(img_bytes))
                import pytesseract
                texte_ocr = pytesseract.image_to_string(image, lang="fra+eng")
                if len(texte_ocr.strip()) > len(texte_page.strip()):
                    texte_page = texte_ocr
            except:
                pass
        if texte_page.strip():
            pages_textes.append(texte_page.strip())
    doc.close()
    texte_final = "\n\n".join(pages_textes)
    return texte_final


# ========== FONCTIONS DE PRETRAITEMENT ==========

def pretraiter_cv(texte):
    """
    Pretraite le texte du CV avant le chunking:
    - Supprime les balises speciales
    - Normalise les retours a la ligne
    - Nettoie les caracteres
    """
    texte = re.sub(r'\[\[(.*?)\]\]', r'\1', texte)
    texte = re.sub(r'>>>|>>|===|---|\*\*\*|///', '\n', texte)
    texte = re.sub(r'\r\n', '\n', texte)
    texte = re.sub(r'\n{3,}', '\n\n', texte)
    texte = re.sub(r'[•●○◆◇■□▶▷→⇒➔]', ' ', texte)
    texte = re.sub(r' +', ' ', texte)
    return texte.strip()


# ========== STRATEGIES DE CHUNKING ==========
# Ces fonctions decoupent le CV en blocs pour l'analyse
# Chaque strategie a une approche differente

def chunk_by_newlines(texte):
    """
    Strategie 1: Decoupage par double saut de ligne.
    Simple et efficace pour les CVs bien structures.
    """
    texte = pretraiter_cv(texte)
    blocs = texte.strip().split("\n\n")
    blocs = [b.strip() for b in blocs if b.strip()]
    return blocs


def chunk_by_words(texte, taille=80, overlap=20):
    """
    Strategie 2: Decoupage par nombre de mots avec chevauchement.
    - taille: nombre de mots par bloc
    - overlap: mots communs entre blocs consecutifs
    Utile pour les tres longs documents.
    """
    texte = pretraiter_cv(texte)
    mots = texte.split()
    blocs = []
    i = 0
    while i < len(mots):
        bloc = " ".join(mots[i:i + taille])
        blocs.append(bloc)
        i += taille - overlap
    return blocs


def chunk_by_sentences(texte, max_mots=100):
    """
    Strategie 3: Regroupement de phrases.
    Decoupe d'abord en phrases, puis regroupe jusqu'a atteindre max_mots.
    """
    texte = pretraiter_cv(texte)
    phrases = re.split(r'(?<=[.!?])\s+', texte)
    blocs = []
    bloc_courant = []
    cpt_mots = 0
    for phrase in phrases:
        n = len(phrase.split())
        if cpt_mots + n > max_mots and bloc_courant:
            blocs.append(" ".join(bloc_courant))
            bloc_courant = []
            cpt_mots = 0
        bloc_courant.append(phrase)
        cpt_mots += n
    if bloc_courant:
        blocs.append(" ".join(bloc_courant))
    return blocs


def chunk_semantic(texte):
    """
    Strategie 4: Decoupage semantique.
    Calcule la similarite entre phrases consecutives.
    Coupe quand la similarite chute (changement de sujet).
    """
    texte = pretraiter_cv(texte)
    phrases = re.split(r'(?<=[.!?\n])\s+', texte.strip())
    phrases = [p.strip() for p in phrases if len(p.strip()) > 10]
    if len(phrases) < 3:
        return phrases
    embeddings = model.encode(phrases)
    sims = []
    for i in range(1, len(phrases)):
        sim = cosine_similarity([embeddings[i-1]], [embeddings[i]])[0][0]
        sims.append(sim)
    seuil = np.mean(sims) - np.std(sims) if sims else 0.5
    seuil = max(0.2, min(seuil, 0.85))
    blocs = []
    bloc_courant = [phrases[0]]
    for i, s in enumerate(sims):
        if s < seuil:
            blocs.append(" ".join(bloc_courant))
            bloc_courant = []
        bloc_courant.append(phrases[i + 1])
    blocs.append(" ".join(bloc_courant))
    return blocs


def chunk_adaptatif(texte, max_words=80, seuil_similarite=None):
    """
    Strategie 5: Chunking adaptatif (RECOMMANDEE).
    Decoupage hierarchique: paragraphes -> lignes -> phrases -> mots.
    Puis fusion semantique des blocs similaires.
    """
    texte = pretraiter_cv(texte)
    if not texte:
        return []
    
    # Niveau 1 : paragraphes
    segments = []
    ordre_global = 0
    a_traiter_n2 = []
    
    for bloc in texte.strip().split("\n\n"):
        a_traiter_n2.append((ordre_global, bloc))
        ordre_global += 1
    
    def nb_mots(txt):
        return len(txt.split())
    
    def traiter_niveau(blocs, max_words):
        """Traite un niveau de decoupage"""
        result = []
        a_traiter_suivant = []
        for idx, bloc in blocs:
            if nb_mots(bloc) <= max_words:
                result.append((idx, bloc.strip()))
            else:
                a_traiter_suivant.append((idx, bloc))
        return result, a_traiter_suivant
    
    # Niveau 2 : lignes
    segments_n2, a_traiter_n3 = traiter_niveau([
        (idx, "\n".join(bloc.split("\n"))) for idx, bloc in a_traiter_n2
    ], max_words)
    
    # Niveau 3 : phrases
    segments_n3, a_traiter_n4 = traiter_niveau([
        (idx, phrase)
        for idx, bloc in a_traiter_n3
        for phrase in re.split(r'(?<=[.!?;:])\s+', bloc)
    ], max_words)
    
    # Niveau 4 : mots (decoupage force)
    segments_n4 = []
    for idx, bloc in a_traiter_n4:
        courant = ""
        for mot in bloc.split():
            if nb_mots(courant) + 1 <= max_words:
                courant += (" " if courant else "") + mot
            else:
                segments_n4.append((idx, courant.strip()))
                courant = mot
        if courant:
            segments_n4.append((idx, courant.strip()))
    
    # Combinaison et tri par ordre original
    segments = segments_n2 + segments_n3 + segments_n4
    segments.sort(key=lambda x: x[0])
    blocs = [texte_s for _, texte_s in segments if len(texte_s) > 0]
    
    if len(blocs) <= 1:
        return blocs
    
    # Fusion semantique: regroupe les blocs dont la similarite est elevee
    embeddings = model.encode(blocs, show_progress_bar=False)
    
    # Calcul des similarites entre blocs consecutifs
    sims = []
    for i in range(1, len(embeddings)):
        sim = cosine_similarity([embeddings[i-1]], [embeddings[i]])[0][0]
        sims.append(sim)
    
    # Seuil adaptatif (median + 0.5 * ecart-type)
    if len(sims) == 0:
        seuil = 0.5
    else:
        if seuil_similarite is None:
            seuil = float(np.median(sims) + 0.5 * np.std(sims))
            seuil = max(0.2, min(seuil, 0.85))
        else:
            seuil = float(seuil_similarite)
    
    # Fusion des blocs similaires
    blocs_fusionnes = []
    groupe_courant = [blocs[0]]
    taille_courante = nb_mots(blocs[0])
    
    for i, sim in enumerate(sims):
        prochain = blocs[i + 1]
        taille_fusionnee = taille_courante + nb_mots(prochain)
        
        if sim >= seuil and taille_fusionnee <= max_words:
            groupe_courant.append(prochain)
            taille_courante = taille_fusionnee
        else:
            blocs_fusionnes.append(" ".join(groupe_courant))
            groupe_courant = [prochain]
            taille_courante = nb_mots(prochain)
    
    blocs_fusionnes.append(" ".join(groupe_courant))
    
    return blocs_fusionnes


# ========== FONCTION PRINCIPALE DE CLASSEMENT ==========

def classer_cvs(offre_texte, cvs_textes, strategy="adaptatif"):
    """
    Classe les CVs par similarite avec l'offre d'emploi.
    
    Etapes:
    1. Encoder l'offre en embedding
    2. Pour chaque CV, appliquer la strategie de chunking
    3. Encoder tous les blocs du CV
    4. Calculer la similarite cosinus entre l'offre et chaque bloc
    5. Garder le meilleur score par CV
    6. Trier les CVs par score decroissant
    """
    embedding_offre = model.encode([offre_texte])
    results = []
    
    for cv_nom, cv_texte in cvs_textes:
        # Selection de la strategie de chunking
        if strategy == "newlines":
            blocs = chunk_by_newlines(cv_texte)
        elif strategy == "words":
            blocs = chunk_by_words(cv_texte)
        elif strategy == "sentences":
            blocs = chunk_by_sentences(cv_texte)
        elif strategy == "semantic":
            blocs = chunk_semantic(cv_texte)
        else:
            blocs = chunk_adaptatif(cv_texte, max_words=80)
        
        # Cas ou aucun bloc n'a ete genere
        if not blocs:
            results.append({
                "nom": cv_nom,
                "score": 0.0,
                "meilleur_bloc": "",
                "meilleur_idx": 0,
                "nb_blocs": 0,
                "taille_cv": len(cv_texte),
                "cv_complet": cv_texte
            })
            continue
        
        # Encodage des blocs et calcul des similarites
        embeddings_blocs = model.encode(blocs)
        scores = cosine_similarity(embedding_offre, embeddings_blocs)[0]
        
        # Meilleur bloc et son score
        meilleur_idx = int(np.argmax(scores))
        meilleur_score = float(scores[meilleur_idx])
        
        results.append({
            "nom": cv_nom,
            "score": meilleur_score,
            "meilleur_bloc": blocs[meilleur_idx],
            "meilleur_idx": meilleur_idx,
            "nb_blocs": int(len(blocs)),
            "taille_cv": len(cv_texte),
            "cv_complet": cv_texte
        })
    
    # Tri par score decroissant
    results.sort(key=lambda x: x["score"], reverse=True)
    return results


# ========== ROUTES FLASK ==========

@app.route('/')
def index():
    """Route principale: affiche la page d'accueil"""
    return render_template('index.html')


@app.route('/upload', methods=['POST'])
def upload_cvs():
    """
    Route qui recoit les donnees du formulaire:
    - offre: texte de l'offre d'emploi
    - strategy: strategie de chunking choisie
    - cvs: liste des fichiers PDF
    
    Extrait le texte des CVs, les classe, et retourne les resultats en JSON.
    """
    # Recuperation de l'offre
    offre = request.form.get('offre', '')
    if not offre:
        return jsonify({"error": "Veuillez fournir une offre d'emploi"}), 400
    
    # Recuperation de la strategie
    strategy = request.form.get('strategy', 'adaptatif')
    
    # Recuperation des fichiers
    fichiers = request.files.getlist('cvs')
    if not fichiers or fichiers[0].filename == '':
        return jsonify({"error": "Veuillez fournir au moins un CV"}), 400
    
    cvs_textes = []
    
    # Traitement de chaque fichier
    for fichier in fichiers:
        if fichier and allowed_file(fichier.filename):
            nom_fichier = secure_filename(fichier.filename)
            chemin_fichier = os.path.join(app.config['UPLOAD_FOLDER'], nom_fichier)
            fichier.save(chemin_fichier)
            
            try:
                # Extraction du texte du CV
                cv_texte = extraire_texte_cv(chemin_fichier, activer_ocr=False)
                nom = nom_fichier.replace('.pdf', '').replace('_', ' ')
                cvs_textes.append((nom, cv_texte))
                print(f"Texte extrait: {nom} ({len(cv_texte)} caracteres)")
            except Exception as e:
                print(f"Erreur extraction {nom_fichier}: {e}")
            finally:
                # Nettoyage: suppression du fichier temporaire
                if os.path.exists(chemin_fichier):
                    os.remove(chemin_fichier)
    
    if not cvs_textes:
        return jsonify({"error": "Aucun CV n'a pu etre traite"}), 400
    
    # Classement des CVs
    results = classer_cvs(offre, cvs_textes, strategy)
    
    return jsonify({
        "results": results,
        "strategy": strategy,
        "total_cvs": len(results)
    })


def allowed_file(filename):
    """Verifie si l'extension du fichier est autorisee (uniquement PDF)"""
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


# ========== LANCEMENT DE L'APPLICATION ==========

if __name__ == '__main__':
    print("=" * 50)
    print("RECRUIT AI - Serveur de classement de CV")
    print("=" * 50)
    print("Serveur demarre sur http://localhost:5000")
    print("Pour arreter: CTRL+C")
    print("=" * 50)
    # use_reloader=False evite le double chargement du modele
    # threaded=True permet a Flask de gerer plusieurs requetes simultanement
    app.run(debug=True, host='0.0.0.0', port=5000, use_reloader=False, threaded=True)