"""
AI Analysis Prompts

All AI prompts centralized in one place for easy maintenance.
"""

# UNIFIED MODE: Complete analysis in one request
UNIFIED_PROMPT = """
Analysiere diesen Inhalt und gib eine umfassende strukturierte Antwort zurück.
{context_info}

DEINE AUFGABEN:

1. SAFETY CHECK (Sicherheitsbewertung)
   - Überprüfe auf NSFW, Gewalt, inappropriate Inhalte
   - Bewerte die Sicherheit (true/false)
   - Gib einen Konfidenz-Wert (0.0-1.0)
   - Erkläre deine Einschätzung kurz
   - Liste eventuelle Probleme in 'flags' auf

2. CLASSIFICATION (Kategorisierung)
   - Klassifiziere in eine Kategorie: 'product', 'person', 'event', 'landscape', 'art', 'document', 'video', 'text', 'other'
   - Bewerte das Gefährdungspotenzial für Kinder (1-10 Skala: 1=völlig sicher, 10=sehr gefährlich)

3. CONTENT ANALYSIS (Inhaltsanalyse)
   - Generiere einen prägnanten Titel (max 50 Zeichen)
   - Erstelle einen emotionalen Untertitel mit Emoji (Instagram-Style)
   - Erkenne 3-5 relevante Tags (OHNE #-Symbol, auf deutsch)
   - Schlage 1-2 einfache Collection-Namen vor (z.B. "Arbeit", "Freizeit", "Familie", "Reisen")

4. KNOWLEDGE GRAPH EXTRACTION (Strukturierte Semantic-Daten)
   - Extrahiere ALLE relevanten strukturierten Daten basierend auf Content-Type
   - Wähle die Felder intelligent basierend auf dem Inhaltstyp:
     * Products: brand, product, year, colors, sizes, materials, certifications, price_range
     * Events: event_name, location, date, year, participants, event_type
     * People: names, age_range, gender, occupation, relationships
     * Landscapes: location, country, region, time_of_day, weather, season
     * Videos/Media: title, creator, date, location, duration_description, subjects
     * Documents: doc_type, author, date, subject, language
   - Füge IMMER ein 'keywords' Array mit semantischen Keywords hinzu

5. SEMANTIC EMBEDDINGS (Intelligente Embedding-Erstellung)
   - embeddingText: Haupttext für das primäre Embedding
   - embeddingsList: OPTIONAL - Nutze dies für strukturierte Daten (CSV, JSON, XLS, komplexe Dokumente)
   - embeddingQuality: WICHTIG - Bewertung der Datenqualität für Embeddings

   EMBEDDING QUALITY BEWERTUNG (KRITISCH!):
   Bewerte die Eignung der Daten für automatische Embeddings:

   * quality_score: 1-10 (1=ungeeignet, 10=perfekt strukturiert)
   * needs_review: boolean - true wenn manuelle Überprüfung nötig
   * issues: Liste von Problemen (z.B. "inkonsistente_struktur", "fehlende_daten", "unklares_format")
   * recommendation: "auto_embed" | "review_required" | "skip_embedding"

   SETZE needs_review=true WENN:
   - Datenstruktur inkonsistent oder unklar
   - Wichtige Informationen fehlen
   - Format ungeeignet für semantische Suche
   - Zu viele Duplikate oder leere Felder
   - Unsicher ob sinnvolle Embeddings entstehen

   WANN embeddingsList verwenden:

   **CSV/Excel DATEIEN (WICHTIG!):**
   - Für tabellarische Daten (Produkte, Events, Personen, Transaktionen, etc.): Erstelle IMMER ein separates Embedding pro Zeile
   - Jede Datenzeile = 1 Embedding in embeddingsList (NICHT nur repräsentative Samples!)
   - Verarbeite ALLE Zeilen von Anfang bis Ende
   - Jede Zeile repräsentiert eine eigenständige Entität die durchsuchbar sein soll
   - Füge Bild-URLs (falls vorhanden) über "uri_groups" im metadata hinzu
   - Bei 100 Zeilen → 100 Embeddings | Bei 274 Zeilen → 274 Embeddings

   **Andere Anwendungsfälle:**
   * JSON Arrays: Jedes wichtige Array-Element als separates Embedding
   * Komplexe Bilder: Haupt-Embedding + Detail-Embeddings (Logo, Text, Objekte)
   * Dokumente: Kapitel, Abschnitte, wichtige Konzepte separat embedden

   Beispiel CSV (beliebiger Typ) - ALLE Zeilen verarbeiten:
   - embeddingsList[0]: "Zeile 1 Daten: Feld1, Feld2, Feld3..."
   - embeddingsList[1]: "Zeile 2 Daten: Feld1, Feld2, Feld3..."
   - embeddingsList[2]: ... (weiter für ALLE Zeilen bis zum Ende!)

   IMAGE URIs in CSV/Excel (INTELLIGENTES HANDLING):
   Wenn Bild-URLs in Spalten vorhanden sind (z.B. "image_url", "thumbnail", "photo"):

   ENTSCHEIDE: Sind die URIs VERSCHIEDENE VERSIONEN oder VERSCHIEDENE OBJEKTE?

   A) VERSCHIEDENE VERSIONEN (gleicher Inhalt):
      - Mehrere Größen desselben Bildes (thumb, medium, large)
      - Verschiedene Formate desselben Bildes (jpg, webp, png)
      - Gleiche Ansicht in verschiedener Qualität
      → Nutze "uri_groups" mit mode="select_best"

   B) VERSCHIEDENE OBJEKTE (unterschiedlicher Inhalt):
      - Verschiedene Ansichten (front, side, back)
      - Verschiedene Produkte/Objekte in einer Zeile
      - Unterschiedliche Bilder die getrennt sein sollten
      → Nutze "uri_groups" mit mode="create_separate"

   METADATA FORMAT für Image URIs:
   {{
     "uri_groups": [
       {{
         "uris": ["url1", "url2"],
         "mode": "select_best",  // oder "create_separate"
         "description": "Product thumbnail in different sizes"  // optional
       }}
     ]
   }}

   Beispiel A - Verschiedene Versionen:
   {{
     "uri_groups": [{{
       "uris": ["helmet_thumb.jpg", "helmet_large.jpg", "helmet_hd.jpg"],
       "mode": "select_best",
       "description": "Helmet front view - multiple resolutions"
     }}]
   }}

   Beispiel B - Verschiedene Objekte:
   {{
     "uri_groups": [
       {{"uris": ["helmet_front.jpg"], "mode": "select_best", "description": "Front view"}},
       {{"uris": ["helmet_side.jpg"], "mode": "select_best", "description": "Side view"}},
       {{"uris": ["helmet_back.jpg"], "mode": "select_best", "description": "Back view"}}
     ]
   }}

   LEGACY-SUPPORT: Alte "image_uris" weiterhin akzeptiert (wird als select_best behandelt)

   WICHTIG für CSV/Excel: IMMER alle Zeilen als separate Embeddings verarbeiten (siehe oben)!
   WICHTIG für andere Datentypen: Nutze embeddingsList nur wenn sinnvoll! Bei Unsicherheit: needs_review=true

WICHTIG: Füge NUR Felder hinzu, die wirklich relevant für diesen spezifischen Inhalt sind!

ANTWORT FORMAT (EXAKT dieses JSON Format zurückgeben):
{{
  "safetyCheck": {{
    "isSafe": boolean,
    "confidence": number (0.0-1.0),
    "reasoning": "string",
    "flags": []
  }},
  "classification": {{
    "category": "string",
    "dangerPotential": integer (1-10)
  }},
  "mediaAnalysis": {{
    "suggestedTitle": "string (max 50 chars)",
    "suggestedSubtitle": "string with emoji",
    "tags": ["tag1", "tag2", "tag3"],
    "collectionSuggestions": ["collection1", "collection2"]
  }},
  "extractedTags": {{
    "keywords": ["keyword1", "keyword2"]
  }},
  "embeddingInfo": {{
    "embeddingText": "string - Rich searchable text combining all key info for semantic search",
    "searchableFields": ["field1", "field2"],
    "metadata": {{}},
    "embeddingQuality": {{
      "quality_score": integer (1-10),
      "needs_review": boolean,
      "issues": ["issue1", "issue2"],
      "recommendation": "auto_embed|review_required|skip_embedding"
    }},
    "embeddingsList": [
      {{
        "text": "string - Semantic text for this specific embedding",
        "type": "product|feature|detail|row|item|other",
        "metadata": {{}}
      }}
    ]
  }}
}}
"""


# SPLIT MODE: Safety check only (fast, focused)
SAFETY_PROMPT = """
Analysiere diesen Inhalt hinsichtlich Sicherheit und grundlegender Kategorisierung.
{context_info}

AUFGABEN:
1. SAFETY CHECK - Überprüfe auf NSFW, Gewalt, inappropriate Inhalte
2. CLASSIFICATION - Kategorisiere in: 'product', 'person', 'event', 'landscape', 'art', 'document', 'video', 'text', 'other'
3. DANGER POTENTIAL - Gefährdungspotenzial für Kinder (1-10)

ANTWORT FORMAT (JSON):
{{
  "safetyCheck": {{
    "isSafe": boolean,
    "confidence": number (0.0-1.0),
    "reasoning": "string",
    "flags": []
  }},
  "classification": {{
    "category": "string",
    "dangerPotential": integer (1-10)
  }}
}}
"""


# SPLIT MODE: Embedding generation (comprehensive semantic analysis)
EMBEDDING_PROMPT = """
Analysiere diesen Inhalt für semantische Suche und Knowledge Graph Extraction.
{context_info}

AUFGABEN:
1. CONTENT ANALYSIS - Titel, Untertitel, Tags, Collections
2. KNOWLEDGE GRAPH EXTRACTION - Strukturierte semantische Daten basierend auf Content-Typ
   - Products: brand, product, year, colors, sizes, materials, certifications, price_range
   - Events: event_name, location, date, year, participants, event_type
   - People: names, age_range, gender, occupation, relationships
   - Landscapes: location, country, region, time_of_day, weather, season
   - Videos/Media: title, creator, date, location, duration_description, subjects
   - Documents: doc_type, author, date, subject, language
3. EMBEDDING INFO - Rich searchable text und Metadaten

   INTELLIGENTE EMBEDDINGS:
   - embeddingText: Haupttext für primäres Embedding
   - embeddingsList: OPTIONAL für strukturierte Daten (CSV, JSON, XLS, komplexe Inhalte)
   - embeddingQuality: WICHTIG - Bewertung der Datenqualität

   EMBEDDING QUALITY (KRITISCH!):
   * quality_score: 1-10 (1=ungeeignet, 10=perfekt)
   * needs_review: boolean - true wenn manuelle Prüfung nötig
   * issues: Liste von Problemen
   * recommendation: "auto_embed" | "review_required" | "skip_embedding"

   SETZE needs_review=true bei:
   - Inkonsistenter Struktur
   - Fehlenden Daten
   - Unklarem Format
   - Unsicherheit über Sinnhaftigkeit

   Nutze embeddingsList für:
   * CSV/Excel: ALLE Zeilen als separate Embeddings (NICHT nur Samples!)
   * JSON Arrays: Jedes wichtige Element einzeln embedden
   * Komplexe Bilder: Haupt-Embedding + Details

   WICHTIG: Bei Unsicherheit needs_review=true setzen!
   WICHTIG für CSV/Excel: IMMER alle Zeilen verarbeiten!

ANTWORT FORMAT (JSON):
{{
  "mediaAnalysis": {{
    "suggestedTitle": "string (max 50 chars)",
    "suggestedSubtitle": "string with emoji",
    "tags": ["tag1", "tag2", "tag3"],
    "collectionSuggestions": ["collection1", "collection2"]
  }},
  "extractedTags": {{
    "keywords": ["keyword1", "keyword2"]
  }},
  "embeddingInfo": {{
    "embeddingText": "Rich searchable text combining all key info",
    "searchableFields": ["field1", "field2"],
    "metadata": {{}},
    "embeddingQuality": {{
      "quality_score": integer (1-10),
      "needs_review": boolean,
      "issues": ["issue1", "issue2"],
      "recommendation": "auto_embed|review_required|skip_embedding"
    }},
    "embeddingsList": [
      {{
        "text": "Semantic text for specific embedding",
        "type": "product|feature|detail|row|item|other",
        "metadata": {{}}
      }}
    ]
  }}
}}
"""


# CHUNKED CSV MODE: For processing large CSVs in batches
CHUNKED_CSV_PROMPT = """
Analysiere diesen CSV-Chunk und erstelle für JEDE Zeile ein separates Embedding.

{context_info}

CHUNK INFO: Dies ist Teil {chunk_index} einer größeren CSV-Datei.

WICHTIG - DEINE AUFGABE:
- Erstelle für JEDE Zeile in diesem Chunk ein separates Embedding in embeddingsList
- Jede Zeile = 1 Embedding (keine Samples, keine Zusammenfassungen!)
- Füge Bild-URLs über "uri_groups" im metadata hinzu (falls vorhanden)
- Extrahiere strukturierte Daten basierend auf dem Inhaltstyp

ANTWORT FORMAT (JSON):
{{
  "embeddingsList": [
    {{
      "text": "Semantic text for row 1",
      "type": "row",
      "metadata": {{
        "uri_groups": [
          {{
            "uris": ["image_url_if_present"],
            "mode": "select_best",
            "description": "Product image"
          }}
        ]
      }}
    }},
    {{
      "text": "Semantic text for row 2",
      "type": "row",
      "metadata": {{}}
    }}
  ],
  "quality_score": integer (1-10)
}}
"""


# VISION MODE: Comprehensive image analysis with product detection
VISION_ANALYSIS_PROMPT = """
Analysiere dieses Bild UMFASSEND und extrahiere ALLE relevanten visuellen und semantischen Informationen.

{context_info}

DEINE AUFGABEN - SEHR DETAILLIERT:

1. PRODUKTERKENNUNG (Product Detection)
   Erkenne ob dieses Bild ein PRODUKT zeigt (für E-Commerce, Kataloge, Shops).

   Falls JA, extrahiere/schätze folgende Eigenschaften:
   - Produkttyp & Kategorie (z.B. Helm, Jacke, Schuhe, Elektronik)
   - Marke/Brand (falls sichtbar)
   - Modellname/Bezeichnung (falls erkennbar)
   - Farben (ALLE sichtbaren Farben, Hauptfarbe zuerst)
   - Materialien (geschätzt aus visueller Erscheinung)
   - Größendimension (klein/mittel/groß, relative Größe schätzen)
   - Gewichtsklasse (leicht/mittel/schwer basierend auf Material/Typ)
   - Preisvolumen (economy/mid-range/premium basierend auf Qualität/Design)
   - Zielgruppe (Kinder/Jugendliche/Erwachsene, Männer/Frauen/Unisex)
   - Verwendungszweck (Sport, Freizeit, Arbeit, etc.)
   - Saison (Sommer/Winter/Ganzjahr)
   - Stil/Ästhetik (sportlich, elegant, casual, technisch)
   - Besondere Features (wasserdicht, gepolstert, reflektierend, etc.)

2. VISUELLE ANALYSE (Visual Analysis)
   Extrahiere ALLE visuellen Eigenschaften für intelligentes Layout in Katalogen:

   A) FARBANALYSE:
   - dominant_colors: Top 3-5 dominante Farben (HEX codes wenn möglich)
   - color_palette: Gesamtes Farbschema (warm/cool/neutral/vibrant/muted)
   - color_harmony: Farbharmonie-Typ (monochromatic/complementary/analogous)
   - background_color: Hintergrundfarbe (white/black/colored/transparent)

   B) KOMPOSITION & LAYOUT:
   - composition_type: Aufbau (centered/rule-of-thirds/asymmetric)
   - viewing_angle: Perspektive (front/side/back/top/angled/3-4-view)
   - product_position: Position im Bild (centered/left/right/floating)
   - image_style: Stil (studio_shot/lifestyle/packshot/detail/in_use)
   - background_type: Hintergrund (plain/gradient/lifestyle/textured)

   C) QUALITÄT & PRÄSENTATION:
   - image_quality: Bildqualität (high/medium/low)
   - lighting: Beleuchtung (studio/natural/soft/dramatic)
   - clarity: Schärfe/Klarheit (crisp/soft/blurred)
   - professional_level: Professionalität (professional/semi-pro/casual)

   D) VISUELLER STIL:
   - aesthetic_style: Ästhetischer Stil (minimal/modern/vintage/playful/technical)
   - mood: Stimmung (energetic/calm/professional/fun)
   - visual_weight: Visuelles Gewicht (light/balanced/heavy)

3. LAYOUT-KOMPATIBILITÄT (Catalog Layout Intelligence)
   Analysiere wie dieses Produkt mit ANDEREN visuell kombiniert werden kann:

   - visual_harmony_tags: Tags für visuelle Harmonie (z.B. "minimal_white", "vibrant_sport", "dark_premium")
   - pairing_suggestions: Welche visuellen Eigenschaften passen gut zusammen
   - contrast_level: Kontraststärke (high/medium/low) - wichtig für Gruppierung
   - visual_complexity: Komplexität (simple/moderate/complex)
   - attention_score: Aufmerksamkeitswert (1-10, wie stark zieht es den Blick)

4. SEMANTISCHE EIGENSCHAFTEN (Semantic Properties)
   Extrahiere semantische Informationen für intelligente Suche:

   - keywords: Semantische Keywords (min. 10-20 relevante Begriffe)
   - use_cases: Anwendungsfälle & Szenarien
   - target_audience: Detaillierte Zielgruppenanalyse
   - emotional_appeal: Emotionale Ansprache (Abenteuer, Sicherheit, Style, etc.)
   - brand_perception: Markenwahrnehmung (luxury/sporty/reliable/innovative)

5. TECHNISCHE METADATEN (Technical Metadata)
   - aspect_ratio: Seitenverhältnis (z.B. 1:1, 4:3, 16:9)
   - orientation: Ausrichtung (landscape/portrait/square)
   - complexity_score: Komplexität (1-10)
   - uniqueness_score: Einzigartigkeit (1-10)

WICHTIG:
- Sei SEHR detailliert und umfassend
- Schätze Eigenschaften intelligent auch wenn nicht explizit sichtbar
- Denke an ALLE Dimensionen die für Katalog-Layout relevant sein könnten
- Extrahiere Informationen die später für Sortierung, Gruppierung und visuelle Harmonie nützlich sind

ANTWORT FORMAT (JSON):
{{
  "safetyCheck": {{
    "isSafe": boolean,
    "confidence": number (0.0-1.0),
    "reasoning": "string",
    "flags": []
  }},
  "classification": {{
    "category": "product|other",
    "dangerPotential": integer (1-10),
    "isProduct": boolean,
    "productType": "string or null"
  }},
  "productAnalysis": {{
    "brand": "string or null",
    "productName": "string or null",
    "productType": "string",
    "category": "string",
    "colors": ["color1", "color2"],
    "materials": ["material1", "material2"],
    "sizeCategory": "small|medium|large",
    "weightClass": "light|medium|heavy",
    "priceRange": "economy|mid-range|premium|luxury",
    "targetAudience": {{}},
    "usageContext": [],
    "season": "summer|winter|all-season",
    "style": "string",
    "features": []
  }},
  "visualAnalysis": {{
    "colorAnalysis": {{
      "dominantColors": ["#hex1", "#hex2", "#hex3"],
      "colorPalette": "string",
      "colorHarmony": "string",
      "backgroundColor": "string"
    }},
    "composition": {{
      "compositionType": "string",
      "viewingAngle": "string",
      "productPosition": "string",
      "imageStyle": "string",
      "backgroundType": "string"
    }},
    "quality": {{
      "imageQuality": "high|medium|low",
      "lighting": "string",
      "clarity": "string",
      "professionalLevel": "string"
    }},
    "aesthetics": {{
      "aestheticStyle": "string",
      "mood": "string",
      "visualWeight": "string"
    }}
  }},
  "layoutIntelligence": {{
    "visualHarmonyTags": ["tag1", "tag2"],
    "pairingSuggestions": "string",
    "contrastLevel": "high|medium|low",
    "visualComplexity": "simple|moderate|complex",
    "attentionScore": integer (1-10)
  }},
  "semanticProperties": {{
    "keywords": ["keyword1", "keyword2", "..."],
    "useCases": [],
    "targetAudience": {{}},
    "emotionalAppeal": [],
    "brandPerception": "string"
  }},
  "technicalMetadata": {{
    "aspectRatio": "string",
    "orientation": "landscape|portrait|square",
    "complexityScore": integer (1-10),
    "uniquenessScore": integer (1-10)
  }},
  "mediaAnalysis": {{
    "suggestedTitle": "string",
    "suggestedSubtitle": "string",
    "tags": [],
    "collectionSuggestions": []
  }},
  "embeddingInfo": {{
    "embeddingText": "SEHR DETAILLIERTER TEXT mit allen extrahierten Informationen für semantische Suche",
    "searchableFields": ["field1", "field2"],
    "metadata": {{
      "all extracted properties": "as structured data"
    }},
    "embeddingQuality": {{
      "quality_score": integer (1-10),
      "needs_review": false,
      "issues": [],
      "recommendation": "auto_embed"
    }}
  }},
  "annotations": [
    {{
      "label": "string",
      "type": "feature|logo|zipper|material|vent|price|title|other",
      "anchor": {{"x": number (0..1), "y": number (0..1)}},
      "box": [number (0..1), number (0..1), number (0..1), number (0..1)],
      "confidence": number (0.0-1.0),
      "source": "vision-llm"
    }}
  ]
}}
"""


def build_context_info(context: dict = None) -> str:
    """Build context information string from context dict"""
    if not context:
        return ""

    context_parts = []

    if "file_path" in context and context["file_path"]:
        context_parts.append(f"File path: {context['file_path']}")

    if "collection_id" in context and context["collection_id"]:
        context_parts.append(f"Collection: {context['collection_id']}")

    if "metadata" in context and context["metadata"]:
        meta = context["metadata"]
        if isinstance(meta, dict):
            meta_str = ", ".join([f"{k}: {v}" for k, v in meta.items()])
            context_parts.append(f"Metadata: {meta_str}")

    if "role" in context and context["role"]:
        context_parts.append(f"Role: {context['role']}")

    if "context_text" in context and context["context_text"]:
        context_parts.append(f"\nFree-form description:\n{context['context_text']}")

    if context_parts:
        return "\n\nContextual Information:\n" + "\n".join(context_parts) + "\n"

    return ""
