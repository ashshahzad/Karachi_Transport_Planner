"""
karachi_transport/graph_data.py
--------------------------------
Karachi transport network: nodes, coordinates, and weighted edges.
All edge weights are travel times in minutes under moderate traffic.
"""

# Node index → name
NODES = {
    0:  "Saddar",
    1:  "Clifton",
    2:  "DHA Phase V",
    3:  "Korangi",
    4:  "Landhi",
    5:  "Malir",
    6:  "Gulshan-e-Iqbal",
    7:  "Johar More",
    8:  "North Nazimabad",
    9:  "Orangi Town",
    10: "SITE Area",
    11: "Nazimabad",
    12: "Lyari",
    13: "Garden",
    14: "Nagan Chowrangi",
    15: "Surjani Town",
    16: "Sohrab Goth",
    17: "Superhighway",
    18: "Shah Faisal",
    19: "Quaidabad",
}

# Approximate lat/lon for heuristic calculations
COORDINATES = {
    "Saddar":           (24.8607, 67.0100),
    "Clifton":          (24.8138, 67.0300),
    "DHA Phase V":      (24.7900, 67.0600),
    "Korangi":          (24.8200, 67.1200),
    "Landhi":           (24.8500, 67.1400),
    "Malir":            (24.8900, 67.1700),
    "Gulshan-e-Iqbal":  (24.9200, 67.0900),
    "Johar More":       (24.9300, 67.0700),
    "North Nazimabad":  (24.9400, 67.0200),
    "Orangi Town":      (24.9600, 66.9700),
    "SITE Area":        (24.9000, 66.9600),
    "Nazimabad":        (24.9100, 67.0000),
    "Lyari":            (24.8800, 67.0000),
    "Garden":           (24.8700, 67.0200),
    "Nagan Chowrangi":  (24.9500, 67.0600),
    "Surjani Town":     (25.0000, 67.0100),
    "Sohrab Goth":      (24.9600, 67.0800),
    "Superhighway":     (24.9700, 67.1300),
    "Shah Faisal":      (24.9000, 67.1100),
    "Quaidabad":        (24.8600, 67.1800),
}

# Undirected edges: (node_a, node_b, weight_minutes)
EDGES = [
    ("Saddar",          "Clifton",          15),
    ("Saddar",          "Garden",           10),
    ("Saddar",          "Lyari",            15),
    ("Saddar",          "Nazimabad",        25),
    ("Clifton",         "DHA Phase V",      20),
    ("Clifton",         "Garden",           20),
    ("DHA Phase V",     "Korangi",          25),
    ("Korangi",         "Landhi",           15),
    ("Korangi",         "Shah Faisal",      20),
    ("Landhi",          "Malir",            20),
    ("Landhi",          "Quaidabad",        25),
    ("Malir",           "Shah Faisal",      20),
    ("Malir",           "Superhighway",     25),
    ("Shah Faisal",     "Gulshan-e-Iqbal",  20),
    ("Shah Faisal",     "Sohrab Goth",      20),
    ("Gulshan-e-Iqbal", "Johar More",       10),
    ("Gulshan-e-Iqbal", "Nagan Chowrangi",  15),
    ("Gulshan-e-Iqbal", "North Nazimabad",  25),
    ("Johar More",      "North Nazimabad",  15),
    ("Johar More",      "Sohrab Goth",      20),
    ("Johar More",      "Nagan Chowrangi",  10),
    ("North Nazimabad", "Nazimabad",        15),
    ("North Nazimabad", "Surjani Town",     30),
    ("North Nazimabad", "Orangi Town",      30),
    ("Nazimabad",       "Lyari",            20),
    ("Nazimabad",       "SITE Area",        15),
    ("Orangi Town",     "SITE Area",        20),
    ("Orangi Town",     "Surjani Town",     25),
    ("Orangi Town",     "DHA Phase V",      65),   # Major bottleneck
    ("SITE Area",       "Lyari",            20),
    ("SITE Area",       "Garden",           25),
    ("Sohrab Goth",     "Superhighway",     15),
    ("Sohrab Goth",     "Surjani Town",     25),
    ("Surjani Town",    "Superhighway",     20),
    ("Nagan Chowrangi", "Sohrab Goth",      15),
    ("Garden",          "Lyari",            10),
    ("Nagan Chowrangi", "North Nazimabad",  20),
]

# Bottleneck annotations (for report commentary)
BOTTLENECKS = [
    ("Orangi Town", "DHA Phase V",    "No direct bridge; Lyari Expressway detour; 65 min"),
    ("DHA Phase V", "Korangi",        "Korangi Creek Bridge; single chokepoint; floods in rain"),
    ("North Nazimabad", "Orangi Town","Orangi nullah crossing; limited bridges; chronic congestion"),
]
