"""Move target encoding block from PGR section to general GP section."""
with open('manuscrito.tex', 'rb') as f:
    raw = f.read()

# ---- Block to move (target encoding, lines 304-323) ----
block = (
    b"Las funciones de covarianza continuas ---incluido el Mat\xc3\xa9rn-$\\nicefrac{3}{2}$--- est\xc3\xa1n definidas sobre espacios de entrada continuos y\r\n"
    b"computan similitudes mediante distancias euclideanas; las variables categ\xc3\xb3ricas nominales carecen de una m\xc3\xa9trica de distancia\r\n"
    b"inherente y requieren transformaci\xc3\xb3n previa al modelado \\citep{GarridoMerchan2020, Rasmussen2006}. Las dos alternativas\r\n"
    b"principales son: (i)~la \\textit{codificaci\xc3\xb3n binaria} (\\textit{one-hot}), que genera un indicador 0/1 por categor\xc3\xada; y \r\n"
    b"(ii)~las \\textit{funciones de covarianza categ\xc3\xb3ricas}, que tratan todos los pares de clases como\r\n"
    b"equidistantes e ignoran gradientes de susceptibilidad entre unidades geol\xc3\xb3gicas o de cobertura.\r\n"
    b"\r\n"
    b"Se adopt\xc3\xb3 en cambio el \\textit{target encoding} \\citep{MicciBarreca2001}, que reemplaza cada categor\xc3\xada por la proporci\xc3\xb3n\r\n"
    b"observada de deslizamientos dentro de esa clase en el conjunto de entrenamiento:\r\n"
    b"\r\n"
    b"\\begin{equation}\r\n"
    b"    X'_{\\mathrm{cat}} = \\frac{1}{n_{\\mathrm{cat}}} \\sum_{i \\in \\mathrm{cat}} y_i\r\n"
    b"    \\label{eq:target}\r\n"
    b"\\end{equation}\r\n"
    b"donde $n_{\\mathrm{cat}}$ es el n\xc3\xbamero de celdas de entrenamiento en la categor\xc3\xada y $y_i \\in \\{0,1\\}$ es el indicador binario de\r\n"
    b"deslizamiento. Esta estrategia presenta tres ventajas sobre las alternativas: (1)~mantiene la parsimonia del modelo, reduciendo\r\n"
    b"las covariables y dimensiones y haciendo viable la optimizaci\xc3\xb3n de hiperpar\xc3\xa1metros; (2)~la codificaci\xc3\xb3n resultante es directamente\r\n"
    b"interpretable como la probabilidad emp\xc3\xadrica de falla de cada unidad litol\xc3\xb3gica o de cobertura, aportando significado f\xc3\xadsico a la\r\n"
    b"longitud de escala ARD de esas variables; y (3)~es compatible con implementaciones est\xc3\xa1ndar del PG sin requerir funciones de covarianza\r\n"
    b"especializadas.\r\n"
)

# ---- Anchor: insert BEFORE this sentence ----
anchor = (
    b"La diferencia fundamental entre un PGR y un PGC radica en la naturaleza de la variable respuesta:"
)

# ---- PGR closing line (what comes right before the block in current file) ----
pgr_close = (
    b"Se utilizaron tres covariables: pendiente, geolog\xc3\xada y cobertura del suelo.\r\n"
    b"\r\n"
)

if block not in raw:
    print("BLOCK NOT FOUND")
    exit()

if anchor not in raw:
    print("ANCHOR NOT FOUND")
    exit()

# 1. Remove block from current position (leave PGR closing intact)
raw = raw.replace(b"\r\n" + block, b"", 1)

# 2. Insert block (with blank line before and after) just before the anchor
insert = b"\r\n" + block + b"\r\n"
raw = raw.replace(anchor, insert + anchor, 1)

with open('manuscrito.tex', 'wb') as f:
    f.write(raw)
print('Done.')
