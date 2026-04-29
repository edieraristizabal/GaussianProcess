"""Restructure GP methodology sections in manuscrito.tex."""
import re

with open('manuscrito.tex', 'r', encoding='utf-8') as f:
    c = f.read()

# -------------------------------------------------------
# New general GP section text
# -------------------------------------------------------
new_gp_section = (
    "\n"
    "\\textbf{Marco formal de los Procesos Gaussianos.}\n"
    "Un Proceso Gaussiano (PG) es una distribucion de probabilidad definida directamente sobre funciones\n"
    "\\citep{Rasmussen2006}. A diferencia de los modelos convencionales ---que producen un unico valor de\n"
    "susceptibilidad por celda---, el PG asigna a cada punto del espacio de covariables una distribucion\n"
    "de probabilidad completa: la \\textit{media posterior} es la estimacion central de susceptibilidad y la\n"
    "\\textit{desviacion estandar posterior} cuantifica de forma nativa la incertidumbre epistemica, es decir,\n"
    "cuan pobremente esta restringida esa estimacion por los datos de entrenamiento disponibles.\n"
    "\n"
    "La funcion latente $f:\\mathbb{R}^D\\!\\to\\mathbb{R}$ que relaciona las $D$ covariables del terreno con el\n"
    "indice de susceptibilidad se modela mediante el prior:\n"
    "\\begin{equation}\n"
    "    f(\\mathbf{x}) \\sim \\mathcal{GP}\\!\\bigl(0,\\; k(\\mathbf{x},\\mathbf{x}')\\bigr)\n"
    "    \\label{eq:gp_prior}\n"
    "\\end{equation}\n"
    "donde se asume media cero y la funcion de covarianza $k(\\mathbf{x},\\mathbf{x}')$ codifica la similitud esperada entre\n"
    "los valores de $f$ en dos puntos del espacio de covariables: sitios con condiciones morfologicas y ambientales\n"
    "similares tendran susceptibilidades correlacionadas. Se empleo el nucleo Matern-$\\nicefrac{3}{2}$ con\n"
    "Determinacion Automatica de Relevancia (ARD) \\citep{Matern1960, Rasmussen2006}:\n"
    "\\begin{equation}\n"
    "    k(\\mathbf{x},\\mathbf{x}') = \\sigma_f^2\\!\\left(1 + \\sqrt{3}\\,r_l\\right)\\exp\\!\\left(-\\sqrt{3}\\,r_l\\right),\n"
    "    \\qquad\n"
    "    r_l = \\sqrt{\\sum_{d=1}^{D}\\!\\left(\\frac{x_d - x'_d}{l_d}\\right)^{\\!2}}\n"
    "    \\label{eq:matern}\n"
    "\\end{equation}\n"
    "donde $\\sigma_f^2$ es la varianza senal y $l_d$ es la longitud de escala de la covariable $d$. La condicion\n"
    "ARD otorga a cada variable su propia escala de influencia: una longitud de escala pequena indica que esa\n"
    "covariable es muy relevante para discriminar la susceptibilidad; la importancia relativa se estima como\n"
    "$w_d = 1/l_d^*$, donde $l_d^*$ es el valor optimo. Los hiperparametros\n"
    "$\\boldsymbol{\\theta}=\\{\\sigma_f^2, l_1,\\ldots,l_D\\}$ se estiman maximizando la log-verosimilitud marginal:\n"
    "\\begin{equation}\n"
    "    \\boldsymbol{\\theta}^* = \\arg\\max_{\\boldsymbol{\\theta}}\\;\\log p\\!\\left(\\mathbf{y}\\mid\\mathbf{X},\\boldsymbol{\\theta}\\right)\n"
    "    \\label{eq:mle}\n"
    "\\end{equation}\n"
    "Este criterio balancea automaticamente el ajuste a los datos y la complejidad del modelo sin necesidad de un\n"
    "conjunto de validacion separado.\n"
    "\n"
    "El costo computacional escala cubicamente con el tamano del conjunto de entrenamiento ($\\mathcal{O}(n^3)$).\n"
    "Por ello, y dado el fuerte desequilibrio de clases (583 celdas con deslizamiento frente a mas de\n"
    "2{,}087{,}000 de fondo) \\citep{Chawla2002, Heckmann2014}, se conforme un conjunto de entrenamiento\n"
    "balanceado de 500 puntos: 250 celdas con deslizamiento y 250 de fondo, seleccionadas aleatoriamente\n"
    "con semilla fija para reproducibilidad. Este conjunto es compartido por el PGR y el PGC. La prediccion\n"
    "se realizo en bloques de 50{,}000 celdas sobre las 2{,}087{,}643 celdas validas de la cuenca.\n"
    "\n"
    "La diferencia fundamental entre ambos modelos radica en la naturaleza de la variable respuesta: continua\n"
    "en el PGR y binaria en el PGC. Esta caracteristica determina la funcion de verosimilitud adoptada y, con\n"
    "ello, la forma de la distribucion posterior.\n"
    "\n"
)

new_pgr = (
    "\\textbf{Proceso Gaussiano de Regresion (PGR).} En el PGR la variable respuesta es\n"
    "\\textit{continua}: la densidad espacial del inventario $\\hat{f}(\\mathbf{s})$ calculada mediante KDE\n"
    "(Ec.~\\ref{eq:kde}). Se asume una verosimilitud gaussiana con ruido de observacion independiente:\n"
    "\\begin{equation}\n"
    "    y_i = f(\\mathbf{x}_i) + \\varepsilon_i, \\qquad \\varepsilon_i \\sim \\mathcal{N}(0,\\sigma_n^2)\n"
    "    \\label{eq:pgr}\n"
    "\\end{equation}\n"
    "Bajo esta verosimilitud, la distribucion posterior $p(f\\mid\\mathbf{X},\\mathbf{y})$ es gaussiana y se\n"
    "obtiene analiticamente sin necesidad de aproximaciones. La media posterior proporciona el mapa de\n"
    "susceptibilidad y la varianza posterior cuantifica la incertidumbre de la estimacion en cada celda.\n"
    "Se utilizaron tres covariables: pendiente, geologia y cobertura del suelo.\n"
)

new_pgc = (
    "\\textbf{Proceso Gaussiano de Clasificacion (PGC).} En el PGC la variable respuesta es\n"
    "\\textit{binaria}: presencia ($y=1$) o ausencia ($y=0$) de deslizamiento en cada celda. La verosimilitud\n"
    "gaussiana del PGR no es apropiada para este caso; en su lugar se adopta la verosimilitud de Bernoulli,\n"
    "conectando la funcion latente $f(\\mathbf{x})$ con la probabilidad de ocurrencia mediante la funcion logistica:\n"
    "\\begin{equation}\n"
    "    p(y=1\\mid\\mathbf{x}) = \\sigma\\!\\left(f(\\mathbf{x})\\right) = \\frac{1}{1 + \\exp\\!\\left(-f(\\mathbf{x})\\right)}\n"
    "    \\label{eq:gpc}\n"
    "\\end{equation}\n"
    "Dado que la verosimilitud de Bernoulli no es conjugada con el prior gaussiano, la distribucion posterior\n"
    "$p(f\\mid\\mathbf{X},\\mathbf{y})$ no tiene forma cerrada y se aproxima mediante el metodo de Laplace, que\n"
    "ajusta una distribucion gaussiana alrededor del modo de la posterior \\citep{Rasmussen2006}. La\n"
    "incertidumbre posterior de la funcion latente se propaga a traves de la funcion logistica para obtener la\n"
    "distribucion de probabilidad de susceptibilidad en cada celda, constituyendo el mapa de\n"
    "incertidumbre nativo del modelo.\n"
)

# Find and replace the old PGR block (with target encoding still after it)
old_pgr_start = "\\textbf{Proceso Gaussiano de Regresión (PGR).} La primera estrategia de modelado"
old_pgr_end   = "entrenamiento balanceado de 500 puntos."

idx_start = c.find(old_pgr_start)
idx_end   = c.find(old_pgr_end)

if idx_start == -1:
    print("PGR start NOT FOUND")
elif idx_end == -1:
    print("PGR end NOT FOUND")
else:
    idx_end += len(old_pgr_end)
    old_pgr_block = c[idx_start:idx_end]
    c = c.replace(old_pgr_block, new_gp_section + new_pgr)
    print("PGR block replaced OK")

# Find and replace old PGC block
old_pgc_start = "\\textbf{Proceso Gaussiano de Clasificación (PGC).} El PGC modela"
old_pgc_end   = "proxy de la incertidumbre epistémica."

idx_start = c.find(old_pgc_start)
idx_end   = c.find(old_pgc_end)

if idx_start == -1:
    print("PGC start NOT FOUND")
elif idx_end == -1:
    print("PGC end NOT FOUND")
else:
    idx_end += len(old_pgc_end)
    old_pgc_block = c[idx_start:idx_end]
    c = c.replace(old_pgc_block, new_pgc)
    print("PGC block replaced OK")

with open('manuscrito.tex', 'w', encoding='utf-8') as f:
    f.write(c)
print("File written.")
