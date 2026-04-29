"""Replace Laplace paragraph in GPC section with clearer version."""
with open('manuscrito.tex', 'rb') as f:
    raw = f.read()

old_b = (
    b"Dado que la verosimilitud de Bernoulli no es conjugada con el prior gaussiano, la distribuci\xc3\xb3n posterior\r\n"
    b"$p(f\\mid\\mathbf{X},\\mathbf{y})$ no tiene forma cerrada y se aproxima mediante el m\xc3\xa9todo de Laplace, que\r\n"
    b"ajusta una distribuci\xc3\xb3n gaussiana alrededor del modo de la posterior \\citep{Rasmussen2006}. La\r\n"
    b"incertidumbre posterior de la funci\xc3\xb3n latente se propaga a trav\xc3\xa9s de la funci\xc3\xb3n log\xc3\xadstica para obtener la\r\n"
    b"distribuci\xc3\xb3n de probabilidad de susceptibilidad en cada celda, constituyendo el mapa de\r\n"
    b"incertidumbre nativo del modelo."
)

new_b = (
    b"Dado que la verosimilitud de Bernoulli no es conjugada con el prior gaussiano, la distribuci\xc3\xb3n posterior\r\n"
    b"$p(f\\mid\\mathbf{X},\\mathbf{y})$ no tiene forma anal\xc3\xadtica exacta. La aproximaci\xc3\xb3n de Laplace resuelve esto\r\n"
    b"localizando el modo $\\mathbf{f}^*$ de la posterior ---el vector de valores latentes que la maximiza--- y\r\n"
    b"ajustando una gaussiana centrada en ese modo usando la curvatura local como medida de dispersi\xc3\xb3n\r\n"
    b"\\citep{Rasmussen2006}. Para cada celda de predicci\xc3\xb3n $\\mathbf{x}^*$ se obtiene as\xc3\xad\r\n"
    b"$f(\\mathbf{x}^*) \\sim \\mathcal{N}(\\mu^*, \\sigma^{*2})$: no un valor \xc3\xbanico sino una distribuci\xc3\xb3n. Al\r\n"
    b"integrar esta distribuci\xc3\xb3n a trav\xc3\xa9s de la funci\xc3\xb3n log\xc3\xadstica se obtiene la probabilidad de deslizamiento\r\n"
    b"junto con su incertidumbre: una varianza peque\xc3\xb1a $\\sigma^{*2}$ indica que el dato de entrenamiento\r\n"
    b"circundante restringe bien la estimaci\xc3\xb3n (predicci\xc3\xb3n confiable); una varianza grande indica una zona\r\n"
    b"pobremente cubierta por el inventario (alta incertidumbre). La desviaci\xc3\xb3n est\xc3\xa1ndar de esta distribuci\xc3\xb3n\r\n"
    b"de salida constituye el mapa de incertidumbre nativo del PGC."
)

if old_b in raw:
    raw = raw.replace(old_b, new_b, 1)
    with open('manuscrito.tex', 'wb') as f:
        f.write(raw)
    print('OK')
else:
    print('NOT FOUND')
