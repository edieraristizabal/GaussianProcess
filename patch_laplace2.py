"""Replace Laplace paragraph with clearer version."""
with open('manuscrito.tex', 'rb') as f:
    raw = f.read()

# Find exact bytes for old paragraph
idx = raw.find(b'Dado que la verosimilitud de Bernoulli')
end_marker = b'nativo del PGC.'
idx_end = raw.find(end_marker, idx) + len(end_marker)

old_b = raw[idx:idx_end]
print('Found:', repr(old_b[:80]))

new_b = (
    b"En el PGR la verosimilitud gaussiana permite obtener la incertidumbre de forma exacta.\r\n"
    b"En el PGC, en cambio, la respuesta es binaria ---deslizamiento s\xc3\xad o no--- y no existe\r\n"
    b"una f\xc3\xb3rmula directa para calcular, dado el conjunto de entrenamiento, cu\xc3\xa1n plausible es\r\n"
    b"cada posible puntuaci\xc3\xb3n de susceptibilidad $f(\\mathbf{x})$ ---la se\xc3\xb1al continua que el modelo\r\n"
    b"genera internamente antes de convertirla a probabilidad mediante la funci\xc3\xb3n log\xc3\xadstica---.\r\n"
    b"La aproximaci\xc3\xb3n de Laplace resuelve esto en dos pasos: primero identifica el valor de\r\n"
    b"$f(\\mathbf{x})$ que mejor explica los datos observados; luego cuantifica la incertidumbre\r\n"
    b"en torno a ese valor midiendo cu\xc3\xa1n abruptamente cae la verosimilitud si uno se aleja de\r\n"
    b"\xc3\xa9l ---una ca\xc3\xadda r\xc3\xa1pida implica poca incertidumbre; una ca\xc3\xadda lenta, mucha \\citep{Rasmussen2006}---.\r\n"
    b"El resultado es que para cada celda no se obtiene un \xc3\xbanico valor sino un rango de\r\n"
    b"puntuaciones plausibles con sus respectivas probabilidades. Al transformar ese rango a\r\n"
    b"escala $[0,1]$ mediante la funci\xc3\xb3n log\xc3\xadstica se obtiene un rango de probabilidades\r\n"
    b"de susceptibilidad: celdas bien representadas por el inventario producen rangos estrechos\r\n"
    b"(predicci\xc3\xb3n confiable); celdas sin eventos pr\xc3\xb3ximos producen rangos amplios (alta\r\n"
    b"incertidumbre). La amplitud de ese rango constituye el mapa de incertidumbre nativo del PGC."
)

raw = raw[:idx] + new_b + raw[idx_end:]
with open('manuscrito.tex', 'wb') as f:
    f.write(raw)
print('OK')
