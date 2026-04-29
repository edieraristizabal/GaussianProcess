"""Insert covariance function motivation before Matern sentence."""
with open('manuscrito.tex', 'rb') as f:
    raw = f.read()

old_b = (
    b"correlacionadas. Se emple\xc3\xb3 el n\xc3\xbacleo Matern-$\\nicefrac{3}{2}$ con\r\n"
    b"Determinaci\xc3\xb3n Autom\xc3\xa1tica de Relevancia (ARD) \\citep{Matern1960, Rasmussen2006}:"
)

new_b = (
    b"correlacionadas. "
    b"La elecci\xc3\xb3n de $k$ es la principal decisi\xc3\xb3n de dise\xc3\xb1o del PG: "
    b"determina la suavidad de la superficie de susceptibilidad, el alcance de la correlaci\xc3\xb3n\r\n"
    b"entre sitios en el espacio de covariables y la capacidad del modelo para capturar patrones no lineales.\r\n"
    b"En ese sentido, cumple el mismo papel estructural que la forma funcional en la regresi\xc3\xb3n log\xc3\xadstica\r\n"
    b"o las reglas de partici\xc3\xb3n en los bosques aleatorios.\r\n"
    b"Se emple\xc3\xb3 el n\xc3\xbacleo Mat\xc3\xa9rn-$\\nicefrac{3}{2}$ con\r\n"
    b"Determinaci\xc3\xb3n Autom\xc3\xa1tica de Relevancia (ARD) \\citep{Matern1960, Rasmussen2006}:"
)

if old_b in raw:
    raw = raw.replace(old_b, new_b, 1)
    with open('manuscrito.tex', 'wb') as f:
        f.write(raw)
    print('OK')
else:
    print('NOT FOUND')
    idx = raw.find(b'correlacionadas.')
    print('FILE bytes:', raw[idx:idx+120])
    print('OLD bytes: ', old_b[:80])
