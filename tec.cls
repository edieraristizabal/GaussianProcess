\LoadClass[11pt,oneside,letterpaper]{article}
\usepackage{vmargin}
\setmargins
{2.5cm}          % margen izquierdo
{1.0cm}           % margen superior
{16cm}          % anchura del texto
{22.5cm}          % altura del texto
{2.5cm}           % altura de los encabezados
{0.5cm}           % espacio entre el texto y los encabezados
{2cm}             % altura del pie de página
{1.25cm}           % espacio entre el texto y el pie de página
\usepackage{cite}
\usepackage[T1]{fontenc}
\usepackage{palatino}
\usepackage{lineno}
 \usepackage{longtable}
 \usepackage{titlesec}
\usepackage{algorithm}
\usepackage{tcolorbox}

\makeatletter
\renewcommand{\@seccntformat}[1]{}
\makeatother

\makeatletter
\renewcommand\@makefntext[1]{%
\setlength\parindent{1em}%
\noindent
\mbox{$^\@thefnmark$}{#1}}
\makeatother
\renewcommand\citeleft{(}
\renewcommand\citeright{)}
\usepackage{amsmath}
\usepackage{lipsum}
\usepackage{amsfonts}
\usepackage[table,xcdraw]{xcolor}
\usepackage{tabularx}
\usepackage{tabulary}
\usepackage{color}
\definecolor{blue2}{RGB}{3,22,240}
\definecolor{blue1}{RGB}{20,160,250}
\definecolor{tinto}{RGB}{ 144, 12, 63}
\usepackage{tikz}
\usepackage{amssymb}
\usepackage{graphicx}
\usepackage[tight,footnotesize]{subfigure} 
\usepackage{cite}
\usepackage{float}
\usepackage{multicol}
\usepackage[spanish, es-tabla,es-nodecimaldot]{babel}
\usepackage[spanish]{babel} 
\usepackage[utf8]{inputenc}
\usepackage{soul}
%\usepackage[none]{hyphenat}
\usepackage{multirow}
\usepackage[figuresright]{rotating}
\usepackage{eso-pic}
\usepackage{color}
\usepackage{fancybox}
\usepackage{wrapfig} 
\usepackage{flushend} 
\usepackage{fancyhdr}
\usepackage[table,xcdraw]{xcolor}
\RequirePackage{titlesec}
\titleformat*{\subsubsection}{\it\large\bfseries}
%\renewcommand\thetable{\textbf{\Roman{table}}} 

\newcommand{\myhyperlink}[2]{%
    \hypersetup{linkcolor=blue1}%
    \hyperlink{#1}{#2}%
    \hypersetup{linkcolor=black}%
}

\addto\captionsenglish{\renewcommand{\figurename}{Fig.}}
\renewcommand{\figurename}{\textbf{Figura}} 
\renewcommand\thefigure{\textbf{\arabic{figure}}} 
\renewcommand{\tablename}{\textbf{Tabla}} 
\renewcommand{\thetable}{\arabic{table}}
%\renewcommand\thetable{\textbf{\Roman{table}}} 
%%---ENCABEZADO Y PIE DE PAGINA PRIMER PAGINA---%%
%%----------------------------------------------%%

\setcounter{page}{1}
\pagestyle{fancy}
\fancyhf{}
\fancyhead[C]{}
\fancyhead[C]{ \fboxsep 8pt 
%\mbox{%
\hspace{-1.5cm}
\begin{minipage}[b][2cm][l]{2cm}
\includegraphics[scale=0.12]{escudo.png}
\end{minipage}
%}
%\mbox {%
\hspace{0.5cm}
\begin{minipage}[b][0.8cm][c]{12cm}
\begin{center}
{\fontencoding{T1}
\fontfamily{pag}
\fontseries{b}
\fontshape{n}
\fontsize{22}{24}
\setlength{\parfillskip}{5pt plus 1fil}
\selectfont
\bf  Tecnura}
\\
\centering
\vspace{0.15cm}
\color{blue1}
\footnotesize{\href{https://revistas.udistrital.edu.co/ojs/index.php/Tecnura/issue/view/1253}{https://revistas.udistrital.edu.co/ojs/index.php/Tecnura/issue/view/1253}}\\\vspace{0.15cm}
\small{{\color{black}DOI:} \href{https://doi.org/10.14483/22487638.xxxxx}{https://doi.org/10.14483/22487638.xxxxx}\\}
\end{center}
\end{minipage}
%}
\\
}




\fancyfoot[C]{\scriptsize{{\bf Tecnura} • p-ISSN: 0123-921X • e-ISSN: 2248-7638 • Volumen 29 Número 84 • Enero - Marzo  de 2025 • pp. xx-xx \\}\normalsize[\thepage]}


