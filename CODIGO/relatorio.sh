jupyter nbconvert  --to latex --template citations.tplx Projeto.ipynb
latex Projeto.tex
bibtex Projeto.aux
pdflatex Projeto.tex
pdflatex Projeto.tex
pdflatex Projeto.tex
