PDF_ENGINE = tectonic  # or lualatex if you prefer
# PDF_ENGINE = xelatex  # or lualatex if you prefer
PANDOC_FLAGS = --from markdown \
               --output paper.pdf \
               --template=templates/default.latex \
               --pdf-engine=$(PDF_ENGINE) \
               --citeproc \
               --bibliography=paper.bib \
               -V geometry:margin=1in

pdf:
	pandoc paper.md $(PANDOC_FLAGS)

clean:
	rm paper.pdf