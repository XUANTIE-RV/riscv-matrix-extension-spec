HEADER_SOURCE := header.adoc
PDF_RESULT := riscv-matrix-spec.pdf

all: build

build:

	@echo "Building asciidoc"
	asciidoctor-pdf --trace \
    --attribute=mathematical-format=svg \
    -a pdf-theme=riscv-matrix.yml \
    -a pdf-fontsdir=GEM_FONTS_DIR \
    --failure-level=ERROR \
    --require=asciidoctor-bibtex \
    --require=asciidoctor-diagram \
    --require=asciidoctor-mathematical \
    --out-file=$(PDF_RESULT) \
    $(HEADER_SOURCE)

clean:
	rm $(PDF_RESULT)
