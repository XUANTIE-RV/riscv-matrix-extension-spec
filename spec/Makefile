SPEC_SOURCE := spec_header.adoc
SPEC_RESULT := RISC-V-Matrix-Extension-spec.pdf

all: build_spec

build_spec:

	@echo "Building specification"
	asciidoctor-pdf \
    --attribute=mathematical-format=svg \
    --attribute=pdf-fontsdir=docs-resources/fonts \
    --attribute=pdf-style=docs-resources/themes/riscv-pdf.yml \
    --failure-level=ERROR \
    --require=asciidoctor-bibtex \
    --require=asciidoctor-diagram \
    --require=asciidoctor-mathematical \
    --out-file=$(SPEC_RESULT) \
    $(SPEC_SOURCE)

clean:
	rm $(SPEC_RESULT) 
