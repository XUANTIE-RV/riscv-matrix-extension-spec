SHL_SOURCE := shl/shl_header.adoc
SHL_RESULT := SHL-USER-GUIDE.pdf
ABI_SOURCE := abi/abi_header.adoc
ABI_RESULT := Matrix-Extension-ABI-Manual.pdf
ITS_SOURCE := intrinsic/intrinsic_header.adoc
ITS_RESULT := Matrix-Extension-intrinsic-API-Reference-Manual.pdf

all: build_shl\
	build_its\
	build_abi

shl: build_shl

its: build_its

abi: build_abi

build_shl:

	@echo "Building shl user guide"
	asciidoctor-pdf \
    --attribute=mathematical-format=svg \
    --attribute=pdf-fontsdir=docs-resources/fonts \
    --attribute=pdf-style=docs-resources/themes/riscv-pdf.yml \
    --failure-level=ERROR \
    --require=asciidoctor-bibtex \
    --require=asciidoctor-diagram \
    --require=asciidoctor-mathematical \
    --out-file=$(SHL_RESULT) \
    $(SHL_SOURCE)

build_its:

	@echo "Building intrinsic manual"
	asciidoctor-pdf \
    --attribute=mathematical-format=svg \
    --attribute=pdf-fontsdir=docs-resources/fonts \
    --attribute=pdf-style=docs-resources/themes/riscv-pdf.yml \
    --failure-level=ERROR \
    --require=asciidoctor-bibtex \
    --require=asciidoctor-diagram \
    --require=asciidoctor-mathematical \
    --out-file=$(ITS_RESULT) \
    $(ITS_SOURCE)

build_abi:

	@echo "Building abi manual"
	asciidoctor-pdf \
    --attribute=mathematical-format=svg \
    --attribute=pdf-fontsdir=docs-resources/fonts \
    --attribute=pdf-style=docs-resources/themes/riscv-pdf.yml \
    --failure-level=ERROR \
    --require=asciidoctor-bibtex \
    --require=asciidoctor-diagram \
    --require=asciidoctor-mathematical \
    --out-file=$(ABI_RESULT) \
    $(ABI_SOURCE)

clean:
	rm $(SHL_RESULT) $(ITS_RESULT) $(ABI_RESULT)
