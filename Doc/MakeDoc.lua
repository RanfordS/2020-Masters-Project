make_pdf = "pdflatex --shell-escape --interaction=nonstopmode Main.tex"

os.execute (make_pdf)
os.execute ("biber Main")
os.execute (make_pdf)

Clean = {"aux", "nlo", "out", "bbl", "bcf", "blg", "xml"}
for _, extension in ipairs (Clean) do
    os.execute ("rm *."..extension)
end
