
os.execute ("pdflatex --interaction=nonstopmode Main.tex")
os.execute ("biber Main")
os.execute ("pdflatex --interaction=nonstopmode Main.tex")

Clean = {"aux", "nlo", "out", "bbl", "bcf", "blg", "xml"}
for _, extension in ipairs (Clean) do
    os.execute ("rm *."..extension)
end
