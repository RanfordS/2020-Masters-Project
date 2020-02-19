jobname = "NeuralNetworksWithPythonAndTensorflow"

make_pdf = 'pdflatex --shell-escape --interaction=nonstopmode -jobname="'..jobname..'" Main.tex'
make_bib = "biber "..jobname
make_pdf_null = make_pdf.." > /dev/null"
make_bib_null = make_bib.." > /dev/null"

print ("\n\27[1;4;33mDoing Repeat Builds\27[0m\n")
for i = 1,2 do
    print ("iteration:", i)
    os.execute (make_pdf_null)
    os.execute (make_bib_null)
end

print ("\n\27[1;4;33mFinal Build Log\27[0m\n")
res = io.popen (make_pdf)
txt = res:read ("*a")
res:close ()
print ("\27[3m"..txt.."\27[0m")

print ("\n\27[1;4;33mTODO:\27[0m\n")
for warn in txt:gmatch ("Package TODO Warning: (.-)%.") do
    print ("\27[1;35m"..warn:gsub('\n','').."\27[0m\n")
end

Clean = {"aux", "nlo", "out", "bbl", "bcf", "blg", "xml", "toc"}
for _, extension in ipairs (Clean) do
    os.execute ("rm -f *."..extension)
end
