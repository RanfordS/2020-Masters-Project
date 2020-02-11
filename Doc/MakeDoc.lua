make_pdf = "pdflatex --shell-escape --interaction=nonstopmode Main.tex"

print ("\n\27[1;4;33mDoing Repeat Builds\27[0m\n")
for i = 1,0 do
    print ("iteration:", i)
    os.execute (make_pdf.." > /dev/null")
    os.execute ("biber Main > /dev/null")
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
    os.execute ("rm *."..extension)
end
