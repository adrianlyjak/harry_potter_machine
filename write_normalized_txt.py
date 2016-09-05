import os
from unidecode import unidecode
import codecs


def writefile(sourcefile, destination):
    if sourcefile is None or not os.path.exists(sourcefile):
        raise FileNotFoundError(
            'no raw.txt already created, and no source text specified or source file/folder not found')

    sourcefiles = [sourcefile]
    if os.path.isdir(sourcefile):
        sourcefiles = [fname for fname in os.listdir(sourcefile) if os.path.isfile(fname)]

    with open(destination, 'w') as outfile:
        for infilename in sourcefiles:

            # not being very efficient here
            text = unidecode(codecs.open(infilename, "r", "utf-8").read())
            while "\n\n" in text:
                text = text.replace("\n\n", "\n")
            outfile.write(text)
