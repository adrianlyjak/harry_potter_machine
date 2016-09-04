from unidecode import unidecode
import codecs


text = unidecode(codecs.open("data/hp-dirty.txt", "r", "utf-8").read())
while "\n\n" in text:
  text = text.replace("\n\n", "\n")

with open("data/hp.txt", "w") as f:
  f.write(text)