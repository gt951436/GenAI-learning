# pyhton code splitiing
# can be done for any language just change language = Language.LANGUAGE_NAME

from langchain.text_splitter import RecursiveCharacterTextSplitter,Language

text1 = """
def scope_test():
    def do_local():
        spam = "local spam"

    def do_nonlocal():
        nonlocal spam
        spam = "nonlocal spam"

    def do_global():
        global spam
        spam = "global spam"

    spam = "test spam"
    do_local()
    print("After local assignment:", spam)
    do_nonlocal()
    print("After nonlocal assignment:", spam)
    do_global()
    print("After global assignment:", spam)

scope_test()
print("In global scope:", spam)
""" 

text2 = """
# Vigore Pasiphaeia supponat nocti aquae quae

## Pollice populi patulos corpore sumpsit ego tota

Lorem markdownum Ligdo, da fero quoque, que campus. Quae fronte rugosis et dotem
plagas fatidicus, omnes cum trabemque Ampycides cuius arvis non, mutabitur qua
[suo](http://www.rumordeus.com/iubebitlaticesque). Ignibus per fila ignoscat et
praemia probas admirantes et grani valerem: structis duces dixit gaudia ordine
deum morte vultum? Commisit orsa cum est viresque, et quid bella visa modo, est
fuit sed Nixosque mecum. Paratis induroque ligno, parcum protinus: a vires
omnes, avoque esse!

Post per erat [hora cura](http://www.arcton.net/parenti-ora.html) ablata, in
videre fortuna aliamve! Ordine feritatis luctibus?

## Vidisset vana medullis evellere lambit induit vis

Terris Saturnus potestis tofis eratque absumptis dedisse, utraque quo illam
pariter. Post tangam ecce; fertis leves Circen, morte in equorum ulciscere
adiere et fui *pressitque glaebas*. Me dedit patri iampridem has curat urbes
currunt, mihi in audita Diamque pulsa: una conducit sanabilis iamque. Manibus
nocendo **satiatae precibus nate**: coepit, consorte et terra rapta [alterius
officioque](http://causa.net/foret-sine.php) natura, Aeacon.

- Locuta senecta
- Sic tibi et nascentur procorum legerat
- Caesa inque factaque triumphos recepit
- Magno nubemque
- Aequalibus et mihi

## Stipe sollicito cacumina nomen Serpens fatentur nobis

Inmergitque cupit, qui ramos commenta. Subito in decimae, locoque. Sed dixit
aegida et vera parentque tempore octavo criminis et Huic non *proxima*, sucis.

Amanti nereides mea fuit valentes ortus vires manibus natalis, bacis. Cognita
remugis cervice fuerant gerens, ingemuit cornua, priameia. Ipse non meus, per
fronte, nullus? Qua quoque Phoebo ne patenti miram repellit adstringit vomeribus
et nequeo quoque, ut insilit freta Methymnaeae.

## Hanc nec

Ense verba capillos quoque, silentia dature. *Casuraque* plebe floresque Oleniae
dedit undam rem voluptas artes, quoque tibi, pro esse vertigine, prior.

1. Manibus stantem libro
2. Et pallae
3. Audacissimus valet
4. Quam versat pervenit

Thalamis quae nullo circumspexit ira pariter deus transit nubigenas nunc quondam
mens sed suscipiunt insigne. Iuvenis [vocant Tiberinaque
deum](http://propriumverum.io/), boves et suam perque in provolat.
"""
splitter1 = RecursiveCharacterTextSplitter.from_language(
    language = Language.PYTHON,
    chunk_size = 300,
    chunk_overlap = 0,
)

splitter2 = RecursiveCharacterTextSplitter.from_language(
    language = Language.MARKDOWN,
    chunk_size = 500,
    chunk_overlap = 0,
)

chunks1 = splitter1.split_text(text1)
chunks2 = splitter2.split_text(text2)

#print(chunks1[0])

print(len(chunks2))
#print(chunks2[7])