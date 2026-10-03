"""Post-process a pandoc-generated DOCX for PLOS ONE formatting.

Adds, by editing the OOXML package directly (no python-docx dependency):
  * double-spaced body text (styles.xml docDefaults),
  * continuous line numbers (sectPr),
  * a centred page-number footer (footer1.xml + content type + relationship).

Usage:  python plos_format_docx.py <file.docx>
"""

import os
import re
import zipfile

FOOTER = (
    '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
    '<w:ftr xmlns:w="http://schemas.openxmlformats.org/wordprocessingml/2006/main"'
    ' xmlns:r="http://schemas.openxmlformats.org/officeDocument/2006/relationships">'
    '<w:p><w:pPr><w:jc w:val="center"/></w:pPr>'
    '<w:r><w:fldChar w:fldCharType="begin"/></w:r>'
    '<w:r><w:instrText xml:space="preserve">PAGE</w:instrText></w:r>'
    '<w:r><w:fldChar w:fldCharType="separate"/></w:r>'
    '<w:r><w:t>1</w:t></w:r>'
    '<w:r><w:fldChar w:fldCharType="end"/></w:r></w:p></w:ftr>'
)
CT_OVERRIDE = ('<Override PartName="/word/footer1.xml" ContentType='
               '"application/vnd.openxmlformats-officedocument.wordprocessingml.footer+xml"/>')
REL = ('<Relationship Type="http://schemas.openxmlformats.org/officeDocument/2006/'
       'relationships/footer" Id="rIdFooter1" Target="footer1.xml"/>')
FOOTER_REF = '<w:footerReference w:type="default" r:id="rIdFooter1"/>'
LNNUM = '<w:lnNumType w:countBy="1" w:restart="continuous" w:distance="360"/>'


def plos_format_docx(path):
    with zipfile.ZipFile(path, "r") as zin:
        order = zin.namelist()
        items = {n: zin.read(n) for n in order}

    dec = lambda b: b.decode("utf-8")  # noqa: E731
    enc = lambda s: s.encode("utf-8")  # noqa: E731

    # 1) double spacing
    st = dec(items["word/styles.xml"])
    st = re.sub(r'<w:spacing w:after="200" />',
                '<w:spacing w:after="120" w:line="480" w:lineRule="auto" />', st, count=1)
    if 'w:line="480"' not in st:
        st = st.replace('<w:pPrDefault><w:pPr>',
                        '<w:pPrDefault><w:pPr>'
                        '<w:spacing w:after="120" w:line="480" w:lineRule="auto"/>', 1)
    items["word/styles.xml"] = enc(st)

    # 2) continuous line numbers + default footer reference
    doc = dec(items["word/document.xml"])
    if "w:lnNumType" not in doc:
        doc = doc.replace("</w:sectPr>", LNNUM + "</w:sectPr>")
    if "footerReference" not in doc:
        doc = re.sub(r"(<w:sectPr[^>]*>)", r"\1" + FOOTER_REF, doc, count=1)
    items["word/document.xml"] = enc(doc)

    # 3) footer part
    items["word/footer1.xml"] = enc(FOOTER)

    # 4) content types
    ct = dec(items["[Content_Types].xml"])
    if "footer1.xml" not in ct:
        ct = ct.replace("</Types>", CT_OVERRIDE + "</Types>")
    items["[Content_Types].xml"] = enc(ct)

    # 5) relationship
    rels = dec(items["word/_rels/document.xml.rels"])
    if "footer1.xml" not in rels:
        rels = rels.replace("</Relationships>", REL + "</Relationships>")
    items["word/_rels/document.xml.rels"] = enc(rels)

    tmp = path + ".tmp"
    with zipfile.ZipFile(tmp, "w", zipfile.ZIP_DEFLATED) as zout:
        for n in order:
            zout.writestr(n, items[n])
        if "word/footer1.xml" not in order:
            zout.writestr("word/footer1.xml", items["word/footer1.xml"])
    os.replace(tmp, path)


if __name__ == "__main__":
    import sys
    plos_format_docx(sys.argv[1])
    print("formatted", sys.argv[1])
