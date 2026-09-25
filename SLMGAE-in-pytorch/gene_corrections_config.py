#!/usr/bin/env python3
"""
Centralized gene corrections and non-coding gene lists.
This ensures consistency across all scripts in the pipeline.
All corrections have been verified to map to genes with UniProt protein sequences.
Last verified: 2024
FILTERED VERSION: Only includes genes from the missing genes list
"""
# Gene name corrections - maps old names to current official names
# FILTERED: Only includes genes that appear in the missing genes list
GENE_CORRECTIONS = {
    # MARCH genes (only the ones in missing list)
    "MAR-02": "MARCHF2",  # https://www.uniprot.org/uniprotkb/Q9P0N8
    "MAR-05": "MARCHF5",  # https://www.uniprot.org/uniprotkb/Q9NX47
    "MAR-06": "MARCHF6",  # https://www.uniprot.org/uniprotkb/O60337
    "MAR-07": "MARCHF7",  # https://www.uniprot.org/uniprotkb/Q9H992
    # SEP genes → SEPT (Septin) family (only the ones in missing list)
    "SEP-01": "SEPT1",  # https://www.uniprot.org/uniprotkb/Q8WYJ6
    "SEP-04": "SEPT4",  # https://www.uniprot.org/uniprotkb/O43236
    "SEP-06": "SEPT6",  # https://www.uniprot.org/uniprotkb/Q14141
    "SEP-14": "SEPT14",  # https://www.uniprot.org/uniprotkb/Q6ZU15
    "SETP10":
    "SEPT10",  # Typo correction, https://www.uniprot.org/uniprotkb/Q9P0V9
    # Other corrections
    "TCL6":
    "TCL6f4",  # T-cell receptor variant (105 aa), https://www.uniprot.org/uniprotkb/A0A024R6J5
    "PRPF4B": "PRP4K",  # https://www.uniprot.org/uniprotkb/Q13523/entry
    "ANKRD32": "SLF1",  # https://www.uniprot.org/uniprotkb/Q9BQI6/entry
    "APITD1": "CENPS",  # https://www.uniprot.org/uniprotkb/Q8N2Z9/entry
    "ATP5J2": "ATP5MF",  # https://www.uniprot.org/uniprotkb/P56134/entry
    "BIVM.ERCC5": "BIVM",  # https://www.uniprot.org/uniprotkb/A0A8C5Y5U2/entry
    "BRE": "BABAM2",  # https://www.uniprot.org/uniprotkb/Q9NXR7/entry
    "C11orf30": "EMSY",  # https://www.uniprot.org/uniprotkb/Q7Z589/entry
    "C12orf45": "NOPCHAP1",  # https://www.uniprot.org/uniprotkb/Q8N5I9/entry
    "C12orf5": "TIGAR",  # https://www.uniprot.org/uniprotkb/Q9NQ88/entry
    "C12orf66": "KICS2",  # https://www.uniprot.org/uniprotkb/Q96MD2/entry
    "C14orf2": "ATP5MJ",  # https://www.uniprot.org/uniprotkb/P56378/entry
    "C14orf39": "SIX6OS1",  # https://www.uniprot.org/uniprotkb/Q8N1H7/entry
    "C16orf59": "TEDC2",  # https://www.uniprot.org/uniprotkb/Q7L2K0/entry
    "C17orf104": "MEIOC",  # https://www.uniprot.org/uniprotkb/A2RUB1/entry
    "C17orf53": "HROB",  # https://www.uniprot.org/uniprotkb/Q8N3J3/entry
    "C17orf70": "FAAP100",  # https://www.uniprot.org/uniprotkb/Q0VG06/entry
    "C19orf40": "FAAP24",  # https://www.uniprot.org/uniprotkb/Q9BTP7/entry
    "C1orf109": "AIRIM",  # https://www.uniprot.org/uniprotkb/Q9NX04/entry
    "C1orf86": "FAAP20",  # https://www.uniprot.org/uniprotkb/Q6NZ36/entry
    "C20orf196": "SHLD1",  # https://www.uniprot.org/uniprotkb/Q8IYI0/entry
    "C4orf27": "HPF1",  # https://www.uniprot.org/uniprotkb/Q9NWY4/entry
    "C5orf45": "MRNIP",  # https://www.uniprot.org/uniprotkb/Q6NTE8/entry
    "C6orf203": "MTRES1",  # https://www.uniprot.org/uniprotkb/Q9P0P8/entry
    "C7orf26": "INTS15",  # https://www.uniprot.org/uniprotkb/Q96N11/entry
    "C7orf49": "CYREN",  # https://www.uniprot.org/uniprotkb/Q9BWK5/entry
    "C9orf114": "SPOUT1",  # https://www.uniprot.org/uniprotkb/Q5T280/entry
    "C9orf142": "PAXX",  # https://www.uniprot.org/uniprotkb/Q9BUH6/entry
    "C9orf41": "CARNMT1",  # https://www.uniprot.org/uniprotkb/Q8N4J0/entry
    "CCDC155": "KASH5",  # https://www.uniprot.org/uniprotkb/Q8N6L0/entry
    "CCDC84": "CENATAC",  # https://www.uniprot.org/uniprotkb/Q86UT8/entry
    "CIRH1A": "UTP4",  # https://www.uniprot.org/uniprotkb/Q969X6/entry
    "CXorf57": "RADX",  # https://www.uniprot.org/uniprotkb/Q6NSI4/entry
    "DARS": "DARS1",  # https://www.uniprot.org/uniprotkb/P14868/entry
    "FAM175A": "ABRAXAS1",  # https://www.uniprot.org/uniprotkb/Q6UWZ7/entry
    "FAM178A": "SLF2",  # https://www.uniprot.org/uniprotkb/Q8IX21/entry
    "FAM19A2": "TAFA2",  # https://www.uniprot.org/uniprotkb/Q8N3H0/entry
    "FAM208A": "TASOR",  # https://www.uniprot.org/uniprotkb/Q9UK61/entry
    "FAM35A": "SHLD2",  # https://www.uniprot.org/uniprotkb/Q86V20/entry
    "FAM96B": "CIAO2B",  # https://www.uniprot.org/uniprotkb/Q9Y3D0/entry
    "GLTSCR2": "NOP53",  # https://www.uniprot.org/uniprotkb/Q9NZM5/entry
    "HIST3H2A": "H2AC25",  # https://www.uniprot.org/uniprotkb/Q7L7L0/entry
    "HIST3H3": "H3-4",  # https://www.uniprot.org/uniprotkb/Q16695/entry
    "HIST4H4": "H4C1",  # https://www.uniprot.org/uniprotkb/P62805/entry
    "ICT1": "MRPL58",  # https://www.uniprot.org/uniprotkb/Q14197/entry
    "KIAA0020": "PUM3",  # https://www.uniprot.org/uniprotkb/Q15397/entry
    "KIAA0101": "PCLAF",  # https://www.uniprot.org/uniprotkb/Q15004/entry
    "KIAA0430": "MARF1",  # https://www.uniprot.org/uniprotkb/Q9Y4F3/entry
    "KIAA1731": "CEP295",  # https://www.uniprot.org/uniprotkb/Q9C0D2/entry
    "LARS": "LARS1",  # https://www.uniprot.org/uniprotkb/Q9P2J5/entry
    "Ltn1": "LTN1",  # https://www.uniprot.org/uniprotkb/O94822/entry
    "MARS": "MARS1",  # https://www.uniprot.org/uniprotkb/P56192/entry
    "MB21D1": "CGAS",  # https://www.uniprot.org/uniprotkb/Q8N884/entry
    "MINOS1": "MICOS10",  # https://www.uniprot.org/uniprotkb/Q5TGZ0/entry
    "MINOS1-NBL1":
    "MICOS10-NBL1",  # https://www.uniprot.org/uniprotkb/R4GMY4/entry
    "MRE11A": "MRE11",  # https://www.uniprot.org/uniprotkb/P49959/entry
    "MRP63": "MRPL57",  # https://www.uniprot.org/uniprotkb/Q9BQC6/entry
    "MTERFD1": "MTERF3",  # https://www.uniprot.org/uniprotkb/Q96E29/entry
    "MYEOV2": "COPS9",  # https://www.uniprot.org/uniprotkb/Q8WXC6/entry
    "NDNL2": "NSMCE3",  # https://www.uniprot.org/uniprotkb/Q96MG7/entry
    "OBFC1": "STN1",  # https://www.uniprot.org/uniprotkb/Q9H668/entry
    "PET112": "GATB",  # https://www.uniprot.org/uniprotkb/O75879/entry
    "PPP2R4": "PTPA",  # https://www.uniprot.org/uniprotkb/Q15257/entry
    "PTPLB": "HACD2",  # https://www.uniprot.org/uniprotkb/Q6Y1H2/entry
    "SHFM1": "SEM1",  # https://www.uniprot.org/uniprotkb/P60896/entry
    "SKIV2L": "SKIC2",  # https://www.uniprot.org/uniprotkb/Q15477/entry
    "SMEK2": "PPP4R3B",  # https://www.uniprot.org/uniprotkb/Q5MIZ7/entry
    "STRA13": "CENPX",  # https://www.uniprot.org/uniprotkb/A8MT69/entry
    "SUV420H1": "KMT5B",  # https://www.uniprot.org/uniprotkb/Q4FZB7/entry
    "SUV420H2": "KMT5C",  # https://www.uniprot.org/uniprotkb/Q86Y97/entry
    "TMEM189.UBE2V1":
    "PEDS1",  # https://www.uniprot.org/uniprotkb/A5PLL7/entry
    "TMEM261": "DMAC1",  # https://www.uniprot.org/uniprotkb/Q96GE9/entry
    "UFD1L": "UFD1",  # https://www.uniprot.org/uniprotkb/Q92890/entry
    "WAPAL": "WAPL",  # https://www.uniprot.org/uniprotkb/Q7Z5K2/entry
    "WHSC1": "NSD2",  # https://www.uniprot.org/uniprotkb/O96028/entry
    "XRCC6BP1": "ATP23",  # https://www.uniprot.org/uniprotkb/Q9Y6H3/entry
    "YAE1D1": "YAE1",  # https://www.uniprot.org/uniprotkb/Q9NRH1/entry
}
# Non-coding genes (pseudogenes, lncRNAs, etc.) that will use placeholder sequences
# NOTE: These non-coding genes typically don't have UniProt entries as they don't encode proteins
NON_CODING_GENES = {
    # Pseudogenes (ending with P + number) - No UniProt entries (non-coding)
    "ANXA2P1",  # Pseudogene, no protein product
    "ATP1B1P1",  # Pseudogene, no protein product
    "ATP5EP1",  # Pseudogene, no protein product
    "BMS1P1",  # Pseudogene, no protein product
    "CLEC4GP1",  # Pseudogene, no protein product
    "CRYZP1",  # Pseudogene, no protein product
    "CSN1S2AP",  # Pseudogene, no protein product
    "CYP2F2P",  # Pseudogene, no protein product
    "GPR53P",  # Pseudogene, no protein product
    "HMGN1P2",  # Pseudogene, no protein product
    "HMGN2P11",  # Pseudogene, no protein product
    "HTR7P1",  # Pseudogene, no protein product
    "KRT88P",  # Pseudogene, no protein product
    "KRT89P",  # Pseudogene, no protein product
    "LDHBP2",  # Pseudogene, no protein product
    "MBL1P",  # Pseudogene, no protein product
    "MRPS36P4",  # Pseudogene, no protein product
    "OR5D2P",  # Pseudogene, no protein product
    "PCBP2P1",  # Pseudogene, no protein product
    "PHKBP2",  # Pseudogene, no protein product
    "PMS2P9",  # Pseudogene, no protein product
    "RLIMP3",  # Pseudogene, no protein product
    "RN7SL8P",  # Pseudogene, no protein product
    "RPL13AP17",  # Pseudogene, no protein product
    "RPL19P1",  # Pseudogene, no protein product
    "RPL37AP1",  # Pseudogene, no protein product
    "RPL7AP13",  # Pseudogene, no protein product
    "RSL24D1P6",  # Pseudogene, no protein product
    "TAS2R62P",  # Pseudogene, no protein product
    "WBP11P1",  # Pseudogene, no protein product
    "ZNF271P",  # Pseudogene, no protein product
    "POLR2J4",  # https://www.genecards.org/cgi-bin/carddisp.pl?gene=POLR2J4
    # Long non-coding RNAs (LINC) - No UniProt entries (non-coding)
    "LINC00029",  # lncRNA, no protein product
    "LINC00200",  # lncRNA, no protein product
    "LINC00452",  # lncRNA, no protein product
    "LINC00909",  # lncRNA, no protein product
    "LINC01003",  # lncRNA, no protein product
    "LINC01623",  # lncRNA, no protein product
    "LINC01743",  # lncRNA, no protein product
    "LINC02145",  # lncRNA, no protein product
    "LINC02288",  # lncRNA, no protein product
    "LINC02486",  # lncRNA, no protein product
    "C1ORF229",  # lncRNA (LINC02897), no protein product
    "C6ORF48",  # lncRNA, no protein product
    # Small nucleolar RNA host genes - No UniProt entries (non-coding)
    "SNHG1",  # snoRNA host gene, no protein product
    "SNHG4",  # snoRNA host gene, no protein product
    "SNHG7",  # snoRNA host gene, no protein product
    "SNORA25",  # Small nucleolar RNA, no protein product
    "SNORA72",  # Small nucleolar RNA, no protein product
    # Antisense RNAs - No UniProt entries (non-coding)
    "ASMTL-AS1",  # Antisense RNA, no protein product
    "NOP14-AS1",  # Antisense RNA, no protein product
    "TP73-AS1",  # Antisense RNA, no protein product
    # Intronic transcripts - No UniProt entries (non-coding)
    "GABPB1-IT1",  # Intronic transcript, no protein product
    "N4BP2L2-IT2",  # Intronic transcript, no protein product
    # Other non-coding - No UniProt entries
    "CERNA2",  # Competing endogenous RNA, no protein product
    "DLEU2",  # Deleted in lymphocytic leukemia 2 (non-coding), no protein product
    "IPW",  # Imprinted in Prader-Willi syndrome (non-coding), no protein product
    "NBR2",  # Neighbor of BRCA1 gene 2 (non-coding), no protein product
    "FAM41C",  # lncRNA without protein sequence, no protein product
    # HLA complex genes (often problematic)
    "HCG27",  # HLA complex group 27 (non-coding), no protein product
    "HCG9",  # HLA complex group 9 (non-coding), no protein product
    # Immunoglobulin genes
    "IGHD3-16",  # Immunoglobulin heavy diversity 3-16, no stable UniProt entry
    # Y chromosome (male-specific)
    "TTTY6",  # Testis-specific transcript Y-linked 6 (non-coding), no protein product
    # Not sure what it is, uniprot doesn't provide much information
    "C19orf48",  # Chromosome 19 open reading frame 48, limited info
    "C19ORF48",  # Alternative name for C19orf48, limited info
    "non-targeting",  # obvisouly...
}


def correct_gene_name(gene: str) -> str:
    """Apply gene name correction if needed."""
    return GENE_CORRECTIONS.get(gene, gene)


def is_non_coding(gene: str) -> bool:
    """Check if a gene is non-coding."""
    return gene in NON_CODING_GENES
