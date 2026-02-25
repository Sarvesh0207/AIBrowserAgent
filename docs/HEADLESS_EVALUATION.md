# Headless Mode Evaluation

This document summarizes which websites work or fail when the agent runs in **headless** mode (no visible browser window).

**Source report:** `headless_report_20260223T223238Z.csv`
**Sites tested:** 10 | **Passed:** 9 | **Failed:** 1

---

## Results Table

| URL | Works in headless? | Error / Notes |
|-----|--------------------|---------------|
| https://example.com | Yes |  |
| https://www.wikipedia.org/ | Yes |  |
| https://www.uic.edu/ | Yes |  |
| https://www.python.org/ | Yes |  |
| https://www.cnn.com/ | Yes |  |
| https://www.cccis.com/ | Yes |  |
| https://www.aloyoga.com/?utm_medium=cpc&utm_source=google&utm_campaign=PPC_GOOG_Branded_US_Search&utm_term=alo&utm_content=20375582717&utm_id=261250877&gad_source=1&gad_campaignid=261250877&gbraid=0AAAAADkcv-OBjLIPOReogrg2TCx46xVjx&gclid=CjwKCAiAkvDMBhBMEiwAnUA9BSVsTjXTrV9jg3MtSgPSC3sN6YMJmYVmfFz2JW0e0frRrH7DqnxrixoCVbgQAvD_BwE | Yes |  |
| https://www.stanley1913.com/?nbt=nb%3Aadwords%3Ag%3A22798611099%3A179539736821%3A778045706833&nb_adtype=&nb_kwd=stanley&nb_ti=kwd-12627822&nb_mi=&nb_pc=&nb_pi=&nb_ppi=&nb_placement=&nb_li_ms=&nb_lp_ms=&nb_fii=&nb_ap=&nb_mt=e&gad_source=1&gad_campaignid=22798611099&gbraid=0AAAAADDIE2LRwbYHNhCvtS-_9zv6kBFQq&gclid=CjwKCAiAkvDMBhBMEiwAnUA9BZN2eeKGHobaaPXLjMp1GM3siRs_WSIn3nLnZNvmebWyzvFR3-JwaxoCjKIQAvD_BwE | Yes |  |
| https://www.ikea.com/us/en/?cid=a1:ps%7Ca2:se%7Ca3:US_LC_A3_ConsumerLed_Search_AO_L1_Google_Core_EN_Search_Brand_HFBMUL_0_Combo%7Ca4:ikea%7Ca5:e%7Ca6:google%7Ca7:cq%7Cid:IKEA%20Branded%20GM%7Ccc:915&gad_source=1&gad_campaignid=11008197745&gbraid=0AAAAAD27g7zkzi3sk5rkI5tbBmX25DHId&gclid=CjwKCAiAkvDMBhBMEiwAnUA9BR7TUaPIQvD6sCLRzgEeHnLjMz_Xua07Y6LG_Fcx0_3uMcL9y53RNxoCTHIQAvD_BwE | Yes |  |
| https://www.dyson.com/deals?ef_id=CjwKCAiAkvDMBhBMEiwAnUA9Bc_Rj7oUIHPrwKkunnu7KG7W9HbndCzso02HryAdpN9mvwYa_3jZbRoC1RkQAvD_BwE:G:s&utm_id=sa_19927058335_150274776920&utm_source=google&utm_medium=cpc&utm_campaign=cc_cross-category-range_always-on&utm_content=do_text&utm_term=dyson&gclsrc=aw.ds&gad_source=1&gad_campaignid=19927058335&gbraid=0AAAAAob0X4bWet5X97vZTMyZQfdSx9hHG&gclid=CjwKCAiAkvDMBhBMEiwAnUA9Bc_Rj7oUIHPrwKkunnu7KG7W9HbndCzso02HryAdpN9mvwYa_3jZbRoC1RkQAvD_BwE | No | Error('Page.goto: net::ERR_HTTP2_PROTOCOL_ERROR at https://www.dyson.com/deals?ef_id=CjwKCAiAkvDMBhBMEiwAnUA9Bc_Rj7oUIHP... |

---

## Summary

- **Passed:** 9 sites completed (browse + summarize) without errors.
- **Failed:** 1 sites raised an error (e.g. timeout, blocked, HTTP/2 error).

## Categorization / Findings (edit manually)

You can add notes here after reviewing the table, for example:

- Sites that **always work**: e.g. simple static pages, documentation.
- Sites that **often fail**: e.g. e‑commerce (bot blocking), heavy JavaScript, login walls.
- Common error types: `ERR_HTTP2_PROTOCOL_ERROR`, timeout, redirect loops.
