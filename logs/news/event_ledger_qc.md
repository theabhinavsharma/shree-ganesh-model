# Event ledger QC · 2026-10-02 17:28

rows 246,045 · symbols 2,931 · sessions 2016-01-01..2026-10-01

## Events by bucket and year

```
session                    2016  2017  2018  2019  2020  2021  2022  2023  2024   2025  2026
bucket                                                                                      
acquisition                4748  4429  3998  4154  2807  1573  1626  4588  5283   6164  5863
agreement                   229   189   223   227   174   269   443   463   697    987   798
approval_launch             153   225   195   233   177    95   164   234   237    330   232
auditor_exit                 11    20    22    14    47     8    17     8     3      8     4
bonus_split                 120   153   138   124    93   178   156   173   254    167    81
buyback                     117   126   192   167   149   159   169   119   103     33    71
capacity_start               86    81    74    70    43   124   176   145   262    556   515
clarification              1706  3726  4366  3840  2595  2660  1599  1639  1936   1799  1923
default_insolvency           35   194   535   740  1173  1365  1688  1382  1110    759   595
delayed_results               0     0    24   111    66    55     4    57    53     69    52
fund_raise                  476   631   498   462   449   653   469   823  1402   1267   887
independent_director_exit   210   284   376   670   462   428   819   778  1272   1029   628
merger_scheme               517   513   552   679   471   586   647   700   777    806   688
mgmt_exit                   174   170   238   258   282   277   356   374   427    564   479
order_cancel                  5    10     9    18     5     8     4     2    10     16    15
order_win                   264   221   271   173   149   189   234   294   643   1553  1472
price_movement_query        111   171   229  1232  2017  1124   895   644   787    474   700
rating_change               505   717   849  1121  1290  1285   984  1549  1859   2162  1842
rating_down                  11     1     3    19    16     0    35     0     0      8     1
rating_up                    58    31    42     9    17     5   246     0     3     31    30
regulatory_action            88    30    60    96    56   143   443   128   920   2751  1753
results                    7191  7580  7611  7420  7773  8125  4832  8399  8835  10998  7181
strike_disruption           134   114   121   116   517   122    14    67    65     31    33
suspension                   25    46    42    62    39    56    47    48    51     63    51
takeover_target             226   176   325   214   186   188   258   105   171    195   139
```

## Stability (events per 1,000 listed companies; only stable buckets may feed a test)

```
                           min_per_yr  max_over_min  max_yoy_factor  stable    2016    2017    2018    2019    2020    2021    2022    2023    2024    2025
bucket                                                                                                                                                     
acquisition                      1573          3.65            2.63   False  2802.8  2546.9  2275.5  2306.5  1521.4   786.9   767.3  2014.9  2217.9  2348.2
agreement                         174          3.99            1.55   False   135.2   108.7   126.9   126.0    94.3   134.6   209.1   203.3   292.6   376.0
approval_launch                    95          2.72            1.63   False    90.3   129.4   111.0   129.4    95.9    47.5    77.4   102.8    99.5   125.7
auditor_exit                        3         19.62            3.27   False     6.5    11.5    12.5     7.8    25.5     4.0     8.0     3.5     1.3     3.0
bonus_split                        93          2.12            1.77    True    70.8    88.0    78.5    68.9    50.4    89.0    73.6    76.0   106.6    63.6
buyback                            33          8.67            1.71   False    69.1    72.5   109.3    92.7    80.8    79.5    79.8    52.3    43.2    12.6
capacity_start                     43          9.09            2.66   False    50.8    46.6    42.1    38.9    23.3    62.0    83.1    63.7   110.0   211.8
clarification                    1599          3.63            2.13   False  1007.1  2142.6  2484.9  2132.1  1406.5  1330.7   754.6   719.8   812.8   685.3
default_insolvency                 35         38.48            5.39   False    20.7   111.6   304.5   410.9   635.8   682.8   796.6   606.9   466.0   289.1
delayed_results                     0           NaN             inf   False     0.0     0.0    13.7    61.6    35.8    27.5     1.9    25.0    22.3    26.3
fund_raise                        449          2.66            1.63   False   281.0   362.9   283.4   256.5   243.4   326.7   221.3   361.4   588.6   482.7
independent_director_exit         210          4.31            1.81   False   124.0   163.3   214.0   372.0   250.4   214.1   386.5   341.7   534.0   392.0
merger_scheme                     471          1.48            1.32    True   305.2   295.0   314.2   377.0   255.3   293.1   305.3   307.4   326.2   307.0
mgmt_exit                         170          2.20            1.39    True   102.7    97.8   135.5   143.3   152.8   138.6   168.0   164.3   179.3   214.9
order_cancel                        2         11.11            4.67   False     3.0     5.8     5.1    10.0     2.7     4.0     1.9     0.9     4.2     6.1
order_win                         149          7.32            2.19   False   155.8   127.1   154.2    96.1    80.8    94.5   110.4   129.1   269.9   591.6
price_movement_query              111         16.69            5.25   False    65.5    98.3   130.3   684.1  1093.2   562.3   422.4   282.8   330.4   180.6
rating_change                     505          2.76            1.46   False   298.1   412.3   483.2   622.4   699.2   642.8   464.4   680.3   780.4   823.6
rating_down                         0           NaN             inf   False     6.5     0.6     1.7    10.5     8.7     0.0    16.5     0.0     0.0     3.0
rating_up                           0           NaN             inf   False    34.2    17.8    23.9     5.0     9.2     2.5   116.1     0.0     1.3    11.8
regulatory_action                  30         60.58            6.87   False    51.9    17.3    34.1    53.3    30.4    71.5   209.1    56.2   386.2  1048.0
results                          4832          1.91            1.62    True  4245.0  4358.8  4331.8  4119.9  4213.0  4064.5  2280.3  3688.6  3709.1  4189.7
strike_disruption                  14         42.45            4.45   False    79.1    65.6    68.9    64.4   280.2    61.0     6.6    29.4    27.3    11.8
suspension                         25          2.32            1.79    True    14.8    26.5    23.9    34.4    21.1    28.0    22.2    21.1    21.4    24.0
takeover_target                   105          4.01            1.83   False   133.4   101.2   185.0   118.8   100.8    94.0   121.8    46.1    71.8    74.3
```

## Completeness vs NSE's own announcements feed (sample days; check_announcements_completeness.py)

```
{
 "2016": 0.993,
 "2017": 0.996,
 "2018": 0.992,
 "2019": 0.994,
 "2020": 0.998,
 "2021": 1.0,
 "2022": 1.0,
 "2023": 0.999,
 "2024": 1.001,
 "2025": 0.99,
 "2026": 0.991
}
```

## Results filings also posted as announcements (NOT a completeness measure: 2016-18 results often weren't)

```
      share
d          
2016  0.248
2017  0.121
2018  0.026
2019  0.600
2020  0.840
2021  0.956
2022  0.590
2023  0.974
2024  0.976
2025  0.949
2026  0.296
```

## Order amounts

order_win rows with an amount: 66.0%

## Examples (5 per bucket, random)

**acquisition**
- 2026-05-27 RBA: Restaurant Brands Asia Limited has Submitted to the Exchange a copy of Disclosure under Regulation 31(4) of the Securities and Exchange Board of India (Substant
- 2026-01-13 LT: Larsen & Toubro Limited has informed the Exchange regarding 'Acquisition of stake held by JV Partner in L&T Sapura Shipping Private Limited'.
- 2017-04-10 JYOTISTRUC: Kalpesh P Kikani has submitted to the Exchange a copy of disclosure under Regulation 30(1) and 30(2) of SEBI (Substantial Acquisition of Shares & Takeovers) Reg
- 2025-04-30 AVANTIFEED: Srinivasa Cystine Private Limited, Indra Kumar Alluri, Nuthakki Ram Prasad HUF has Submitted to the Exchange a copy of Disclosure under Regulation 31(4) of the 
- 2017-02-14 MUTHOOTFIN: Muthoot Finance Limited has informed the Exchange that Board of Directors of the Company in its meeting held on February 13, 2017, has decided to make an additi

**agreement**
- 2023-03-06 SOLEX: Pursuant to Regulation 30 of the Securities and Exchange Board of India (Listing Obligations and Disclosure Requirements) Regulations, 2015 read with Para B of 
- 2026-04-01 IREDA: Intimation under Regulation 30 of the SEBI LODR Regulations, 2015
- 2022-06-08 RITES: RITES Limited has informed the Exchange about press release dated 07-Jun-2022 titled RITES, Senegal firm sign MoU for cooperation in Rail sector
- 2026-03-18 NIPPOBATRY: Indo-National Limited has informed the Exchange about Agreements
- 2024-04-03 CHOICEIN: Choice International Limited has informed the Exchange regarding a press release dated April 02, 2024, titled "We are glad to announce that our wholly owned Sub

**approval_launch**
- 2023-03-20 UNICHEMLAB: Unichem Laboratories Limited has informed the Exchange regarding a press release dated March 17, 2023, titled "Press release pertaining to the receipt of ANDA a
- 2016-08-25 STAR: Strides Shasun Limited has informed the Exchange regarding a press release dated August 24, 2016, titled "Strides Shasun receives USFDA Approval for Ranitidine 
- 2022-01-21 ZYDUSLIFE: Cadila Healthcare Limited has informed the Exchange regarding a press release dated January 20, 2022, titled "Zydus receives final approval from USFDA for Vigab
- 2022-08-04 STAR: Strides Pharma Science Limited has informed the Exchange about press release dated 03-Aug-2022 titled Strides receives USFDA approval for Cyclosporine Softgel C
- 2025-08-05 TRITURBINE: Triveni Turbine Limited has informed the Exchange about Product launch

**auditor_exit**
- 2018-09-06 ELAND: E-Land Apparel Limited has informed the Exchange regarding Change in Auditors of the company.Pursuant to Regulation 30(7) read with Schedule III Part A Para A o
- 2018-11-02 AEGISLOG: Aegis Logistics Limited has informed the Exchange regarding Change in Auditors of the company.This is to inform you pursuant to the regulation 30 of SEBI (Listi
- 2020-11-18 SCHAND: SCHAND:The Exchange has sought clarification from S Chand And Company Limited with respect to announcement dated 13-Nov-2020, regarding " S Chand And Company Li
- 2020-07-23 COFFEEDAY: COFFEEDAY: The Exchange has sought clarification from Coffee Day Enterprises Limited with respect to announcement dated 18-Jul-2020, regarding change in Auditor
- 2026-05-13 STOVEKRAFT: Stove Kraft Limited has informed the Exchange about change in Management - Reappointment of Statutory Auditor, Resignation of CFO and appointment of new CFO

**bonus_split**
- 2026-04-28 KIRLPNU: Kirloskar Pneumatic Company Limited has informed the Exchange that the Board of Directors at its meeting held on April 27, 2026, has considered and approved sub
- 2026-08-06 PGIL: Pearl Global Industries Limited has informed the Exchange that the Board of Directors at its meeting held on August 05, 2026, have considered and approved bonus
- 2016-12-13 GULPOLY: Gulshan Polyols Limited has informed the Exchange that in the meeting share Allotment Committee of the Board of directors held on December 12, 2016 and Committe
- 2018-11-05 GUJGASLTD: Gujarat Gas Limited has informed the Exchange that the Board of Directors at its meeting held on November 03, 2018, has considered and approved subdivision of 1
- 2019-03-07 AIRAN: Airan Limited has informed the Exchange that the Board of Directors at its meeting held on March 06, 2019, have considered and approved bonus at the ratio of 1 

**buyback**
- 2023-01-27 KDDL: KDDL Limited has informed the Exchange Commencement of the Buyback
- 2024-02-13 ZYDUSLIFE: Zydus Lifesciences Limited has informed the Exchange about Buyback
- 2026-09-21 EMAMILTD: Emami Limited has informed the Exchange about public Announcement - Buyback of Shares
- 2016-12-06 EMBDL: Indiabulls Real Estate Limited has submitted to the Exchange a copy of the Public Announcement for Buy Back of Equity Shares of the Company.
- 2025-11-19 INFY: Infosys Limited has informed the Exchange about despatch of Letter of Offer

**capacity_start**
- 2024-07-05 DALBHARAT: Dalmia Bharat Limited has informed the Exchange regarding 'Commencement of commercial production at Kadapa, Andhra Pradesh'.
- 2025-04-11 EPACK: Pursuant to Regulation 30 of SEBI (LODR) Regulations, 2015, we are pleased to inform about the Capacity Addition.
- 2023-07-31 SHREECEM: SHREE CEMENT LIMITED has informed the Exchange regarding 'Commencement of Commercial Production '.
- 2024-04-22 STARCEMENT: Star Cement Limited has informed the Exchange regarding 'Commencement of Commercial Production from New Clinker Line of the Company at Lumshnong, Meghalaya'.
- 2023-04-17 PCBL: PCBL LIMITED has informed the Exchange regarding 'Commencement of commercial production of first phase ( 63,000 MT ) of 147,000 MT Greenfield carbon black manuf

**clarification**
- 2019-11-25 CGPOWER: The Exchange has sought clarification from CG Power and Industrial Solutions Limited with respect to recent news item captioned CG Power plans to raise Rs 800 c
- 2020-02-17 PEL: PEL:The Exchange has sought clarification from Piramal Enterprises Limited with respect to announcement dated 04-Feb-2020, regarding Resignation of Mr Siddharth
- 2017-10-26 YESBANK: The Exchange has sought clarification from Yes Bank Limited with respect to recent news item captioned "RBI slaps penalty on Yes Bank". In this regard, Exchange
- 2021-07-02 VERTOZ: The Exchange had sought clarification from Vertoz Advertising Limited for the quarter ended 31-Mar-2021 with respect to Regulation 33 of the SEBI (Listing Oblig
- 2019-11-07 BANARBEADS: The Exchange has sought clarification from Banaras Beads Limited for the quarter ended 30-Sep-2019 with respect to Regulation 33 of the SEBI (Listing Obligation

**default_insolvency**
- 2022-04-01 RELCAPITAL: Post facto intimation of 6th (Sixth) meeting of the Committee of Creditors (CoC) Reliance Capital Limited 
- 2025-09-16 ARSSINFRA: ARSS Infrastructure Projects Limited has informed the Exchange about Corporate Insolvency Resolution Process
- 2024-09-02 ARSSINFRA: ARSS Infrastructure Projects Limited has informed the Exchange about Corporate Insolvency Resolution Process
- 2024-01-20 JPINFRATEC: Jaypee Infratech Limited has informed the Exchange about Corporate Insolvency Resolution Process
- 2022-05-27 JPINFRATEC: Jaypee Infratech Limited has informed the Exchange about update on corporate insolvency resolution process - Next date of hearing at Hon'ble NCLT, Principal Ben

**delayed_results**
- 2024-08-16 VALUEIND: Value Industries Limited has informed the Exchange about reasons for Delayed/Non-submission of Financial Results for the quarter ended 30th June 2024
- 2019-09-04 EASUNREYRL: Easun Reyrolle Limited has informed the Exchange regarding Compliance as per Regulation 33 of the SEBI (Listing Obligations and Disclosure Requirements) Regulat
- 2020-08-24 EON: Eon Electric Limited has informed the Exchange about reasons for Delayed/Non-submission of Financial Results. Request for grant of extension of time for submiss
- 2020-11-17 ATLASCYCLE: Atlas Cycles (Haryana) Limited has informed the Exchange about reasons for Delayed/Non-submission of Financial Results
- 2024-02-19 VALUEIND: Value Industries Limited has informed the Exchange about reasons for Delayed/Non-submission of Financial Results for the quarter/ Half year ended 30th September

**fund_raise**
- 2016-04-01 DHFL: Dewan Housing Finance Corporation Limited has informed the Exchange that the Company proposes to issue 250 Secured Non-Convertible Redeemable Debentures with a 
- 2017-03-28 BANKINDIA: Bank Of India has informed the Exchange that  the Bank has raised Rs.1,000 Crore by issue of Basel-III compliant Tier-II Bonds (Series XIV) on March 27, 2017. T
- 2019-02-22 BAJFINANCE: Bajaj Finance Limited has informed the Exchange regarding Intimation of allotment of Secured Redeemable Non-Convertible Debentures on Private Placement basis.
- 2016-03-08 RAJSREESUG: Rajshree Sugars & Chemicals Limited has informed the Exchange that the company proposes to obtain shareholders approvals, by passing of special resolutions thro
- 2025-04-24 FUSION: Fusion Finance Limited has informed the Exchange about Copy of Newspaper Publication  of Corrigendum to LOF in respect of Rights issue

**independent_director_exit**
- 2019-08-13 TALWALKARS: Talwalkars Better Value Fitness Limited has informed the Exchange regarding Cessation of Mr MANOHAR BHIDE as Independent Director of the company w.e.f. August 0
- 2019-05-29 PREMIERPOL: Premier Polyfilm Limited has informed the Exchange regarding Resignation of Mr Ratnesh Kumar Gupta as Non- Executive Independent Director of the company w.e.f. 
- 2022-05-10 SRF: SRF Limited  has informed the Exchange about resignation of Vellayan Subbiah as Independent Director of the company w.e.f. 09-May-2022
- 2025-02-20 BOHRAIND: Bohra Industries Limited has informed the Exchange regarding Resignation of Ms KALPANA MEHTA as Non- Executive Independent Director of the company w.e.f. Februa
- 2018-05-07 HINDALCO: Hindalco Industries Limited has informed the Exchange regarding Resignation of Mr JAGDISH KHATTAR as Independent Director of the company w.e.f. May 04, 2018.

**merger_scheme**
- 2022-09-30 TEJASNET: Tejas Networks Limited  has informed the Exchange about Board Meeting held on 29-Sep-2022 to consider and approve Draft Scheme of Amalgamation
- 2025-09-19 SFL: Sheela Foam Limited has informed the Exchange about the Sanction of the Composite Scheme of Arrangement
- 2019-12-19 RUCHISOYA: Ruchi Soya Industries Limited has informed the Exchange regarding 'Closing date for the purpose of Implementation of Scheme of restructuring and Amalgamation un
- 2024-12-16 HMAAGRO: HMAAGRO:The Exchange had sought clarification from Hma Agro Industries Limited with respect to announcement dated 30-Sep-2024, regarding Board meeting held on S
- 2020-12-24 GALLANTT: Gallantt Metal Limited has informed the Exchange regarding 'Information that Company has successfully filed application with the Hon'ble National Company Law Tr

**mgmt_exit**
- 2017-09-05 ABFRL: Aditya Birla Fashion and Retail Limited has informed the Exchange regarding Resignation of Mr Shital Mehta as Chief Executive Officer of the company w.e.f. Sept
- 2018-11-27 MATRIMONY: Matrimony.Com Limited has informed the Exchange regarding Resignation of Mr K Balasubramanian as Chief Financial Officer of the company w.e.f. December 14, 2018
- 2024-08-02 NECCLTD: North Eastern Carrying Corporation Limited has informed the Exchange regarding Cessation of Mr Shyam Lal Yadav as Chief Financial Officer of the company w.e.f. 
- 2019-02-18 OPTIEMUS: Optiemus Infracom Limited has informed the Exchange regarding Resignation of Mr Anoop Singhal  as Chief Financial Officer of the company w.e.f. February 16, 201
- 2025-05-12 GRINDWELL: Grindwell Norton Limited has informed the Exchange regarding Cessation of Mr Hari Singudasu as Chief Financial Officer of the company w.e.f. May 09, 2025.

**order_cancel**
- 2026-05-22 RAILTEL: Railtel Corporation Of India Limited has informed the Exchange about cancellation of an order.
- 2025-06-02 PRSMJOHNSN: Prism Johnson Limited has informed the Exchange about Rescission/termination(s) - Termination of Captive Wind Power Project
- 2018-02-05 BLS: BLS:The Exchange had sought clarification from BLS International Services Limited with respect to announcement dated 30-Jan-2018, regarding " BLS International 
- 2026-08-28 RPPINFRA: R.P.P. Infra Projects Limited has informed the Exchange regarding 'Termination of Work Order from Sports Development Authority of Tamil Nadu'.
- 2019-07-02 NMDC: The Exchange has sought clarification from NMDC Limited with respect to news item captioned ��NMDC terminates contract with BHEL for delay in Rs 1,395 cr projec

**order_win**
- 2018-09-05 MBECL: Mcnally Bharat Engineering Company Limited has informed the Exchange regarding 'Others - Receipt of Order'. We are pleased to inform you that the Company has re
- 2021-12-09 RAILTEL: Railtel Corporation Of India Limited has informed the Exchange regarding 'Pursuant to Regulation 30 read with Part A (B) of Schedule III of SEBI (Listing Obliga
- 2026-05-13 VSTL: Vibhor Steel Tubes Limited has informed the Exchange about Bagging/Receiving of orders/contracts
- 2026-01-28 MARINE: Marine Electricals (India) Limited has informed the Exchange about Bagging/Receiving of orders/contracts
- 2026-04-01 JYOTISTRUC: Jyoti Structures Limited has informed the Exchange about Bagging/Receiving of orders/contracts

**price_movement_query**
- 2021-03-22 ADSL: Significant movement in price has been observed in Allied Digital Services Limited. The Exchange, in order to ensure that investors have latest relevant informa
- 2020-07-30 SHREEPUSHK: Significant movement in price has been observed in Shree Pushkar Chemicals & Fertilisers Limited. The Exchange, in order to ensure that investors have latest re
- 2022-08-18 SCHAND: SCHAND : Significant movement in price has been observed in S Chand And Company Limited. The Exchange, in order to ensure that investors have latest relevant in
- 2020-08-14 EMAMIREAL: Significant movement in price has been observed in Emami Realty Limited. The Exchange, in order to ensure that investors have latest relevant information about 
- 2020-06-10 DEN: Significant movement in price has been observed in Den Networks Limited. The Exchange, in order to ensure that investors have latest relevant information about 

**rating_change**
- 2026-09-15 EPL: EPL Limited has informed the Exchange about Credit Rating
- 2026-09-24 JSWINFRA: JSW Infrastructure Limited has informed the Exchange about revision in the credit rating of one of the subsidiaries- JSW Jaigarh Port Limited.
- 2024-10-15 STEELXIND: STEEL EXCHANGE INDIA LIMITED has informed the Exchange about Credit Rating
- 2024-12-26 MODISONLTD: MODISON LIMITED has informed the Exchange about Credit Rating- Reaffirmed
- 2022-01-14 HINDCOMPOS: Hindustan Composites Limited has informed the Exchange about Credit Rating

**rating_down**
- 2020-06-04 NTPC: NTPC Limited has informed the Exchange regarding Credit RatingRevision of Ratings assigned by Moody sThis is to inform that consequent upon Moody's Investors Se
- 2019-11-19 ZEEL: Zee Entertainment Enterprises Limited has informed the Exchange regarding Credit Rating Downgrade on Preference Shares with outstanding of Rs. 1210.16 Crores
- 2020-07-15 HEG: HEG Limited has informed the Exchange about Credit Rating:  India Ratings and Research (Ind-Ra) has downgraded HEG Limited''s (HEG) Long-Term Issuer Rating to '
- 2025-02-24 FUSION: Fusion Finance Limited has informed the Exchange about Credit Rating- Downgraded by ICRA
- 2022-12-01 TARC: TARC Limited has informed the Exchange about Downgrade of Credit Rating

**rating_up**
- 2022-03-10 ABCOTS: A B Cotspin India Limited has informed the Exchange about Upgrade of Credit Rating
- 2022-07-26 SHREERAMA: Shree Rama Multi-Tech Limited has informed the Exchange about Upgrade of Credit Rating by CRISIL Limited 
- 2016-08-22 BAJAJELEC: Bajaj Electricals Limited has informed the Exchange that ICRA Limited ( Rating Agency ) vide its letter dated August 18, 2016 has communicated that it has upgra
- 2026-04-27 VARROC: Varroc Engineering Limited has informed the Exchange about Credit Rating- Credit Rating Upgraded
- 2022-06-07 EVEREADY: Eveready Industries India Limited has informed the Exchange about Upgrade of Credit Rating

**regulatory_action**
- 2025-03-04 JKTYRE: JK Tyre & Industries Limited has informed the Exchange about Action(s) taken or orders passed
- 2025-10-01 RML: Rane (Madras) Limited has informed the Exchange about Pendency of Litigation(s)/dispute(s) or the outcome impacting the Company
- 2024-11-29 JSWDULUX: Akzo Nobel India Limited has informed the Exchange about Pendency of Litigation(s)/dispute(s) or the outcome impacting the Company
- 2024-09-27 SCHNEIDER: Schneider Electric Infrastructure Limited has informed the Exchange about Action(s) taken or orders passed
- 2026-01-22 AMBER: Amber Enterprises India Limited has informed that in furtherance to our earlier intimation dated 05th March 2025 and pursuant to Regulation 30(13) of SEBI (LODR

**results**
- 2019-02-15 VENUSREM: Venus Remedies Limited has submitted to the Exchange, the financial results for the period ended December 31, 2018.
- 2020-08-13 KINGFA: Kingfa Science & Technology (India) Limited has submitted to the Exchange, the financial results for the period ended June 30, 2020.
- 2020-02-11 PETRONET: Petronet LNG Limited has submitted to the Exchange, the financial results for the period ended December 31, 2019.
- 2026-02-09 RAJRILTD: Raj Rayon Industries Limited has submitted to the Exchange, the financial results for the period ended December 31, 2025.
- 2024-08-13 SOUTHWEST:  South West Pinnacle Exploration Limited has submitted to the Exchange, the financial results for the period ended June 30, 2024.

**strike_disruption**
- 2020-07-29 KECL: Kirloskar Electric Company Limited has informed the Exchange about strikes/lockouts/disturbances
- 2018-12-31 ORIENTBANK: Reg: Notice of Strike on 08th and 09th January, 2019.This is to inform that All India Bank Employees' Association (AIBEA) and Bank Employees Federation of India
- 2020-03-30 VSTIND: Novel Coronavirus (COVID-19) - Intimation under Regulation 30 of SEBI (Listing Obligations andDisclosure Requirements) Regulations, 2015 as amended
- 2023-12-11 BAFNAPH: Bafna Pharmaceuticals Limited has informed the Exchange about strikes/lockouts/disturbances
- 2020-03-30 SFL: Pursuant to Regulation 30 of SEBI (Listing Obligations and Disclosure Requirements), Regulations 2015, we would like to inform you that in view of the direction

**suspension**
- 2017-03-07 MYSOREBANK: MYSOREBANK: Members of the Exchange are hereby informed that the trading in Securities of State Bank of Mysore shall be suspended w.e.f. March 16, 2017 (i.e. cl
- 2024-08-05 ISMTLTD: Members of the Exchange are hereby informed that the trading in Equity Shares of ISMT Limited shall be suspended w.e.f. August 06, 2024 (i.e., closing hours of 
- 2021-06-29 SYNCOM: Members of the Exchange are hereby informed that trading in equity shares of Syncom Healthcare Limited (ISIN: INE602K01014) shall be suspended w.e.f. June 29, 2
- 2020-01-16 MANAPPURAM: MANAPPURAM: Members of the Exchange are hereby informed that the trading in Non-Convertible Debentures (NCDs) of Manappuram Finance Limited shall be suspended w
- 2017-02-07 SHRIRAMFIN: SRTRANSFIN: Members of the Exchange are hereby informed that the trading in Non-Convertible Debentures of Shriram Transport Finance Company Limited shall be sus

**takeover_target**
- 2020-10-05 ACCELYA: Accelya Solutions India Limited has informed the Exchange regarding 'Update on Open offer to the Public Shareholders of Accelya Solutions India Limited (the Tar
- 2026-06-18 TRU:  Sundae capital Advisors Private Ltd has submitted to  the Exchange a copy of Interim order of The Hon ble SAT in the matter of Open Offer given by Marwadi Chan
- 2019-06-19 SURANACORP: Members of the Exchange are hereby informed that the Exchange on June 06,2019 has issued public notice in the newspaper for proposed compulsory delisting of equ
- 2022-10-17 GTECJAINX: Navigant Corporate Advisors Limited has informed the Exchange regarding Letter of Offer of Keerti Knowledge and Skills Limited (Target Company) 
- 2018-03-01 ACROPETAL: Members of the Exchange are hereby informed that the Exchange on February 28, 2018 has issued public notice in the newspaper for�� proposed compulsory delisting
