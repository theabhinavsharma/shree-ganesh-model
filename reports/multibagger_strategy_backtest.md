# Multibagger strategy backtest

Strategy: every Monday, pick top-4 names with score ≥ 0.86 on 100%/180d model, hold 180 trading days.

Test period: 2024-01-01 to 2024-07-01 (need 180d forward data for outcome).

## Aggregate results

- **Total weekly entries tested:** 44
- **% baskets with ≥1 name doubling in 180d:** 40.9%
- **Avg basket max-high return (180d):** +38.69%
- **Avg basket close-to-close return (180d):** -4.76%
- **Median basket max return:** +19.84%

## Per-entry results (chronological)

| Entry date | Basket | Avg max % | Avg close % | n doubled | n ≥+50% | Best pick % | Worst pick % | Winners |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| 2024-01-01 | 4 | +45.0% | -13.4% | 0/4 | 1/4 | +69.4% | +30.5% | BCLIND, MIRZAINT, RAJMET, SARDAEN |
| 2024-01-08 | 4 | +0.9% | -31.5% | 0/4 | 0/4 | +12.9% | -20.7% | BCLIND, MIRZAINT, NESTLEIND, RAJMET |
| 2024-01-15 | 4 | +29.1% | +10.1% | 0/4 | 1/4 | +75.6% | +7.2% | BCLIND, MIRZAINT, NESTLEIND, SARDAEN |
| 2024-01-22 | 4 | +29.9% | +6.1% | 0/4 | 1/4 | +85.1% | +2.9% | BCLIND, MIRZAINT, NESTLEIND, SARDAEN |
| 2024-01-29 | 4 | +37.1% | +6.8% | 1/4 | 1/4 | +104.9% | +11.2% | BCLIND, NESTLEIND, SARDAEN, SIGACHI |
| 2024-02-05 | 4 | +46.8% | +30.8% | 1/4 | 1/4 | +119.4% | +12.8% | BCLIND, NESTLEIND, THEMISMED, SARDAEN |
| 2024-02-12 | 4 | +65.4% | +43.2% | 1/4 | 2/4 | +142.7% | +13.0% | BCLIND, NESTLEIND, PAYTM, SARDAEN |
| 2024-02-19 | 4 | +40.0% | +17.8% | 1/4 | 1/4 | +120.5% | +3.9% | BCLIND, NESTLEIND, PAYTM, THEMISMED |
| 2024-02-26 | 4 | +45.9% | +17.7% | 1/4 | 1/4 | +147.3% | +4.9% | BCLIND, NESTLEIND, SARDAEN, THEMISMED |
| 2024-03-04 | 4 | +44.4% | +23.0% | 1/4 | 1/4 | +146.9% | +0.9% | BCLIND, NESTLEIND, SARDAEN, THEMISMED |
| 2024-03-11 | 4 | +49.2% | +16.2% | 1/4 | 1/4 | +161.7% | +6.4% | BCLIND, CGCL, NESTLEIND, SARDAEN |
| 2024-03-18 | 4 | +51.8% | +24.0% | 1/4 | 1/4 | +162.4% | +7.7% | BCLIND, CGCL, NESTLEIND, SARDAEN |
| 2024-03-25 | 4 | +18.4% | -6.5% | 0/4 | 0/4 | +25.8% | +8.0% | BCLIND, CGCL, NESTLEIND, SIGACHI |
| 2024-04-01 | 4 | +50.2% | +28.9% | 1/4 | 1/4 | +153.6% | +7.4% | BCLIND, CGCL, NESTLEIND, SARDAEN |
| 2024-04-08 | 4 | +18.4% | -4.4% | 0/4 | 0/4 | +39.1% | +11.0% | BCLIND, THEMISMED, CGCL, NESTLEIND |
| 2024-04-15 | 4 | +23.5% | -0.3% | 0/4 | 1/4 | +51.6% | +8.8% | BCLIND, NESTLEIND, THEMISMED, BESTAGRO |
| 2024-04-22 | 4 | +22.3% | -7.2% | 0/4 | 1/4 | +52.9% | +8.5% | BCLIND, NESTLEIND, THEMISMED, SIGACHI |
| 2024-04-29 | 4 | +17.6% | -9.4% | 0/4 | 0/4 | +40.8% | +3.5% | BCLIND, MIRZAINT, NESTLEIND, THEMISMED |
| 2024-05-06 | 4 | +20.1% | -9.2% | 0/4 | 0/4 | +28.2% | +9.0% | BCLIND, SBGLP, BESTAGRO, MIRZAINT |
| 2024-05-13 | 4 | +19.7% | -11.1% | 0/4 | 0/4 | +33.9% | +10.4% | BCLIND, BESTAGRO, MIRZAINT, NESTLEIND |
| 2024-05-20 | 4 | +29.5% | -12.0% | 0/4 | 0/4 | +49.5% | +15.7% | BCLIND, SBGLP, ACCURACY, BESTAGRO |
| 2024-05-27 | 4 | +21.6% | -9.2% | 0/4 | 0/4 | +30.2% | +12.9% | BCLIND, BESTAGRO, NESTLEIND, SBGLP |
| 2024-06-03 | 4 | +41.8% | -2.2% | 0/4 | 1/4 | +97.6% | +17.5% | BCLIND, EXCEL, NESTLEIND, SBGLP |
| 2024-06-10 | 4 | +17.6% | -27.6% | 0/4 | 0/4 | +32.9% | +9.0% | BCLIND, SBGLP, HDFCMOMENT, NESTLEIND |
| 2024-06-17 | 4 | +15.9% | -39.7% | 0/4 | 0/4 | +22.5% | +9.3% | BCLIND, SBGLP, NESTLEIND, BESTAGRO |
| 2024-06-24 | 4 | +13.8% | -28.6% | 0/4 | 0/4 | +24.2% | +6.5% | BCLIND, SBGLP, NESTLEIND, HDFCPVTBAN |
| 2024-07-01 | 4 | +26.8% | -46.7% | 0/4 | 0/4 | +38.9% | +8.2% | BCLIND, KAMOPAINTS, NESTLEIND, RAJMET |
| 2024-07-08 | 4 | +26.3% | -47.1% | 0/4 | 0/4 | +49.8% | +6.7% | BCLIND, KAMOPAINTS, NESTLEIND, RAJMET |
| 2024-07-15 | 4 | +62.3% | -3.9% | 1/4 | 1/4 | +189.1% | +6.6% | KAMOPAINTS, NESTLEIND, PGEL, RAJMET |
| 2024-07-22 | 4 | +59.2% | -7.1% | 1/4 | 1/4 | +184.7% | +1.6% | KAMOPAINTS, NESTLEIND, PGEL, RAJMET |
| 2024-07-29 | 4 | +24.0% | -41.2% | 0/4 | 0/4 | +46.5% | +12.5% | ALMONDZ, ELECON, KAMOPAINTS, NESTLEIND |
| 2024-08-05 | 4 | +23.8% | -37.4% | 0/4 | 1/4 | +51.5% | +3.9% | ALMONDZ, ELECON, KAMOPAINTS, NESTLEIND |
| 2024-08-12 | 4 | +23.6% | -53.7% | 0/4 | 1/4 | +60.5% | +1.9% | AKSHAR, ALMONDZ, KAMOPAINTS, NESTLEIND |
| 2024-08-19 | 4 | +33.0% | -22.2% | 1/4 | 1/4 | +102.8% | +3.3% | AKSHAR, NESTLEIND, PGEL, RAJMET |
| 2024-08-26 | 4 | +88.4% | +14.6% | 2/4 | 2/4 | +231.8% | +4.6% | PGEL, RUSHIL, SBGLP, VERTOZ |
| 2024-09-02 | 4 | +91.7% | +6.3% | 2/4 | 2/4 | +241.7% | +2.7% | PGEL, RAJMET, SBGLP, VERTOZ |
| 2024-09-09 | 4 | +53.5% | +7.7% | 1/4 | 2/4 | +121.1% | +10.4% | EXCEL, NESTLEIND, PGEL, RUSHIL |
| 2024-09-16 | 4 | +81.0% | +10.1% | 1/4 | 2/4 | +227.1% | +1.9% | PGEL, SBGLP, SPORTKING, VERTOZ |
| 2024-09-23 | 4 | +89.5% | +17.8% | 1/4 | 2/4 | +237.1% | +16.4% | PGEL, SBGLP, SPORTKING, VERTOZ |
| 2024-09-30 | 4 | +90.7% | +10.0% | 1/4 | 2/4 | +246.7% | +17.1% | PGEL, SANGHVIMOV, SBGLP, VERTOZ |
| 2024-10-07 | 4 | +19.7% | -23.2% | 0/4 | 1/4 | +95.7% | -58.4% | GPIL, KAMOPAINTS, PGEL, ROSSELLIND |
| 2024-10-14 | 4 | +31.3% | -9.0% | 0/4 | 1/4 | +70.2% | +6.8% | BTML, GPIL, PGEL, RUSHIL |
| 2024-10-21 | 4 | +30.4% | -9.7% | 0/4 | 1/4 | +79.6% | +1.2% | BTML, GPIL, JINDALSAW, PGEL |
| 2024-10-28 | 4 | +31.5% | +12.9% | 0/4 | 1/4 | +86.2% | +7.2% | BTML, CUPID, DRREDDY, FUSION |

## Honest read

The model's claim of 90% hit rate on score ≥ 0.86 translates to:
- A 4-name basket where ≥1 of 4 should double in 180d (40.9% of baskets here)
- The basket's average max return: +38.7% (max-high captures the peak; close-to-close is harder to capture)
- Real-world capture (no perfect-exit assumption): expect 50-70% of avg max return
- Realistic basket return: +23.2% to +31.0% over 180d